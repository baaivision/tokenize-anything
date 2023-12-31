# ------------------------------------------------------------------------
# Copyright (c) 2023-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Evaluate bare segmentation using detection boxes."""

import argparse
import collections
import datetime
import json
import multiprocessing as mp
import os

import cv2
import numpy as np
import torch

from tokenize_anything import engine
from tokenize_anything.utils.image import im_rescale
from tokenize_anything.utils.image import im_vstack
from tokenize_anything.utils.mask import encode_masks
from tokenize_anything.utils.profiler import Timer


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Evaluate bare segmentation.")
    parser.add_argument("--model-type", type=str, required=True, help="Model type.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint.")
    parser.add_argument("--images-dir", type=str, required=True, help="Path of images folder.")
    parser.add_argument("--det-json-file", type=str, required=True, help="Detection json file.")
    parser.add_argument("--gt-json-file", type=str, required=True, help="Ground-truth json file.")
    parser.add_argument("--read-every", type=int, default=100, help="Read every-n images")
    parser.add_argument("--prompt-size", type=int, default=256, help="Maximum prompts per batch")
    parser.add_argument("--device", nargs="+", type=int, default=[0], help="Index of devices.")
    return parser.parse_args()


class Predictor(object):
    """Predictor."""

    def __init__(self, model, kwargs):
        self.model = model
        self.kwargs = kwargs
        self.prompt_size = kwargs.get("prompt_size", 256)
        self.timers = collections.defaultdict(Timer)

    def batch_iterator(self, *args):
        iters = len(args[0]) // self.prompt_size + (len(args[0]) % self.prompt_size != 0)
        for i in range(iters):
            yield [arg[i * self.prompt_size : (i + 1) * self.prompt_size] for arg in args]

    def preprocess_images(self, imgs):
        """Preprocess the inference images."""
        im_batch, im_shapes, im_scales = [], [], []
        for img in imgs:
            scaled_imgs, scales = im_rescale(img, scales=[1024])
            im_batch.__iadd__(scaled_imgs), im_scales.__iadd__(scales)
            im_shapes += [x.shape[:2] for x in scaled_imgs]
        im_batch = im_vstack(im_batch, self.model.pixel_mean_value, size=(1024, 1024))
        im_shapes = np.array(im_shapes)
        im_scales = np.array(im_scales).reshape((len(im_batch), -1))
        im_info = np.hstack([im_shapes, im_scales]).astype("float32")
        return im_batch, im_info

    @torch.inference_mode()
    def get_results(self, examples):
        """Return the inference results."""
        # Preprocess images and prompts.
        imgs = [example["img"] for example in examples]
        boxes = [example.get("boxes", None) for example in examples]
        boxes = np.concatenate(boxes) if len(boxes) > 1 else boxes[0]
        if boxes is None or len(boxes) == 0:
            return [{"boxes": None, "masks": None}]
        im_batch, im_info = self.preprocess_images(imgs)
        box_labels = np.array([2, 3], "float32").reshape((1, 2, 1))
        box_labels = np.tile(box_labels, (len(boxes), 1, 1))
        box_points = boxes[:, :4].reshape((-1, 2, 2))
        box_points = np.concatenate([box_points, box_labels], -1)
        box_points[:, :, :2] *= im_info[:, None, 2:4]
        # Predict tokens and upscale masks.
        self.timers["im_process"].tic()
        inputs = self.model.get_inputs({"img": im_batch})
        inputs.update(self.model.get_features(inputs))
        input_size = im_info[0, :2].astype("int")
        data = collections.defaultdict(list)
        for (points,) in self.batch_iterator(box_points):
            outputs = self.model.get_outputs(dict(**inputs, **{"points": points}))
            mask_pred = outputs["mask_pred"][:, 0:1]
            mask_pred = self.model.upscale_masks(mask_pred, im_batch.shape[1:-1])
            masks = mask_pred[:, :, : input_size[0], : input_size[1]]
            data["masks"].append(masks.flatten(0, 1))
        # Upscale masks to the original image resolution.
        masks = torch.cat(data["masks"])[:, None]
        masks = self.model.upscale_masks(masks, imgs[0].shape[:2])[:, 0]
        masks = masks.gt(0).cpu().numpy()
        self.timers["im_process"].toc(n=len(imgs))
        # Encode masks.
        self.timers["misc"].tic()
        masks = encode_masks(masks.transpose((1, 2, 0)))
        outputs = {"boxes": boxes, "masks": masks}
        self.timers["misc"].toc(n=len(imgs))
        return [outputs]


def main(args):
    # Prepare dataset.
    with open(args.gt_json_file, "r") as f:
        json_dataset = json.load(f)
    with open(args.det_json_file, "r") as f:
        det_results = json.load(f)
    if "annotations" in det_results:
        det_results = det_results["annotations"]
    img_list, img_recs = [], collections.defaultdict(list)
    for info in json_dataset["images"]:
        file_name = info["coco_url"].split("/")[-1]
        images_dir = args.images_dir
        if "train" in info["coco_url"]:
            images_dir = images_dir.replace("val2017", "train2017")
        img_list.append((info["id"], os.path.join(images_dir, file_name)))
        assert os.path.exists(img_list[-1][1])
    for res in det_results:
        x, y, w, h = res["bbox"]
        res["bbox_xyxy"] = [x, y, x + w, y + h]
        img_recs[res["image_id"]].append(res)
    print("%d instances in %d images." % (len(det_results), len(img_list)))

    # Build environment.
    num_images = len(img_list)
    num_devices = len(args.device)
    read_every = int(np.ceil(args.read_every / num_devices) * num_devices)
    queues = [mp.Queue() for _ in range(num_devices + 1)]
    commands = [
        engine.InferenceCommand(
            queues[i],
            queues[-1],
            kwargs={
                "model_type": args.model_type,
                "weights": args.checkpoint,
                "prompt_size": args.prompt_size,
                "device": args.device[i],
                "predictor_type": Predictor,
                "verbose": i == 0,
            },
        )
        for i in range(num_devices)
    ]
    actors = [mp.Process(target=command.run, daemon=True) for command in commands]
    for actor in actors:
        actor.start()

    # Collect results.
    all_boxes, all_masks = [], []
    timers = collections.defaultdict(Timer)
    for count in range(1, len(img_list) + 1):
        img_id, img_path = img_list[count - 1]
        inputs = {"img": cv2.imread(img_path)}
        boxes = [x["bbox_xyxy"] + [x["score"], x["category_id"]] for x in img_recs[img_id]]
        inputs["boxes"] = np.array(boxes, "float32") if len(boxes) > 0 else None
        queues[count % num_devices].put((count - 1, inputs))
        if count % read_every > 0 and count < num_images:
            continue
        if count == num_images:
            for i in range(num_devices):
                queues[i].put((-1, None))
        for _ in range(((count - 1) % read_every + 1)):
            index, time_diffs, outputs = queues[-1].get()
            for _ in range(index - len(all_boxes) + 1):
                all_boxes.append([])
                all_masks.append([])
            all_boxes[index] = outputs["boxes"]
            all_masks[index] = outputs["masks"]
            for name, diff in time_diffs.items():
                timers[name].add_diff(diff)
        avg_time = sum([t.average_time for t in timers.values()])
        eta_seconds = avg_time * (num_images - count)
        print(
            "\rim_process: {:d}/{:d} [{:.3f}s + {:.3f}s] (eta: {})".format(
                count,
                num_images,
                timers["im_process"].average_time,
                timers["misc"].average_time,
                str(datetime.timedelta(seconds=int(eta_seconds))),
            ),
            end="",
        )

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../outputs")
    prefix = "coco_" if "coco" in args.gt_json_file else "lvis_"
    segm_res_file = os.path.join(output_dir, prefix + "segmentations.json")
    print("\nWriting segmentations to {}".format(segm_res_file))
    results = []
    for i, (img_id, _) in enumerate(img_list):
        boxes, masks = all_boxes[i], all_masks[i]
        if boxes is None:
            continue
        results += [
            {
                "image_id": img_id,
                "category_id": int(boxes[j, 5]),
                "segmentation": masks[j],
                "score": float(boxes[j, 4]),
            }
            for j in range(boxes.shape[0])
        ]
    os.makedirs(output_dir, exist_ok=True)
    with open(segm_res_file, "w") as fid:
        json.dump(results, fid)

    if "coco" in args.gt_json_file:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        print("\nEvaluating COCO segmentations...")
        coco_Gt = COCO(args.gt_json_file)
        coco_Dt = coco_Gt.loadRes(segm_res_file)
        coco_eval = COCOeval(coco_Gt, coco_Dt, "segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        print("Summary:")
        coco_eval.summarize()
    elif "lvis" in args.gt_json_file:
        from lvis import LVIS
        from lvis import LVISEval
        from lvis import LVISResults

        print("\nEvaluating LVIS segmentations...")
        lvis_Gt = LVIS(args.gt_json_file)
        lvis_Dt = LVISResults(lvis_Gt, segm_res_file)
        lvis_eval = LVISEval(lvis_Gt, lvis_Dt, "segm")
        lvis_eval.run()
        print("Summary:")
        lvis_eval.print_results()


if __name__ == "__main__":
    main(parse_args())

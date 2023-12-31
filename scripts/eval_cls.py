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
"""Evalute bare classification using ground-truth boxes."""

import argparse
import collections
import datetime
import json
import multiprocessing as mp
import os

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from tokenize_anything import engine
from tokenize_anything.utils.image import im_rescale
from tokenize_anything.utils.image import im_vstack
from tokenize_anything.utils.profiler import Timer


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Evaluate bare segmentation.")
    parser.add_argument("--model-type", type=str, required=True, help="Model type.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint.")
    parser.add_argument("--concept", type=str, required=True, help="Concept weights.")
    parser.add_argument("--images-dir", type=str, required=True, help="Path of images folder.")
    parser.add_argument("--gt-json-file", type=str, required=True, help="Ground-truth json file.")
    parser.add_argument("--max-dets", type=int, default=300, help="Maximum detections per images.")
    parser.add_argument("--read-every", type=int, default=100, help="Read every-n images")
    parser.add_argument("--prompt-size", type=int, default=256, help="Maximum prompts per batch")
    parser.add_argument("--device", nargs="+", type=int, default=[0], help="Index of devices.")
    return parser.parse_args()


def filter_outputs(outputs, max_dets=300):
    """Limit the max number of detections."""
    if max_dets <= 0:
        return outputs
    boxes = outputs.pop("boxes")
    scores, num_classes = [], len(boxes)
    for i in range(num_classes):
        if len(boxes[i]) > 0:
            scores.append(boxes[i][:, -1])
    scores = np.hstack(scores) if len(scores) > 0 else []
    if len(scores) > max_dets:
        thresh = np.sort(scores)[-max_dets]
        for i in range(num_classes):
            if len(boxes[i]) < 1:
                continue
            keep = np.where(boxes[i][:, -1] >= thresh)[0]
            boxes[i] = boxes[i][keep]
    outputs["boxes"] = boxes
    return outputs


def extend_results(index, collection, results, begin=0):
    """Add image results to the collection."""
    if results is None:
        return
    for _ in range(len(results) - len(collection)):
        collection.append([])
    for i in range(begin, len(results)):
        for _ in range(index - len(collection[i]) + 1):
            collection[i].append([])
        collection[i][index] = results[i]


class Predictor(object):
    """Predictor."""

    def __init__(self, model, kwargs):
        self.model = model
        self.kwargs = kwargs
        self.prompt_size = kwargs.get("prompt_size", 256)
        self.max_dets = kwargs.get("max_dets", 300)
        self.nms_thresh = kwargs.get("nms_thresh", 0.5)
        self.model.concept_projector.reset_weights(kwargs["concept_weights"])
        self.num_classes = len(self.model.concept_projector.concepts)
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
            return [{"boxes": None}]
        im_batch, im_info = self.preprocess_images(imgs)
        box_labels = np.array([2, 3], "float32").reshape((1, 2, 1))
        box_labels = np.tile(box_labels, (len(boxes), 1, 1))
        box_points = boxes[:, :4].reshape((-1, 2, 2))
        box_points = np.concatenate([box_points, box_labels], -1)
        box_points[:, :, :2] *= im_info[:, None, 2:4]
        # Predict tokens.
        self.timers["im_process"].tic()
        inputs = self.model.get_inputs({"img": im_batch})
        inputs.update(self.model.get_features(inputs))
        data = collections.defaultdict(list)
        for (points,) in self.batch_iterator(box_points):
            outputs = self.model.get_outputs(dict(**inputs, **{"points": points}))
            data["sem_embeds"].append(outputs["sem_embeds"][:, 0])
            data["iou_scores"].append(outputs["iou_pred"][:, 0].cpu().numpy())
        # Split categorical results.
        sem_embeds = torch.cat(data["sem_embeds"])
        iou_scores = np.concatenate(data["iou_scores"])
        cls_prob = self.model.concept_projector.decode(sem_embeds, return_prob=True)
        box_ind, cls_ind = np.where(cls_prob > 0.001)
        scores = cls_prob[box_ind, cls_ind, None] * np.clip(iou_scores[box_ind], 0, 1)
        dets = np.concatenate([boxes[box_ind], np.sqrt(scores)], axis=1)
        self.timers["im_process"].toc(n=len(imgs))
        self.timers["misc"].tic()
        cls_boxes = [[]] * self.num_classes
        valid_classes = np.unique(cls_ind)
        for j in valid_classes:
            inds = np.where(cls_ind == j)[0]
            cls_dets = dets[inds]
            cls_dets_tensor = torch.from_numpy(cls_dets)
            keep = nms(cls_dets_tensor[:, :4], cls_dets_tensor[:, 4], self.nms_thresh).numpy()
            cls_boxes[j] = cls_dets[keep]
        outputs = {"boxes": cls_boxes}
        outputs = filter_outputs(outputs, self.max_dets)
        self.timers["misc"].toc(n=len(imgs))
        return [outputs]


def main(args):
    # Prepare dataset.
    with open(args.gt_json_file, "r") as f:
        json_dataset = json.load(f)
    det_results = json_dataset["annotations"]
    img_list, img_recs = [], collections.defaultdict(list)
    cls2cat = dict((cls_ind, info["id"]) for cls_ind, info in enumerate(json_dataset["categories"]))
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
                "concept_weights": args.concept,
                "prompt_size": args.prompt_size,
                "max_dets": args.max_dets,
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
    all_boxes = []
    timers = collections.defaultdict(Timer)
    for count in range(1, len(img_list) + 1):
        img_id, img_path = img_list[count - 1]
        inputs = {"img": cv2.imread(img_path)}
        boxes = [x["bbox_xyxy"] for x in img_recs[img_id]]
        inputs["boxes"] = np.array(boxes, "float32") if len(boxes) > 0 else None
        queues[count % num_devices].put((count - 1, inputs))
        if count % read_every > 0 and count < num_images:
            continue
        if count == num_images:
            for i in range(num_devices):
                queues[i].put((-1, None))
        for _ in range(((count - 1) % read_every + 1)):
            index, time_diffs, outputs = queues[-1].get()
            extend_results(index, all_boxes, outputs["boxes"])
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
    segm_res_file = os.path.join(output_dir, prefix + "detections.json")
    print("\nWriting detections to {}".format(segm_res_file))
    results = []
    for cls_ind, boxes in enumerate(all_boxes):
        for i, (img_id, _) in enumerate(img_list):
            if len(boxes[i]) == 0:
                continue
            dets = boxes[i].astype("float64")
            xs, ys = dets[:, 0], dets[:, 1]
            ws, hs = dets[:, 2] - xs, dets[:, 3] - ys
            scores = dets[:, -1]
            results += [
                {
                    "image_id": img_id,
                    "category_id": cls2cat[cls_ind],
                    "bbox": [xs[j], ys[j], ws[j], hs[j]],
                    "score": scores[j],
                }
                for j in range(dets.shape[0])
            ]
    os.makedirs(output_dir, exist_ok=True)
    with open(segm_res_file, "w") as fid:
        json.dump(results, fid)

    if "coco" in args.gt_json_file:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        print("\nEvaluating COCO detections...")
        coco_Gt = COCO(args.gt_json_file)
        coco_Dt = coco_Gt.loadRes(segm_res_file)
        coco_eval = COCOeval(coco_Gt, coco_Dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        print("Summary:")
        coco_eval.summarize()
    elif "lvis" in args.gt_json_file:
        from lvis import LVIS
        from lvis import LVISEval
        from lvis import LVISResults

        print("\nEvaluating LVIS detections...")
        lvis_Gt = LVIS(args.gt_json_file)
        lvis_Dt = LVISResults(lvis_Gt, segm_res_file)
        lvis_eval = LVISEval(lvis_Gt, lvis_Dt, "bbox")
        lvis_eval.run()
        print("Summary:")
        lvis_eval.print_results()


if __name__ == "__main__":
    main(parse_args())

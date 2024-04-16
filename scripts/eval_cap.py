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
"""Evaluate region caption using ground-truth boxes."""

import argparse
import collections
import datetime
import json
import multiprocessing as mp
import os

import cv2
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import torch

from tokenize_anything import engine
from tokenize_anything.utils.image import im_rescale
from tokenize_anything.utils.image import im_vstack
from tokenize_anything.utils.profiler import Timer


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Evaluate region caption")
    parser.add_argument("--model-type", type=str, help="Model type")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint")
    parser.add_argument("--images-dir", type=str, help="Path of images folder")
    parser.add_argument("--gt-json-file", type=str, help="Ground-truth json file")
    parser.add_argument("--read-every", type=int, default=100, help="Read every-n images")
    parser.add_argument("--prompt-size", type=int, default=256, help="Maximum prompts per batch")
    parser.add_argument("--device", nargs="+", type=int, default=[0], help="Index of devices")
    return parser.parse_args()


class Predictor(object):
    """Predictor."""

    def __init__(self, model, kwargs):
        self.model = model
        self.kwargs = kwargs
        self.prompt_size = kwargs.get("prompt_size", 256)
        self.model.text_decoder.reset_cache(max_batch_size=self.prompt_size)
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
        if boxes is None:
            return [{"captions": None}]
        im_batch, im_info = self.preprocess_images(imgs)
        box_labels = np.array([2, 3], "float32").reshape((1, 2, 1))
        box_labels = np.tile(box_labels, (len(boxes), 1, 1))
        box_points = boxes[:, :4].reshape((-1, 2, 2))
        box_points = np.concatenate([box_points, box_labels], -1)
        box_points[:, :, :2] *= im_info[:, None, 2:4]
        # Predict tokens and generate captions.
        self.timers["im_process"].tic()
        inputs = self.model.get_inputs({"img": im_batch})
        inputs.update(self.model.get_features(inputs))
        data = collections.defaultdict(list)
        for (points,) in self.batch_iterator(box_points):
            outputs = self.model.get_outputs(dict(**inputs, **{"points": points}))
            data["texts"].append(self.model.generate_text(outputs["sem_tokens"][:, 0]))
        self.timers["im_process"].toc(n=len(imgs))
        outputs = {"captions": np.concatenate(data["texts"])}
        return [outputs]


def main(args):
    # Prepare dataset.
    with open(args.gt_json_file, "r") as f:
        json_dataset = json.load(f)
    img_list = [(x["id"], x["file_name"]) for x in json_dataset["images"]]
    img_list = [(x[0], os.path.abspath(os.path.join(args.images_dir, x[1]))) for x in img_list]
    img_recs = collections.defaultdict(list)
    for ann in json_dataset["annotations"]:
        x, y, w, h = ann["bbox"]
        ann["bbox_xyxy"] = [x, y, x + w, y + h]
        img_recs[ann["image_id"]].append(ann)
    print("%d instances in %d images." % (len(json_dataset["annotations"]), len(img_list)))

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
    all_captions = []
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
            for _ in range(index - len(all_captions) + 1):
                all_captions.append([])
            all_captions[index] = outputs["captions"]
            for name, diff in time_diffs.items():
                timers[name].add_diff(diff)
        avg_time = sum([t.average_time for t in timers.values()])
        eta_seconds = avg_time * (num_images - count)
        print(
            "\rim_process: {:d}/{:d} [{:.3f}s] (eta: {})".format(
                count,
                num_images,
                timers["im_process"].average_time,
                str(datetime.timedelta(seconds=int(eta_seconds))),
            ),
            end="",
        )

    print("\nEvaluating captions...")

    # Tokenize results and references.
    res, gts = {}, {}
    for i, (img_id, _) in enumerate(img_list):
        captions, recs = all_captions[i], img_recs[img_id]
        assert len(captions) == len(recs)
        for j in range(len(captions)):
            key = f"{img_id}_region_{j}"
            res[key] = [{"caption": captions[j]}]
            gts[key] = [{"caption": recs[j]["caption"]}]
    tokenizer = PTBTokenizer()
    res, gts = tokenizer.tokenize(res), tokenizer.tokenize(gts)

    # Evaluate results for each metric.
    for metric in (Bleu(), Meteor(), Rouge(), Cider()):
        kwargs = {"verbose": 0} if isinstance(metric, Bleu) else {}
        score, _ = metric.compute_score(gts, res, **kwargs)
        print(metric.method(), score)


if __name__ == "__main__":
    main(parse_args())

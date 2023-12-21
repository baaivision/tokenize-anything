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
"""Gradio application."""

import argparse
import multiprocessing as mp
import os
import time

import numpy as np
import torch

from tokenize_anything import test_engine
from tokenize_anything.utils.image import im_rescale
from tokenize_anything.utils.image import im_vstack


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Launch gradio app.")
    parser.add_argument("--model-type", type=str, required=True, help="Model type.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint.")
    parser.add_argument("--concept", type=str, required=True, help="Concept weights.")
    parser.add_argument("--device", nargs="+", type=int, default=[0], help="Index of devices.")
    parser.add_argument("--port", type=int, default=2030, help="Server port.")
    return parser.parse_args()


class Predictor(object):
    """Predictor."""

    def __init__(self, model, kwargs):
        self.model = model
        self.kwargs = kwargs
        self.batch_size = kwargs.get("batch_size", 256)
        self.model.concept_projector.reset_weights(kwargs["concept_weights"])
        self.model.text_decoder.reset_cache(max_batch_size=self.batch_size)

    def preprocess_images(self, imgs):
        """Preprocess the inference images."""
        im_batch, im_shapes, im_scales = [], [], []
        for img in imgs:
            scaled_imgs, scales = im_rescale(img, scales=[1024])
            im_batch += scaled_imgs
            im_scales += scales
            im_shapes += [x.shape[:2] for x in scaled_imgs]
        im_batch = im_vstack(im_batch, self.model.pixel_mean_value, size=(1024, 1024))
        im_shapes = np.array(im_shapes)
        im_scales = np.array(im_scales).reshape((len(im_batch), -1))
        im_info = np.hstack([im_shapes, im_scales]).astype("float32")
        return im_batch, im_info

    @torch.inference_mode()
    def get_results(self, examples):
        """Return the results."""
        # Preprocess images and prompts.
        imgs = [example["img"] for example in examples]
        points = np.concatenate([example["points"] for example in examples])
        im_batch, im_info = self.preprocess_images(imgs)
        num_prompts = points.shape[0] if len(points.shape) > 2 else 1
        batch_shape = im_batch.shape[0], num_prompts // im_batch.shape[0]
        batch_points = points.reshape(batch_shape + (-1, 3))
        batch_points[:, :, :, :2] *= im_info[:, None, None, 2:4]
        batch_points = batch_points.reshape(points.shape)
        # Predict tokens and masks.
        inputs = self.model.get_inputs({"img": im_batch})
        inputs.update(self.model.get_features(inputs))
        outputs = self.model.get_outputs(dict(**inputs, **{"points": batch_points}))
        # Select final mask.
        iou_pred = outputs["iou_pred"].cpu().numpy()
        point_score = batch_points[:, 0, 2].__eq__(2).__sub__(0.5)[:, None]
        rank_scores = iou_pred + point_score * ([1000] + [0] * (iou_pred.shape[1] - 1))
        mask_index = np.arange(rank_scores.shape[0]), rank_scores.argmax(1)
        iou_scores = outputs["iou_pred"][mask_index].cpu().numpy().reshape(batch_shape)
        # Upscale masks to the original image resolution.
        mask_pred = outputs["mask_pred"][mask_index].unsqueeze_(1)
        mask_pred = self.model.upscale_masks(mask_pred, im_batch.shape[1:-1])
        mask_pred = mask_pred.view(batch_shape + mask_pred.shape[2:])
        # Predict concepts.
        concepts, scores = self.model.predict_concept(outputs["sem_embeds"][mask_index])
        concepts, scores = [x.reshape(batch_shape) for x in (concepts, scores)]
        # Generate captions.
        sem_tokens = outputs["sem_tokens"][mask_index].unsqueeze_(1)
        captions = self.model.generate_text(sem_tokens).reshape(batch_shape)
        # Postprecess results.
        results = []
        for i in range(batch_shape[0]):
            pred_h, pred_w = im_info[i, :2].astype("int")
            masks = mask_pred[i : i + 1, :, :pred_h, :pred_w]
            masks = self.model.upscale_masks(masks, imgs[i].shape[:2]).flatten(0, 1)
            results.append(
                {
                    "scores": np.stack([iou_scores[i], scores[i]], axis=-1),
                    "masks": masks.gt(0).cpu().numpy().astype("uint8"),
                    "concepts": concepts[i],
                    "captions": captions[i],
                }
            )
        return results


class ServingCommand(object):
    """Command to run serving."""

    def __init__(self, output_queue):
        self.output_queue = output_queue
        self.output_dict = mp.Manager().dict()
        self.output_index = mp.Value("i", 0)

    def postprocess_outputs(self, outputs):
        """Main the detection objects."""
        scores, masks = outputs["scores"], outputs["masks"]
        concepts, captions = outputs["concepts"], outputs["captions"]
        text_template = "{} ({:.2f}, {:.2f}): {}"
        text_contents = concepts, scores[:, 0], scores[:, 1], captions
        texts = np.array([text_template.format(*vals) for vals in zip(*text_contents)])
        return masks, texts

    def run(self):
        """Main loop to make the serving outputs."""
        while True:
            img_id, outputs = self.output_queue.get()
            self.output_dict[img_id] = self.postprocess_outputs(outputs)


def build_gradio_app(queues, command):
    """Build the gradio application."""
    import gradio as gr
    import gradio_image_prompter as gr_ext

    title = "Tokenize Anything"
    header = (
        "<div align='center'>"
        "<h1>Tokenize Anything via Prompting</h1>"
        "<h3><a href='https://arxiv.org/abs/2312.09128' target='_blank' rel='noopener'>[paper]</a>"
        "<a href='https://github.com/baaivision/tokenize-anything' target='_blank' rel='noopener'>[code]</a></h3>"  # noqa
        "<h3>A promptable model capable of simultaneous segmentation, recognition and caption.</h3>"  # noqa
        "</div>"
    )
    theme = "soft"
    css = """#anno-img .mask {opacity: 0.5; transition: all 0.2s ease-in-out;}
             #anno-img .mask.active {opacity: 0.7}"""

    def get_click_examples():
        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        app_images = list(filter(lambda x: x.startswith("app_image"), os.listdir(assets_dir)))
        app_images.sort()
        return [{"image": os.path.join(assets_dir, x)} for x in app_images]

    def on_reset_btn():
        click_img, draw_img = gr.Image(None), gr.ImageEditor(None)
        anno_img = gr.AnnotatedImage(None)
        return click_img, draw_img, anno_img

    def on_submit_btn(click_img, mask_img, prompt, multipoint):
        img, points = None, np.array([[[0, 0, 4]]])
        if prompt == 0 and click_img is not None:
            img, points = click_img["image"], click_img["points"]
            points = np.array(points).reshape((-1, 2, 3))
            if multipoint == 1:
                points = points.reshape((-1, 3))
                lt = points[np.where(points[:, 2] == 2)[0]][None, :, :]
                rb = points[np.where(points[:, 2] == 3)[0]][None, :, :]
                poly = points[np.where(points[:, 2] <= 1)[0]][None, :, :]
                points = [lt, rb, poly] if len(lt) > 0 else [poly, np.array([[[0, 0, 4]]])]
                points = np.concatenate(points, axis=1)
        elif prompt == 1 and mask_img is not None:
            img, points = mask_img["background"], []
            for layer in mask_img["layers"]:
                ys, xs = np.nonzero(layer[:, :, 0])
                if len(ys) > 0:
                    keep = np.linspace(0, ys.shape[0], 11, dtype="int64")[1:-1]
                    points.append(np.stack([xs[keep][None, :], ys[keep][None, :]], 2))
            if len(points) > 0:
                points = np.concatenate(points).astype("float32")
                points = np.pad(points, [(0, 0), (0, 0), (0, 1)], constant_values=1)
                pad_points = np.array([[[0, 0, 4]]], "float32").repeat(points.shape[0], 0)
                points = np.concatenate([points, pad_points], axis=1)
        img = img[:, :, (2, 1, 0)] if img is not None else img
        img = np.zeros((480, 640, 3), dtype="uint8") if img is None else img
        points = np.array([[[0, 0, 4]]]) if (len(points) == 0 or points.size == 0) else points
        inputs = {"img": img, "points": points.astype("float32")}
        with command.output_index.get_lock():
            command.output_index.value += 1
            img_id = command.output_index.value
        queues[img_id % len(queues)].put((img_id, inputs))
        while img_id not in command.output_dict:
            time.sleep(0.005)
        masks, texts = command.output_dict.pop(img_id)
        annotations = [(x, y) for x, y in zip(masks, texts)]
        return inputs["img"][:, :, ::-1], annotations

    app, _ = gr.Blocks(title=title, theme=theme, css=css).__enter__(), gr.Markdown(header)
    container, column = gr.Row().__enter__(), gr.Column().__enter__()
    click_tab, click_img = gr.Tab("Point+Box").__enter__(), gr_ext.ImagePrompter(show_label=False)
    interactions = "LeftClick (FG) | MiddleClick (BG) | PressMove (Box)"
    gr.Markdown("<h3 style='text-align: center'>[🖱️ | 🖐️]: 🌟🌟 {} 🌟🌟 </h3>".format(interactions))
    point_opt = gr.Radio(["Batch", "Ensemble"], label="Multipoint", type="index", value="Batch")
    gr.Examples(get_click_examples(), inputs=[click_img])
    _, draw_tab = click_tab.__exit__(), gr.Tab("Sketch").__enter__()
    draw_img, _ = gr.ImageEditor(show_label=False), draw_tab.__exit__()
    prompt_opt = gr.Radio(["Click", "Draw"], type="index", visible=False, value="Click")
    row, reset_btn, submit_btn = gr.Row().__enter__(), gr.Button("Reset"), gr.Button("Execute")
    _, _, column = row.__exit__(), column.__exit__(), gr.Column().__enter__()
    anno_img = gr.AnnotatedImage(elem_id="anno-img", show_label=False)
    reset_btn.click(on_reset_btn, [], [click_img, draw_img, anno_img])
    submit_btn.click(on_submit_btn, [click_img, draw_img, prompt_opt, point_opt], [anno_img])
    click_tab.select(lambda: "Click", [], [prompt_opt])
    draw_tab.select(lambda: "Draw", [], [prompt_opt])
    column.__exit__(), container.__exit__(), app.__exit__()
    return app


if __name__ == "__main__":
    args = parse_args()
    queues = [mp.Queue(1024) for _ in range(len(args.device) + 1)]
    commands = [
        test_engine.InferenceCommand(
            queues[i],
            queues[-1],
            kwargs={
                "model_type": args.model_type,
                "weights": args.checkpoint,
                "concept_weights": args.concept,
                "device": args.device[i],
                "predictor_type": Predictor,
                "verbose": i == 0,
            },
        )
        for i in range(len(args.device))
    ]
    commands += [ServingCommand(queues[-1])]
    actors = [mp.Process(target=command.run, daemon=True) for command in commands]
    for actor in actors:
        actor.start()
    app = build_gradio_app(queues[:-1], commands[-1])
    app.queue()
    app.launch(server_port=args.port, share=False)

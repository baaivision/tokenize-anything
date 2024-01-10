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
"""Image tokenizer."""

import numpy as np
import torch
from torch import nn


class ImageTokenizer(nn.Module):
    """Tokenize image regions with visual prompts."""

    def __init__(
        self,
        image_encoder,
        prompt_encoder,
        image_decoder,
        concept_projector=None,
        text_tokenizer=None,
        text_decoder=None,
        pixel_mean=(103.53, 116.28, 123.675),
        pixel_std=(57.375, 57.12, 58.395),
    ):
        super(ImageTokenizer, self).__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.image_decoder = image_decoder
        self.concept_projector = concept_projector
        self.text_tokenizer = text_tokenizer
        self.text_decoder = text_decoder
        self.pixel_mean_value = pixel_mean  # BGR order.
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean))
        self.register_buffer("pixel_rsig", torch.Tensor(pixel_std).reciprocal_())

    def get_inputs(self, inputs, dtype=None):
        """Return the model inputs.

        Parameters
        ----------
        inputs : dict
            The initial inputs.
        dtype : torch.dtype, optional
            The optional input dtype.

        Returns
        -------
        dict
            The model inputs.

        """
        img_dtype, img_device = self.pixel_mean.dtype, self.pixel_mean.device
        inputs["img"] = torch.as_tensor(inputs["img"], dtype=img_dtype, device=img_device)
        inputs["img"] = inputs["img"].sub(self.pixel_mean).mul_(self.pixel_rsig).permute(0, 3, 1, 2)
        inputs["img"] = inputs["img"].to(dtype=dtype) if dtype else inputs["img"]
        return inputs

    def get_features(self, inputs):
        """Return the image features.

        Parameters
        ----------
        inputs : dict
            The inputs.

        Returns
        -------
        dict
            The image features.

        """
        features = self.image_encoder(inputs["img"])
        img_embeds = features[0].permute(0, 2, 3, 1).unsqueeze_(1)
        return {"features": features, "img_embeds": img_embeds}

    def get_outputs(self, inputs):
        """Return the model outputs.

        Parameters
        ----------
        inputs : dict
            The model inputs.

        Returns
        -------
        dict
            The model outputs.

        """
        inputs.update(self.prompt_encoder(inputs))
        return self.image_decoder(inputs)

    def forward(self, inputs):
        """Define the computation performed at every call.

        Parameters
        ----------
        inputs : dict
            The initial inputs.

        Returns
        -------
        dict
            The model outputs.

        """
        inputs = self.get_inputs(inputs)
        inputs.update(self.get_features(inputs))
        return self.get_outputs(inputs)

    def upscale_masks(self, masks, size):
        """Upscale masks using bilinear interpolation.

        Parameters
        ----------
        masks : torch.Tensor
            The input masks.
        size : Union[int, Tuple[int]]
            The output size.

        Returns
        -------
        torch.Tensor
            The output masks.

        """
        return nn.functional.interpolate(masks, size, mode="bilinear", align_corners=False)

    @torch.inference_mode()
    def predict_concept(self, visual_embeds, k=1):
        """Predict top-k concepts based on visual embeddings.

        Parameters
        ----------
        visual_embeds: torch.Tensor
            The embeddings to predict visual content.
        k : int, optional, default=1
            The k value.

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray]
            The concept scores and indices.

        """
        return self.concept_projector.decode(visual_embeds, k)

    @torch.inference_mode()
    def generate_text(self, visual_tokens, max_gen_len=None, temperature=0):
        """Generate text sequences based on visual tokens.

        Parameters
        ----------
        visual_tokens: torch.Tensor
            The tokens to prompt visual context.
        max_gen_len : int, optional
            The maximum length of the generated text sequences.
        temperature : float, optional
            The temperature for controlling randomness in sampling.

        Returns
        -------
        np.ndarray
            An array of generated texts.

        """
        max_gen_len = max_gen_len or self.text_decoder.max_text_len
        prompts = self.text_decoder.get_prompts(visual_tokens)
        out_shape = (prompts.size(0), self.text_decoder.max_text_len)
        tokens = np.full(out_shape, self.text_tokenizer.pad_id, "int64")
        tokens[:, 0], prev_pos = self.text_tokenizer.bos_id, 0
        eos_reached = np.array([False] * tokens.shape[0])
        for cur_pos in range(1, max_gen_len):
            decode_seq_len = cur_pos - prev_pos
            x = torch.as_tensor(tokens[:, prev_pos:cur_pos], device=prompts.device)
            logits = self.text_decoder.transformer(prompts, x, prev_pos)
            next_logits = logits[: x.size(0), decode_seq_len - 1]
            if temperature > 0:
                p = nn.functional.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(p, 1).cpu().numpy().flatten()
            else:
                next_token = next_logits.argmax(-1).cpu().numpy()
            tokens[:, cur_pos] = next_token
            eos_reached |= next_token == self.text_tokenizer.eos_id
            prev_pos, logits, next_logits = cur_pos, None, None
            if eos_reached.all():
                break
        return np.array(self.text_tokenizer.detokenize(tokens))

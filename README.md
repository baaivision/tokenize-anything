<div align="center">

<h1>Tokenize Anything via Prompting</h1>

[Ting Pan](https://github.com/PhyscalX/)<sup>1,2*</sup>, &nbsp; [Lulu Tang]([https://github.com/lulutang0608](https://scholar.google.com/citations?authuser=1&user=o2fG4xUAAAAJ))<sup>2*</sup>, &nbsp; [Xinlong Wang](https://www.xloong.wang/)<sup>2¶</sup>, &nbsp; [Shiguang Shan](https://scholar.google.com/citations?user=Vkzd7MIAAAAJ&hl=en)<sup>1</sup>

<sup>1</sup>[ICT-CAS](http://english.ict.cas.cn/), &nbsp; <sup>2</sup>[BAAI](https://www.baai.ac.cn/english.html)<br>
<sup>*</sup> Equal Contribution, <sup>¶</sup>Project Lead

[[`Paper`](https://arxiv.org/pdf/2312.09128.pdf)] [[`🤗 Demo`](https://huggingface.co/spaces/BAAI/tokenize-anything)]
<br><br><image src="assets/model_overview.png"/>

</div>

We present **T**okenize **A**nything via **P**rompting, a unified and promptable model capable of simultaneously segmenting, recognizing, and captioning arbitrary regions, with flexible visual prompts (point, box and sketch). The model is trained with exhaustive segmentation masks sourced from SA-1B, coupled with semantic priors from a pre-trained EVA-CLIP with 5 billion parameters.

## Installation

### Preliminaries

``torch`` >= 2.1

``flash-attn`` >= 2.3.3 (for TextGeneration)

``gradio-image-prompter`` (for GradioApp, Install from [URL](https://github.com/PhyscalX/gradio-image-prompter))

### Installing Package

Clone this repository to local disk and install:

```bash
cd tokenize-anything && pip install .
```

You can also install from the remote repository: 

```bash
pip install git+ssh://git@github.com/baaivision/tokenize-anything.git
```

## Quick Start

### Development

The **TAP** models can be used for diverse vision and language tasks. 

We adopt a modular design that decouples all components and predictors.

As a best practice, implement your custom predictor and asynchronous pipeline as follows:

```python
from tokenize_anything import model_registry

with <distributed_actor>:
    model = model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
    results = <custom_predictor>(model, *args, **kwargs)

server.collect_results()
```

See builtin examples (web-demo and evaluations) provided in [scripts](scripts/) for more details.

### Inference

See [Inference Guide](notebooks/inference.ipynb).

See [Concept Guide](notebooks/concept.ipynb).

### Evaluation

See [Evaluation Guide for TAP-H](notebooks/evaluation_tap_vit_h_v1_1.ipynb).

See [Evaluation Guide for TAP-L](notebooks/evaluation_tap_vit_l_v1_1.ipynb).

See [Evaluation Guide for TAP-B](notebooks/evaluation_tap_vit_b_v1_1.ipynb).

## Models

### Model weights

#### V1.1 Release Notes

- Three versions of the model are available with different image encoders.
- Use a longer pre-training and fine-tuning schedule (improved segmentation and caption performance).
- Apply weight decay for all bias parameters (avoid FP16 overflow in QK matmul).
- Sample point prompts from predicted mask instead of GT box during VG training.

| Model | Description | Schedule | MD5 | Weights |
| ----- | ------------| ------ | ----| ------ |
| **tap_vit_h** | ViT-H TAP v1.1 model | (100% SA-1B, 180k), (VG, 50ep) | 4bdfb9 | [🤗 HF link](https://huggingface.co/BAAI/tokenize-anything/blob/main/models/tap_vit_h_v1_1.pkl) |
| **tap_vit_l** | ViT-L TAP v1.1 model | (100% SA-1B, 180k), (VG, 50ep) | c1d41f | [🤗 HF link](https://huggingface.co/BAAI/tokenize-anything/blob/main/models/tap_vit_l_v1_1.pkl) |
| **tap_vit_b** | ViT-B TAP v1.1 model | (100% SA-1B, 180k), (VG, 50ep) | 707f80 | [🤗 HF link](https://huggingface.co/BAAI/tokenize-anything/blob/main/models/tap_vit_b_v1_1.pkl) |

#### V1.0 Release Notes

- Two versions of the model are available with different image encoders.
- Original paper results.

| Model | Description | Schedule | MD5 | Weights |
| ----- | ------------| ------ | ----| ------ |
| **tap_vit_l** | ViT-L TAP v1.0 model | (50% SA-1B, 90k), (VG, 25ep) | 03f8ec | [🤗 HF link](https://huggingface.co/BAAI/tokenize-anything/blob/main/models/tap_vit_l_v1_0.pkl) |
| **tap_vit_b** | ViT-B TAP v1.0 model | (50% SA-1B, 90k), (VG, 25ep) | b45cbf | [🤗 HF link](https://huggingface.co/BAAI/tokenize-anything/blob/main/models/tap_vit_b_v1_0.pkl) |

### Concept weights

***Note***: You can generate these weights following the [Concept Guide](notebooks/concept.ipynb).

| Concept | Description | Weights |
| ------- | ------------| ------ |
| **Merged-2560** | Merged concepts | [🤗 HF link](https://huggingface.co/BAAI/tokenize-anything/blob/main/concepts/merged_2560.pkl) |
| **LVIS-1203**   | LVIS concepts | [🤗 HF link](https://huggingface.co/BAAI/tokenize-anything/blob/main/concepts/lvis_1203.pkl) |
| **COCO-80**   | COCO concepts  | [🤗 HF link](https://huggingface.co/BAAI/tokenize-anything/blob/main/concepts/coco_80.pkl) |

## License
[Apache License 2.0](LICENSE)

## Citation

```
@article{pan2023tap,
  title={Tokenize Anything via Prompting},
  author={Pan, Ting and Tang, Lulu and Wang, Xinlong and Shan, Shiguang},
  journal={arXiv preprint arXiv:2312.09128},
  year={2023}
}
```

## Acknowledgement

We thank the repositories: [SAM](https://github.com/facebookresearch/segment-anything), [EVA](https://github.com/baaivision/EVA), [LLaMA](https://github.com/facebookresearch/llama), [FlashAttention](https://github.com/Dao-AILab/flash-attention), [Gradio](https://github.com/gradio-app/gradio), [Detectron2](https://github.com/facebookresearch/detectron2) and [CodeWithGPU](https://github.com/seetacloud/codewithgpu).

<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# MobileCLIP: Fast Image-Text Models Through Multi-Modal Reinforced Training

[![Ultralytics Actions](https://github.com/ultralytics/velocity/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/velocity/actions/workflows/format.yml)
[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)

This repository is an Ultralytics fork of Apple's official [MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training](https://arxiv.org/pdf/2311.17049.pdf) (CVPR 2024)  
_Pavan Kumar Anasosalu Vasu, Hadi Pouransari, Fartash Faghri, Raviteja Vemulapalli, Oncel Tuzel._

It provides code for inference, training, and evaluation of MobileCLIP models trained on DataCompDR datasets.

<p align="center">
<img src="docs/fig_accuracy_latency.png" alt="Accuracy vs latency figure." width="400"/>
</p>

- **Update 2024/11/22:** iOS app released for real-time zero-shot image classification with MobileCLIP. Explore the [iOS app](./ios_app/).
- **Update 2024/06/13:** Training scripts for [OpenCLIP](https://github.com/mlfoundations/open_clip/tree/main/src/open_clip) on DataCompDR datasets are now available. See [training/](./training/).
- **Update 2024/06/13:** MobileCLIP models and DataCompDR datasets are hosted on Hugging Face in the [MobileCLIP/DataCompDR Collection](https://huggingface.co/collections/apple/mobileclip-models-datacompdr-data-665789776e1aa2b59f35f7c8).

## 🚀 Highlights

- The smallest variant, **MobileCLIP-S0**, achieves comparable zero-shot performance to [OpenAI's ViT-B/16](https://arxiv.org/abs/2103.00020) while being 4.8x faster and 2.8x smaller.
- **MobileCLIP-S2** surpasses [SigLIP's ViT-B/16](https://arxiv.org/abs/2303.15343) in average zero-shot performance, is 2.3x faster and 2.1x smaller, and is trained with 3x fewer seen samples.
- **MobileCLIP-B (LT)** attains a zero-shot ImageNet accuracy of **77.2%**, outperforming recent models like [DFN](https://arxiv.org/abs/2309.17425), [SigLIP](https://arxiv.org/abs/2303.15343), and even [OpenAI's ViT-L/14@336](https://arxiv.org/abs/2103.00020).
- Dedicated iOS app demonstrates high performance on mobile devices.

![MobileCLIP iOS App Examples](ios_app/docs/app_screenshots/examples.png)

## ⚡ Getting Started

### Setup

```bash
conda create -n clipenv python=3.10
conda activate clipenv
pip install -e .
```

To download pretrained checkpoints:

```bash
source get_pretrained_models.sh # Downloads files to the `checkpoints` directory.
```

### Usage Example

To use models from the official repository:

```python
import torch
from PIL import Image

import mobileclip

model, _, preprocess = mobileclip.create_model_and_transforms("mobileclip_s0", pretrained="/path/to/mobileclip_s0.pt")
tokenizer = mobileclip.get_tokenizer("mobileclip_s0")

image = preprocess(Image.open("docs/fig_accuracy_latency.png").convert("RGB")).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
```

For an example of loading data from Hugging Face, see [hf_dataset_example.py](./hf_dataset_example.py).

## 🔗 OpenCLIP Support

MobileCLIP models are natively supported in [OpenCLIP](https://github.com/mlfoundations/open_clip). To use them:

```bash
conda create -n clipenv python=3.10
conda activate clipenv

pip install git+https://github.com/mlfoundations/open_clip
pip install git+https://github.com/huggingface/pytorch-image-models
```

Example inference:

```python
import open_clip

from mobileclip.modules.common.mobileone import reparameterize_model

model, _, preprocess = open_clip.create_model_and_transforms("MobileCLIP-S2", pretrained="datacompdr")
tokenizer = open_clip.get_tokenizer("MobileCLIP-S2")

# For inference/model exporting, reparameterize first
model.eval()
model = reparameterize_model(model)

# ... follow further examples in the OpenCLIP repository ...
```

Available variants on OpenCLIP:

- MobileCLIP-S1 (`datacompdr`)
- MobileCLIP-S2 (`datacompdr`)
- MobileCLIP-B (`datacompdr`)
- MobileCLIP-B (`datacompdr_lt`)

## 📊 Evaluation

Comprehensive evaluation results are available in the [results directory](./results).  
To reproduce results, use the provided script for zero-shot evaluation on the ImageNet-1k dataset.  
For evaluation on all 38 datasets, follow the instructions in the [DataComp repository](https://github.com/mlfoundations/datacomp).

```bash
# Run evaluation with a single GPU
python eval/zeroshot_imagenet.py --model-arch mobileclip_s0 --model-path /path/to/mobileclip_s0.pt
```

Compare with other models using the [OpenCLIP Results CSV](https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv).

| Model             | # Seen <br>Samples (B) | # Params (M) <br> (img + txt) | Latency (ms) <br> (img + txt) | IN-1k Zero-Shot <br> Top-1 Acc. (%) | Avg. Perf. (%) <br> on 38 datasets |                                            PyTorch Checkpoint (URL)                                            |
| :---------------- | :--------------------: | :---------------------------: | :---------------------------: | :---------------------------------: | :--------------------------------: | :------------------------------------------------------------------------------------------------------------: |
| MobileCLIP-S0     |           13           |          11.4 + 42.4          |           1.5 + 1.6           |                67.8                 |                58.1                |  [mobileclip_s0.pt](https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s0.pt)  |
| MobileCLIP-S1     |           13           |          21.5 + 63.4          |           2.5 + 3.3           |                72.6                 |                61.3                |  [mobileclip_s1.pt](https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s1.pt)  |
| MobileCLIP-S2     |           13           |          35.7 + 63.4          |           3.6 + 3.3           |                74.4                 |                63.7                |  [mobileclip_s2.pt](https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s2.pt)  |
| MobileCLIP-B      |           13           |          86.3 + 63.4          |          10.4 + 3.3           |                76.8                 |                65.2                |   [mobileclip_b.pt](https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_b.pt)   |
| MobileCLIP-B (LT) |           36           |          86.3 + 63.4          |          10.4 + 3.3           |                77.2                 |                65.8                | [mobileclip_blt.pt](https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt) |

**Note:** MobileCLIP-B (LT) is trained for 300k iterations with a constant learning rate schedule and 300k iterations with a cosine learning rate schedule.

## 📚 Citation

If you find this code useful, please cite:

```bibtex
@InProceedings{mobileclip2024,
  author = {Pavan Kumar Anasosalu Vasu, Hadi Pouransari, Fartash Faghri, Raviteja Vemulapalli, Oncel Tuzel},
  title = {MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2024},
}
```

## 🙏 Acknowledgements

This codebase builds upon multiple open-source contributions. See [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for details.

---

We welcome your contributions! If you have suggestions, improvements, or want to get involved, please open an issue or submit a pull request.

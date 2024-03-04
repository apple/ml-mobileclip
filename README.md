# MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training

This is the official repository of 

**MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training. (CVPR 2024)**
*Pavan Kumar Anasosalu Vasu, Hadi Pouransari, Fartash Faghri, Raviteja Vemulapalli, Oncel Tuzel.*

[//]: # (![MobileCLIP Performance]&#40;docs/fig_accuracy_latency.png&#41;)
<p align="center">
<img src="docs/fig_accuracy_latency.png" alt="Accuracy vs latency figure." width="400"/>
</p>

### Highlights
* Our smallest variant `MobileCLIP-S0` obtains similar zero-shot performance as [OpenAI](https://arxiv.org/abs/2103.00020)'s ViT-B/16 model while being 4.8x faster and 2.8x smaller.
* `MobileCLIP-S2` obtains better avg zero-shot performance than [SigLIP](https://arxiv.org/abs/2303.15343)'s ViT-B/16 model while being 2.3x faster and 2.1x smaller, and trained with 3x less seen samples.
* `MobileCLIP-B`(LT) attains zero-shot ImageNet performance of **77.2%** which is significantly better than recent works like [DFN](https://arxiv.org/abs/2309.17425) and [SigLIP](https://arxiv.org/abs/2303.15343) with similar architectures or even [OpenAI's ViT-L/14@336](https://arxiv.org/abs/2103.00020).

## Setup
```bash
conda create -n clipenv python=3.10
conda activate clipenv
pip install -e .
```
To download pretrained checkpoints follow the code snippet below
```bash
source get_pretrained_models.sh   # Files will be downloaded to `checkpoints` directory.
```

## Getting Started
For easy adoption, our model has similar API as [open_clip](https://github.com/mlfoundations/open_clip) models.
To use our model, follow the code snippet below
```python
import torch
from PIL import Image
import mobileclip

model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained='/path/to/mobileclip_s0.pt')
tokenizer = mobileclip.get_tokenizer('mobileclip_s0')

image = preprocess(Image.open("docs/fig_accuracy_latency.png").convert('RGB')).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
```

## Evaluation
To reproduce results, we provide script to perform zero-shot evaluation on ImageNet-1k dataset. 
To evaluate on all the 38 datasets, please follow instructions in [datacomp](https://github.com/mlfoundations/datacomp).
```bash
# Run evaluation with single GPU
python eval/zeroshot_imagenet.py --model-arch mobileclip_s0 --model-path /path/to/mobileclip_s0.pt
```

Please refer to [Open CLIP Results](https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv) to compare with other models.

| Model             |   # Seen <BR>Samples (B)   | # Params (M) <BR> (img + txt) | Latency (ms) <BR> (img + txt)  | IN-1k Zero-Shot <BR> Top-1 Acc. (%) | Avg. Perf. (%) <BR> on 38 datasets |                                            Pytorch Checkpoint (url)                                            |
|:------------------|:----------------------:|:-----------------------------:|:------------------------------:|:-----------------------------------:|:----------------------------------:|:--------------------------------------------------------------------------------------------------------------:|
| MobileCLIP-S0     |           13           |          11.4 + 42.4          |           1.5 + 1.6            |                67.8                 |                58.1                |  [mobileclip_s0.pt](https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s0.pt)  |
| MobileCLIP-S1     |           13           |          21.5 + 63.4          |           2.5 + 3.3           |                72.6                 |                61.3                |  [mobileclip_s1.pt](https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s1.pt)  |
| MobileCLIP-S2     |           13           |          35.7 + 63.4          |           3.6 + 3.3           |                74.4                 |                63.7                |  [mobileclip_s2.pt](https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s2.pt)  |
| MobileCLIP-B      |           13           |          86.3 + 63.4          |          10.4 + 3.3           |                76.8                 |                65.2                |   [mobileclip_b.pt](https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_b.pt)   |
| MobileCLIP-B (LT) |           36           |          86.3 + 63.4          |          10.4 + 3.3           |                77.2                 |                65.8                | [mobileclip_blt.pt](https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt) |

Note: MobileCLIP-B(LT) is trained for 300k iterations with constant learning rate schedule and 300k iterations with cosine learning rate schedule.

## Citation
If you found this code useful, please cite the following paper:
```
@InProceedings{mobileclip2024,
  author = {Pavan Kumar Anasosalu Vasu, Hadi Pouransari, Fartash Faghri, Raviteja Vemulapalli, Oncel Tuzel},
  title = {MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2024},
}
```

## Acknowledgements
Our codebase is built using multiple opensource contributions, please see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for more details. 

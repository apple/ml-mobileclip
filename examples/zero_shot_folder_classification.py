#
# For licensing see accompanying LICENSE file.
#
"""Run MobileCLIP zero-shot classification over a folder of images.

This example scores each image in a folder against user-provided text labels and
writes the results to a CSV file.

Example:
    python examples/zero_shot_folder_classification.py \
        --image-dir /path/to/images \
        --checkpoint /path/to/mobileclip2_s0.pt \
        --label "a diagram" \
        --label "a dog" \
        --label "a cat"
"""

import argparse
import csv
import re
from pathlib import Path

import open_clip
import torch
from PIL import Image

from mobileclip.modules.common.mobileone import reparameterize_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def load_labels(args):
    labels = []
    if args.labels_file is not None:
        with open(args.labels_file, encoding="utf-8") as handle:
            labels.extend(line.strip() for line in handle if line.strip())
    if args.label is not None:
        labels.extend(args.label)
    if not labels:
        raise ValueError("Provide at least one --label or --labels-file entry.")
    return labels


def iter_images(image_dir):
    for path in sorted(Path(image_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def chunked(items, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def slugify(label):
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return slug[:48] or "label"


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def unique_score_columns(labels):
    columns = []
    used = set()
    for label in labels:
        base = f"score_{slugify(label)}"
        column = base
        suffix = 2
        while column in used:
            column = f"{base}_{suffix}"
            suffix += 1
        columns.append(column)
        used.add(column)
    return columns


def model_kwargs_for(model_name):
    if model_name in {"MobileCLIP2-S3", "MobileCLIP2-S4"} or model_name.endswith("L-14"):
        return {}
    return {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}


def resolve_device(device):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode_text(model, tokenizer, labels, device):
    tokens = tokenizer(labels).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
    return features


def encode_images(model, preprocess, image_paths, device):
    images = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(preprocess(image.convert("RGB")))
    image_batch = torch.stack(images).to(device)
    with torch.no_grad():
        features = model.encode_image(image_batch)
        features /= features.norm(dim=-1, keepdim=True)
    return features


def run(args):
    image_paths = list(iter_images(args.image_dir))
    if not image_paths:
        raise ValueError(f"No images found under {args.image_dir}")

    labels = load_labels(args)
    top_k = min(args.top_k, len(labels))
    device = resolve_device(args.device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.checkpoint,
        **model_kwargs_for(args.model_name),
    )
    model.eval()
    model = reparameterize_model(model).to(device)
    tokenizer = open_clip.get_tokenizer(args.model_name)
    text_features = encode_text(model, tokenizer, labels, device)

    score_columns = unique_score_columns(labels)
    rank_columns = []
    for rank in range(1, top_k + 1):
        rank_columns.extend([f"rank_{rank}_label", f"rank_{rank}_confidence"])

    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "top_label", "confidence"]
            + rank_columns
            + score_columns,
        )
        writer.writeheader()

        for image_batch_paths in chunked(image_paths, args.batch_size):
            image_features = encode_images(model, preprocess, image_batch_paths, device)
            batch_scores = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            for image_path, scores in zip(image_batch_paths, batch_scores):
                top_scores, top_indexes = torch.topk(scores, k=top_k)
                row = {
                    "image": str(image_path),
                    "top_label": labels[int(top_indexes[0].item())],
                    "confidence": f"{top_scores[0].item():.6f}",
                }
                for rank, (score, index) in enumerate(zip(top_scores, top_indexes), start=1):
                    row[f"rank_{rank}_label"] = labels[int(index.item())]
                    row[f"rank_{rank}_confidence"] = f"{score.item():.6f}"
                row.update(
                    {
                        column: f"{score.item():.6f}"
                        for column, score in zip(score_columns, scores)
                    }
                )
                writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, help="Folder of images to classify.")
    parser.add_argument("--checkpoint", required=True, help="Path to a MobileCLIP checkpoint.")
    parser.add_argument("--output-csv", default="mobileclip_folder_classification.csv")
    parser.add_argument("--model-name", default="MobileCLIP2-S0")
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--top-k", type=positive_int, default=3)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device. Defaults to CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Text label to score. Repeat to provide multiple labels.",
    )
    parser.add_argument(
        "--labels-file",
        help="Optional newline-delimited text labels. Combined with repeated --label values.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    run(parse_args())

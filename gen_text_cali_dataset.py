import numpy as np
import torch
import random
from imagenet_dataset import ImagenetDataset, imagenet_classes, imagenet_templates
import mobileclip

tokenizer = mobileclip.get_tokenizer('mobileclip_s2')

for i, classname in enumerate(imagenet_classes):
    if i>=64:
        break

    idx = random.randint(0, 79)

    texts = [imagenet_templates[idx].format(classname)]
    # format with class
    texts = tokenizer(texts)  # tokenize
    texts = texts.to(torch.int32)
    s_path = f"dataset/text_quant_data/{idx}.npy"
    print("save: ", s_path, texts.shape)
    np.save(s_path, np.array(texts))
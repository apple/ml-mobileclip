import numpy as np
from PIL import Image
import mobileclip
# import onnxruntime as ort
import axengine as axe
import torch

def softmax(x, axis=-1):
    """
    对 numpy 数组在指定维度上应用 softmax 函数
    
    参数:
        x: numpy 数组，输入数据
        axis: 计算 softmax 的维度，默认为最后一个维度 (-1)
    
    返回:
        经过 softmax 处理的 numpy 数组，与输入形状相同
    """
    # 减去最大值以防止数值溢出（数值稳定化）
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    # 计算每个元素的指数与所在维度总和的比值
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

_, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s2', pretrained=None)
tokenizer = mobileclip.get_tokenizer('mobileclip_s2')

image = preprocess(Image.open("fig_accuracy_latency.png").convert('RGB')).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])
text = text.to(torch.int32)


onnx_image_encoder = axe.InferenceSession("mobileclip_s2_image_encoder.axmodel")
onnx_text_encoder = axe.InferenceSession("mobileclip_s2_text_encoder.axmodel")

image_features = onnx_image_encoder.run(["unnorm_image_features"],{"image":np.array(image)})[0]
text_features = []
for i in range(text.shape[0]):
    text_feature = onnx_text_encoder.run(["unnorm_text_features"],{"text":np.array([text[i]])})[0]
    text_features.append(text_feature)
text_features = np.array([t[0] for t in text_features])
image_features /= np.linalg.norm(image_features, ord=2, axis=-1, keepdims=True)
text_features /= np.linalg.norm(text_features, ord=2, axis=-1, keepdims=True)

text_probs = softmax(100.0 * image_features @ text_features.T)

print("Label probs:", text_probs)
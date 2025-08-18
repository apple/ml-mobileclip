import torch

from PIL import Image
import mobileclip

model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s2', pretrained='models/mobileclip_s2.pt')
tokenizer = mobileclip.get_tokenizer('mobileclip_s2')

image = preprocess(Image.open("docs/fig_accuracy_latency.png").convert('RGB')).unsqueeze(0)
text = tokenizer(["a diagram"])
text = text.to(torch.int32)

# export image onnx
torch.onnx.export(model.image_encoder,
            image,
            "./models/mobileclip_s2_image_encoder.onnx",
            input_names=['image'],
            output_names=['unnorm_image_features'],
            export_params=True,
            opset_version=14,)

# import text onnx
torch.onnx.export(model.text_encoder,
            text,
            "./models/mobileclip_s2_text_encoder.onnx",
            input_names=['text'],
            output_names=['unnorm_text_features'],
            export_params=True,
            opset_version=14,)

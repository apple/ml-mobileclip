# axera.ml-mobileclip

[ml-mobileclip](https://github.com/apple/ml-mobileclip) demo on axera

## 支持平台
- [x] AX650N
- [ ] AX630C

### env

根据原repo配置运行环境
```
conda create -n clipenv python=3.10
conda activate clipenv
pip install -e .
```
补充onnx相关包
```
pip install onnx
pip install onnxruntime
pip install opencv-python
```

### 导出模型(PyTorch -> ONNX)
```
python export_onnx.py
```
导出成功后会生成两个onnx模型:
- image encoder: mobileclip_s2_image_encoder.onnx
- text encoder: mobileclip_s2_text_encoder.onnx


#### 转换模型(ONNX -> Axera)
使用模型转换工具 Pulsar2 将 ONNX 模型转换成适用于 Axera 的 NPU 运行的模型文件格式 .axmodel，通常情况下需要经过以下两个步骤：

- 生成适用于该模型的 PTQ 量化校准数据集
- 使用 Pulsar2 build 命令集进行模型转换（PTQ 量化、编译），更详细的使用说明请参考[AXera Pulsar2 工具链指导手册](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)


#### 量化数据集准备
此处仅用作demo，建议使用实际参与训练的数据
- image数据:

    dataset/imagenet-calib.tar

- text数据:
    ```
    python gen_text_cali_dataset.py
    cd dataset
    zip -r text_quant_data.zip text_quant_data/
    ```
最终得到两个数据集：

\- dataset/dataset_v04.zip

\- dataset/text_quant_data.zip

注：对分数据集建议用实际使用场景的数据，此处仅用于演示

#### 模型编译
修改配置文件
检查config.json 中 calibration_dataset 字段，将该字段配置的路径改为上一步准备的量化数据集存放路径



在编译环境中，执行pulsar2 build参考命令：
```
# image encoder
pulsar2 build --config build_config/mobileclip_s2_image_u16.json --input models/mobileclip_s2_image_encoder.onnx --output_dir build_output/image_encoder --output_name mobileclip_s2_image_encoder.axmodel

# text encoder
pulsar2 build --config build_config/mobileclip_s2_text_u16.json --input models/mobileclip_s2_text_encoder.onnx --output_dir build_output/text_encoder --output_name mobileclip_s2_text_encoder.axmodel
```

编译完成后得到两个axmodel模型：


\- mobileclip_s2_image_encoder.axmodel

\- mobileclip_s2_text_encoder.axmodel


### Python API 运行
需基于[PyAXEngine](https://github.com/AXERA-TECH/pyaxengine)在AX650N上进行部署

demo基于原repo中的提取图文特征向量并计算相似度，将两个axmodel和run_onboard中的文件拷贝到开发板上后，运行run_axmodel.py文件

1. 输入图片：

    ![](docs/fig_accuracy_latency.png)

2. 输入文本：

    ["a diagram", "a dog", "a cat"]

3. 输出类别置信度：

    Label probs: [[9.9999499e-01 9.3656269e-07 4.0868617e-06]]



## 技术讨论

- Github issues
- QQ 群: 139953715
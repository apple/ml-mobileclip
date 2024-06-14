# Training on DataCompDR with OpenCLIP
We provide release code and a patch to
[OpenCLIP](https://github.com/mlfoundations/open_clip/tree/main/src/open_clip) 
for training models on DataCompDR.

## Data
Our reinforcements to DataComp are available on HuggingFace.
- [DataCompDR-12M](https://huggingface.co/datasets/apple/DataCompDR-12M)
- [DataCompDR-12M-BFloat16](https://huggingface.co/datasets/apple/DataCompDR-12M-bf16)
- [DataCompDR-1B](https://huggingface.co/datasets/apple/DataCompDR-1B)

## Installing dependencies

We use OpenCLIP for training. We have made minor modifications to OpenCLIP for 
support of loading reinforcements and the training loss. To checkout the 
specific version of each library and apply our corresponding patch run the 
following commands in order:
```bash
# Clone MobileCLIP repository
git clone git@github.com:apple/ml-mobileclip.git
cd ml-mobileclip/

# Clone OpenCLIP repository, apply patch, and install
git clone https://github.com/mlfoundations/open_clip.git
cd open_clip
git checkout cf86ee7ec4658845f640858ecd34d0f15588271a
git apply ../open_clip.patch  # Support for sampling without replacement
cp ../configs/ ./ -r
cp ../dr/ ./src/training/ -r
```

## Training

We provide scripts for training on DataCompDR-12M and DataCompDR-1B.

```bash
cd open_clip/
bash configs/run_datacomp12m.sh  # Train a ViT-B/16 on DataComp-12M without DR
bash configs/run_datacompdr12m.sh  # Train a ViT-B/16 on DataComp-12M with DR
bash configs/run_datacompdr1B.sh  # Train a ViT-B/16 on DataComp-1B with DR
```

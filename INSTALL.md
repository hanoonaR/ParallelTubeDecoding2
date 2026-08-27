# Installation

## Environment

PTD uses Python 3.11. The provided requirements install the training stack used
by this repository, including PyTorch 2.8, Transformers 5.3, DeepSpeed, TRL,
and the video-processing dependencies.

```bash
conda create -n ptd python=3.11 -y
conda activate ptd

pip install --upgrade pip
pip install -r requirements.txt
pip install qwen-vl-utils
```

The requirements include the CUDA 12.8 PyTorch packages. If your system uses a
different CUDA or ROCm version, install the matching PyTorch build for your
system and adjust the PyTorch package pins in `requirements.txt` before
installing the remaining dependencies.

## Check the installation

Run this command from the repository root:

```bash
python -c "import torch, transformers, qwen_vl_utils; print(torch.__version__, transformers.__version__)"
```

## Next steps

- [Prepare the VidSTG and HC-STVG training annotations](data/README.md).
- [Run SFT, merge the adapter, and run GRPO](ptd_scripts/README.md).
- [Add the PTD evaluation tasks to lmms-eval](evaluation/README.md).

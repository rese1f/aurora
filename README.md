# Aurora Series
A more efficient multimodal large language model series.

> **AuroraCap** &emsp; A Detailed Captioning Baseline and Benchmark for Video  

[![](https://img.shields.io/badge/docs-922133)](docs/auroracap/README.md)
[![](https://img.shields.io/badge/web-922133)](https://rese1f.github.io/aurora-web/)
[![](http://img.shields.io/badge/arXiv-922133)](https://arxiv.org/abs/2409.)
[![](https://img.shields.io/badge/%F0%9F%A4%97%20_model-ffc107?color=ffc107&logoColor=white)]()
[![](https://img.shields.io/badge/%F0%9F%A4%97%20_VDC_benchmark-ffc107?color=ffc107&logoColor=white)]()

## News

- [2024/09/15] 

## Quick Start  

### Installation

We recommend installing aurora in a virtual environment from Conda (Python>=3.10).
```
conda create -n aurora python=3.10
conda activate aurora
```

Install PyTorch following [instruction](https://pytorch.org/get-started/locally/).
```
pip install torch torchvision
```

For quick usage only for deploy, install aurora via pip.
```
pip install aurora
```

For further development, clone this repository and install from source.
```
git clone https://github.com/rese1f/aurora.git && cd aurora
pip install -r requirements.txt
```

For training, install additional dependencies.
```
cd src/xtuner && pip install -e '.[all]'
```

For evaluation, install additional dependencies.
```
cd src/lmms-eval && pip install -e .
```

Since transformers version confilct, we recommand using seperated virtual environment for deploy, install addttional dependencies.
```
cd src/sglang && pip install -e "python[all]"
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

### Play with AuroraCap

#### with huggingface transformers

#### with SGLang

#### with Gradio GUI

## Citation

```bibtex
```

## License

This project is released under the [Apache License 2.0](LICENSE). Please also adhere to the Licenses of models and datasets being used.

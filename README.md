# Aurora Series
A more efficient multimodal large language model series.

> [**AuroraCap**](docs/auroracap/README.md) &emsp; Efficient, Performant Video Detailed Captioning and a New Benchmark

[![](https://img.shields.io/badge/docs-922133)](docs/auroracap/README.md)
[![](https://img.shields.io/badge/web-922133)](https://rese1f.github.io/aurora-web/)
[![](http://img.shields.io/badge/arXiv-922133)](https://arxiv.org/abs/2409.)
[![](https://img.shields.io/badge/%F0%9F%A4%97%20_AuroraCap_model-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/collections/wchai/auroracap-66d117ffe13bedda96702013)
[![](https://img.shields.io/badge/%F0%9F%A4%97%20_VDC_benchmark-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/datasets/wchai/Video-Detailed-Caption)
[![](https://img.shields.io/badge/%F0%9F%A4%97%20_Trainset-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/datasets/wchai/AuroraCap-trainset)

<img src="assets/auroracap/vdc_baseline.png" align="center">

## News

- [2024/] Release AuroraCap model and VDC benchmark, as well as the training and evaluation code.

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

Clone this repository and install from source.
```
git clone https://github.com/rese1f/aurora.git && cd aurora
```

For training, install additional dependencies.
```
cd src/xtuner && pip install -e '.[all]'
```

For evaluation, install additional dependencies.
```
cd src/lmms-eval && pip install -e .
```

### Play with AuroraCap

```
python inference.py \
    --model_path wchai/AuroraCap-7B-VID-hf \
    --prompt "Describe the video in detail." \
    --visual_input assets/auroracap/test.mp4 \
    --num_frm 8 \
    --token_kept_ratio 0.2 \
    --temperature 0.0 \
    --top_p 1.0 \
    --num_beams 1 \
    --max_new_tokens 2048
```

with Gradio GUI

```
python gradio_gui.py
```

## FAQ

Q: Can I only use token merging during inference?

A: No, our experiments show that token merging is also a way to accelerate training while maintaining similar performance. Additionally, besides auroracap, you can also use token merging on other llava-like models.

Q: How can I find the official LLaVA-format checkpoint for AuroraCap?

A: We will release the official LLaVA-format checkpoint at a later time. While Xtuner supports saving checkpoints in both Huggingface and LLaVA formats, it currently only supports continued training with Huggingface. However, we provide the code for evaluation using the Huggingface model.

## Citation

```bibtex
```

## License

This project is released under the [Apache License 2.0](LICENSE). Please also adhere to the Licenses of models and datasets being used.

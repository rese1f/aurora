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

For quick usage only for deploy, install aurora via pip.
```
pip install aurora
```

For further development, clone this repository and install from source.
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

Since transformers version confilct, we recommand using seperated virtual environment for deploy, install addttional dependencies.
```
cd src/sglang && pip install -e "python[all]"
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

### Play with AuroraCap

```
import os.path as osp
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor
from huggingface_hub import snapshot_download

from src.xtuner.xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, PROMPT_TEMPLATE
from src.xtuner.xtuner.model.aurora import AuroraEncoder, AuroraModel
from src.xtuner.xtuner.tools.load_video import read_video_pyav

def process_text(inputs, tokenizer):
    chunk_encode = []
    for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
        if idx == 0:
            cur_encode = tokenizer.encode(chunk)
        else:
            cur_encode = tokenizer.encode(chunk, add_special_tokens=False)
        chunk_encode.append(cur_encode)
    # assert len(chunk_encode) == 2 # for single image
    ids = []
    for idx, cur_chunk_encode in enumerate(chunk_encode):
        ids.extend(cur_chunk_encode)
        if idx != len(chunk_encode) - 1:
            ids.append(IMAGE_TOKEN_INDEX)
    ids = torch.tensor(ids).cuda().unsqueeze(0)
    return ids

if __name__ == "__main__":
    model_path = 'wchai/AuroraCap-7B-VID'
    pretrained_pth = snapshot_download(repo_id=model_path) if not osp.isdir(model_path) else model_path
    pretrained_vit = osp.join(pretrained_pth, "visual_encoder")
    projector_path = osp.join(pretrained_pth, "projector")

    auroracap = AuroraModel(
        llm=AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_pth,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ),
        visual_encoder=AuroraEncoder.from_pretrained(
            pretrained_model_name_or_path=pretrained_vit,
            torch_dtype=torch.float16,
        ),
    ).cuda()
    auroracap.projector = AutoModel.from_pretrained(projector_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
    image_processor = CLIPImageProcessor.from_pretrained(
        pretrained_model_name_or_path="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",  # use standard CLIP processor
        trust_remote_code=True,
        size=378,
        crop_size=378,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=pretrained_pth,
        trust_remote_code=True,
        padding_side='right',
    )

    prompt = "describe the video in detail"
    visual_input = "test.mp4"
    data = dict()
    if visual_input.endswith('mp4'):
        video_frames = read_video_pyav(visual_input, num_frm=8)
        image_tensor = image_processor(video_frames, return_tensors='pt')['pixel_values']
        image_tensor = [_image.to(dtype=torch.float16).cuda() for _image in image_tensor]
        data["pixel_values"] = torch.stack(image_tensor).unsqueeze(0)
        image_tokens = [DEFAULT_IMAGE_TOKEN] * len(video_frames)
        image_tokens = " ".join(image_tokens)
    elif visual_input.endswith('png') or input_file.endswith('jpg'):
        image_tensor = image_processor(images, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
        image_tokens = [DEFAULT_IMAGE_TOKEN]
        data["pixel_values"] = image_tensor

    text_input = image_tokens + "\n" + prompt
    prompt_text = PROMPT_TEMPLATE.vicuna["INSTRUCTION"].format(input=text_input, round=1)
    data["input_ids"] = process_text(prompt_text, tokenizer).cuda()
    token_kept_ratio = 0.4 # modify the token kept ratio here
    auroracap.visual_encoder.reset_tome_r(token_kept_ratio)
    output = auroracap(data, mode="inference")
    cont = auroracap.llm.generate(
        **output,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        num_beams=1,
        max_new_tokens=2048,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    print(text_outputs)
```

**with Gradio GUI**

## FAQ

Q: Can I only use token merging during inference?

A: No, our experiments show that token merging is also a way to accelerate training while maintaining similar performance. Additionally, besides auroracap, you can also use token merging on other llava-like models.

## Citation

```bibtex
```

## License

This project is released under the [Apache License 2.0](LICENSE). Please also adhere to the Licenses of models and datasets being used.

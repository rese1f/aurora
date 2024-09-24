<img src="../../assets/auroracap/teaser.png" align="center">

## Resources

- [Website](https://rese1f.github.io/aurora-web/)
- [arXiv: Paper]()
- [GitHub: Code](https://github.com/rese1f/aurora)
- [Huggingface: AuroraCap Model](https://huggingface.co/collections/wchai/auroracap-66d117ffe13bedda96702013)
- [Huggingface: VDC Benchmark](https://huggingface.co/datasets/wchai/Video-Detailed-Caption)
- [Huggingface: Trainset](https://huggingface.co/datasets/wchai/AuroraCap-trainset)

## Docs

- [Train Docs](TRAIN.md) powered by [Xtuner](https://github.com/InternLM/xtuner).
- [Eval Docs](EVAL.md) powered by [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).
- [Deploy Docs](DEPLOY.md) powered by [SGLang](https://github.com/sgl-project/sglang).

[RETURN TO MAIN README](../../README.md)

## Features

AuroraCap is a efficient captioning model for image and video, achieving the best trade-off between performance and efficiency.

<img src="../../assets/auroracap/vdc_baseline.png" align="center">


## Quick Start
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

## Citation

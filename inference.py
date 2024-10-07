import os.path as osp
import argparse
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image  

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to the model', default='wchai/AuroraCap-7B-IMG-xtuner')
    parser.add_argument('--prompt', type=str, help='prompt for the model', default='Describe the video in detail.')
    parser.add_argument('--visual_input', type=str, help='path to the video or image file', default='output.png')
    parser.add_argument('--num_frm', type=int, help='number of frames to sample from the video', default=8)
    parser.add_argument('--token_kept_ratio', type=float, help='token merge ratio', default=0.8)
    parser.add_argument('--temperature', type=float, help='temperature', default=0.0)
    parser.add_argument('--top_p', type=float, help='top p', default=1.0)
    parser.add_argument('--num_beams', type=int, help='number of beams', default=1)
    parser.add_argument('--max_new_tokens', type=int, help='max new tokens', default=2048)
    args = parser.parse_args()
    
    pretrained_pth = snapshot_download(repo_id=args.model_path) if not osp.isdir(args.model_path) else args.model_path
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

    data = dict()
    if args.visual_input.endswith('mp4'):
        video_frames = read_video_pyav(args.visual_input, args.num_frm)
        image_tensor = image_processor(video_frames, return_tensors='pt')['pixel_values']
        image_tensor = [_image.to(dtype=torch.float16).cuda() for _image in image_tensor]
        data["pixel_values"] = torch.stack(image_tensor).unsqueeze(0)
        image_tokens = [DEFAULT_IMAGE_TOKEN] * len(video_frames)
        image_tokens = " ".join(image_tokens)
    elif args.visual_input.endswith('png') or args.visual_input.endswith('jpg'):
        image = Image.open(args.visual_input)
        image_tensor = image_processor(image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16).cuda()
        image_tokens = DEFAULT_IMAGE_TOKEN
        data["pixel_values"] = image_tensor

    text_input = image_tokens + "\n" + args.prompt
    prompt_text = PROMPT_TEMPLATE.vicuna["INSTRUCTION"].format(input=text_input, round=1)
    data["input_ids"] = process_text(prompt_text, tokenizer).cuda()
    auroracap.visual_encoder.reset_tome_r(args.token_kept_ratio)
    output = auroracap(data, mode="inference")
    cont = auroracap.llm.generate(
        **output,
        do_sample=False,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
    print(text_outputs)

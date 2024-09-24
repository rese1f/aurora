import os.path as osp
import gradio as gr
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

class Backend():
    def __init__(self):
        self.model_path = None
    
    def load_model(self, model_path):
        self.model_path = model_path
        
        pretrained_pth = snapshot_download(repo_id=model_path) if not osp.isdir(model_path) else model_path
        pretrained_vit = osp.join(pretrained_pth, "visual_encoder")
        projector_path = osp.join(pretrained_pth, "projector")

        self.auroracap = AuroraModel(
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
        self.auroracap.projector = AutoModel.from_pretrained(projector_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
        self.image_processor = CLIPImageProcessor.from_pretrained(
            pretrained_model_name_or_path="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",  # use standard CLIP processor
            trust_remote_code=True,
            size=378,
            crop_size=378,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_pth,
            trust_remote_code=True,
            padding_side='right',
        )
    
    def generate_text(self, model_path, prompt, visual_input, num_frm, token_kept_ratio, temperature, top_p, num_beams, max_new_tokens):
        if model_path != self.model_path:
            self.load_model(model_path)

        data = dict()
        if visual_input.endswith('mp4'):
            video_frames = read_video_pyav(visual_input, num_frm)
            image_tensor = self.image_processor(video_frames, return_tensors='pt')['pixel_values']
            image_tensor = [_image.to(dtype=torch.float16).cuda() for _image in image_tensor]
            data["pixel_values"] = torch.stack(image_tensor).unsqueeze(0)
            image_tokens = [DEFAULT_IMAGE_TOKEN] * len(video_frames)
            image_tokens = " ".join(image_tokens)
        elif visual_input.endswith('png') or visual_input.endswith('jpg'):
            image_tensor = self.image_processor(visual_input, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            image_tokens = [DEFAULT_IMAGE_TOKEN]
            data["pixel_values"] = image_tensor

        text_input = image_tokens + "\n" + prompt
        prompt_text = PROMPT_TEMPLATE.vicuna["INSTRUCTION"].format(input=text_input, round=1)
        data["input_ids"] = process_text(prompt_text, self.tokenizer).cuda()
        self.auroracap.visual_encoder.reset_tome_r(token_kept_ratio)
        output = self.auroracap(data, mode="inference")
        cont = self.auroracap.llm.generate(
            **output,
            do_sample=False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
    
        return text_outputs


if __name__ == "__main__":
    backend = Backend()

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# AuroraCap")
        
        with gr.Row():
            with gr.Column():
                video = gr.Video(width=640, height=360)
                video_button = gr.Button("Process Video")
            with gr.Column():
                image = gr.Image(width=640, height=360)
                image_button = gr.Button("Process Image")
        
        with gr.Row():
            with gr.Column():
                output = gr.Textbox(label="Output")
                prompt = gr.Textbox(label="Prompt", value="Describe the video in detail.")
            with gr.Column():
                model_path = gr.Textbox(label="Model Path", value="wchai/AuroraCap-7B-VID-xtuner")
                token_kept_ratio = gr.Slider(0, 1, value=0.2, step=0.01, label="Token Kept Ratio")
                num_frm = gr.Slider(1, 16, value=8, step=1, label="Num Frames (only for video)")
                with gr.Accordion("Advanced Options", open=False):
                    temperature = gr.Slider(0, 1, value=0., step=0.01, label="Temperature")
                    top_p = gr.Slider(0, 1, value=1., step=0.01, label="Top P")
                    num_beams = gr.Slider(1, 10, value=1, step=1, label="Num Beams")
                    max_new_tokens = gr.Slider(1, 4096, value=2048, step=1, label="Max New Tokens")

        video_button.click(backend.generate_text, inputs=[model_path, prompt, video, num_frm, token_kept_ratio, temperature, top_p, num_beams, max_new_tokens], outputs=[output])
        image_button.click(backend.generate_text, inputs=[model_path, prompt, image, num_frm, token_kept_ratio, temperature, top_p, num_beams, max_new_tokens], outputs=[output])
        
    demo.launch(share=True)
# Copyright (c) OpenMMLab. All rights reserved.
import json
import logging
import os

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine import print_log
from mmengine.dataset import DefaultSampler
from mmengine.config import Config, ConfigDict
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from xtuner.registry import BUILDER
from .huggingface import process_hf_dataset
from .utils import expand2square

PATCH_SIZE = 14

def load_jsonl(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


class AuroraDataset(Dataset):

    def __init__(
        self,
        image_folder,
        image_processor,
        data_path=None,
        tokenizer=None,
        offline_processed_text_folder=None,
        max_dataset_length=None,
        dataset_map_fn=None,
        template_map_fn=None,
        max_length=2048,
        pad_image_to_square=False,
        keep_aspect_ratio=True,
    ):
        super().__init__()

        if offline_processed_text_folder is not None:
            self.text_data = load_from_disk(offline_processed_text_folder)
            print_log(f"Loaded {len(self.text_data)} samples from {offline_processed_text_folder}")
        else:
            if data_path.endswith('.json'):
                json_data = json.load(open(data_path))
            elif data_path.endswith('.jsonl'):
                json_data = load_jsonl(data_path)
            else:
                raise NotImplementedError

            print("start filter data without id")
            json_data = [
                {**item, 'id': str(item['id'])} if 'id' in item and isinstance(item['id'], int) else item
                for item in json_data
                if 'id' in item
            ]
            print("finish filter data without id")

            print("start transform int id")
            for idx in range(len(json_data)):
                if isinstance(json_data[idx]['id'], int):
                    json_data[idx]['id'] = str(json_data[idx]['id'])
            print("finish transform int id")


            json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
            self.text_data = process_hf_dataset(
                dataset=json_data,
                tokenizer=tokenizer,
                max_length=max_length,
                dataset_map_fn=dataset_map_fn,
                template_map_fn=template_map_fn,
                split='train',
                max_dataset_length=max_dataset_length,
                remove_unused_columns=False,
                pack_to_max_length=False,
                with_image_token=True)
        self.image_folder = image_folder
        if (
            isinstance(image_processor, dict)
            or isinstance(image_processor, Config)
            or isinstance(image_processor, ConfigDict)
        ):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor
        self.pad_image_to_square = pad_image_to_square
        self.keep_aspect_ratio = keep_aspect_ratio


    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.text_data:
            cur_len = len(data_dict['input_ids'])
            if data_dict.get('image', None) is None:
                cur_len = -cur_len
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):
        data_dict = self.text_data[index]

        if data_dict.get("image", None) is not None:
            image_files = data_dict["image"]
            
            if not isinstance(image_files, list):
                image_files = [image_files]  

            processed_images = []

            for image_file in image_files:
                try:
                    if os.path.exists(image_file):
                        image = Image.open(image_file).convert("RGB")
                    else:
                        image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
                except:
                    crop_size = self.image_processor.crop_size
                    processed_images.append(torch.zeros(3, crop_size["height"], crop_size["width"]))
                    continue

                if self.pad_image_to_square:
                    image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))

                if self.keep_aspect_ratio:
                    h, w = image.size
                    scale = self.image_processor.size['shortest_edge'] / min(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    image = image.resize((new_w, new_h))

                    pad_h = (PATCH_SIZE - new_h % PATCH_SIZE) % PATCH_SIZE
                    pad_w = (PATCH_SIZE - new_w % PATCH_SIZE) % PATCH_SIZE
                    image = ImageOps.expand(image, (0, 0, pad_w, pad_h), fill=(0, 0, 0))

                    image = self.image_processor.preprocess(image, do_center_crop=False, do_resize=False, return_tensors='pt')['pixel_values'][0]
                else:
                    image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

                processed_images.append(image)

            data_dict["pixel_values"] = torch.stack(processed_images)
        else:
            crop_size = self.image_processor.crop_size
            data_dict["pixel_values"] = torch.stack([torch.zeros(3, crop_size["height"], crop_size["width"])])

        return data_dict
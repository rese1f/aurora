# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import warnings
import os
import shutil
import json

from mmengine import Config

from xtuner.registry import BUILDER

# ignore FutureWarning in hf datasets
warnings.simplefilter(action='ignore', category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--save-folder', help='The folder to save data order.')
    parser.add_argument('--input-file', help='The folder to save data order.')
    parser.add_argument('--jsonl-output-folder', help='The folder to save data order.')
    args = parser.parse_args()
    return args

def split_large_jsonl(input_file, output_folder, chunk_size=100000, prefix="data_part_"):
    os.makedirs(output_folder, exist_ok=True)
    with open(input_file, 'r') as infile:
        for i, line in enumerate(infile):
            if i % chunk_size == 0:
                if 'outfile' in locals():
                    outfile.close()
                output_file = os.path.join(output_folder, f"{prefix}{i // chunk_size:05d}.jsonl")
                outfile = open(output_file, 'w')
            outfile.write(line)
        if 'outfile' in locals():
            outfile.close()



def build_dataset_from_jsonl(config, jsonl_folder, arrow_folder, prefix="data_part_"):
    os.makedirs(arrow_folder, exist_ok=True)
    for f in os.listdir(jsonl_folder):
        jsonl_path = os.path.join(jsonl_folder, f)
        config.train_dataloader.dataset['data_path'] = jsonl_path
        arrow_path = os.path.join(arrow_folder, f"{f.split('.')[0]}")
        
        if os.path.exists(arrow_path):
            print(f"arrow_path {arrow_path} already exists, skipping...")
            continue

        dataset = BUILDER.build(config.train_dataloader.dataset)
        text_data = dataset.text_data
        text_data.save_to_disk(arrow_path)
        print(f"Finish processing {jsonl_path}")





if __name__ == "__main__":
    args = parse_args()
    # cfg = Config.fromfile(args.config)

    config_path = args.config
    config_data = None
    
    
    # Try reading the config file with utf-8 encoding first
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = f.read()
    except UnicodeDecodeError:
        pass
    
    # If utf-8 decoding fails, try with latin1 encoding
    if config_data is None:
        try:
            with open(config_path, 'r', encoding='latin1') as f:
                config_data = f.read()
        except UnicodeDecodeError as e:
            print(f"Failed to read the config file with both utf-8 and latin1 encodings: {e}")
            exit(1)
    
    # Load the configuration
    cfg = Config.fromstring(config_data, file_format=os.path.splitext(config_path)[1])
    

    input_file = args.input_file
    jsonl_output_folder = args.jsonl_output_folder
    arrow_output_folder = args.save_folder

    
    split_large_jsonl(input_file, jsonl_output_folder, chunk_size=100000)
    
    build_dataset_from_jsonl(cfg, jsonl_output_folder, arrow_output_folder, prefix="data_part_")


import datasets
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Merge datasets.')
parser.add_argument('--data_folder', type=str, required=True, help='Path to the data folder')
parser.add_argument('--save_folder', type=str, required=True, help='Path to save the merged dataset')
args = parser.parse_args()

data_folder = args.data_folder

data_parts = [f.path for f in os.scandir(data_folder) if f.is_dir()]

datasets_list = [datasets.load_from_disk(part) for part in data_parts]

required_columns = ['id', 'image', 'conversation', 'input_ids', 'labels']

def align_features(dataset, required_columns):
    new_data = {col: dataset[col] if col in dataset.column_names else [None] * len(dataset) for col in required_columns}
    return datasets.Dataset.from_dict(new_data)


aligned_datasets_list = [align_features(dataset, required_columns) for dataset in datasets_list]

full_dataset = datasets.concatenate_datasets(aligned_datasets_list)

print(len(full_dataset))

print(full_dataset[:1])

# Save the merged dataset
full_dataset.save_to_disk(args.save_folder)
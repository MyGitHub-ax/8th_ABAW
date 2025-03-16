import math
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, CLIPImageProcessor
from concurrent.futures import ThreadPoolExecutor
from torch.nn import DataParallel
import argparse

class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        return img


def parse_args():
    parser = argparse.ArgumentParser(description='Feature extraction script.')
    parser.add_argument('--batch_size', type=str, default=16)
    parser.add_argument('--dim', type=int, default=3200) 
    return parser.parse_args()

# Model setup
os.environ['CUDA_VISIBLE_DEVICES'] ='0,1,2,3'
model_dir = '/data/wenzhuofan/Data/tools/InternViT-6B-448px-V1-5'
model = AutoModel.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).cuda().eval()
model = DataParallel(model)
image_processor = CLIPImageProcessor.from_pretrained(model_dir)

def split_into_batch(inputs, bsize=32):
    batches = []
    for ii in range(math.ceil(len(inputs) / bsize)):
        batch = inputs[ii * bsize:(ii + 1) * bsize]
        batches.append(batch)
    return batches
def load_image(path):
    return Image.open(path).convert('RGB')
def predict(paths,batch_size):
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(load_image, [os.path.join(paths,file)for file in os.listdir(paths)]))
    with torch.no_grad():
        # 使用 image_processor 处理一批图像
        pixel_values = image_processor(images=images, return_tensors='pt').pixel_values
        batches = split_into_batch(pixel_values, bsize=batch_size)
        all_features = []
        for batch in batches:
            batch = batch.to(torch.bfloat16).cuda()
            output = model(batch)
            features = output.pooler_output
            features = features.float()
            all_features.append(features)
        all_features=torch.cat(all_features,dim=0)
    return all_features.detach().cpu().numpy()


args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = r'/data/wenzhuofan/Data/MuSe2024DA/data_extract'
face_dir = os.path.join(root, 'faces')
overlap_length='o10l15'
save_path=f'/data/wenzhuofan/Data/MuSe2024DA/c1_muse_perception/feature_segments/InternViT-UTT-{overlap_length}'
if not os.path.exists(save_path):
    os.makedirs(save_path)

for face_file in tqdm(os.listdir(face_dir)):
    if os.path.exists(os.path.join(save_path,f'{face_file}.npy')):
        continue
    print(f"Predicting features for image: {os.path.join(face_dir, face_file)}")
    embeddings = predict(os.path.join(face_dir, face_file),args.batch_size)
    embeddings = np.mean(embeddings, axis=0)
    np.save(os.path.join(save_path,f'{face_file}.npy'), embeddings)


# if not os.path.exists(os.path.dirname(target_file)):
#     os.makedirs(os.path.dirname(target_file))
# feature_data.to_csv(target_file, index=False)
# print(f"Saved processed data to {target_file}")
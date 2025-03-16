import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# Each input text should start with "query: " or "passage: ", even for non-English texts.
# For tasks other than retrieval, you can simply use the "query: " prefix.
# input_texts = ['query: how much protein should a female eat',
#                'query: 南瓜的家常做法',
#                "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
#                "passage: 1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"]

root = r'/data/wenzhuofan/Data/MuSe2024DA/data_extract'
model_dir='/data/wenzhuofan/Data/tools/multilingual-e5-large'
transcriptions_dir = os.path.join(root, 'transcriptions')
overlap_length='o10l15'
save_path=f'/data/wenzhuofan/Data/MuSe2024DA/c1_muse_perception/feature_segments/multilingual-e5-large-UTT-{overlap_length}'
if not os.path.exists(save_path):
    os.makedirs(save_path)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir)

for transcriptions_file in tqdm(os.listdir(transcriptions_dir)):
    file_name=transcriptions_file.split('.')[0]
    if os.path.exists(os.path.join(save_path,f'{file_name}.npy')):
        continue
    # Tokenize the input texts
    df = pd.read_csv(os.path.join(transcriptions_dir, transcriptions_file))
    sentences=[]
    print(f'Processing {transcriptions_file}...')
    for idx, row in df.iterrows():
        sentence = 'query: '+ row['sentence']
        sentences.append(sentence)

        # extract embedding from sentences
    if  len(sentences) > 0:
        batch_dict = tokenizer(sentences, max_length=1024, padding=True, truncation=True, return_tensors='pt')

        outputs = model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1).detach().cpu().numpy()
        embeddings = np.mean(embeddings, axis=0)
        np.save(os.path.join(save_path,f'{file_name}.npy'), embeddings)

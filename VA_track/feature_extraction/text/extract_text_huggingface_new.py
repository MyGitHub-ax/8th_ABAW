# *_*coding:utf-8 *_*
import os
import time
import argparse
import numpy as np
import pandas as pd

import torch
from transformers import AutoModel, BertTokenizer, AutoTokenizer # version: 4.5.1, pip install transformers
from transformers import GPT2Tokenizer, GPT2Model, AutoModelForCausalLM

# local folder
import sys
sys.path.append('../../')
import config

##################### English #####################
BERT_BASE = 'bert-base-cased'
BERT_LARGE = 'bert-large-cased'
BERT_BASE_UNCASED = 'bert-base-uncased'
BERT_LARGE_UNCASED = 'bert-large-uncased'
ALBERT_BASE = 'albert-base-v2'
ALBERT_LARGE = 'albert-large-v2'
ALBERT_XXLARGE = 'albert-xxlarge-v2'
ROBERTA_BASE = 'roberta-base'
ROBERTA_LARGE = 'roberta-large'
ELECTRA_BASE = 'electra-base-discriminator'
ELECTRA_LARGE = 'electra-large-discriminator'
XLNET_BASE = 'xlnet-base-cased'
XLNET_LARGE = 'xlnet-large-cased'
T5_BASE = 't5-base'
T5_LARGE = 't5-large'
DEBERTA_BASE = 'deberta-base'
DEBERTA_LARGE = 'deberta-large'
DEBERTA_XLARGE = 'deberta-v2-xlarge'
DEBERTA_XXLARGE = 'deberta-v2-xxlarge'

##################### Chinese #####################
BERT_BASE_CHINESE = 'bert-base-chinese' # https://huggingface.co/bert-base-chinese
ROBERTA_BASE_CHINESE = 'chinese-roberta-wwm-ext' # https://huggingface.co/hfl/chinese-roberta-wwm-ext
ROBERTA_LARGE_CHINESE = 'chinese-roberta-wwm-ext-large' # https://huggingface.co/hfl/chinese-roberta-wwm-ext-large
DEBERTA_LARGE_CHINESE = 'deberta-chinese-large' # https://huggingface.co/WENGSYX/Deberta-Chinese-Large
ELECTRA_SMALL_CHINESE = 'chinese-electra-180g-small' # https://huggingface.co/hfl/chinese-electra-180g-small-discriminator
ELECTRA_BASE_CHINESE  = 'chinese-electra-180g-base' # https://huggingface.co/hfl/chinese-electra-180g-base-discriminator
ELECTRA_LARGE_CHINESE = 'chinese-electra-180g-large' # https://huggingface.co/hfl/chinese-electra-180g-large-discriminator
XLNET_BASE_CHINESE = 'chinese-xlnet-base' # https://huggingface.co/hfl/chinese-xlnet-base
MACBERT_BASE_CHINESE = 'chinese-macbert-base' # https://huggingface.co/hfl/chinese-macbert-base
MACBERT_LARGE_CHINESE = 'chinese-macbert-large' # https://huggingface.co/hfl/chinese-macbert-large
PERT_BASE_CHINESE = 'chinese-pert-base' # https://huggingface.co/hfl/chinese-pert-base
PERT_LARGE_CHINESE = 'chinese-pert-large' # https://huggingface.co/hfl/chinese-pert-large
LERT_SMALL_CHINESE = 'chinese-lert-small' # https://huggingface.co/hfl/chinese-lert-small
LERT_BASE_CHINESE  = 'chinese-lert-base' # https://huggingface.co/hfl/chinese-lert-base
LERT_LARGE_CHINESE = 'chinese-lert-large' # https://huggingface.co/hfl/chinese-lert-large
GPT2_CHINESE = 'gpt2-chinese-cluecorpussmall' # https://huggingface.co/uer/gpt2-chinese-cluecorpussmall
CLIP_CHINESE = 'taiyi-clip-roberta-chinese' # https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese
WENZHONG_GPT2_CHINESE = 'wenzhong2-gpt2-chinese' # https://huggingface.co/IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese
ALBERT_TINY_CHINESE = 'albert_chinese_tiny' # https://huggingface.co/clue/albert_chinese_tiny
ALBERT_SMALL_CHINESE = 'albert_chinese_small' # https://huggingface.co/clue/albert_chinese_small
SIMBERT_BASE_CHINESE = 'simbert-base-chinese' # https://huggingface.co/WangZeJun/simbert-base-chinese

##################### Multilingual #####################
MPNET_BASE = 'paraphrase-multilingual-mpnet-base-v2' # https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2

##################### LLM #####################
LLAMA_7B  = 'llama-7b-hf' # https://huggingface.co/decapoda-research/llama-7b-hf
LLAMA_13B = 'llama-13b-hf' # https://huggingface.co/decapoda-research/llama-13b-hf
LLAMA2_7B = 'llama-2-7b' # https://huggingface.co/meta-llama/Llama-2-7b
LLAMA2_13B = 'Llama-2-13b-hf' # https://huggingface.co/NousResearch/Llama-2-13b-hf
VICUNA_7B  = 'vicuna-7b-v0' # https://huggingface.co/lmsys/vicuna-7b-delta-v0
VICUNA_13B = 'stable-vicuna-13b' # https://huggingface.co/CarperAI/stable-vicuna-13b-delta
ALPACE_13B = 'chinese-alpaca-2-13b' # https://huggingface.co/ziqingyang/chinese-alpaca-2-13b
MOSS_7B = 'moss-base-7b' # https://huggingface.co/fnlp/moss-base-7b
STABLEML_7B = 'stablelm-base-alpha-7b-v2' # https://huggingface.co/stabilityai/stablelm-base-alpha-7b-v2
BLOOM_7B = 'bloom-7b1' # https://huggingface.co/bigscience/bloom-7b1
CHATGLM2_6B = 'chatglm2-6b' # https://huggingface.co/THUDM/chatglm2-6b
# reley on pytorch=2.0 => env: videollama4 + cpu
FALCON_7B = 'falcon-7b' # https://huggingface.co/tiiuae/falcon-7b
# Baichuan: pip install transformers_stream_generator
BAICHUAN_7B = 'Baichuan-7B' # https://huggingface.co/baichuan-inc/Baichuan-7B
BAICHUAN_13B = 'Baichuan-13B-Base' # https://huggingface.co/baichuan-inc/Baichuan-13B-Base
# BAICHUAN2_7B: conda install xformers -c xformers
BAICHUAN2_7B = 'Baichuan2-7B-Base' # https://huggingface.co/baichuan-inc/Baichuan2-7B-Base
# BAICHUAN2_13B: pip install accelerate
BAICHUAN2_13B = 'Baichuan2-13B-Base' # https://huggingface.co/baichuan-inc/Baichuan2-13B-Base
OPT_13B = 'opt-13b' # https://huggingface.co/facebook/opt-13b


################################################################
# 自动删除无意义token对应的特征
# def find_start_end_pos(tokenizer):
#     sentence = 'The weather is great today' # 句子中没有空格
#     input_ids = tokenizer(sentence, return_tensors='pt')['input_ids'][0]
#     start, end = None, None
#
#     # find start, must in range [0, 1, 2]
#     for start in range(0, 3, 1):
#         # 因为decode有时会出现空格，因此我们显示的时候把这部分信息去掉看看
#         outputs = tokenizer.decode(input_ids[start:]).replace(' ', '')
#         if outputs == sentence:
#             print (f'start: {start};  end: {end}')
#             return start, None
#
#         if outputs.startswith(sentence):
#             break
#
#     # find end, must in range [-1, -2]
#     for end in range(-1, -3, -1):
#         outputs = tokenizer.decode(input_ids[start:end]).replace(' ', '')
#         if outputs == sentence:
#             break
#
#     assert tokenizer.decode(input_ids[start:end]).replace(' ', '') == sentence
#     print (f'start: {start};  end: {end}')
#     return start, end
def find_start_end_pos(tokenizer, language='english'):
    """
    确定分词后 input_ids 中有效的起始和结束位置，以去除无意义的 token。

    参数：
    - tokenizer: 分词器对象。
    - language: 处理的语言，支持 'english' 和 'chinese'。

    返回：
    - start: 有效 token 的起始索引。
    - end: 有效 token 的结束索引（可为 None）。
    """
    if language == 'chinese':
        sentence = '今天天气真好'  # 中文句子
    elif language == 'english':
        sentence = 'The weather is great today.'  # 英文句子
    else:
        raise ValueError(f"Unsupported language: {language}")

    # 编码句子
    encoding = tokenizer(sentence, return_tensors='pt', add_special_tokens=True)
    input_ids = encoding['input_ids'][0]
    print(f"Input IDs: {input_ids}")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(input_ids)}")

    start, end = None, None

    # 寻找起始位置，必须在范围 [0, 1, 2]
    for start in range(0, 3, 1):
        # 解码并去除空格，跳过特殊标记
        outputs = tokenizer.decode(input_ids[start:], skip_special_tokens=True).replace(' ', '')
        print(f"start={start}, decoded_full='{outputs}'")
        if outputs == sentence.replace(' ', ''):
            print(f'start: {start};  end: {end}')
            return start, None

        if outputs.startswith(sentence.replace(' ', '')):
            print(f"outputs.startswith(sentence): start={start}")
            break

    # 寻找结束位置，必须在范围 [-1, -2]
    for end in range(-1, -3, -1):
        outputs = tokenizer.decode(input_ids[start:end], skip_special_tokens=True).replace(' ', '')
        print(f"end={end}, decoded_end='{outputs}'")
        if outputs == sentence.replace(' ', ''):
            print(f"Match found at end={end}")
            break

    decoded = tokenizer.decode(input_ids[start:end], skip_special_tokens=True).replace(' ', '')
    print(f"Decoded output: '{decoded}'")
    print(f"Original sentence: '{sentence.replace(' ', '')}'")
    assert decoded == sentence.replace(' ',
                                       ''), f"Decoded output '{decoded}' does not match original sentence '{sentence.replace(' ', '')}'"
    print(f'start: {start};  end: {end}')
    return start, end


def find_batchpos_embdim(tokenizer, model, gpu, language='english'):
    """
    确定嵌入输出的批次维度位置和特征维度。

    参数：
    - tokenizer: 分词器对象。
    - model: 预训练模型对象。
    - gpu: GPU ID，如果为 -1 则使用 CPU。
    - language: 处理的语言，支持 'english' 和 'chinese'。

    返回：
    - batch_pos: 批次维度的位置（0 或 1）。
    - feature_dim: 嵌入的特征维度。
    """
    if language == 'chinese':
        sentence = '今天天气真好'
    elif language == 'english':
        sentence = 'The weather is great today.'
    else:
        raise ValueError(f"Unsupported language: {language}")

    inputs = tokenizer(sentence, return_tensors='pt', add_special_tokens=True)
    if gpu != -1:
        inputs = inputs.to('cuda')

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True).hidden_states  # 获取所有隐藏层输出
        outputs = torch.stack(outputs)[[-1]].sum(dim=0)  # 取最后一层并求和 => [batch, T, D=768]
        outputs = outputs.cpu().numpy()  # 转为 NumPy 数组
        batch_pos = None
        if outputs.shape[0] == 1:
            batch_pos = 0
        if outputs.shape[1] == 1:
            batch_pos = 1
        assert batch_pos in [0, 1], f"Unexpected batch_pos: {batch_pos}"
        feature_dim = outputs.shape[2]
    print(f'batch_pos: {batch_pos}, feature_dim: {feature_dim}')
    return batch_pos, feature_dim


def extract_embedding(model_name, trans_dir, save_dir, feature_level, gpu=-1, punc_case=None, language='english',
                      model_dir=None):
    """
    提取文本嵌入并保存为 .npy 文件。

    参数：
    - model_name: 预训练模型名称。
    - trans_dir: 包含 .txt 文件的目录路径。
    - save_dir: 保存嵌入的目录路径。
    - feature_level: 'UTTERANCE' 或 'FRAME'。
    - gpu: GPU ID，如果为 -1 则使用 CPU。
    - punc_case: 标点符号处理情况（可选）。
    - language: 处理的语言，支持 'english' 和 'chinese'。
    - model_dir: 预训练模型的自定义路径（可选）。
    """
    print('=' * 30 + f' Extracting "{model_name}" ' + '=' * 30)
    start_time = time.time()

    # 定义要保存的最后四层的 ID
    layer_ids = [-4, -3, -2, -1]

    # 确定保存目录
    if punc_case is None and language == 'chinese' and model_dir is None:
        save_dir = os.path.join(save_dir, f'{model_name}-{feature_level[:3]}')
    elif punc_case is not None:
        save_dir = os.path.join(save_dir, f'{model_name}-punc{punc_case}-{feature_level[:3]}')
    elif language == 'english':
        save_dir = os.path.join(save_dir, f'{model_name}-langeng-{feature_level[:3]}')
    elif model_dir is not None:
        prefix_name = "-".join(model_dir.split('/')[-2:])
        save_dir = os.path.join(save_dir, f'{prefix_name}-{model_name}-{feature_level[:3]}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 加载预训练模型和分词器
    print('Loading pre-trained tokenizer and model...')
    if model_dir is None:
        model_dir = model_name  # 使用 HuggingFace 上的模型

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModel.from_pretrained(model_dir, output_hidden_states=True)

    # 如果使用 GPU，则将模型移动到 GPU
    if gpu != -1:
        torch.cuda.set_device(gpu)
        model.cuda()

    # 如果是大规模语言模型，转换为半精度
    large_models = [
        'llama-7b', 'llama-13b', 'llama-2-7b', 'llama-2-13b',
        'vicuna-7b', 'vicuna-13b', 'alpace-13b', 'opt-13b',
        'bloom-7b1', 'chatglm2-6b', 'moss-base-7b', 'falcon-7b',
        'baichuan-7b', 'baichuan-13b', 'baichuan2-7b', 'baichuan2-13b',
        'stablelm-base-alpha-7b-v2'
    ]
    if gpu != -1 and model_name.lower() in large_models:
        model = model.half()

    model.eval()

    print('Calculate embeddings...')
    start, end = find_start_end_pos(tokenizer, language=language)  # 传递语言参数
    batch_pos, feature_dim = find_batchpos_embdim(tokenizer, model, gpu, language=language)  # find batch pos

    # 读取 trans_dir 下的所有 .txt 文件
    print(f"trans_dir: {trans_dir}")
    if os.path.isdir(trans_dir):
        print(f"trans_dir is a directory. Processing all .txt files within the directory...")
        txt_files = [f for f in os.listdir(trans_dir) if f.endswith('.txt') and not f.startswith('.')]
        if not txt_files:
            print(f"No .txt files found in the directory: {trans_dir}")
            sys.exit(1)
    elif os.path.isfile(trans_dir) and trans_dir.endswith('.txt'):
        print(f"trans_dir is a single .txt file. Processing the file...")
        txt_files = [trans_dir]
    else:
        print(f"Error: {trans_dir} is not a valid directory or .txt file.")
        sys.exit(1)

    # 处理每个 .txt 文件
    for file in txt_files:
        if os.path.isdir(trans_dir):
            filepath = os.path.join(trans_dir, file)
            name = os.path.splitext(file)[0]  # 使用文件名（不含扩展名）作为 'name'
            with open(filepath, 'r', encoding='utf-8') as f:
                sentence = f.read().strip()  # 读取文本
        else:
            # 如果 trans_dir 是单个 .txt 文件
            filepath = trans_dir
            name = os.path.splitext(os.path.basename(file))[0]
            with open(filepath, 'r', encoding='utf-8') as f:
                sentence = f.read().strip()

        print(f'Processing {name} ({file})...')

        # 提取嵌入
        embeddings = []
        if sentence and len(sentence) > 0:
            inputs = tokenizer(sentence, return_tensors='pt', add_special_tokens=True)
            if gpu != -1:
                inputs = inputs.to('cuda')
            with torch.no_grad():
                outputs = model(**inputs).hidden_states  # 获取所有隐藏层输出
                # 取指定层的输出并求和
                selected_layers = [outputs[i] for i in layer_ids]  # 修改这里
                summed_layers = torch.stack(selected_layers).sum(dim=0)
                embeddings = summed_layers.cpu().numpy()  # 转为 NumPy 数组

                if batch_pos == 0:
                    embeddings = embeddings[start:end]
                elif batch_pos == 1:
                    embeddings = embeddings[start:end, 0]

        # 保存嵌入为 .npy 文件
        print(f'feature dimension: {feature_dim}')
        npy_file = os.path.join(save_dir, f'{name}.npy')
        if feature_level == 'FRAME':
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((1, feature_dim))
            elif len(embeddings.shape) == 1:
                embeddings = embeddings[np.newaxis, :]
            np.save(npy_file, embeddings)
        else:  # 'UTTERANCE'
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((feature_dim,))
            elif len(embeddings.shape) == 2:
                embeddings = embeddings.mean(axis=0)
            np.save(npy_file, embeddings)

    end_time = time.time()
    print(f'Total {len(txt_files)} files done! Time used ({model_name}): {end_time - start_time:.1f}s.')


def parse_arguments():
    """
    解析命令行参数。

    返回：
    - args: 解析后的参数对象。
    """
    parser = argparse.ArgumentParser(description='Extract text embeddings using HuggingFace models.')
    parser.add_argument('--trans_dir', type=str, required=True,
                        help='Path to directory containing .txt files or a single .txt file.')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to directory to save the extracted embeddings.')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the pretrained model (e.g., bert-base-cased).')
    parser.add_argument('--feature_level', type=str, default='UTTERANCE', choices=['UTTERANCE', 'FRAME'],
                        help='Output feature level.')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID to use (default: -1 for CPU).')
    parser.add_argument('--punc_case', type=str, default=None,
                        help='Test punctuation impact to the performance (optional).')
    parser.add_argument('--language', type=str, default='english', choices=['english', 'chinese'],
                        help='Language of the text.')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Path to a user-defined model directory (optional).')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    # 检查 trans_dir 是否存在
    if not os.path.exists(args.trans_dir):
        print(f"Error: trans_dir '{args.trans_dir}' does not exist.")
        sys.exit(1)

    # 检查 save_dir 是否存在，不存在则创建
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"Created save_dir at '{args.save_dir}'.")

    extract_embedding(
        model_name=args.model_name,
        trans_dir=args.trans_dir,
        save_dir=args.save_dir,
        feature_level=args.feature_level,
        gpu=args.gpu,
        punc_case=args.punc_case,
        language=args.language,
        model_dir=args.model_dir
    )
import os
import time
import math
import shutil
import sys
import torch
import pandas as pd
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from data.abaw_data import HumeDatasetEval, HumeDatasetTrain, HumeDatasetTest
from data.utils import setup_system, Logger
from data.trainer import train
from data.evaluate import evaluate, predict_test
from data.loss import MSE, CCC, MSECCC, CORR
from data.model import Model
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, \
    get_cosine_schedule_with_warmup
import pickle
import numpy as np

##### VISION new STANDARD 1e-4


@dataclass
class TrainingConfiguration:
    '''
    Describes configuration of the training process
    '''

    # Model
    model: tuple = ('linear',
                    'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim',
                    'bert-base-uncased')
    #model: tuple = ('timm/convnext_base.fb_in22k_ft_in1k', 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim')#'facebook/wav2vec2-large-960h') # ('facebook/dinov2-small', 'hf-audio/wav2vec2-bert-CV16-en') or ('linear', 'linear')
    # model: tuple = ('trpakov/vit-face-expression',
    #                 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim')  # 'facebook/wav2vec2-large-960h') # ('facebook/dinov2-small', 'hf-audio/wav2vec2-bert-CV16-en') or ('linear', 'linear')

    # Training
    mixed_precision: bool = True
    seed = 1
    epochs: int = 10
    batch_size: int = 2  # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple = (0,)  # GPU ids for training

    # Eval
    batch_size_eval: int = 16
    eval_every_n_epoch: int = 1  # eval every n Epoch

    # Optimizer
    clip_grad = 100.  # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False  # Gradient Checkpointing

    # Loss
    loss: str = 'MSE'  # MSE, CCC, MSECCC choice wise

    # Learning Rate
    lr: float = 5e-5  # 1
    scheduler: str = "cosine"  # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 1
    lr_end: float = 0.0001  # only for "polynomial"
    gradient_accumulation: int = 1

    # Dataset
    data_folder = "./data/test/"

    # Savepath for model checkpoints
    model_path: str = "./hume_model"

    # Checkpoint to start from
    checkpoint_start = "weights_e10_0.3650.pth"     # USE CHECKPOINT HERE

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4

    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for better performance
    cudnn_benchmark: bool = True

    # make cudnn deterministic
    cudnn_deterministic: bool = False


# -----------------------------------------------------------------------------#
# Train Config                                                                #
# -----------------------------------------------------------------------------#

config = TrainingConfiguration()

# %%

if __name__ == '__main__':

    model_path = "{}/{}/{}".format(config.model_path,
                                   config.model,
                                   time.strftime("%H%M%S"))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=config.seed,
                 cudnn_benchmark=config.cudnn_benchmark,
                 cudnn_deterministic=config.cudnn_deterministic)

    # -----------------------------------------------------------------------------#
    # Model                                                                       #
    # -----------------------------------------------------------------------------#

    print("\nModel: {}".format(config.model))

    model = Model(config.model)

    # load pretrained Checkpoint
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)

        # Data parallel
    print("GPUs available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    # Model to device
    model = model.to(config.device)

    # -----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    # -----------------------------------------------------------------------------#

    eval_dataset = HumeDatasetTest(data_folder=config.data_folder,
                                   model=config.model,
                                   )

    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=config.batch_size_eval,
                                 num_workers=config.num_workers,
                                 shuffle=False,
                                 pin_memory=True,
                                 collate_fn=eval_dataset.collate_fn)

    print("Test Length:", len(eval_dataset))

    preds, filenames = predict_test(config=config,
                                    model=model,
                                    dataloader=eval_dataloader)

    preds_array = preds.numpy()
    preds_array = np.hstack([np.array(filenames).reshape(len(filenames), 1) , preds_array])
    preds_df = pd.DataFrame(preds_array, columns=['Filename', 'Admiration', 'Amusement', 'Determination', 'Empathic Pain', 'Excitement', 'Joy'])
    preds_df = preds_df.sort_values(by='Filename')
    preds_df['Filename'] = preds_df['Filename'].astype(int)
    preds_csv_path = f'{str(config.checkpoint_start).replace(".pth", "")}_test_preds.csv'
    preds_df.to_csv(preds_csv_path, index=False)

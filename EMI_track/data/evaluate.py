import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
from torchmetrics.regression import PearsonCorrCoef

def evaluate(config, model, eval_dataloader):
    with torch.no_grad():
        preds, labels, filenames = predict(config, model, eval_dataloader)
        r = PearsonCorrCoef(num_outputs=6)
        r = r(preds, labels)
        r = r.mean()
    return r.cpu().numpy(), preds, filenames


def predict(train_config, model, dataloader):
    model.eval()

    # wait before starting progress bar
    time.sleep(0.1)

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    r = PearsonCorrCoef(num_outputs=6).cuda()
    preds = []
    labels = []
    filenames = []
    with torch.no_grad():

        for audio, vision, text, label, filename in bar:  # 新增 text

            with autocast():

                audio = audio.to(train_config.device)
                vision = vision.to(train_config.device)
                text = {k: v.to(train_config.device) for k, v in text.items()}
                label = label.to(train_config.device)
                pred = model(audio, vision, text)

            # save features in fp32 for sim calculation
            labels.append(label.detach().cpu())
            preds.append(pred.to(torch.float32).detach().cpu())
            filenames.extend(filename)
            r.update(pred, label)
            bar.set_postfix(ordered_dict={'corr': f'{r.compute().mean().cpu().numpy():.4f}'})
        # keep Features on GPU
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

    if train_config.verbose:
        bar.close()

    return preds, labels, filenames

def predict_test(config, model, dataloader):
    model.eval()

    # wait before starting progress bar
    time.sleep(0.1)

    if config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    preds = []
    filenames = []
    with torch.no_grad():
        for audio, vision, text, filename in bar:
            with autocast():
                audio = audio.to(config.device)
                vision = vision.to(config.device)
                text = {k: v.to(config.device) for k, v in text.items()}
                pred = model(audio, vision, text)

            # save predictions
            preds.append(pred.to(torch.float32).detach().cpu())
            filenames.extend(filename)

        # concatenate predictions
        preds = torch.cat(preds, dim=0)

    if config.verbose:
        bar.close()

    return preds, filenames
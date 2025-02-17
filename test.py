import torch
from configs.option import get_option
from tools.datasets.dataset_m2e.dataset import *
from models import LAST, Huggingface_LAST, TrOCRConfig
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from tqdm import tqdm
import cv2
from utils.vocab import vocab
import editdistance

def last_load(model_path="./checkpoints/baseline_LAST_V1.0/epoch_53-loss_0.330.ckpt"):
    model = LAST(
        d_model=256,
        growth_rate=24,
        num_layers=16,
        nhead=8,
        num_decoder_layers=3,
        dim_feedforward=1024, 
        dropout=0.3,
        nline=16,
        )
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint['state_dict']
    # print(state_dict.keys())
    # print(model.state_dict().keys())
    # assert 0
    update_dict = {}
    for k, v in state_dict.items():
        update_dict[k[6:]] = v
    model.load_state_dict(update_dict)
    model.eval()
    return model 


model = last_load()
regressor=model.bar
opt = get_option("config_last_m2edataset.yaml")
opt.batch_size = 1
train_dataloader, valid_dataloader = get_dataloader(opt)

correct = 0
total = 0
pbar = tqdm(valid_dataloader, desc="Progress")

for i, batch in enumerate(pbar):
    image, image_mask, task_batches = (batch["imgs"], batch["img_mask"], batch["task_batches"])
    for k, task_seq in task_batches.items():
        if k.startswith('bline'):
            ans, ti, n = regressor(image, image_mask, task_name=k)
            target_list = task_seq.tgt[0].view(-1).tolist()

            target_list = [value for value in target_list if value < 121]
            pred = vocab.indices2label(target_list)
            target = vocab.lindices2llabel(ans).replace(' \\n ', ' ')

            total += 1
            if pred == target:
                correct += 1

            pbar.set_postfix({'ExpRate': f'{correct / total:.4f}'})


print("ExpRate:", correct / total)



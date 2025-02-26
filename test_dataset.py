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

# model = LAST(
#         d_model=256,
#         growth_rate=24,
#         num_layers=16,
#         nhead=8,
#         num_decoder_layers=3,
#         dim_feedforward=1024, 
#         dropout=0.3,
#         nline=16,
#     )

model = Huggingface_LAST(
            config=TrOCRConfig(vocab_size=415, max_length=80, d_model=256,
                decoder_layers=3,
                decoder_attention_heads=8,
                decoder_ffn_dim=1024, 
                max_position_embeddings=100,
                dropout=0.3,
                ),
            nline=16,
            num_layers=16,
            growth_rate=24,
            )

opt = get_option("config_last_m2edataset.yaml")
opt.batch_size = 1
train_dataloader, valid_dataloader = get_dataloader(opt)
ce_loss = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
for i, batch in enumerate(tqdm(valid_dataloader)):
    image, image_mask, task_batches = (batch["imgs"], batch["img_mask"], batch["task_batches"])
    for k, task_seq in task_batches.items():

        out = model(image, image_mask, task_seq)


    break
        


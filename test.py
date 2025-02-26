import torch
from configs.option import get_option
from tools.datasets.dataset_m2e.dataset import *
from models import LAST, Huggingface_LAST, TrOCRConfig
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from tqdm import tqdm
import cv2
from utils.vocab import vocab

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

def huggingface_last_load(model_path):
    model = Huggingface_LAST(
            config=TrOCRConfig(vocab_size=415, max_length=100, d_model=256,
            decoder_layers=3,
            decoder_attention_heads=8,
            decoder_ffn_dim=1024, 
            max_position_embeddings=100,
            dropout=0.3,
            # use_cache=True,
            ),
            nline=16,
            num_layers=16,
            growth_rate=24,
            )
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint['state_dict']
    update_dict = {}
    for k, v in state_dict.items():
        update_dict[k[6:]] = v
    msg = model.load_state_dict(update_dict)
    model.eval()
    return model 

# 加载模型
# model = huggingface_last_load("./checkpoints/baseline_LAST_V1.1/epoch_36-loss_0.367.ckpt").cuda()
model = last_load().cuda()
mode = "bline"  # 支持 bline， mline_l2r， mline_r2l， sline_l2r，sline_r2l，plain_l2r，plain_r2l，

# 加载数据集
opt = get_option("config_last_m2edataset.yaml")
train_dataloader, valid_dataloader = get_dataloader(opt)

correct = 0
total = 0
pbar = tqdm(valid_dataloader, desc="Progress")

with torch.no_grad():
    for i, batch in enumerate(pbar):
        image, image_mask, task_batches = (batch["imgs"], batch["img_mask"], batch["task_batches"])
        for k, task_seq in task_batches.items():
            if k.startswith(mode):
                task_seq.input = task_seq.input.cuda()
                task_seq.pos = task_seq.pos.cuda()
                task_seq.li = task_seq.li.cuda()
                task_seq.attn_mask = task_seq.attn_mask.cuda()
                out = model(image.cuda(), image_mask.cuda(), task_seq)
                softmax_out = torch.nn.functional.softmax(out, dim=2)
                pred_indices = softmax_out.argmax(dim=2)

                # 计算批量大小
                batch_size = pred_indices.size(0)

                for sample_idx in range(batch_size):
                    # 将预测和目标值转化为一维列表
                    pred_list = pred_indices[sample_idx].tolist()
                    target_list = task_seq.tgt[sample_idx].tolist()
                    pred_list = [value for value in pred_list if value < 121]
                    target_list = [value for value in target_list if value < 121]

                    # 直接比较两个列表是否完全一致
                    total += 1
                    if pred_list == target_list:
                        correct += 1

        pbar.set_postfix({'ExpRate': f'{correct / total:.4f}'})




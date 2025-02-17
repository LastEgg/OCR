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
# print(vocab.word2idx)

opt = get_option("config_last_m2edataset.yaml")
opt.batch_size = 1
train_dataloader, valid_dataloader = get_dataloader(opt)
ce_loss = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
for i, batch in enumerate(tqdm(valid_dataloader)):
    image, image_mask, task_batches = (batch["imgs"], batch["img_mask"], batch["task_batches"])
    for k, task_seq in task_batches.items():
        print(k)
        out = model(image, image_mask, task_seq)
        print(out.logits)
        # loss = ce_loss(out.view(-1, 415), task_seq.tgt.view(-1)) 

        # softmax_out = torch.nn.functional.softmax(out, dim=2)

        # # 2. 将每个时间步选择为最高概率的类别索引
        # pred_indices = softmax_out.argmax(dim=2)  # [batch_size, sequence_length]

        # # 3. 将 `pred_indices` 和 `task_seq_tgt` 转换为列表格式
        # pred_list = pred_indices.view(-1).tolist()
        # target_list = task_seq.tgt[0].view(-1).tolist()
        # input_list = task_seq.input[0].view(-1).tolist()
        # target_list = [value for value in target_list if value < 121]
        # input_list = [value for value in input_list if value < 121]
        # print(task_seq.input)
        # # print("预测结果（列表格式）：", pred_list)
        # print("目标标签（列表格式）：", target_list)
        # print(task_seq.tgt.view(-1).shape)
        # print(out.shape)
        # out = model.ar(image, image_mask, "plain_l2r")
        # print(len(out[0]))
        # print(vocab.indices2label(input_list).replace("<pad>", ""))
        # print(vocab.indices2label(target_list).replace("<pad>", ""))
        # dist = editdistance.eval(pred_list, target_list)
        # print(dist)
        # dist = editdistance.eval(target_list, target_list)
        # print(dist)
        # print(image[0].unsqueeze(0).shape)
        # print(image.shape)
        # break
    break
        
#     break
    
    # break
    # pass


import torch
from models import LAST, Huggingface_LAST, TrOCRConfig, Huggingface_LAST_infer, TrOCRConfig_infer
from torchvision.transforms import transforms
import cv2
import imutils
from utils.vocab import vocab
from tqdm import tqdm
import os
import time

def data_preprocess(img, image_size=256):
    if img.shape[0]>img.shape[1]:
        img=imutils.resize(img,height=image_size)
    else:
        img=imutils.resize(img,width=image_size)

    top_size=(image_size-img.shape[0])//2
    bottom_size=image_size-top_size-img.shape[0]
    left_size=(image_size-img.shape[1])//2
    right_size=image_size-left_size-img.shape[1]
    
    img=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_CONSTANT, value = (255,255,255))
    x_mask = torch.zeros((img.shape[0],img.shape[1]), dtype=torch.bool).unsqueeze(0)
    img = transforms.ToTensor()(img).unsqueeze(0)
    return img, x_mask

# 论文原版的LAST模型
def last_load(model_path="./checkpoints/baseline_LAST_V1.0/epoch_23-loss_0.451.ckpt"):
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
    update_dict = {}
    for k, v in state_dict.items():
        update_dict[k[6:]] = v
    msg = model.load_state_dict(update_dict)
    print(f"模型加载：{msg}")
    model.eval()
    return model 

# 将LAST的decoder替换为hugging face的TrOCR decoder；但不支持kv cache
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
    print(f"模型加载：{msg}")
    model.eval()
    return model 

# 将LAST的decoder替换为hugging face的TrOCR decoder；支持kv cache
def huggingface_last_load_kvcache(model_path):
    model = Huggingface_LAST_infer(
            config=TrOCRConfig_infer(vocab_size=415, max_length=100, d_model=256,
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
    print(f"模型加载：{msg}")
    model.eval()
    return model 


def infer_single(model, img_path, mode):
    with torch.no_grad():
        img = cv2.imread(img_path)
        img, img_mask = data_preprocess(img)
        if mode.startswith('plain'): regressor=model.ar
        if mode.startswith('bline'): regressor=model.bar
        ans, ti, n = regressor(img, img_mask, task_name=mode)
        if mode.startswith('plain'):
            pr=vocab.indices2label(ans)
        else:
            pr=vocab.lindices2llabel(ans)
        print("--------------------------------------------------")
        print(f"推理时间：{ti} 推理模式：{mode}")
        print(f"推理结果：{pr}")
        print("--------------------------------------------------")

def infer_folders(model, im_folder, mode):
    for im_name in tqdm(os.listdir(im_folder)):
        print(im_name)
        im_path = os.path.join(im_folder, im_name)
        infer_single(model, im_path, mode)
        


if __name__ == '__main__':
    # model = last_load("/datassd/hz/gdx_ocr/checkpoints/baseline_LAST_V1.0/epoch_53-loss_0.330.ckpt")
    # model = huggingface_last_load("/datassd/hz/gdx_ocr/checkpoints/baseline_LAST_V1.1/epoch_36-loss_0.367.ckpt")
    model = huggingface_last_load_kvcache("/datassd/hz/gdx_ocr_1/checkpoints/baseline_LAST_V1.1/epoch_9-loss_1.973.ckpt")
    infer_single(model, "/datassd/hz/gdx_ocr/M2E/images/22272.jpg", "bline") # 支持 bline，plain_l2r，plain_r2l 三种模式

    # im_folder = "/datassd/hz/gdx_ocr/test_dir/img"
    # infer_folders(model, im_folder, "bline")
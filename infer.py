import torch
from models import LAST, Huggingface_LAST, TrOCRConfig
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
    print(msg)
    model.eval()
    return model 

def infer_single(model, img_path, mode):
    img = cv2.imread(img_path)
    img, img_mask = data_preprocess(img)
    if mode.startswith('plain'): regressor=model.ar
    if mode.startswith('bline'): regressor=model.bar
    ans, ti, n = regressor(img, img_mask, task_name=mode)
    print(ti)
    pr=vocab.indices2label(ans)
    # pr=vocab.lindices2llabel(ans)
    print(pr.replace("\\n", "\n"))

def infer_test(model, img_path, mode):
    img = cv2.imread(img_path)
    img, img_mask = data_preprocess(img)
    if mode.startswith('plain'): regressor=model.ar
    if mode.startswith('bline'): regressor=model.bar
    ans, ti, n = regressor(img, img_mask, task_name=mode)
    print(ti)
    vocab.indices2label(input_list)
    # pr=vocab.lindices2llabel(ans)
    print(pr.replace("\\n", "\n"))

def infer_folders(model, im_folder, mode):
    for im_name in tqdm(os.listdir(im_folder)):
        print(im_name)
        im_path = os.path.join(im_folder, im_name)
        infer_single(model, im_path, mode)


if __name__ == '__main__':

    # img = cv2.imread("/datassd/hz/gdx_ocr/visual_data/1370/image.jpg")
    # data_preprocess(img)
    model = huggingface_last_load("/datassd/hz/gdx_ocr/checkpoints/baseline_LAST_V1.1/epoch_73-loss_0.280.ckpt")
    infer_single(model, "/datassd/hz/gdx_ocr/M2E/images/73268.jpg", "plain_l2r")
    # im_folder = "/datassd/hz/gdx_ocr/test_dir/img"
    # infer_folders(model, im_folder, "bline")
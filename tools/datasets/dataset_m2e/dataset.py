import torch
from torch.utils.data import Dataset
import json
import cv2
import imutils
from utils.line_pos_enc import InLinePos
from .augments import get_transform
from .dataset_util import *

import os

class M2EData(Dataset):
    def __init__(self, opt, phase,  train_transform=None, valid_transform=None):
        super(M2EData, self).__init__()
        self.list=[]
        self.image_size = opt.image_size
        self.dataRoot=opt.data_path
        self.inposer=InLinePos(256)
        with open(self.dataRoot+f'/{phase}.jsonl','r',encoding='utf8')as f:
            for line in f:
                json_object = json.loads(line)
                name, tex = (json_object['name'], json_object['tex'])
                tokens=tex.replace('\\n \\t', '\\t').split(' ')
                try:
                    two_dimensionalize(tokens) # 将一维的标签转换为二维
                except AssertionError:
                    print(f'{name} is too long, ignore') # 行数大于16行，不处理该数据
                    continue
                self.list.append((name,tokens))
        self.list = self.list[0:10]
        
    def __getitem__(self, index):
        name, tex=self.list[index]
        img = cv2.imread(self.dataRoot+f'/images/{name}', cv2.IMREAD_COLOR)
        assert img is not None
        if img.shape[0]>img.shape[1]:
            img=imutils.resize(img,height=self.image_size)
        else:
            img=imutils.resize(img,width=self.image_size)

        top_size=(self.image_size-img.shape[0])//2
        bottom_size=self.image_size-top_size-img.shape[0]
        left_size=(self.image_size-img.shape[1])//2
        right_size=self.image_size-left_size-img.shape[1]

        img=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_CONSTANT, value = (255,255,255))
        
        seqs=make_seqs(tex, self.inposer)
        
        # data_dir = f"/datassd/hz/gdx_ocr/visual_data/{name.split('.')[0]}"
        # os.makedirs(data_dir, exist_ok=True)
        # cv2.imwrite(f"{data_dir}/image.jpg", img)
        # for k, v in seqs.items():
        #     # v.save_to_json(f"{data_dir}/{k}.json")
        #     v.visualize_to_file(f"{data_dir}/{k}.txt")
        #     v.visualize_attn_mask(f"{data_dir}/{k}.jpg")

        return name, img, seqs

    def __len__(self):
        return len(self.list)

def get_dataloader(opt):
    train_transform, valid_transform = get_transform(opt)
    train_dataset = M2EData(
        phase="train",
        opt=opt,
        train_transform=train_transform,
        valid_transform=valid_transform,
    )
    valid_dataset = M2EData(
        phase="val",
        opt=opt,
        train_transform=train_transform,
        valid_transform=valid_transform,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    return train_dataloader, valid_dataloader



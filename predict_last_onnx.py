import torch
from torchvision.transforms import transforms
import cv2
import imutils
from utils.vocab import vocab
import os
import time
import onnx
import onnxruntime as ort
from models.KV_LAST_ONNX.datamodule import BiMultinline

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

def inference(img_path, 
            encoder_path="./checkpoints/onnx/encoder.onnx", 
            decoder_path= "./checkpoints/onnx/decoder.onnx", 
            decoder_kvcache_path = "./checkpoints/onnx/decoder_kvcache.onnx", 
            device="cpu"):
    img = cv2.imread(img_path)
    img, img_mask = data_preprocess(img)
    # 加载 ONNX 模型
    ort_session = ort.InferenceSession(encoder_path)
    # 执行推理
    outputs = ort_session.run(
        None,
        {
            "image": img.numpy(), 
            "image_mask": img_mask.numpy()
        }
    )
    # 从 logits 中选择下一个标记 ID

    feature = outputs[0]
    mask = outputs[1]

    ort_decoder_session = ort.InferenceSession(decoder_path)
    ort_decoder_session_kvcache = ort.InferenceSession(decoder_kvcache_path)

    BAR=BiMultinline("bline", device, 256, 16)

    ti=0
    n=0
    past_key_values = None
    while not BAR.is_done():
        BAR.make_input()
        task_seq_input = BAR.batch.input
        task_seq_pos = BAR.batch.pos
        task_seq_li = BAR.batch.li

        st=time.perf_counter()
        if past_key_values == None:
            outputs = ort_decoder_session.run(
                None,
                {
                    "encoder_input": feature,
                    "task_seq_input":task_seq_input.cpu().numpy(),
                    "task_seq_pos":task_seq_pos.cpu().numpy(),
                    "task_seq_li":task_seq_li.cpu().numpy(),
                }
            )
        else:
            outputs = ort_decoder_session_kvcache.run(
                None,
                {
                    "task_seq_input":task_seq_input.cpu().numpy(),
                    "task_seq_pos":task_seq_pos.cpu().numpy(),
                    "task_seq_li":task_seq_li.cpu().numpy(),
                    **{f'kvcache_{i}': past_key_values[i] for i in range(len(past_key_values))},
                }
            )
        new_char_outputs = outputs[0]
        past_key_values = outputs[1:]
        et=time.perf_counter()
        ti+=et-st
        BAR.update(new_char_outputs)
        n+=1
    ans=BAR.return_ans()
    

    pr=vocab.lindices2llabel(ans)

    print(pr)


if __name__ == '__main__':

    inference("/datassd/hz/gdx_ocr/M2E/images/22272.jpg", device="cpu")


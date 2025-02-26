import torch
from torchmetrics import ConfusionMatrix, F1Score
import lightning.pytorch as pl
from tools.datasets.dataset_m2e.dataset import *
import wandb
import editdistance
from utils.vocab import vocab
from torchvision.utils import make_grid

torch.set_float32_matmul_precision("high")


class LightningModule(pl.LightningModule):
    def __init__(self, opt, model, len_trainloader, len_valloader):
        super().__init__()
        self.learning_rate = opt.learning_rate  # 学习率
        self.len_trainloader = len_trainloader  # 训练数据加载器长度
        self.len_valloader = len_valloader
        self.opt = opt  # 配置参数
        self.model = model  

        self.ce_loss = torch.nn.CrossEntropyLoss()  

        self.log_train_data = []
        self.log_val_data = []
        self.log_label_format = ["seq_name", "image", "pred", "tgt", "equal"]


    def forward(self, x):
        """前向传播"""
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        """配置优化器和学习率 Scheduler"""
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            weight_decay=self.opt.weight_decay,
            lr=self.learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            epochs=self.opt.epochs,
            steps_per_epoch=self.len_trainloader,
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
            },
        }
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        image, image_mask, task_batches = (batch["imgs"], batch["img_mask"], batch["task_batches"])
        ce_loss = 0
        for k, task_seq in task_batches.items():
            out = self.model(image, image_mask, task_seq).logits
            # if k = "bline":
            #     ce_loss +=  self.ce_loss(out.view(-1, self.opt.num_classes), task_seq.tgt.view(-1))
            k_ce_loss = self.ce_loss(out.view(-1, self.opt.num_classes), task_seq.tgt.view(-1))
            self.log(f"loss/train_{k}_ce_loss", k_ce_loss) 
            ce_loss += k_ce_loss
            if batch_idx == 0:
                softmax_out = torch.nn.functional.softmax(out, dim=2)
                pred_indices = softmax_out.argmax(dim=2)[0]
                pred_list = pred_indices.view(-1).tolist()
                target_list = task_seq.tgt[0].view(-1).tolist()
                pred_list = [value for value in pred_list if value < 121]
                target_list = [value for value in target_list if value < 121]
                dist = editdistance.eval(pred_list, target_list)
                pred_str = vocab.indices2label(pred_list).replace("<pad>", "")
                target_str = vocab.indices2label(target_list).replace("<pad>", "")
                self.log_train_data.append([k, wandb.Image(image[0].unsqueeze(0)), pred_str, target_str, dist])
                self.logger.log_table(key="visual/train", columns=self.log_label_format, data=self.log_train_data)
        
        self.log("loss/train_loss", ce_loss)  # 记录训练损失
        return ce_loss
        

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        image, image_mask, task_batches = (batch["imgs"], batch["img_mask"], batch["task_batches"])
        ce_loss = 0
        for k, task_seq in task_batches.items():
            out = self.model(image, image_mask, task_seq).logits
            # if k = "bline":
            #     ce_loss +=  self.ce_loss(out.view(-1, self.opt.num_classes), task_seq.tgt.view(-1))
            k_ce_loss = self.ce_loss(out.view(-1, self.opt.num_classes), task_seq.tgt.view(-1))
            self.log(f"loss/val_{k}_ce_loss", k_ce_loss) 
            ce_loss += k_ce_loss
            if batch_idx == 0:
                softmax_out = torch.nn.functional.softmax(out, dim=2)
                pred_indices = softmax_out.argmax(dim=2)[0]
                pred_list = pred_indices.view(-1).tolist()
                target_list = task_seq.tgt[0].view(-1).tolist()
                pred_list = [value for value in pred_list if value < 121]
                target_list = [value for value in target_list if value < 121]
                dist = editdistance.eval(pred_list, target_list)
                pred_str = vocab.indices2label(pred_list).replace("<pad>", "")
                target_str = vocab.indices2label(target_list).replace("<pad>", "")
                self.log_val_data.append([k, wandb.Image(image[0].unsqueeze(0)), pred_str, target_str, dist])
                self.logger.log_table(key="visual/val", columns=self.log_label_format, data=self.log_val_data)
        
        self.log("trainer/learning_rate", self.optimizer.param_groups[0]["lr"])
        self.log("loss/val_loss", ce_loss)  # 记录训练损失

    def on_train_epoch_end(self):
        """训练周期结束时执行"""
        self.log_train_data.clear()

    def on_validation_epoch_end(self):
        """验证周期结束时执行"""
        self.log_val_data.clear()
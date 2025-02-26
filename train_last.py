'''
文档增强: drnet + docres数据预处理 训练代码
'''
import torch
from configs.option import get_option
from tools.datasets.dataset_m2e.dataset import *
from tools.pl_tools.pl_tool_LAST import *
from models import LAST, Huggingface_LAST, TrOCRConfig
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import wandb

torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    opt = get_option("config_last_m2edataset.yaml")
    """定义网络"""
    # model = LAST(
    #     d_model=256,
    #     growth_rate=24,
    #     num_layers=16,
    #     nhead=8,
    #     num_decoder_layers=3,
    #     dim_feedforward=1024, 
    #     dropout=0.3,
    #     nline=16,
    # )
    model = Huggingface_LAST(
            config=TrOCRConfig(vocab_size=415, max_length=256, d_model=256,
            decoder_layers=3,
            decoder_attention_heads=8,
            decoder_ffn_dim=1024, 
            max_position_embeddings=256,
            dropout=0.3,
            ),
            nline=16,
            num_layers=16,
            growth_rate=24,
            )
    """模型编译"""
    # model = torch.compile(model)
    """导入数据集"""
    train_dataloader, valid_dataloader = get_dataloader(opt)

    """Lightning 模块定义"""
    wandb_logger = WandbLogger(
        project=opt.project,
        name=opt.exp_name,
        offline=not opt.save_wandb,
        config=opt,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices=[opt.devices],
        strategy="auto",
        max_epochs=opt.epochs,
        precision=opt.precision,
        default_root_dir="./",
        logger=wandb_logger,
        val_check_interval=opt.val_check,
        log_every_n_steps=opt.log_step,
        accumulate_grad_batches=opt.accumulate_grad_batches,
        gradient_clip_val=opt.gradient_clip_val,
        callbacks=[
            # pl.callbacks.ModelCheckpoint(
            #     dirpath=os.path.join("./checkpoints", opt.exp_name),
            #     monitor="loss/val_loss",
            #     mode="min",
            #     save_top_k=1,
            #     save_last=False,
            #     filename="epoch_{epoch}-loss_{loss/val_loss:.3f}",
            #     auto_insert_metric_name=False,  # 使用 f-string 和 replace
            # ),
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join("./checkpoints", opt.exp_name),
                filename="last",  # 指定文件名为 "last"
                save_top_k=0,  # 不基于任何指标保存
                save_last=True,  # 保存最后一个模型
            ),
        ],
    )

    # Start training
    trainer.fit(
        LightningModule(opt, model, len(train_dataloader), len(valid_dataloader)),
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        ckpt_path=opt.resume,
    )
    wandb.finish()

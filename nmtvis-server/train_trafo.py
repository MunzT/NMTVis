# For data loading
import logging
import os
import pytorch_lightning as pl
import time
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import data
from shared import MODELS_FOLDER
from transformer.transformer import TransformerModel, TranslationDataModule


class PeriodicCheckpoint(Callback):
    def __init__(self, save_every_minutes, src_lang, tgt_lang, resume_from_ckpt = 0) -> None:
        super().__init__()
        self.counter = resume_from_ckpt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.save_every_minutes = save_every_minutes
        self.last_ckpt_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if time.time() - self.last_ckpt_time >= self.save_every_minutes * 60:
            print("SAVE")
            trainer.save_checkpoint(f'.data/models/transformer_small/{self.src_lang}_{self.tgt_lang}_{self.counter}.pt')
            self.last_ckpt_time = time.time()
            self.counter += 1


class ValidationOnStartCallback(pl.callbacks.Callback):
    def on_train_start(self, trainer, pl_module):
        trainer.checkpoint_callback.on_train_start(trainer, pl_module)
        return trainer.run_evaluation(test_mode=False)


if __name__ == "__main__":
    SRC_LANG = "en"
    TGT_LANG = "de"
    CONTINUE_TRAINING_CKPT = -1
    print('Translating from', SRC_LANG, 'to', TGT_LANG)

    STEPS = 100000
    EPOCH_TIME_MINUTES = 30 # Epoch time in minutes
    print("Saving every ", EPOCH_TIME_MINUTES, "minutes")

    D_MODEL = 512
    D_FF = 2048
    HEADS = 8
    P_DROP = 0.1
    LAYERS = 6

    BATCH_SIZE = 4096 # * torch.cuda.device_count()

    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logger = TensorBoardLogger(
        save_dir='lightning_logs',
        version=8,
        name=f'transformer_small_{SRC_LANG}_{TGT_LANG}'
    )
    lr_logger = LearningRateMonitor(logging_interval='step')

    dm = TranslationDataModule(batch_size=BATCH_SIZE, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    dm.prepare_data()
    dm.setup()

    model = TransformerModel(src_vocab_size=dm.src_vocab_size, tgt_vocab_size=dm.tgt_vocab_size,
                                src_pad_key=dm.src_pad_key, tgt_pad_key=dm.tgt_pad_key, max_len=data.MAX_LEN,
                                d_model=D_MODEL, nhead=HEADS, num_encoder_layers=LAYERS,
                                num_decoder_layers=LAYERS, dim_feedforward=D_FF, dropout=P_DROP,
                                smoothing=0.1)
    if CONTINUE_TRAINING_CKPT >= 0:
        ckpt = os.path.join(MODELS_FOLDER, 'transformer', f'trafo_{SRC_LANG}_{TGT_LANG}_{CONTINUE_TRAINING_CKPT}.pt')
        trainer = pl.Trainer(gpus=torch.cuda.device_count(), accumulate_grad_batches=2,
                        callbacks=[PeriodicCheckpoint(EPOCH_TIME_MINUTES, SRC_LANG, TGT_LANG, resume_from_ckpt=CONTINUE_TRAINING_CKPT), lr_logger],
                        checkpoint_callback=False, max_steps=STEPS, logger=logger, accelerator='ddp',
                        resume_from_checkpoint=ckpt, val_check_interval=15000, log_every_n_steps=100, progress_bar_refresh_rate=1000)
    else:
        trainer = pl.Trainer(gpus=torch.cuda.device_count(), accumulate_grad_batches=2,
                        callbacks=[PeriodicCheckpoint(EPOCH_TIME_MINUTES, SRC_LANG, TGT_LANG), lr_logger],
                        checkpoint_callback=False, max_steps=STEPS, logger=logger, accelerator='ddp', val_check_interval=15000, log_every_n_steps=100, progress_bar_refresh_rate=1000)

    trainer.fit(model, dm)

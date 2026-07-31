from __future__ import annotations
import math
from pathlib import Path
import lightning as L
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
import torch
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, IterableDataset
from data_pipeline import StreamingPipeline
from gpt_model import GPTModel
from config import GPT2_SMALL, TRAIN_CONFIG


tokens_per_step = TRAIN_CONFIG['context_length'] * TRAIN_CONFIG['batch_size'] * TRAIN_CONFIG['gradient_accumalation_steps']
max_steps = math.ceil(TRAIN_CONFIG['max_tokens'] / tokens_per_step)

checkpoint_directory = Path(TRAIN_CONFIG['checkpoint_directory'])
last_checkpoint = checkpoint_directory / 'last.ckpt'


class PipelineAdapter(IterableDataset):
    def __init__(self, pipeline: StreamingPipeline):
        super().__init__()
        self.pipeline = pipeline

    def __iter__(self):
        while True:
            yield self.pipeline.next_batch()


class MayFeiTrainer(L.LightningModule):
    def __init__(self, model: GPTModel, pipeline: StreamingPipeline):
        super().__init__()

        self.model = model
        self.pipeline = pipeline
        self.validation_batches = []
        self.tokens_seen = 0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)

    def calculate_loss(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        logits = self.forward(input_ids)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1)
        )

    def training_step(self, batch) -> torch.Tensor:
        input_ids, target_ids = batch
        loss = self.calculate_loss(input_ids, target_ids)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non Finite training loss: {loss.item()}')

        self.tokens_seen += input_ids.numel()
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=input_ids.size(0), )
        self.log('tokens_seen', float(self.tokens_seen), on_step=True, on_epoch=False, prog_bar=True, batch_size=input_ids.size(0))
        return loss

    def validation_step(self, batch) -> torch.Tensor:
        input_ids, target_ids = batch
        loss = self.calculate_loss(input_ids, target_ids)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=input_ids.size(0), )

    def configure_optimizers(self):
        decay_parameters = []
        non_decay_parameters = []

        for parameter in self.model.parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim >= 2:
                decay_parameters.append(parameter)
            else:
                non_decay_parameters.append(parameter)

        optimizer = torch.optim.AdamW(
            [
                {
                    'params': decay_parameters,
                    'weight_decay': TRAIN_CONFIG['weight_decay']
                },
                {
                    'params': non_decay_parameters,
                    'weight_decay': 0.0,
                },
            ],
            lr=TRAIN_CONFIG['learning_rate'],
        )

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=TRAIN_CONFIG['warmup_steps']
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=max(1, max_steps - TRAIN_CONFIG['warmup_steps']),
            eta_min=TRAIN_CONFIG['min_learning_rate'],
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[TRAIN_CONFIG['warmup_steps']],
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }

    def train_dataloader(self):
        dataset = PipelineAdapter(self.pipeline)
        return DataLoader(dataset=dataset, batch_size=None, num_workers=0, pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        return DataLoader(self.validation_batches, batch_size=None, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    def reserve_validation_batches(self):
        self.validation_batches = []
        for _ in range(TRAIN_CONFIG['validation_batches']):
            input_ids, target_ids = self.pipeline.next_batch()
            self.validation_batches.append((input_ids.cpu(), target_ids.cpu()))

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint['streaming_pipeline'] = self.pipeline._state_dict()
        checkpoint['validation_batches'] = self.validation_batches
        checkpoint['tokens_seen'] = self.tokens_seen

    def restore_data_state(self, checkpoint: dict) -> None:
        self.pipeline.load_state_dict(checkpoint['streaming_pipeline'])
        self.validation_batches = checkpoint['validation_batches']

        self.tokens_seen = int(checkpoint['tokens_seen'])


def main():
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    model_config = GPT2_SMALL.copy()

    data_pipeline = StreamingPipeline(context_length=model_config['context_length'],
                                      batch_size=TRAIN_CONFIG['batch_size'], 
                                      shuffle_buffer=0
                                      )
    model = GPTModel(cfg=model_config)
    training_module = MayFeiTrainer(model=model, pipeline=data_pipeline)
    resume_checkpoint = None

    if last_checkpoint.exists():
        saved_checkpoint = torch.load(last_checkpoint, map_location='cpu', weights_only=False)
        training_module.restore_data_state(saved_checkpoint)
        resume_checkpoint = str(last_checkpoint)
        print("Resuming | "
            f"step={saved_checkpoint['global_step']} | "
            f"tokens={training_module.tokens_seen:,}")

    else:
        training_module.reserve_validation_batches()
        print(f'Reserved {TRAIN_CONFIG['validation_batches']} validation batches')

    best_model_callback = ModelCheckpoint(
        dirpath=checkpoint_directory,
        filename='best',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=False,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )

    last_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_directory,
        filename='step-{step:08d}',
        monitor=None,
        save_top_k=1,
        save_last=True,
        every_n_train_steps=TRAIN_CONFIG['checkpoint_interval'],
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=TRAIN_CONFIG['early_stopping_patience'],
        min_delta=TRAIN_CONFIG['early_stopping_min_delta'],
        check_finite=True,
        check_on_train_epoch_end=False,
    )

    learning_rate_callback = LearningRateMonitor(logging_interval='step')
    logger = CSVLogger(save_dir='logs', name='mayfei')

    trainer = L.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu', 
        devices=1,
        precision='16-mixed' if torch.cuda.is_available() else '32-true',
        max_epochs=-1,
        max_steps=max_steps,
        accumulate_grad_batches=TRAIN_CONFIG['gradient_accumalation_steps'],
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',
        check_val_every_n_epoch=None,

        # lighting counts micro batches
        val_check_interval=TRAIN_CONFIG['evaluation_interval'] * TRAIN_CONFIG['gradient_accumalation_steps'],
        num_sanity_val_steps=0,
        log_every_n_steps=TRAIN_CONFIG['log_interval'],
        callbacks=[
            best_model_callback,
            last_checkpoint_callback,
            early_stopping_callback,
            learning_rate_callback,
        ],
        logger=logger,
        default_root_dir='training_output',
    )

    trainer.fit(training_module, ckpt_path=resume_checkpoint)

    trainer.save_checkpoint(str(last_checkpoint))

    print(f'Training Finished | steps = {trainer.global_step:,} | tokens={training_module.tokens_seen:,}')


if __name__ == '__main__':
    main()

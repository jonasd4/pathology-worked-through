import os
import json
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
import argparse
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from model import Model
from datasets import TissueDataModule
from torchvision.models import resnet18

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='resources/datasets')
    parser.add_argument('--logdir', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--augmentations', default='none')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--loss', choices=['bce', 'ce'], default='ce')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--val-interval', type=int, default=5)
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--dataset')
    parser.add_argument('--classes', type=int, default=3)
    parser.add_argument('--train-split', default='train')
    parser.add_argument('--test-split', default='test')
    parser.add_argument('--tissue-type', default='kidney')
    parser.add_argument('--gradient-clip', default=None, type=float)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--color-aug', action='store_true')
    args = parser.parse_args()

    dl_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers)

    dataset = TissueDataModule(args.data_root, dl_kwargs=dl_kwargs,
                               image_size=args.image_size,
                               color_aug=args.color_aug,
                               train_split=args.train_split,
                               test_split=args.test_split,
                               tissue_type=args.tissue_type)
    num_classes = args.classes

    backbone = resnet18(pretrained=args.pretrained)
    backbone.fc = torch.nn.Linear(backbone.fc.in_features, num_classes)
    backbone.train()

    model = Model(model=backbone, learning_rate=args.learning_rate,
                  epochs=args.epochs,
                  num_classes=num_classes,
                  weight_decay=args.weight_decay, loss=args.loss)

    tb_logger = pl_loggers.TensorBoardLogger(args.logdir)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    best_checkpoint = ModelCheckpoint(
        monitor='test/accuracy',
        mode='max',
        dirpath=args.logdir,
        filename='best-{epoch:02d}'
    )
    latest_checkpoint = ModelCheckpoint(
        monitor='epoch',
        mode='max',
        dirpath=args.logdir,
        filename='latest-{epoch:02d}'
    )

    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, 'params.json'), 'w+') as f:
        json.dump(vars(args), f, indent=4)

    trainer = Trainer(gpus=1, precision=32,
                      check_val_every_n_epoch=args.val_interval,
                      gradient_clip_val=args.gradient_clip,
                      callbacks=[lr_monitor, best_checkpoint, latest_checkpoint],
                      max_epochs=args.epochs, logger=tb_logger)
    trainer.fit(model, dataset)

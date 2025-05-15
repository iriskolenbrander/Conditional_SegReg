import argparse
import os

from datetime import datetime

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from Registration.dataset import MyRegistrationDataset
from Registration.model import init_model
from Pytorch_lightning.lightning_module import MyLightningModuleWeakSupervision

import lightning.pytorch as pl
import ml_collections
from lightning import seed_everything

torch.backends.cudnn.benchmark = True

ROOT_CHECKPOINTS = 'checkpoints'
ROOT_OUTPUT = 'output'
ROOT_DATA = 'Example_data/DatasetRegistration'

def parse_arguments():
    parser = argparse.ArgumentParser(description='Trainer ')

    # Hyper-parameters for training
    parser.add_argument('-lr', '--learning_rate', type=float, metavar='', default=1e-4, help='learning rate')
    parser.add_argument("-rw", '--reg_weight', type=float, metavar='', default=5, help='regularization (smoothing) weight in loss function')
    parser.add_argument("-sw", '--seg_weight', type=float, metavar='', default=0.4, help='segmentation weight in loss function')
    parser.add_argument('-ep', '--epochs', type=int, metavar='', default=100, help='nr. of training epochs')

    # Set remaining
    parser.add_argument('-seed', '--random_seed', type=int, metavar='', default=1000, help='random seed')
    parser.add_argument('-dev', '--device', type=str, metavar='', default='cuda', help='device / gpu used')
    parser.add_argument('--in_channels', type=int, metavar='', default=3, help='number of input channels (two for RegBase and three for Reg')

    config = parser.parse_args()
    config.imgsize = (128, 128, 128)
    config.voxel_spacing = (2.0, 2.0, 2.0)
    config.root_output = ROOT_OUTPUT
    config.root_checkpoints = ROOT_CHECKPOINTS
    config.root_data = ROOT_DATA
    config = ml_collections.ConfigDict(dict(**vars(config)))
    print(config)

    # Set seed
    seed_everything(config.random_seed)
    return config


def init_lightning_model(config, path_to_pretrained_weights='None'):
    # Init model
    model = init_model(config)

    # Load pretrained model
    if path_to_pretrained_weights != 'None':
        lightning_model = MyLightningModuleWeakSupervision.load_from_checkpoint(
            checkpoint_path=path_to_pretrained_weights,
            config=config,
            model=model,
            map_location=config.device)
        # state_dict = torch.load(path_to_pretrained_weights, map_location=config.device)["state_dict"]
        # state_dict_new = dict()
        # for key in list(state_dict.keys()):
        #     state_dict_new[key.replace('model.', '', 1)] = state_dict.pop(key)  #
        # model.load_state_dict(state_dict_new)
    else:
        lightning_model = MyLightningModuleWeakSupervision(config, model)
    return lightning_model


def init_dataset(config):
    train_dataset = MyRegistrationDataset(folder=config.root_data,
                                        train_val_test='train',
                                        target_spacing=config.voxel_spacing,
                                        device=config.device)
    val_dataset = MyRegistrationDataset(folder=config.root_data,
                                        train_val_test='val',
                                        target_spacing=config.voxel_spacing,
                                        device=config.device)
    return train_dataset, val_dataset, config


if __name__ == '__main__':
    now = datetime.now()
    date_time = now.strftime("%m%d-%H%M%S")
    config = parse_arguments()

    """ INIT DATA """
    train_dataset, val_dataset, config = init_dataset(config)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=False)

    """ INIT MODEL AND LOGGER """
    lightning_model = init_lightning_model(config)
    lightning_model.train()
    config.run_name = f"Reg-{date_time}" if config.in_channels == 3 else f"RegBase-{date_time}"
    # wandb_logger = WandbLogger(
    #     project="project_name",
    #     name=config.run_name,
    #     mode='online',
    #     config=dict(**vars(config))['_fields'],
    #     save_dir=config.root_checkpoints,
    #     log_model=False,
    # )

    checkpoint_callback = ModelCheckpoint(monitor="val-dice", mode="max",
                                          dirpath=os.path.join(config.root_checkpoints, config.run_name),
                                          filename="{epoch:05d}",
                                          every_n_epochs=10,
                                          save_last=True,
                                          save_top_k=-1)

    """ TRAIN MODEL """
    # Overfitting first in debugging
    # trainer = pl.Trainer(overfit_batches=1, max_epochs=config.epochs, log_every_n_steps=1,
    #                      logger=[wandb_logger],
    #                      devices=1, accelerator=config.device)
    # trainer.fit(lightning_model, train_dataloaders=train_dataloader)

    # Them on complete set
    trainer = pl.Trainer(max_epochs=config.epochs,
                         # logger=[wandb_logger],
                         callbacks=[checkpoint_callback],
                         devices=-1, accelerator=config.device,
                         check_val_every_n_epoch=5,
                         strategy=DDPStrategy(find_unused_parameters=True))
    trainer.fit(lightning_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


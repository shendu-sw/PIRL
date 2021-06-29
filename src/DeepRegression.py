# encoding: utf-8
import math
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt
import numpy as np

from src.data.layout import LayoutDataset
import src.utils.np_transforms as transforms
import src.models as models
import src.loss as loss
from src.data.default_layout import layout_generate


class Model(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self._build_model()
        self.criterion = nn.L1Loss()
        self.pointloss = loss.PointLoss()
        self.tvloss = loss.TVLoss()
        #self.laplaceloss = loss.LaplaceLoss(length=self.hparams.length, nx = self.hparams.nx, bcs=[[[0,4.5],[0,5.5]]])
        self.laplaceloss = loss.LaplaceLoss(length=self.hparams.length, nx = self.hparams.nx, bcs=self.hparams.bcs)
        self.outsideloss = loss.OutsideLoss(length = self.hparams.length, bcs = self.hparams.bcs, nx = self.hparams.nx)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.default_layout = None

    def _build_model(self):
        model_list = ["SegNet_AlexNet", "SegNet_VGG", "SegNet_ResNet18", "SegNet_ResNet50",
                      "SegNet_ResNet101", "SegNet_ResNet34", "SegNet_ResNet152",
                      "FPN_ResNet18", "FPN_ResNet50", "FPN_ResNet101", "FPN_ResNet34", "FPN_ResNet152",
                      "FCN_AlexNet", "FCN_VGG", "FCN_ResNet18", "FCN_ResNet50", "FCN_ResNet101",
                      "FCN_ResNet34", "FCN_ResNet152",
                      "UNet_VGG"]
        layout_model = self.hparams.model_name + '_' + self.hparams.backbone
        assert layout_model in model_list
        self.model = getattr(models, layout_model)(in_channels=1)
        #self.model_1 = models.FCN_AlexNet(in_channels=1)

    def forward(self, x):
        x = self.model(x)
        x = torch.flip(x,dims=[2,3])
        #x = self.model_1(x)
        x = self.model(x)
        return x

    def __dataloader(self, dataset, shuffle=False):
        loader = DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
        return loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.lr)
        scheduler = ExponentialLR(optimizer, gamma=0.99)
        return [optimizer], [scheduler]

    def prepare_data(self):
        """Prepare dataset
        """
        size: int = self.hparams.input_size
        transform_layout = transforms.Compose(
            [
                transforms.Resize(size=(size, size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    torch.tensor([self.hparams.mean_layout]),
                    torch.tensor([self.hparams.std_layout]),
                ),
            ]
        )
        transform_heat = transforms.Compose(
            [
                transforms.Resize(size=(size, size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    torch.tensor([self.hparams.mean_heat]),
                    torch.tensor([self.hparams.std_heat]),
                ),
            ]
        )

        # here only support format "mat"
        assert self.hparams.data_format == "mat"
        trainval_dataset = LayoutDataset(
            self.hparams.data_root,
            list_path=self.hparams.train_list,
            train=True,
            transform=transform_layout,
            target_transform=transform_heat,
        )
        test_dataset = LayoutDataset(
            self.hparams.data_root,
            list_path=self.hparams.test_list,
            train=False,
            transform=transform_layout,
            target_transform=transform_heat,
        )

        # split train/val set
        train_length, val_length = int(len(trainval_dataset) * 0.8), int(len(trainval_dataset) * 0.2)
        train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset,
                                                                   [train_length, val_length])
        print(
            f"Prepared dataset, train:{int(len(train_dataset))},\
                val:{int(len(val_dataset))}, test:{len(test_dataset)}"
        )

        # assign to use in dataloaders
        self.train_dataset = self.__dataloader(train_dataset, shuffle=True)
        self.val_dataset = self.__dataloader(val_dataset, shuffle=False)
        self.test_dataset = self.__dataloader(test_dataset, shuffle=False)

        self.default_layout = layout_generate(nx=self.hparams.nx,length=self.hparams.length, \
            positions=self.hparams.positions,default_powers=self.hparams.default_powers, \
                units=self.hparams.units,angles=self.hparams.angles)
        
        self.default_layout = torch.from_numpy(self.default_layout).unsqueeze(0).unsqueeze(0)

    def train_dataloader(self):
        return self.train_dataset

    def val_dataloader(self):
        return self.val_dataset

    def test_dataloader(self):
        return self.test_dataset

    def training_step(self, batch, batch_idx):

        heat_obs, heat = batch
        #print("device:", heat_obs.device)
        # print(torch.equal(layout,heat))

        ## flip operation
        #heat_obs_flip = torch.flip(heat_obs,dims=[2,3])
        #heat_flip = torch.flip(heat,dims=[2,3])

        heat_pred = self(heat_obs)

        #heat_pred_flip = self(heat_obs_flip)
        default_layout = torch.repeat_interleave(self.default_layout, repeats=heat_pred.size(0), dim=0).float().to(device=heat.device)
        #print("device1:", default_layout.device)
        #print(heat_obs)
        #loss_1 = self.criterion(heat, heat_pred)
        loss_obs = self.pointloss(heat_obs, heat_pred) * 1e5
        #loss_obs_flip = self.pointloss(heat_obs_flip, heat_pred_flip) * 1e5
        loss_tv = self.tvloss(heat_pred) * 1e4
        loss_laplace = self.laplaceloss(default_layout, heat_pred) * 1e2
        #loss_deviationlaplace = self.deviationlaplaceloss(default_layout, heat_pred) * 1e3
        #loss_laplace_flip = self.laplaceloss(default_layout, torch.flip(heat_pred_flip, dims=[2,3]))
        loss_outside = self.outsideloss(heat_pred) * 1e2

        ## 翻转不变性　flip invariant
        #loss_inv = self.criterion(heat_pred, torch.flip(heat_pred_flip, dims=[2,3]))

        loss = loss_obs + loss_outside + loss_laplace + loss_tv #+ loss_obs_flip + loss_inv# loss_obs + loss_outside + 
        self.log("train/training_mae", loss_obs * self.hparams.std_heat)
        self.log("train/training_laplace", loss_laplace * self.hparams.std_heat)
        #self.log("train/training_obs", loss_outside * self.hparams.std_heat)

        if batch_idx == 0:
            grid = torchvision.utils.make_grid(
                heat_pred[:4, ...], normalize=True
            )
            self.logger.experiment.add_image(
                "train_pred_heat_field", grid, self.global_step
            )
            if self.global_step == 0:
                grid = torchvision.utils.make_grid(
                    heat[:4, ...], normalize=True
                )
                self.logger.experiment.add_image(
                    "train_heat_field", grid, self.global_step
                )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        layout, heat = batch
        heat_pred = self(layout)
        loss = self.criterion(heat, heat_pred)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val/val_mae", val_loss_mean.item() * self.hparams.std_heat)

    def test_step(self, batch, batch_idx):
        layout, heat = batch
        heat_pred = self(layout)

        loss = self.criterion(heat_pred, heat) * self.hparams.std_heat
        #---------------------------------
        
        #---------------------------------
        default_layout = torch.repeat_interleave(self.default_layout, repeats=heat_pred.size(0), dim=0).float().to(device=heat.device)
        ones = torch.ones_like(default_layout).to(device=layout.device)
        zeros = torch.zeros_like(default_layout).to(device=layout.device)
        layout_ind = torch.where(default_layout<1e-2,zeros,ones)
        #print(ones)
        loss_2 = torch.sum(torch.abs(torch.sub(heat, heat_pred)) *layout_ind )* self.hparams.std_heat/ torch.sum(layout_ind)
        #print(loss_2)
        #---------------------------------
        loss_1 = torch.sum(torch.max(torch.max(torch.max(torch.abs(torch.sub(heat,heat_pred)) * layout_ind, 3).values, 2).values * self.hparams.std_heat,1).values)/heat_pred.size(0)
        #---------------------------------
        boundary_ones = torch.zeros_like(default_layout).to(device=layout.device)
        boundary_ones[..., -2:, :] = ones[..., -2:, :]
        boundary_ones[..., :2, :] = ones[..., :2, :]
        boundary_ones[..., :, :2] = ones[..., :, :2]
        boundary_ones[..., :, -2:] = ones[..., :, -2:]
        loss_3 = torch.sum(torch.abs(torch.sub(heat, heat_pred)) *boundary_ones )* self.hparams.std_heat/ torch.sum(boundary_ones)
        return {"test_loss": loss, "test_loss_1": loss_1, "test_loss_2": loss_2, "test_loss_3": loss_3}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x["test_loss"] for x in outputs]).mean()
        self.log("test_loss (" + "MAE" +")", test_loss_mean.item())
        #test_loss_max = torch.max(torch.stack([x["test_loss_1"] for x in outputs]))
        test_loss_max = torch.stack([x["test_loss_1"] for x in outputs]).mean()
        self.log("test_loss_1 (" + "M-CAE" +")", test_loss_max.item())
        test_loss_com_mean = torch.stack([x["test_loss_2"] for x in outputs]).mean()
        self.log("test_loss_2 (" + "CMAE" +")", test_loss_com_mean.item())
        test_loss_bc_mean = torch.stack([x["test_loss_3"] for x in outputs]).mean()
        self.log("test_loss_3 (" + "BMAE" +")", test_loss_bc_mean.item())

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no-cover
        """Parameters you define here will be available to your model through `self.hparams`.
        """
        # data
        parser.add_argument("--bcs", type=yaml.safe_load, action="append", help="boundary conditions of heat source layout")
        parser.add_argument("--length", type=float, default=10.0, help="the length of layout board")
        parser.add_argument("--nx", type=int, default=200, help="the dimension of matrix for modeling the layout board")
        parser.add_argument("--positions", type=yaml.safe_load, action="append", help="the positions of heat sources")
        parser.add_argument("--units",type=yaml.safe_load, action="append", help="the shape of heat sources")
        parser.add_argument("--default_powers", type=yaml.safe_load, action="append", help="the default powers of heat sources")
        parser.add_argument("--angles", type=yaml.safe_load, action="append", help="angles of heat sources")

        # dataset args
        parser.add_argument("--data_root", type=str, required=True, help="path of dataset")
        parser.add_argument("--train_list", type=str, required=True, help="path of train dataset list")
        parser.add_argument("--train_size", default=0.8, type=float, help="train_size in train_test_split")
        parser.add_argument("--test_list", type=str, required=True, help="path of test dataset list")
        #parser.add_argument("--boundary", type=str, default="rm_wall", help="boundary condition")
        parser.add_argument("--data_format", type=str, default="mat", choices=["mat", "h5"], help="dataset format")

        # Normalization params
        parser.add_argument("--mean_layout", default=0, type=float)
        parser.add_argument("--std_layout", default=1, type=float)
        parser.add_argument("--mean_heat", default=0, type=float)
        parser.add_argument("--std_heat", default=1, type=float)

        # Model params (opt)
        parser.add_argument("--input_size", default=200, type=int)
        parser.add_argument("--model_name", type=str, default='SegNet', help="the name of chosen model")
        parser.add_argument("--backbone", type=str, default='ResNet18', help="the used backbone in the regression model")
        parser.add_argument("--model_2_name", type=str, default='SegNet', help="the name of chosen model")
        parser.add_argument("--backbone_2", type=str, default='ResNet18', help="the used backbone in the regression model")
        
        return parser

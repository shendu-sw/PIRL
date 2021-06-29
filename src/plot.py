"""
Runs a model on a single node across multiple gpus.
"""
import os
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
import scipy.io as sio
import matplotlib.pyplot as plt
import configargparse

from src.DeepRegression import Model


def main(hparams):

    if hparams.gpu == 0:
        device = torch.device("cpu")
    else:
        ngpu = "cuda:"+str(hparams.gpu-1)
        print(ngpu)
        device = torch.device(ngpu)
    model = Model(hparams).to(device)

    print(hparams)
    print()

    # Model loading
    model_path = os.path.join(f'lightning_logs/version_' +
                              hparams.test_check_num, 'checkpoints/')
    ckpt = list(Path(model_path).glob("*.ckpt"))[0]
    print(ckpt)

    model = model.load_from_checkpoint(str(ckpt))

    model.eval()
    model.to(device)
    mae_test = []

    # Testing Set
    root = hparams.data_root
    test_list = hparams.test_list
    file_path = os.path.join(root, test_list)
    root_dir = os.path.join(root, 'test', 'test')

    with open(file_path, 'r') as fp:
        for line in fp.readlines():
            # Data Reading
            data_path = line.strip()
            path = os.path.join(root_dir, data_path)
            data = sio.loadmat(path)
            u_true, u_obs = data["u"], data["u_obs"]
            hs_F = data["F"]
            
            # Plot u_obs and Real Temperature Field
            fig = plt.figure(figsize=(22,5))

            grid_x = np.linspace(0, 0.1, num=200)
            grid_y = np.linspace(0, 0.1, num=200)
            X, Y = np.meshgrid(grid_x, grid_y)

            plt.subplot(141)
            plt.title('Real Time Power')
            #im = plt.pcolormesh(X, Y, u_obs,cmap='jet')
            im = plt.pcolormesh(X,Y,hs_F)
            plt.colorbar(im)
            fig.tight_layout(pad=2.0, w_pad=3.0,h_pad=2.0)

            u_obs = torch.Tensor((u_obs - hparams.mean_layout) / hparams.std_layout).unsqueeze(0).unsqueeze(0).to(device)
            print(u_obs.size())
            heat = torch.Tensor((u_true - hparams.mean_heat) / hparams.std_heat).unsqueeze(0).unsqueeze(0).to(device)

            #assert torch.equal(u_obs, heat)

            with torch.no_grad():
                #u_obs=torch.flip(u_obs,dims=[2,3])
                #heat = torch.flip(heat,dims=[2,3])
                heat_pre = model(u_obs)
                mae = F.l1_loss(heat, heat_pre) * hparams.std_heat
                print('MAE:', mae)
            mae_test.append(mae.item())
            heat_pre = heat_pre.squeeze(0).squeeze(0).cpu().numpy() * hparams.std_heat + hparams.mean_heat
            hmax = max(np.max(heat_pre), np.max(u_true))
            hmin = min(np.min(heat_pre), np.min(u_true))

            plt.subplot(143)
            plt.title('Real Temperature Field')
            if "xs" and "ys" in data.keys():
                xs, ys = data["xs"], data["ys"]
                im = plt.pcolormesh(xs, ys, u_true,cmap='jet')
                plt.axis('equal')
            else:
                #im = plt.pcolormesh(X, Y, heat.squeeze(0).squeeze(0).cpu().numpy() * hparams.std_heat + hparams.mean_heat)
                #im = plt.pcolormesh(X, Y, u_true,cmap='jet')
                im = plt.contourf(X,Y,u_true,levels=150,cmap='jet')
            plt.colorbar(im)

            plt.subplot(142)
            plt.title('Reconstructed Temperature Field')
            if "xs" and "ys" in data.keys():
                xs, ys = data["xs"], data["ys"]
                im = plt.pcolormesh(xs, ys, heat_pre,cmap='jet')
                plt.axis('equal')
            else:
                im = plt.contourf(X, Y, heat_pre,levels=150,cmap='jet')
            plt.colorbar(im)

            plt.subplot(144)
            plt.title('Absolute Error')
            if "xs" and "ys" in data.keys():
                xs, ys = data["xs"], data["ys"]
                #im = plt.pcolormesh(xs, ys, np.abs(heat_pre-u_true)/u_true)
                im = plt.pcolormesh(xs, ys, np.abs(heat_pre-u_true),cmap='jet')
                plt.axis('equal')
            else:
                im = plt.contourf(X, Y, np.abs(heat_pre-u_true),levels=150,cmap='jet')
                #im = plt.pcolormesh(X, Y, np.abs(heat_pre-u_true)/u_true)
            plt.colorbar(im)

            save_name = os.path.join('outputs/predict_plot', os.path.splitext(os.path.basename(path))[0]+'.png')
            fig.savefig(save_name, dpi=300)
            plt.close()

    mae_test = np.array(mae_test)
    print(mae_test.mean())
    np.savetxt('outputs/mae_test.csv', mae_test, fmt='%f', delimiter=',')


#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

from scipy.stats import norm, multivariate_normal
from sklearn.metrics import r2_score, root_mean_squared_error
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tqdm import tqdm

import sys
sys.path.append('../')

from models.fno import FNO
from models.losses import LpLoss, H1Loss


#########################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


data_dir = '../data/2024-07-30/'
directories = os.listdir(data_dir)
datasets = []

for directory in directories:
    ax_path = os.path.join(data_dir, directory, 'ax.pt')
    ux_path = os.path.join(data_dir, directory, 'ux_analytic.pt')
    hw_path = os.path.join(data_dir, directory, 'h_weights.pt')
    hb_path = os.path.join(data_dir, directory, 'h_bias.pt')
    
    try:
        ax = torch.load(ax_path, map_location=torch.device('cpu'))
        ux = torch.load(ux_path, map_location=torch.device('cpu'))
        hw = torch.load(hw_path, map_location=torch.device('cpu'))
        hb = torch.load(hb_path, map_location=torch.device('cpu'))
        datasets.append({'ax': ax,
                         'ux': ux,
                         'hw': hw,
                         'hb': hb})
    except Exception as e:
        directories.remove(directory)
        print(e)

n_dataset = len(datasets)



class DictDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        return {'x': x, 'y': y}
    
    def __len__(self):
        return len(self.data)



def training_loop(model, train_dl, train_loss, optimizer, scheduler, n_epochs=500):

    train_losses = []
    test_losses = []

    for epoch in tqdm(range(n_epochs)):

        avg_loss = 0
        avg_lasso_loss = 0
        model.train()
        train_err = 0.0

        avg_test_loss = 0
        test_err = 0.0

        for idx, sample in enumerate(train_dl):

            # load everything from the batch onto self.device if 
            # no callback overrides default load to device

            for k,v in sample.items():
                if hasattr(v, 'to'):
                    sample[k] = v.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(**sample)

            loss = 0.

            if isinstance(out, torch.Tensor):
                loss = train_loss(out.float(), **sample)
            elif isinstance(out, dict):
                loss += train_loss(**out, **sample)

            del out

            loss.backward()

            optimizer.step()
            train_err += loss.item()

            with torch.no_grad():
                avg_loss += loss.item()

        if (epoch + 1) % 5:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_err)
            else:
                scheduler.step()

        train_err /= len(train_dl)
        avg_loss  /= n_epochs


        train_losses.append(avg_loss)
        # test_losses.append(avg_test_loss)

        # print(f'Epoch: {epoch+1} loss: {avg_loss:.4f}  test loss: {avg_test_loss:.4f} lr: {scheduler.get_last_lr()[0]:.4f}, {optimizer.state_dict()["param_groups"][0]["lr"]:.4f}')
        if (epoch + 1) == n_epochs:
            print(f'Epoch: {epoch+1} loss: {avg_loss:.4f} lr: {scheduler.get_last_lr()[0]:.4f}, {optimizer.state_dict()["param_groups"][0]["lr"]:.4f}')
        
    return model


def train_model(train_loader, n_modes=(16, 10), in_channels=2, hidden_channels=64, 
                 projection_channels=64, n_epochs=200):
    
    # Losses
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)

    train_loss = h1loss
    eval_losses={'h1': h1loss, 'l2': l2loss}
    
    # Model configuration
    n_modes = n_modes
    model = FNO(n_modes=n_modes, in_channels=in_channels, hidden_channels=hidden_channels, 
                 projection_channels=projection_channels).double()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=1e-3,
                                weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    # print(model)
    
    model = training_loop(model, train_loader, 
                          train_loss, optimizer, 
                          scheduler, n_epochs=n_epochs)
    
    return model


def normalise(dataset, key, channel_mean=None, channel_std=None):
    if channel_mean is None:
        arr = dataset[key].permute(1, 0, 2, 3).flatten(1)
        channel_mean = arr.mean(1)
        channel_std = arr.std(1)
    normalised_dataset = ((dataset[key].permute(0, 2, 3, 1) - channel_mean)/channel_std).permute(0, 3, 1, 2)
    return normalised_dataset, channel_mean, channel_std



def plot_results(ax, ux, ux_hat, filepath):

    # Extents
    xmin, xmax = ax[0, 0].min(), ax[0].max()
    ymin, ymax = ax[1, :, 0].min(), ax[1, :, 0].max()

    fig, axes = plt.subplots(1, 4, figsize=(18, 3))
    
    im = axes[0].imshow(ax[2], extent=(xmin, xmax, ymin, ymax), origin='lower')
    axes[0].set_title('$h(x)$')
    fig.colorbar(im, orientation='vertical')
    
    
    im = axes[1].imshow(ax[3], extent=(xmin, xmax, ymin, ymax), origin='lower')
    axes[1].set_title('$p(x)$')
    fig.colorbar(im, orientation='vertical')
    
    im = axes[2].imshow(ux[0], extent=(xmin, xmax, ymin, ymax), origin='lower')
    axes[2].set_title('$\phi(x)$')
    fig.colorbar(im, orientation='vertical')
    
    im = axes[3].imshow(ux_hat[0], vmax=ux.max(), vmin=ux.min(), 
                        extent=(xmin, xmax, ymin, ymax), origin='lower')
    axes[3].set_title('$\phi_{pred}(x)$')
    fig.colorbar(im, orientation='vertical')

    fig.savefig(filepath, bbox_inches='tight')

    return fig



def validate(model, val_dataset, directory, ux_mean, ux_std, interpolate=True):
    val_ax = val_dataset['ax']
    val_ux = val_dataset['ux']
    
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    
    with torch.no_grad():
        ϕx_hat = model(val_ax.to(device))
        error = h1loss(ϕx_hat, val_ux.to(device))
        print(f"Validation H1Loss: {error.item():.2f}")

    ϕx_hat = ϕx_hat.reshape((10, 10, 1, 32, 32))

    if interpolate:
        for i in range(2, 9, 2):
            for j in range(2, 9, 2):
                ϕx_hat[i, j, :, :, :16] = (ϕx_hat[i, j, :, :, :16] + ϕx_hat[i, j-1, :, :, 16:])/2
                ϕx_hat[i, j, :, :, 16:] = (ϕx_hat[i, j, :, :, 16:] + ϕx_hat[i, j+1, :, :, :16])/2
                ϕx_hat[i, j, :, :16, :] = (ϕx_hat[i, j, :, :16, :] + ϕx_hat[i-1, j, :, 16:, :])/2
                ϕx_hat[i, j, :, 16:, :] = (ϕx_hat[i, j, :, 16:, :] + ϕx_hat[i+1, j, :, :16, :])/2
            

    # skip alternate rows and cols
    ϕx_hat = ϕx_hat[::2, ::2].cpu().numpy()
    val_ax = val_ax.reshape((10, 10, 4, 32, 32))[::2, ::2].numpy()
    val_ux = val_ux.reshape((10, 10, 1, 32, 32))[::2, ::2].numpy()


    ux_unscaled = val_ux.flatten() * ux_std.numpy()[0] + ux_mean.numpy()[0]
    ux_hat_unscaled = ϕx_hat.flatten() * ux_std.numpy()[0] + ux_mean.numpy()[0]

    r2_error = r2_score(ux_unscaled, ux_hat_unscaled)
    rmse = root_mean_squared_error(ux_unscaled, ux_hat_unscaled)

    # ux_hat
    ϕx_hat = np.concatenate([ϕx_hat[:, i] for i in range(5)], axis=-1)
    ϕx_hat = np.concatenate([ϕx_hat[i] for i in range(5)], axis=-2)
    
    # ax
    val_ax = np.concatenate([val_ax[:, i] for i in range(5)], axis=-1)
    val_ax = np.concatenate([val_ax[i] for i in range(5)], axis=-2)
    
    # ux
    val_ux = np.concatenate([val_ux[:, i] for i in range(5)], axis=-1)
    val_ux = np.concatenate([val_ux[i] for i in range(5)], axis=-2)

    # plot results
    os.makedirs(f'../plots/{str(dt.date.today())}', exist_ok=True)
    plot_results(val_ax, val_ux, ϕx_hat, filepath=f'../plots/{str(dt.date.today())}/{directory}.png')
    
    return error.item(), r2_error, rmse


res_list = []
for k in range(n_dataset):
    train_dataset = {}
    train_dataset['ax'] = torch.concat([datasets[i]['ax'] for i in range(n_dataset) if i!=k], dim=0)
    train_dataset['ux'] = torch.concat([datasets[i]['ux'] for i in range(n_dataset) if i!=k], dim=0)
    val_dataset = datasets[k]
    
    # Normalise
    train_dataset['ax'], ax_channel_mean, ax_channel_std = normalise(train_dataset, 'ax')
    train_dataset['ux'], ux_channel_mean, ux_channel_std = normalise(train_dataset, 'ux')

    val_dataset['ax'], _, _ = normalise(val_dataset, 'ax', ax_channel_mean, ax_channel_std)
    val_dataset['ux'], _, _ = normalise(val_dataset, 'ux', ux_channel_mean, ux_channel_std)
    
    train_ds = DictDataset(train_dataset['ax'], train_dataset['ux'])
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    
    model = train_model(train_dl, in_channels=4, n_modes=(8, 8), n_epochs=1000)
    
    h1_error, r2_error, rmse = validate(model, val_dataset, 
                                        directory=directories[k], 
                                        ux_mean=ux_channel_mean,
                                        ux_std=ux_channel_std,
                                        interpolate=True)
    h1_error, r2_error_noint, rmse_noint = validate(model, val_dataset, 
                                                    directory=f'{directories[k]}_noint',
                                                    ux_mean=ux_channel_mean,
                                                    ux_std=ux_channel_std,
                                                    interpolate=False)

    μ, σ = list(map(float, directories[k].split('_')))
    
    res = {'$\mu$': μ,
           '$\sigma$': σ,
           'h1_error': h1_error,
           'r2_int': r2_error,
           'r2_noint': r2_error_noint,
           'rmse_int': rmse,
           'rmse_noint': rmse_noint,
           'w1': val_dataset['hw'].flatten().numpy().tolist()[0],
           'w2': val_dataset['hw'].flatten().numpy().tolist()[1],
           'w0': val_dataset['hb'].flatten().numpy().tolist()[0]}
    
    res_list.append(res)


res_df = pd.DataFrame(res_list)
res_df.to_csv(f'../results_{str(dt.date.today())}.csv', index=False)
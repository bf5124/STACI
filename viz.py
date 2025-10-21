import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pyproj import Transformer
from scipy.spatial import KDTree
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from global_land_mask import globe

def plotOptHis(logger, log_freq, epochs, save_dir, temporal):
    mse_train = logger['mse_train']
    mse_val = logger['mse_val']
    r2_train = logger['r2_train']
    r2_val = logger['r2_val']
    mnlp_val = logger['mnlp_val']
    cov_val = logger['coverage_val']
    nu_Z = logger['nu_Z']
    rho_Z = logger['rho_Z']
    sig2_Z = logger['sig2_Z']
    tau = logger['tau']
    bayes_IS = logger['bayes_is_val']
    bayes_IW = logger['bayes_iw_val']

    mpl.rcParams.update({'font.family': 'serif', 'font.size': 7})
    x_axis = np.arange(log_freq, epochs + log_freq, log_freq)
    
    plt.figure(figsize=(10, 7))
    plt.plot(x_axis, r2_train, 'k-')
    plt.plot(x_axis, r2_val, 'r--')
    plt.xlabel('Epoch')
    plt.ylabel(r'$R^2$-score')
    plt.grid(True)
    plt.ylim(0,1)
    plt.savefig(save_dir + '/r2.png', dpi=500, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 7))
    plt.plot(x_axis, mse_train, 'k-')
    plt.plot(x_axis, mse_val, 'r--')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.savefig(save_dir + '/rmse.png', dpi=500, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 7))
    plt.plot(x_axis, cov_val)
    plt.xlabel('Epoch')
    plt.ylabel('Coverage')
    plt.grid(True)
    plt.savefig(save_dir + '/cov.png', dpi=500, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 7))
    plt.plot(x_axis, nu_Z)
    plt.xlabel('Epoch')
    plt.ylabel('Full Smoothness')
    plt.grid(True)
    plt.savefig(save_dir + '/nuZ.png', dpi=500, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 7))
    plt.plot(x_axis, rho_Z)
    plt.xlabel('Epoch')
    plt.ylabel('Full Range')
    plt.grid(True)
    plt.savefig(save_dir + '/rhoZ.png', dpi=500, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 7))
    plt.plot(x_axis, sig2_Z)
    plt.xlabel('Epoch')
    plt.ylabel('Full Spatial Variance')
    plt.grid(True)
    plt.savefig(save_dir + '/sig2Z.png', dpi=500, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 7))
    plt.plot(x_axis, tau)
    plt.xlabel('Epoch')
    plt.ylabel('Nugget')
    plt.grid(True)
    plt.savefig(save_dir + '/tau.png', dpi=500, bbox_inches='tight')
    plt.close()

    np.savetxt(save_dir + '/r2_train.txt', r2_train)
    np.savetxt(save_dir + '/r2_val.txt', r2_val)
    np.savetxt(save_dir + '/mse_train.txt', mse_train)
    np.savetxt(save_dir + '/mse_val.txt', mse_val)
    np.savetxt(save_dir + '/mnlp_val.txt', mnlp_val)
    np.savetxt(save_dir + '/coverage_val.txt', cov_val)
    np.savetxt(save_dir + '/nu_Z.txt', nu_Z)
    np.savetxt(save_dir + '/rho_Z.txt', rho_Z)
    np.savetxt(save_dir + '/sig2_Z.txt', sig2_Z)
    np.savetxt(save_dir + '/tau.txt', tau)
    np.savetxt(save_dir + '/bayes_is.txt', bayes_IS)
    np.savetxt(save_dir + '/bayes_iw.txt', bayes_IW)

    if temporal == True:
 
        rhot_Z = logger['rhot_Z']

        plt.figure(figsize=(10, 7))
        plt.plot(x_axis, rhot_Z)
        plt.xlabel('Epoch')
        plt.ylabel('Full Temporal Range')
        plt.grid(True)
        plt.savefig(save_dir + '/rhotZ.png', dpi=500, bbox_inches='tight')
        plt.close()

        np.savetxt(save_dir + '/rhot_Z.txt', rhot_Z)
        

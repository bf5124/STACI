import torch
import numpy as np
import os
import torch.nn.functional as F
from scipy.stats import norm
from args import args
from svgd import SVGD
from BayesNN import BayesNet
from time import time
from getData_Real import prepare_MSS_data, data_transform_MSS
from torch.utils.data import DataLoader
from utilities import getModel, getOpt, R2_value, dataTranform, toNumpy, saveData, interval_score
from viz import plotOptHis
from conformal import conformal, conformal_parallel, eta_cv
from sklearn.neighbors import KDTree


#####################################
# step 1: load data
#####################################
print('----------- Load data -----------')

data_dir = os.path.join('./data',args.data_dir)
Train_Data, Test_Data, Y_mean, Y_std = prepare_MSS_data(data_dir, args.ttsplit, args.seed)
train_loader, val_loader, test_loader = data_transform_MSS(Train_Data, Test_Data, args.ttsplit, [args.btrain, args.bval, args.btest], args.seed)
args.t_dim = len(np.unique(Train_Data[:,3]))


print('----------- Data loaded! -----------\n')

#####################################
# step 2: define the model
#####################################
print("Aggregation device:", args.device)
if args.gpu_devices:
    print("Available GPU devices:", args.gpu_devices)
    for device_i in args.gpu_devices:
        torch.inverse(torch.ones((1, 1), device= device_i))

print('----------- Setup model -----------')
model_RFF_Net = getModel(args)

model = BayesNet(args, model_RFF_Net).to(args.device) 

criterion, optimizer, scheduler, logger = getOpt(args, model)
print('----------- '+ args.latent_mod + ' is selected -----------\n')

#####################################
# step 3: define training
#####################################
if args.conformal == True:
    model_SVGD = SVGD(args, model, train_loader, val_loader, Train_Data[:,0], Train_Data[:,1:]) 
else:
    model_SVGD = SVGD(args, model, train_loader, val_loader)

#####################################
# step 4: define test
#####################################
def test():
    model.eval()
    start = time()

    mse_test, r2_score_test, nlp_test, covered_count, total_samples = 0., 0., 0., 0., 0.
    bayes_IS, bayes_IW = 0., 0.
    if args.conformal:
        train_x_sc = Train_Data[:, 1:].copy()
        rho_Z = np.exp(model.log_rho_Z.mean().item())
        rhot_Z = np.exp(model.log_rhot_Z.mean().item())
    
        train_x_sc[:, :2] /= rho_Z
        train_x_sc[:, 2] /= rhot_Z
        conf_start = time()
    
        X_train_neigh = KDTree(train_x_sc)
        conformal_IS, conf_covered_count, conformal_IW, = 0., 0., 0.
    
        m_search = np.linspace(args.conformal_neigh[0], args.conformal_neigh[1], args.conformal_neigh[2])
    
        if args.conformal_par_device == "gpu":
            opt_neigh = eta_cv(Train_Data[:, 1:], Train_Data[:, 0], X_train_neigh, m_search, model, rho_Z,
                               rhot_Z, args.conformal_grid, args.alpha, args.device, args.conformal_parallel,
                               args.gpu_devices)
        else:
            opt_neigh = eta_cv(Train_Data[:, 1:], Train_Data[:, 0], X_train_neigh, m_search, model, rho_Z,
                               rhot_Z, args.conformal_grid, args.alpha, args.device, args.conformal_parallel,
                               args.cpu_devices)
        conf_end = time()
        conf_time_cv= conf_end - conf_start
        conf_time = conf_time_cv

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        outputs = model.predict(inputs)

        outputs_mean = outputs.mean(0)

        EyyT = (outputs ** 2).mean(0)
        EyEyT = outputs_mean ** 2
        outputs_noise_var = model.log_tau.exp().mean()
        outputs_var = EyyT - EyEyT + outputs_noise_var

        
        mse_test += np.sqrt(F.mse_loss(outputs_mean, torch.unsqueeze(targets, 1)).item())
        r2_score_test += R2_value(outputs_mean, torch.unsqueeze(targets, 1))
        nlp_test += F.gaussian_nll_loss(outputs_mean, torch.unsqueeze(targets,1), outputs_var).item()

        
        inputs_x = inputs.detach().cpu().numpy()
        pred_y = torch.squeeze(outputs_mean).detach().cpu().numpy()
        target_y = targets.detach().cpu().numpy()
        resids = np.abs(pred_y - target_y)
        CI_length = np.sqrt(toNumpy(torch.squeeze(outputs_var))) * norm.ppf(1-(args.alpha/2))

        coverage = ((target_y >= (pred_y - CI_length)) & (target_y <= (pred_y + CI_length)))
        covered_count += np.sum(coverage)
        total_samples += len(target_y)

        bayes_IS_i = interval_score(pred_y- CI_length, pred_y+ CI_length, target_y, args.alpha)
        bayes_IS += bayes_IS_i
        bayes_IW += np.mean((CI_length)*2)
        
        if args.conformal:
            conf_st = time()
            if args.conformal_parallel:
                if args.conformal_par_device == "gpu":
                    conf_CI = conformal_parallel(inputs_x, Train_Data[:, 1:], Train_Data[:, 0], X_train_neigh,
                                                 opt_neigh, \
                                                 model, rho_Z, rhot_Z, args.conformal_grid, args.alpha,
                                                 args.gpu_devices)
                else:
                    conf_CI = conformal_parallel(inputs_x, Train_Data[:, 1:], Train_Data[:, 0], X_train_neigh,
                                                 opt_neigh, \
                                                 model, rho_Z, rhot_Z, args.conformal_grid, args.alpha,
                                                 args.cpu_devices)
            else:
                conf_CI = conformal(inputs_x, Train_Data[:, 1:], Train_Data[:, 0], X_train_neigh, opt_neigh, \
                                    model, rho_Z, rhot_Z, args.conformal_grid, args.alpha, args.device)
            conf_end = time()
            conf_time += conf_end - conf_st
            conf_IS_i = interval_score(conf_CI[:,0], conf_CI[:,1], target_y, args.alpha)
            conformal_IS += conf_IS_i
            conf_coverage = ((target_y >= (conf_CI[:,0])) & (target_y <= (conf_CI[:,1])))
            conf_covered_count += np.sum(conf_coverage)

            conformal_IW += np.mean((conf_CI[:,1]) -(conf_CI[:,0]))

        if batch_idx == 0:
            loc_save = inputs_x
            out_save = pred_y
            tar_save = target_y
            err_save = resids
            uncer_save = (CI_length)*2
            if args.conformal:
                conf_save = (conf_CI[:, 1]) - (conf_CI[:, 0])
        else:
            loc_save = np.concatenate((loc_save, inputs_x))
            out_save = np.concatenate((out_save, pred_y))
            tar_save = np.concatenate((tar_save, target_y))
            err_save = np.concatenate((err_save, resids))
            uncer_save = np.concatenate((uncer_save, CI_length * 2))
            if args.conformal:
                conf_save = np.concatenate((conf_save, (conf_CI[:, 1]) - (conf_CI[:, 0])))

    end = time()

    mse_test = mse_test / (batch_idx + 1)
    r2_test = r2_score_test / (batch_idx + 1)
    mnlp_test = nlp_test / (batch_idx + 1)
    coverage_test = covered_count / total_samples
    bayes_IS = bayes_IS / (batch_idx + 1)
    bayes_IW = bayes_IW / (batch_idx + 1)
    if args.conformal:
        conf_coverage = conf_covered_count / total_samples
        conformal_IS = conformal_IS/(batch_idx + 1)
        conformal_IW = conformal_IW / (batch_idx + 1)

    logger['r2_test'].append(r2_test)
    logger['mse_test'].append(mse_test)
    logger['mnlp_test'].append(mnlp_test)
    logger['coverage_test'].append(coverage_test)
    logger['bayes_is_test'].append(bayes_IS)
    logger['bayes_iw_test'].append(bayes_IW)

    if args.conformal:
        logger['conf_cov_test'].append(conf_coverage)
        logger['conf_is_test'].append(conformal_IS)
        logger['conf_iw_test'].append(conformal_IW)

    np.save(os.path.join(args.save_dir, 'pred_locs.npy'), loc_save)
    np.save(os.path.join(args.save_dir, 'preds.npy'), out_save)
    np.save(os.path.join(args.save_dir, 'targets.npy'), tar_save)
    np.save(os.path.join(args.save_dir, 'resids.npy'), err_save)
    np.save(os.path.join(args.save_dir, 'pred_uncer.npy'), uncer_save)

    np.savetxt(args.save_dir + '/r2_test.txt', logger['r2_test'])
    np.savetxt(args.save_dir + '/mse_test.txt', logger['mse_test'])
    np.savetxt(args.save_dir+ '/mnlp_test.txt', logger['mnlp_test'])
    np.savetxt(args.save_dir+ '/coverage_test.txt', logger['coverage_test'])
    np.savetxt(args.save_dir + '/bayes_is_test.txt', logger['bayes_is_test'])
    np.savetxt(args.save_dir + '/bayes_iw_test.txt', logger['bayes_iw_test'])
    if args.conformal == True:
        np.savetxt(args.save_dir + '/conf_cov.txt', logger['conf_cov_test'])
        np.savetxt(args.save_dir + '/conf_is.txt',  logger['conf_is_test'])
        np.savetxt(args.save_dir + '/conf_iw.txt', logger['conf_iw_test'])
        np.savetxt(args.save_dir + '/conf_time_cv.txt', [conf_time_cv])
        np.savetxt(args.save_dir + '/conf_time.txt', [conf_time])
    
    if args.conformal:
        np.save(os.path.join(args.save_dir, 'pred_uncer_conf.npy'), conf_save)
        
        print("test r2: {:.6f}, test rmse: {:.6f}, test nll: {:.6f}, test bayes coverage: {:.3f}, test conf coverage: {:.3f}, test bayes IS: {:.3f}, test conf IS: {:.3f}, test bayes IW: {:.3f}, test conf IW: {:.3f}, conf total time: {:.3f}, conf cv time: {:.3f}".format(
                        r2_test, mse_test, mnlp_test, coverage_test, conf_coverage, \
                    bayes_IS, conformal_IS, bayes_IW, conformal_IW, conf_time, conf_time_cv))
    else:
        print("test r2: {:.6f}, test rmse: {:.6f}, test nll: {:.6f}, test bayes coverage: {:.2f}, test bayes IS: {:.3f}, test bayes IW: {:.3f}".format(r2_test, mse_test, mnlp_test, coverage_test, bayes_IS, bayes_IW))

#####################################
# step 5: start training and test
#####################################
print('------------- Start training -------------')
if args.load_weights == False:
    tic = time()
    for epoch in range(1, args.epochs + 1):
        model_SVGD.train(epoch, logger)
    tic2 = time()
    np.savetxt(args.save_dir + '/avg_epoch_time.txt', [(tic2 - tic)/args.epochs])
    print('Finished training {} epochs with {} SVGD samples using {:.3f} seconds'.format(args.epochs, args.nSVGD, tic2 - tic))
    with torch.no_grad():
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'mod_weights.pt')))
        test()


else:
    with torch.no_grad():
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'mod_weights.pt')))
        test()

#####################################
# step 6: summarize results
#####################################
plotOptHis(logger, args.log_freq, args.epochs, args.save_dir, args.temporal)

del model, train_loader, val_loader, test_loader



import numpy as np
import torch
import torch.nn.functional as F
import math
import os
from tqdm import tqdm
import concurrent.futures
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utilities import log_sum_exp, parameters_to_vector, vector_to_parameters, getModel, getOpt, R2_value, dataTranform, toNumpy, saveData, interval_score
from args import args
from conformal import conformal, conformal_parallel, eta_cv
from sklearn.neighbors import KDTree
from time import time
from scipy.stats import norm

class SVGD(object):
    def __init__(self, args, bayes_nn, train_loader, val_loader, train_y = None, train_x = None):
        self.bayes_nn = bayes_nn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_samples = args.nSVGD
        self.lr = args.lr
        self.lr_noise = args.lr_noise
        self.lr_latent = args.lr_latent
        self.ntrain = int(int(args.n_dim*args.ttsplit)*args.ttsplit)
        self.batch_train_size = args.btrain
        self.out_channels = args.output_size
        self.device = args.device
        self.optimizers, self.schedulers = self._optimizers_schedulers(self.lr, self.lr_noise, self.lr_latent)
        self.epochs = args.epochs
        self.alpha = args.alpha
        self.conformal = args.conformal
        if args.conformal == True:
            self.train_x = train_x
            self.train_y = train_y
            self.neigh_range = args.conformal_neigh
            self.grid_y = args.conformal_grid
            self.conformal_par = args.conformal_parallel
            if args.conformal_par_device == 'gpu':
                self.devices = args.gpu_devices
            else:
                self.devices = args.cpu_devices

    def _squared_dist(self, X):
        XXT = torch.mm(X, X.t())
        XTX = XXT.diag()
        return -2.0 * XXT + XTX + XTX.unsqueeze(1)

    def _Kxx_dxKxx(self, X):
        squared_dist = self._squared_dist(X)
        l_square = 0.5 * squared_dist.median() / math.log(self.n_samples)
        Kxx = torch.exp(-0.5 / l_square * squared_dist)
        dxKxx = (Kxx.sum(1).diag() - Kxx).matmul(X) / l_square
        return Kxx, dxKxx

    def _optimizers_schedulers(self, lr, lr_noise, lr_latent):
        optimizers = []
        schedulers = []
        for i in range(self.n_samples):
            parameters = [{'params': [self.bayes_nn[i].log_tau], 'lr': lr_noise},
                          {'params': self.bayes_nn[i].latent.parameters(), 'lr': lr_latent},
                          {'params': self.bayes_nn[i].fc3.parameters()},
                          {'params': self.bayes_nn[i].fc4a.parameters()}, {'params': self.bayes_nn[i].fc4b.parameters()},
                          {'params': [self.bayes_nn[i].log_nu_Z], 'lr': lr_noise}, {'params': [self.bayes_nn[i].log_rho_Z], 'lr': lr_noise},
                          {'params': [self.bayes_nn[i].log_rhol_Z], 'lr': lr_noise}, {'params': [self.bayes_nn[i].log_sig2_Z]}]
            if hasattr(self.bayes_nn[i], 'log_rhot_Z'):
                parameters.append({'params': [self.bayes_nn[i].log_rhot_Z], 'lr': lr_noise})
            optimizer_i = torch.optim.AdamW(parameters, lr=lr, weight_decay= args.weight_decay)
            optimizers.append(optimizer_i)
            schedulers.append(ReduceLROnPlateau(optimizer_i, mode='min', factor=0.1, patience=5))
        return optimizers, schedulers

    def train(self, epoch, logger):
        self.bayes_nn.train()
        mse_train, r2_score = 0., 0.
        progress = tqdm(enumerate(self.train_loader), desc="Loss: ", total=len(self.train_loader))
        for batch_idx, (input_x, true_y) in progress:
            input_x, true_y = input_x.to(self.device), true_y.to(self.device)
            true_y = torch.unsqueeze(true_y, 1)
            self.bayes_nn.zero_grad()
            pred_y = torch.zeros_like(true_y)
            grad_log_joint = []
            theta = []
            log_joint = 0.

            for idx in range(self.n_samples):
                pred_y_i= self.bayes_nn[idx].forward(input_x)
                pred_y += pred_y_i

                log_joint_i = self.bayes_nn._log_joint(idx, pred_y_i, true_y, self.ntrain)
                log_joint_i.backward()
                log_joint += log_joint_i.item()

                vec_param, vec_grad_log_joint = parameters_to_vector(self.bayes_nn[idx].parameters(), both=True)

                grad_log_joint.append(vec_grad_log_joint.unsqueeze(0))
                theta.append(vec_param.unsqueeze(0))



            theta = torch.cat(theta)
            Kxx, dxKxx = self._Kxx_dxKxx(theta)
            grad_log_joint = torch.cat(grad_log_joint)
            grad_logp = torch.mm(Kxx, grad_log_joint)
            grad_theta = - (grad_logp + dxKxx) / self.n_samples
            for idx in range(self.n_samples):
                vector_to_parameters(grad_theta[idx], self.bayes_nn[idx].parameters(), grad=True)
                self.optimizers[idx].step()

                with torch.no_grad():
                    for name, param in self.bayes_nn[idx].named_parameters():
                        if 'log_nu_Z' in name:
                            param.clamp_(min=math.log(0.5), max = math.log(4.5))
                        if 'log_rho_Z' in name:
                            param.clamp_(min=math.log(0.0001), max = math.log(1.0))
                        if 'log_rhot_Z' in name:
                            param.clamp_(min=math.log(0.0001))
                        if 'log_rhol_Z' in name:
                            param.clamp_(min=math.log(0.0001))

            #mse_train += np.sqrt(F.l1_loss(pred_y / self.n_samples, true_y).item())
            mse_train += np.sqrt(F.mse_loss(pred_y / self.n_samples, true_y).item())
            r2_score += R2_value(pred_y / self.n_samples, true_y)


        mse_train = mse_train / len(self.train_loader)
        r2_train = r2_score / len(self.train_loader)



        logger['mse_train'].append(mse_train)
        logger['r2_train'].append(r2_train)
        logger['nu_Z'].append(np.exp(self.bayes_nn.log_nu_Z.mean().item()))
        logger['rho_Z'].append(np.exp(self.bayes_nn.log_rho_Z.mean().item()))
        logger['sig2_Z'].append(np.exp(self.bayes_nn.log_sig2_Z.mean().item()))
        logger['tau'].append(np.exp(self.bayes_nn.log_tau.mean().item()))
        if args.temporal == True:
            logger['rhot_Z'].append(np.exp(self.bayes_nn.log_rhot_Z.mean().item()))

        

        for idx in range(self.n_samples):
            self.schedulers[idx].step(mse_train)

        best_val = None
        
        with torch.no_grad():
            self.bayes_nn.eval()
            mse_val, r2_score_val, nlp_val, covered_count, total_samples = 0., 0., 0., 0., 0.
            bayes_IS, bayes_IW = 0., 0.

            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                
                inputs, targets = inputs.to(args.device), targets.to(args.device)

                mse, nlp, outputs = self.bayes_nn._compute_mse_nlp(inputs, targets, size_average=True, out=True)

                outputs_mean = outputs.mean(0)
        
                EyyT = (outputs ** 2).mean(0)
                EyEyT = outputs_mean ** 2
                outputs_noise_var = self.bayes_nn.log_tau.exp().mean()
                outputs_var =  EyyT - EyEyT + outputs_noise_var
                #outputs_var = torch.clamp(outputs_var, min=1e-6)
        
                #mse_val += np.sqrt(F.l1_loss(outputs_mean, torch.unsqueeze(targets, 1)).item())
                mse_val += np.sqrt(F.mse_loss(outputs_mean, torch.unsqueeze(targets, 1)).item())
                r2_score_val += R2_value(outputs_mean, torch.unsqueeze(targets, 1))
                #nlp_val += nlp.item()
                nlp_val += F.gaussian_nll_loss(outputs_mean, torch.unsqueeze(targets,1), outputs_var).item()

                inputs_x = inputs.detach().cpu().numpy()
                pred_y = torch.squeeze(outputs_mean).detach().cpu().numpy()
                target_y = targets.detach().cpu().numpy()
                resids = pred_y - target_y
                CI_length = np.sqrt(toNumpy(torch.squeeze(outputs_var))) * norm.ppf(1-(self.alpha/2))
        
                coverage = ((target_y >= (pred_y - CI_length)) & (target_y <= (pred_y + CI_length)))
                covered_count += np.sum(coverage)
                total_samples += len(target_y)

                bayes_IS_i = interval_score(pred_y-CI_length, pred_y+CI_length, target_y, self.alpha)
                bayes_IS += bayes_IS_i
                bayes_IW += np.mean(CI_length*2)

            mse_val = mse_val / (batch_idx + 1)
            r2_val = r2_score_val / (batch_idx + 1)
            mnlp_val = nlp_val / (batch_idx + 1)
            coverage_val = covered_count / total_samples
            bayes_IS = bayes_IS / (batch_idx + 1)
            bayes_IW = bayes_IW / (batch_idx + 1)
            
            #stop = time()
            if best_val is None:
                best_val = mse_val
                torch.save(self.bayes_nn.state_dict(), os.path.join(args.save_dir, 'mod_weights.pt')) 
            else:
                if best_val < mse_val:
                    torch.save(self.bayes_nn.state_dict(), os.path.join(args.save_dir, 'mod_weights.pt')) 
            
            if epoch % args.log_freq == 0:
                logger['r2_val'].append(r2_val)
                logger['mse_val'].append(mse_val)
                logger['mnlp_val'].append(mnlp_val)
                logger['coverage_val'].append(coverage_val)
                logger['bayes_is_val'].append(bayes_IS)
                logger['bayes_iw_val'].append(bayes_IW)
            
        if epoch == self.epochs:
            print("epoch {}, training r2: {:.6f}, training rmse: {:.6f}, validation r2: {:.6f}, validation rmse: {:.6f}, validation nll: {:.6f}, validation bayes coverage: {:.2f}, validation bayes IS: {:.3f}, validation bayes IW: {:.3f}".format(epoch, r2_train, mse_train, r2_val, mse_val, mnlp_val, coverage_val, bayes_IS, bayes_IW))
        else:
            print("epoch {}, training r2: {:.6f}, training rmse: {:.6f}, validation r2: {:.6f}, validation rmse: {:.6f}, validation nll: {:.6f}, validation bayes coverage: {:.2f}, validation bayes IS: {:.3f}, validation bayes IW: {:.3f}".format(epoch, r2_train, mse_train, r2_val, mse_val, mnlp_val, coverage_val, bayes_IS, bayes_IW))

      

            



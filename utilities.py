import os
import torch
import numpy as np

from sklearn.metrics import r2_score
from model import RFF_Net
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau


def getModel(args):
    model = (RFF_Net(input_size = args.input_size,
                     aux_size = args.aux_size,
                     gp_size = args.gp_size,
                     latent_mod = args.latent_mod,
                     n_blocks = args.block_size,
                     output_size = args.output_size,
                     latent_size = args.latent_size,
                     latent_temporal=args.latent_temporal,
                     mod_temporal = args.mod_temporal,
                     t_steps = args.t_dim))
    return model

def getOpt(args, model):
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    if args.lrs == 'StepLR':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.lrs == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=0.995)
    elif args.lrs == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=args.lr, threshold_mode='rel', cooldown=0, min_lr=1e-12, eps=1e-12)

    logger = {}
    logger['mse_train'] = []
    logger['mse_val'] = []
    logger['r2_train'] = []
    logger['r2_val'] = []
    logger['mnlp_val'] = []
    logger['coverage_val'] = []
    logger['nu_Z'] = []
    logger['rho_Z'] = []
    logger['sig2_Z'] = []
    logger['tau'] = []
    logger['bayes_is_val'] = []
    logger['bayes_iw_val'] = []

    logger['mse_test'] = []
    logger['r2_test'] = []
    logger['mnlp_test'] = []
    logger['coverage_test'] = []
    logger['bayes_is_test'] = []
    logger['bayes_iw_test'] = []
    logger['crps_mean_test'] = []

    if args.temporal == True:
        logger['rhot_Z'] = []

    if args.conformal == True:
        logger['conf_cov_test'] = []
        logger['conf_is_test'] = []
        logger['conf_iw_test'] = []
    
    return criterion, optimizer, scheduler, logger

def R2_value(output, targets):
    output = output.detach().cpu().numpy().squeeze()
    targets = targets.detach().cpu().numpy().squeeze()

    #r2 = r2_score(output, targets)
    corr_matrix = np.corrcoef(output, targets, rowvar=False)
    r2 = corr_matrix[0, 1]**2
    return r2

def interval_score(lower, upper, y, alpha):
    width = upper - lower
    sl = 2/alpha * (lower - y)* (y < lower)
    su = 2/alpha * (y - upper)* (y > upper)
    return np.mean(width + sl + su)

def dataTranform(tensor, data_map):
    data = tensor.detach().cpu().numpy()
    for idx in range(data.shape[0]):
        if idx == 0:
            temp = np.expand_dims(data_map.inverse_transform(data[idx,:,:]), axis=0)
        else:
            temp = np.concatenate((temp, np.expand_dims(data_map.inverse_transform(data[idx,:,:]), axis=0)))
    return temp

def toNumpy(tensor):
    return tensor.detach().cpu().numpy()

def saveData(x, y, y_hat, error, uncer, save_dir):
    np.savez(os.path.join(save_dir, 'results'), locs = x, outputs=y_hat, targets=y, uncertainties=uncer, error=error)

def log_sum_exp(input, dim=None, keepdim=False):
    if dim is None:
        input = input.view(-1)
        dim = 0
    max_val = input.max(dim=dim, keepdim=True)[0]
    output = max_val + (input - max_val).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        output = output.squeeze(dim)
    return output

def parameters_to_vector(parameters, grad=False, both=False):
    param_device = None
    if not both:
        vec = []
        if not grad:
            for param in parameters:
                param_device = _check_param_device(param, param_device)
                vec.append(param.data.view(-1))
        else:
            for param in parameters:
                param_device = _check_param_device(param, param_device)
                vec.append(param.grad.data.view(-1))
        return torch.cat(vec)
    else:
        vec_params, vec_grads = [], []
        for param in parameters:
            param_device = _check_param_device(param, param_device)
            vec_params.append(param.data.view(-1))
            vec_grads.append(param.grad.data.view(-1))
        return torch.cat(vec_params), torch.cat(vec_grads)

def vector_to_parameters(vec, parameters, grad=True):
    param_device = None
    pointer = 0
    
    if grad:
        for param in parameters:
            param_device = _check_param_device(param, param_device)
            num_param = torch.prod(torch.LongTensor(list(param.size())))
            param.grad.data = vec[pointer:pointer + num_param].view(param.size())
            pointer += num_param
    else:
        for param in parameters:
            param_device = _check_param_device(param, param_device)
            num_param = torch.prod(torch.LongTensor(list(param.size())))
            param.data = vec[pointer:pointer + num_param].view(param.size())
            pointer += num_param

def _check_param_device(param, old_param_device):
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:
            warn = (param.get_device() != old_param_device)
        else:
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device



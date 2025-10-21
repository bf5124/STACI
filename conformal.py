import numpy as np
import torch
from utilities import interval_score
import copy
from joblib import Parallel, delayed

# Global variables that will hold the model and its assigned device in each worker process.
# Global variables for thread workers
global_models = None
global_devices = None


def process_single(i, pred_x, train_locs, Y, Y_precision, m, alpha):
    """
    Process a single prediction index.
    Each task picks its model/device based on a round-robin assignment
    using the task index.
    """
    global global_models, global_devices
    # Select the model copy corresponding to this task
    model_index = i % len(global_models)
    model = global_models[model_index]
    device = global_devices[model_index]

    # Get the prediction location and its neighbors
    pred_loc = pred_x[i, :]
    tot_locs = np.concatenate((pred_loc.reshape(-1, 3), train_locs[i]), axis=0)

    # Convert to tensor on the assigned device
    x_neigh = torch.tensor(tot_locs, dtype=torch.float32).to(device)

    # Compute predictions and related statistics
    out_neigh = model.predict(x_neigh)
    out_neigh_mean = out_neigh.mean(0)

    EyyT = (out_neigh ** 2).mean(0)
    EyEyT = out_neigh_mean ** 2
    outputs_noise_var = model.log_tau.exp().mean().to(device)
    outputs_var = torch.squeeze(EyyT - EyEyT + outputs_noise_var).detach().cpu().numpy()

    y_pred = torch.squeeze(out_neigh_mean).detach().cpu().numpy()

    # Compute conformity scores for candidate values
    Y_cand = np.linspace(Y[i].min(), Y[i].max(), Y_precision)
    Y_aug = np.concatenate(
        (Y_cand.reshape(-1, 1), np.tile(Y[i], (Y_precision, 1))),axis=1
    )
    deltas = np.abs((Y_aug - y_pred) / np.sqrt(outputs_var))

    py = np.sum(deltas >= deltas[:, [0]], axis=1)

    threshold = np.floor((m + 1) * alpha) / (m + 1)
    selected = Y_cand[py >= threshold]
    return (selected.min(), selected.max())


def conformal_parallel(pred_x, train_x, train_y, neigh_tree, m, model, rho_s, rho_t, Y_precision, alpha, devices):
    """
    Computes confidence intervals for predictions in parallel using threads.

    A separate copy of the model is created for each device in the provided list.
    Tasks are assigned in round-robin fashion based on the task index.

    Parameters:
      - pred_x, train_x, train_y, neigh_tree, m, rho_s, rho_t, Y_precision, alpha:
          as in your original code.
      - model: a PyTorch model that has a .predict() method and a .log_tau attribute.
      - devices: a list of devices (e.g., [torch.device('cpu')]*d or list of GPU devices).

    Returns:
      - A NumPy array of shape (n_preds, 2) with the computed confidence intervals.
    """
    global global_models, global_devices
    # Save the devices list and create a copy of the model per device.

    if global_models is None:
        global_devices = devices
        global_models = [copy.deepcopy(model).to(device) for device in devices]

    # Pre-scale pred_x for querying neighbors.
    pred_x_sc = pred_x.copy()
    pred_x_sc[:, :2] /= rho_s
    pred_x_sc[:, 2] /= rho_t

    dists, idx = neigh_tree.query(pred_x_sc, k=m)
    train_locs = train_x[idx]
    Y = train_y[idx]
    n_preds = pred_x.shape[0]

    results = Parallel(n_jobs=len(devices), backend='threading')(
        delayed(process_single)(i, pred_x, train_locs, Y, Y_precision, m, alpha)
        for i in range(n_preds)
    )

    conf_intervals = np.array(results)
    return conf_intervals

def conformal(pred_x, train_x, train_y, neigh_tree, m, model, rho_s, rho_t, Y_precision, alpha, device):
    pred_x_sc = pred_x.copy()
    pred_x_sc[:, :2] /= rho_s
    pred_x_sc[:, 2] /= rho_t

    dists, idx = neigh_tree.query(pred_x_sc, k= m)

    train_locs = train_x[idx]
    Y = train_y[idx]

    n_preds = pred_x.shape[0]
    conf_intervals = np.zeros((n_preds, 2))

    for i in range(n_preds):
        pred_loc = pred_x[i,:]

        tot_locs = np.concatenate((pred_loc.reshape(-1,3), train_locs[i]), axis = 0)
        x_neigh = torch.tensor(tot_locs, dtype=torch.float32).to(device)
        out_neigh = model.predict(x_neigh)

        out_neigh_mean = out_neigh.mean(0)

        EyyT = (out_neigh ** 2).mean(0)
        EyEyT = out_neigh_mean ** 2
        outputs_noise_var = model.log_tau.exp().mean()
        outputs_var = torch.squeeze(EyyT - EyEyT + outputs_noise_var).detach().cpu().numpy()

        y_pred = torch.squeeze(out_neigh_mean).detach().cpu().numpy()

        Y_cand = np.linspace(Y[i].min(), Y[i].max(), Y_precision)

        Y_aug = np.concatenate(
            (Y_cand.reshape(-1, 1), np.tile(Y[i], (Y_precision, 1))), axis=1
        )
        deltas = np.abs((Y_aug - y_pred) / np.sqrt(outputs_var))

        py = np.sum(deltas >= deltas[:, [0]], axis=1)
        threshold = np.floor((m + 1)*alpha) / (m+1)
        selected = Y_cand[py >= threshold]
        conf_intervals[i, :] = (selected.min(), selected.max())

    return conf_intervals

def eta_cv(train_x, train_y, neigh_tree, m_range, model, rho_s, rho_t, Y_precision, alpha, device, parallel, devices):

    pred_idx = np.random.choice(train_x.shape[0], 100, replace=False)
    pred_locs = train_x[pred_idx,:]

    opt_int_score = None
    opt_m = 0
    for m in m_range:
        m = int(m)
        if parallel:
            PI = conformal_parallel(pred_locs, train_x, train_y, neigh_tree, m, model, rho_s, rho_t, Y_precision, alpha, devices)
        else:
            PI = conformal(pred_locs, train_x, train_y, neigh_tree, m, model, rho_s, rho_t, Y_precision, alpha, device)
        int_score = interval_score(PI[:,0], PI[:,1], train_y[pred_idx], alpha)
        if opt_int_score is None:
            opt_int_score = int_score
            opt_m = m
        else:
            if int_score < opt_int_score:
                opt_int_score = int_score
                opt_m = m

    return opt_m


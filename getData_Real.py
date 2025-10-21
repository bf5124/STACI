import os
import zipfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from pyproj import Transformer
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split
from einops import rearrange

def add_grid_indices_and_date(test_locs):
    """
    Adds grid indices and date information to the test locations DataFrame.

    Args:
        test_locs (pd.DataFrame): DataFrame containing test location information.

    Returns:
        pd.DataFrame: Updated DataFrame with grid indices and date information.
    """
    date = np.array(np.datetime64("2020-03-05"))
    t = date.astype("float")

    delta_x = np.diff(np.sort(test_locs["x"].unique())).min()
    delta_y = np.diff(np.sort(test_locs["y"].unique())).min()

    x_start = test_locs["x"].min()
    x_end = test_locs["x"].max()
    x_coords = np.arange(x_start, x_end + delta_x, delta_x)

    y_start = test_locs["y"].min()
    y_end = test_locs["y"].max()
    y_coords = np.arange(y_start, y_end + delta_y, delta_y)

    test_locs["x_idxs"] = match(test_locs["x"].values, x_coords)
    test_locs["y_idxs"] = match(test_locs["y"].values, y_coords)

    test_locs["date"] = date
    test_locs["t"] = t

    return test_locs


def match(x, y, exact=True, tol=1e-9):
    """
    Matches elements of array x with elements in array y based on exact or approximate matching.

    Args:
        x (np.ndarray): Array of values to be matched.
        y (np.ndarray): Array of reference values to match against.
        exact (bool, optional): Whether to match exactly. Defaults to True.
        tol (float, optional): Tolerance for approximate matching. Defaults to 1e-9.

    Returns:
        np.ndarray: Array of indices in y that correspond to matches for x.

    Raises:
        AssertionError: If any value in x cannot be matched within the tolerance.
    """
    if exact:
        mask = x[:, None] == y
    else:
        dif = np.abs(x[:, None] - y)
        mask = dif < tol

    row_mask = mask.any(axis=1)
    assert (
        row_mask.all()
    ), f"{(~row_mask).sum()} not found, uniquely: {np.unique(np.array(x)[~row_mask])}"
    return np.argmax(mask, axis=1)

def WGS84toEASE2(lon, lat, return_vals="both", lon_0=0, lat_0=90):
    """
    Converts WGS84 longitude and latitude coordinates to EASE2 grid coordinates.
    """

    valid_return_vals = ['both', 'x', 'y']
    assert return_vals in ['both', 'x', 'y'], f"return_val: {return_vals} is not in valid set: {valid_return_vals}"
    EASE2 = f"+proj=laea +lon_0={lon_0} +lat_0={lat_0} +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
    WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    transformer = Transformer.from_crs(WGS84, EASE2)
    x, y = transformer.transform(lon, lat)
    if return_vals == 'both':
        return x, y
    elif return_vals == "x":
        return x
    elif return_vals == "y":
        return y

def data_transform_MSS(Train_Data, Test_Data, ttsplit, batch_size, seed):
    n = Train_Data.shape[0]
    val_size = int(n * ttsplit)
    train_size = n-val_size

    X_Train_full = Train_Data[:, 1:]
    Y_Train_full = Train_Data[:, 0]
    X_Test = Test_Data[:, 1:]
    Y_Test = Test_Data[:, 0]

    X_Train = torch.tensor(X_Train_full, dtype=torch.float32)
    Y_Train = torch.tensor(Y_Train_full, dtype=torch.float32)

    X_Test = torch.tensor(X_Test, dtype=torch.float32)
    Y_Test = torch.tensor(Y_Test, dtype=torch.float32)

    train_full_ds = TensorDataset(X_Train, Y_Train)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    # Use random_split to split the dataset into training and validation sets.
    train_ds, val_ds = random_split(train_full_ds, [train_size, val_size], generator=g)
    train_loader = DataLoader(train_ds, batch_size=batch_size[0], shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size[1], shuffle=False)

    test_ds = TensorDataset(X_Test, Y_Test)
    test_loader = DataLoader(test_ds, batch_size=batch_size[2], shuffle= False)

    return train_loader, val_loader, test_loader

def data_transform_AOD(
    Train_Data: np.ndarray,
    Val_Data:   np.ndarray,
    Test_Data:  np.ndarray,
    batch_size: tuple[int,int,int],
    seed:       int | None = None
):
    """
    Turn Train/Val/Test numpy arrays into DataLoaders, and Plot_Data into a tensor.

    Args:
      Train_Data : (N_train, 1+M) array, first col = targets, rest = features
      Val_Data   : (N_val,   1+M) array
      Test_Data  : (N_test,  1+M) array
      batch_size : tuple of (bs_train, bs_val, bs_test)
      seed       : random seed for reproducibility of shuffling

    Returns:
      train_loader, val_loader, test_loader, Plot_tensor
    """
    # unpack batch sizes
    bs_tr, bs_val, bs_te = batch_size

    # ---- 1) convert numpy → torch tensors ----
    # Train
    X_tr = torch.tensor(Train_Data[:, 1:], dtype=torch.float32)
    Y_tr = torch.tensor(Train_Data[:, 0],   dtype=torch.float32)
    # Val
    X_val = torch.tensor(Val_Data[:, 1:], dtype=torch.float32)
    Y_val = torch.tensor(Val_Data[:, 0],   dtype=torch.float32)
    # Test
    X_te = torch.tensor(Test_Data[:, 1:], dtype=torch.float32)
    Y_te = torch.tensor(Test_Data[:, 0],   dtype=torch.float32)


    # ---- 2) create TensorDatasets ----
    train_ds = TensorDataset(X_tr, Y_tr)
    val_ds   = TensorDataset(X_val, Y_val)
    test_ds  = TensorDataset(X_te, Y_te)

    # ---- 3) build DataLoaders ----
    # ensure reproducible shuffling
    if seed is not None:
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
    else:
        g = None

    train_loader = DataLoader(
        train_ds, batch_size=bs_tr, shuffle=True, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs_val, shuffle=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs_te, shuffle=False
    )

    return train_loader, val_loader, test_loader

    
def prepare_MSS_data(datapath, ttsplit, seed):
    sim_data = pd.read_hdf(os.path.join(datapath, 'along_track_sample_from_mss_ground_ABC.h5'), "data")

    obs_data, test_locs = train_test_split(
        sim_data, 
        test_size=ttsplit,     # 10% goes to test
        random_state=seed,    # for reproducibility
        shuffle=True        # shuffle before split (default)
    )
        
    spatial_X = obs_data[["x", "y"]].to_numpy()
    temporal_X = obs_data[["t"]].to_numpy()
    Y = obs_data["obs"].to_numpy()
    
    min_vals = spatial_X.min(axis=0)  # shape (2,)
    max_vals = spatial_X.max(axis=0)  # shape (2,)
    
    # Compute the range for each column:
    ranges = max_vals - min_vals
    
    # The scale is the maximum range among the two columns:
    scale = np.max(ranges)

    # Normalize each column:
    # For the first column:
    spatial_X[:, 0] = (spatial_X[:, 0] - min_vals[0]) / scale
    # For the second column:
    spatial_X[:, 1] = (spatial_X[:, 1] - min_vals[1]) / scale
    
    Y_mean = Y.mean()
    Y_std = Y.std()
    norm_Y = (Y - Y_mean) / Y_std

    spatial_X_test = test_locs[["x", "y"]].to_numpy()

    # Normalize each column:
    # For the first column:
    spatial_X_test[:, 0] = (spatial_X_test[:, 0] - min_vals[0]) / scale
    # For the second column:
    spatial_X_test[:, 1] = (spatial_X_test[:, 1] - min_vals[1]) / scale

    temporal_X_test = test_locs[["t"]].to_numpy()

    temporal_X_mean = temporal_X.mean()
    temporal_X_std = temporal_X.std()
    temporal_X = (
                         temporal_X - temporal_X_mean
                 ) / temporal_X_std


    temporal_X_test = (
                              temporal_X_test - temporal_X_mean
                      ) / temporal_X_std

    Train_Data = np.column_stack((norm_Y, spatial_X, temporal_X))

    Y_test = test_locs["obs"].to_numpy()
    
    norm_Y_test  = (Y_test  - Y_mean) / Y_std

    Test_Data = np.column_stack((norm_Y_test, spatial_X_test, temporal_X_test))

    return Train_Data, Test_Data, Y_mean, Y_std



def prepare_AOD_data(datapath, sample_frac, val_time, test_time, seed):
    """
    Splits AOD_data.csv into train/val/test (with overlap),
    normalizes, encodes time, and builds a regular Plot_Data grid.

    Args:
      datapath    : directory containing 'AOD_data.csv'
      sample_frac : fraction of points PER DAY to sample into training
      val_time    : integer Day threshold; days 1..val_time go into validation set
      test_time   : Day value to hold out as test set
      grid_nx     : number of grid points along S1 (x) axis
      grid_ny     : number of grid points along S2 (y) axis
      seed        : RNG seed for reproducibility

    Returns:
      Train_Data : np.ndarray, shape (N_train,4) = [norm_Y, S1_norm, S2_norm, t_enc]
      Val_Data   : np.ndarray, shape (N_val,4)   = [norm_Y, S1_norm, S2_norm, t_enc]
      Test_Data  : np.ndarray, shape (N_test,4)  = [norm_Y, S1_norm, S2_norm, t_enc]
      Plot_Data  : np.ndarray, shape (grid_ny*grid_nx,3) = [S1_norm, S2_norm, t_enc]
      Y_mean     : float
      Y_std      : float
      grid_shape : (grid_ny, grid_nx)
    """
    # 1) load & drop missing
    df = pd.read_csv(os.path.join(datapath, 'AOD_data.csv'), index_col=0)
    df = df.dropna(subset=['AOD'])

    # 2) define val/test subsets (but do NOT remove them from train sampling)
    val_df  = df[df['Day'].between(1, val_time)]
    test_df = df[df['Day'] == test_time]

    # 3) sample training rows: sample_frac per-day from the entire df
    rng = np.random.default_rng(seed)
    train_parts = []
    for day in df['Day'].unique():
        sub = df[df['Day'] == day]
        k   = int(np.ceil(sample_frac * len(sub)))
        idx = rng.choice(sub.index, size=k, replace=False)
        train_parts.append(df.loc[idx])

    train_df = pd.concat(train_parts, axis=0)

    # 4) extract raw arrays
    def extract(df_):
        X_sp = df_[['S1','S2']].to_numpy()          # (N,2)
        T_t  = df_['Day'].to_numpy().reshape(-1,1)  # (N,1)
        Y_y  = df_['AOD'].to_numpy()               # (N,)
        return X_sp, T_t, Y_y

    X_tr_sp, T_tr, Y_tr = extract(train_df)
    X_val_sp, T_val, Y_val = extract(val_df)
    X_te_sp,  T_te,  Y_te  = extract(test_df)

    # 5) spatial normalization (train stats)
    min_xy    = X_tr_sp.min(axis=0)
    max_xy    = X_tr_sp.max(axis=0)
    scale_xy  = (max_xy - min_xy).max()
    X_tr_nm   = (X_tr_sp - min_xy) / scale_xy
    X_val_nm  = (X_val_sp - min_xy) / scale_xy
    X_te_nm   = (X_te_sp  - min_xy) / scale_xy

    # 6) target normalization (train z-score)
    Y_mean    = Y_tr.mean()
    Y_std     = Y_tr.std() + 1e-10

    Y_tr_nm   = Y_tr
    Y_val_nm  = Y_val
    Y_te_nm   = Y_te
    Y_tr_nm   = (Y_tr - Y_mean) / Y_std
    Y_val_nm  = (Y_val - Y_mean) / Y_std
    Y_te_nm   = (Y_te  - Y_mean) / Y_std

    # 7) time encoding
    t_mean    = T_tr.mean()
    t_std     = T_tr.std() + 1e-10
    T_tr_enc  = ((T_tr - t_mean) / t_std).reshape(-1,1)
    T_val_enc = ((T_val - t_mean) / t_std).reshape(-1,1)
    T_te_enc  = ((T_te - t_mean) / t_std).reshape(-1,1)

    # 8) assemble Train/Val/Test
    Train_Data = np.column_stack([ Y_tr_nm.reshape(-1,1),
                                   X_tr_nm,
                                   T_tr_enc ])
    Val_Data   = np.column_stack([ Y_val_nm.reshape(-1,1),
                                   X_val_nm,
                                   T_val_enc ])
    Test_Data  = np.column_stack([ Y_te_nm.reshape(-1,1),
                                   X_te_nm,
                                   T_te_enc ])



    return Train_Data, Val_Data, Test_Data, Y_mean, Y_std

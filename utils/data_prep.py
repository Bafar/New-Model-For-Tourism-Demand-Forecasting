import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def create_xy_windows(X_data, y_data, input_len, pred_horizon, step):
    X_list, y_list = [], []
    data_len = min(len(X_data), len(y_data))
    for i in range(0, data_len - input_len - pred_horizon + 1, step):
        X_list.append(X_data[i: i + input_len])
        y_list.append(y_data[i + input_len: i + input_len + pred_horizon])
    return torch.from_numpy(np.array(X_list)).float(), torch.from_numpy(np.array(y_list)).float()


def create_exo_windows(X_data, input_len, pred_horizon, pre_exo, step):
    X_list = []
    data_len = len(X_data)
    for i in range(0, data_len - input_len - pred_horizon + 1, step):
        X_list.append(X_data[i: i + input_len + pre_exo])
    return torch.from_numpy(np.array(X_list)).float()


def split_existing_data(all_inputs, all_targets, exo_data, batch_size=32):
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen = torch.Generator(device=device_str)

    total = len(all_inputs)
    train_size = int(total * 0.7)
    val_size = int(total * 0.1)

    train_x = all_inputs[:train_size]
    train_exo = exo_data[:train_size]
    train_y = all_targets[:train_size]

    val_x = all_inputs[train_size: train_size + val_size]
    val_exo = exo_data[train_size: train_size + val_size]
    val_y = all_targets[train_size: train_size + val_size]

    test_x = all_inputs[train_size + val_size:]
    test_exo = exo_data[train_size + val_size:]
    test_y = all_targets[train_size + val_size:]

    train_dataset = TensorDataset(train_x, train_exo, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=gen)

    val_dataset = TensorDataset(val_x, val_exo, val_y)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(test_x, test_exo, test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
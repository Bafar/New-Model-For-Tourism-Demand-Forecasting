import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL

from models.net import STGNet
from utils.data_prep import create_xy_windows, create_exo_windows, split_existing_data
from utils.metrics import CustomLoss, Evaluator

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    data0 = pd.read_excel(args.data_path, sheet_name=args.sheet_name)

    series = data0[args.target_col]
    stl = STL(series, period=7, robust=True)
    res = stl.fit()
    df_vmd = pd.DataFrame({'Trend': res.trend, 'Seasonal': res.seasonal, 'Resid': res.resid})

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(df_vmd)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(series.values.reshape(-1, 1))

    all_inputs, all_targets = create_xy_windows(X_scaled, y_scaled, args.batch_len, args.pred_horizon, args.step)

    exo = data0[args.exo_cols]
    scaler_exo = StandardScaler()
    exo_scaled = scaler_exo.fit_transform(exo)
    exo_data = create_exo_windows(exo_scaled, args.batch_len, args.pred_horizon, args.pre_exo, args.step)

    train_loader, val_loader, test_loader = split_existing_data(all_inputs, all_targets, exo_data,
                                                                batch_size=args.batch_size)

    T_list = [3, 7, 30, 90, 365]
    data_dim = len(T_list) * 4
    hide_node_num = 2 * data_dim

    model = STGNet(
        origin_dim=1,
        output_dim=1,
        time_max=args.time_max,
        data_dim=data_dim,
        heads_num=args.heads_num,
        hide_node_num=hide_node_num,
        T=T_list,
        batch_len=args.batch_len,
        prediction_horizon=args.pred_horizon,
        exo_dim=len(args.exo_cols),
        dropout=args.dropout,
        g_flags=[1, 1, 1],
        s_flags=[0, 0, 0]
    ).to(device)

    criterion = CustomLoss(gamma=0, loss_type='mae')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = 0
        all_train_outputs = []
        all_train_targets = []
        graph_last = None

        for i, (batch_input, batch_exo, batch_target) in enumerate(train_loader):
            batch_input, batch_target, batch_exo = batch_input.to(device), batch_target.to(device), batch_exo.to(device)
            optimizer.zero_grad()
            output, graph = model(batch_input, batch_exo, 0, batch_input.size(0), epoch, args.step)

            if i == 0 and epoch == 0:
                loss = criterion(output, batch_target, graph, torch.zeros_like(graph).to(device) + 10)
            else:
                loss = criterion(output, batch_target, graph, graph_last)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            graph_last = graph.detach()
            epoch_train_loss += loss.item()
            all_train_outputs.append(output.detach().cpu())
            all_train_targets.append(batch_target.detach().cpu())

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_outputs_tensor = torch.cat(all_train_outputs)
        train_targets_tensor = torch.cat(all_train_targets)

        num_train_samples = train_outputs_tensor.shape[0] * train_outputs_tensor.shape[1]
        train_output_inv = torch.from_numpy(
            scaler_y.inverse_transform(train_outputs_tensor.reshape(num_train_samples, 1))).reshape(
            train_outputs_tensor.shape)
        train_target_inv = torch.from_numpy(
            scaler_y.inverse_transform(train_targets_tensor.reshape(num_train_samples, 1))).reshape(
            train_targets_tensor.shape)
        train_estimate = Evaluator(true=train_target_inv, predict=train_output_inv)
        train_mape = train_estimate.MAPE()

        model.eval()
        epoch_val_loss = 0
        all_val_outputs = []
        all_val_targets = []

        with torch.no_grad():
            for batch_input, batch_exo, batch_target in val_loader:
                batch_input, batch_target, batch_exo = batch_input.to(device), batch_target.to(device), batch_exo.to(
                    device)
                output, graph = model(batch_input, batch_exo, 0, batch_input.size(0), epoch, args.step)
                val_loss = torch.mean(torch.abs(output - batch_target))
                epoch_val_loss += val_loss.item()
                all_val_outputs.append(output.cpu())
                all_val_targets.append(batch_target.cpu())

        avg_val_loss = epoch_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        val_outputs_tensor = torch.cat(all_val_outputs)
        val_targets_tensor = torch.cat(all_val_targets)
        num_val_samples = val_outputs_tensor.shape[0] * val_outputs_tensor.shape[1]
        val_output_inv = torch.from_numpy(
            scaler_y.inverse_transform(val_outputs_tensor.reshape(num_val_samples, 1))).reshape(
            val_outputs_tensor.shape)
        val_target_inv = torch.from_numpy(
            scaler_y.inverse_transform(val_targets_tensor.reshape(num_val_samples, 1))).reshape(
            val_targets_tensor.shape)
        val_estimate = Evaluator(true=val_target_inv, predict=val_output_inv)
        val_mape = val_estimate.MAPE()

        print(
            f"Epoch:{epoch + 1:02d} | Val Loss:{avg_val_loss:.4f} | Train MAPE:{train_mape:.4f} | Val MAPE:{val_mape:.4f}")

        if avg_val_loss < best_val_loss and epoch >= 1:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        elif avg_val_loss >= best_val_loss and epoch >= 1:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
        if args.save_path:
            torch.save(best_model_state, args.save_path)

    model.eval()
    all_test_outputs, all_test_targets = [], []
    with torch.no_grad():
        for batch_input, batch_exo, batch_target in test_loader:
            batch_input, batch_target, batch_exo = batch_input.to(device), batch_target.to(device), batch_exo.to(device)
            output, _ = model(batch_input, batch_exo, 0, batch_input.size(0), epoch=0, step=args.step)
            all_test_outputs.append(output.cpu())
            all_test_targets.append(batch_target.cpu())

    test_outputs_tensor = torch.cat(all_test_outputs)
    test_targets_tensor = torch.cat(all_test_targets)
    batch_dim, steps_dim, feature_dim = test_outputs_tensor.shape
    num_test_samples = test_outputs_tensor.shape[0] * test_outputs_tensor.shape[1]

    output_inv_flat = scaler_y.inverse_transform(test_outputs_tensor.reshape(num_test_samples, 1))
    target_inv_flat = scaler_y.inverse_transform(test_targets_tensor.reshape(num_test_samples, 1))

    output_inv = torch.from_numpy(output_inv_flat).reshape(-1, 1)
    output_inv = torch.clamp(output_inv, max=41000)
    target_inv = torch.from_numpy(target_inv_flat).reshape(-1, 1)

    test_estimate = Evaluator(true=target_inv, predict=output_inv)
    print("\n--- Test Metrics Evaluation ---")
    print(f"MAE:   {test_estimate.MAE().item():.4f}")
    print(f"MAPE:  {test_estimate.MAPE().item():.4f}")
    print(f"RMSE:  {test_estimate.RMSE().item():.4f}")
    print(f"SMAPE: {test_estimate.SMAPE().item():.4f}%")
    print(f"MDA:   {test_estimate.MDA().item():.4f}")
    print(f"R2:    {test_estimate.R2().item():.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--sheet_name', type=str, default='sheet1')
    parser.add_argument('--target_col', type=str, default='人数')
    parser.add_argument('--exo_cols', type=str, nargs='+',
                        default=['百度指数', '节假日', '最高温', '最低温', '天气', 'Attractions', 'Accessibility',
                                 'Amenities', 'Available Packages', 'Activities', 'Ancillary Services'])
    parser.add_argument('--pred_horizon', type=int, default=1)
    parser.add_argument('--batch_len', type=int, default=7)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--pre_exo', type=int, default=0)
    parser.add_argument('--time_max', type=int, default=3000)
    parser.add_argument('--heads_num', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='stgnet_model.pth')

    args = parser.parse_args()
    main(args)
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self, gamma, loss_type):
        super(CustomLoss, self).__init__()
        self.gamma = gamma
        self.loss_type = loss_type

    def log_cosh(self, y_pred, y_true):
        error = y_pred - y_true
        return torch.mean(torch.log(torch.cosh(error)))

    def mse(self, y_pred, y_true):
        return torch.sqrt(torch.mean(torch.square(y_pred - y_true)))

    def mae(self, y_pred, y_true):
        return torch.mean(torch.abs(y_pred - y_true))

    def forward(self, output, target, G_present, G_previous):
        if self.loss_type == 'mse':
            loss1 = self.mse(output, target)
        elif self.loss_type == 'log_cosh':
            loss1 = self.log_cosh(output, target)
        elif self.loss_type == 'mae':
            loss1 = self.mae(output, target)
        else:
            loss1 = self.mae(output, target)

        loss2 = torch.sqrt(torch.mean(torch.square(G_present - G_previous)))
        return loss1 + self.gamma * loss2


class Evaluator:
    def __init__(self, true, predict):
        self.true = true
        self.predict = predict

    def MSE(self):
        return F.mse_loss(self.predict, self.true)

    def MAE(self):
        return F.l1_loss(self.predict, self.true)

    def RMSE(self):
        return torch.sqrt(self.MSE())

    def MAPE(self):
        return torch.mean(torch.abs((self.true - self.predict) / (self.true + 1e-8)))

    def SMAPE(self):
        denominator = (torch.abs(self.true) + torch.abs(self.predict)) / 2.0
        denominator = torch.where(denominator == 0.0, torch.ones_like(denominator), denominator)
        return torch.mean(torch.abs(self.true - self.predict) / denominator) * 100.0

    def MDA(self, threshold=100.0):
        true_diff = self.true[1:] - self.true[:-1]
        pred_diff = self.predict[1:] - self.true[:-1]

        def get_tolerant_sign(diff, tol):
            sign = torch.zeros_like(diff)
            sign = torch.where(diff > tol, torch.ones_like(diff), sign)
            sign = torch.where(diff < -tol, -torch.ones_like(diff), sign)
            return sign

        true_sign = get_tolerant_sign(true_diff, threshold)
        pred_sign = get_tolerant_sign(pred_diff, threshold)
        return torch.mean((true_sign == pred_sign).float())

    def R2(self):
        ss_tot = torch.sum((self.true - torch.mean(self.true)) ** 2)
        ss_res = torch.sum((self.true - self.predict) ** 2)
        if ss_tot == 0:
            return torch.tensor(0.0)
        return 1.0 - ss_res / ss_tot
#GCDLOSS
import torch
def gcd_loss(P, T, eps=1e-7):

        # 计算边界框中心
        Cp = (P[:, :2] + P[:, 2:]) / 2  # P的中心
        Ct = (T[:, :2] + T[:, 2:]) / 2  # T的中心

        # 计算宽度和高度
        Wp, Hp = P[:, 2] - P[:, 0], P[:, 3] - P[:, 1]
        Wt, Ht = T[:, 2] - T[:, 0], T[:, 3] - T[:, 1]

        # 计算中心差异
        deltaC = Cp - Ct

        # 计算距离 D1 和 D2
        D1 = torch.sqrt((deltaC[:, 0] / (Wp + eps)) ** 2 + (deltaC[:, 1] / (Hp + eps)) ** 2)
        D2 = torch.sqrt((deltaC[:, 0] / (Wt + eps)) ** 2 + (deltaC[:, 1] / (Ht + eps)) ** 2)

        # 计算宽度和高度差异
        Dw = torch.sqrt(((Wp - Wt) / (Wp + eps)) ** 2 + ((Hp - Ht) / (Hp + eps)) ** 2)

        # 计算 GCD 损失度量
        GCD2 = (D1 + D2 + Dw) / 2

        # 计算最终的 GCD 损失
        Metric = 1 - torch.exp(-torch.sqrt(GCD2))
        return Metric


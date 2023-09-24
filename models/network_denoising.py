from typing import Any, List, Tuple
import torch
from torch import Tensor
import torch.nn as nn
from torch import autograd
import models.basicblock as B

import torch.nn.functional as F
import numpy as np
from math import ceil
from .utils import *
from torch.nn.functional import conv2d as Conv2d


class HeadNet(nn.Module):
    def __init__(self, in_nc: int, nc_x: List[int], out_nc: int, d_size: int):
        super(HeadNet, self).__init__()
        self.head_x = nn.Sequential(
            nn.Conv2d(in_nc + 1,
                      nc_x[0],
                      d_size,
                      padding=(d_size - 1) // 2,
                      bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(nc_x[0], nc_x[0], 3, padding=1, bias=False))

        self.head_d = torch.zeros(1, out_nc, nc_x[0], d_size, d_size)

    def forward(self, y: Any, sigma: Any) -> Tuple[Tensor, Tensor]:
        sigma = sigma.repeat(1, 1, y.size(2), y.size(3))
        x = self.head_x(torch.cat([y, sigma], dim=1))
        d = self.head_d.repeat(y.size(0), 1, 1, 1, 1).to(y.device)
        return x, d

class HeadNetM(nn.Module):
    def __init__(self, in_nc: int, nc_x: List[int], out_nc: int, d_size: int):
        super(HeadNetM, self).__init__()
        self.head_x = nn.Sequential(
            nn.Conv2d(in_nc,
                      nc_x[0],
                      1,
                      padding=0,
                      bias=False),
            nn.ReLU(inplace=True),
            )

        self.head_d = torch.zeros(1, out_nc, nc_x[0], d_size, d_size)

    def forward(self, y: Any) -> Tuple[Tensor, Tensor]:

        x = self.head_x(y)
        d = self.head_d.repeat(y.size(0), 1, 1, 1, 1).to(y.device)
        return x, d

# 主干结构
class BodyNet(nn.Module):
    def   __init__(self, in_nc: int, nc_x: List[int], nc_d: List[int],
                 out_nc: int, nb: int):
        super(BodyNet, self).__init__()

        self.net_x = NetX(in_nc=in_nc, nc_x=nc_x, nb=nb)
        self.solve_fft = SolveFFT()
        self.net_d = NetD(nc_d=nc_d, out_nc=out_nc)
        self.solve_ls = SolveLS()


    def forward(self, x: Tensor, d: Tensor, y: Tensor, Y: Tensor,
                alpha_x: Tensor, beta_x: Tensor, alpha_d: float, beta_d: float,
                reg: float):
        """
            x: N, C_in, H, W
            d: N, C_out, C_in, d_size, d_size
            Y: N, C_out, 1, H, W, 2
            y: N, C_out, H, W
            alpha/beta: 1, 1, 1, 1
            reg: float
        """

        # Solve X
        X, D = self.rfft_xd(x, d)
        size_x = np.array(list(x.shape[-2:]))
        x = self.solve_fft(X, D, Y, alpha_x, size_x)
        # beta_x.size torch.Size([4, 1, 1, 1])
        beta_x = (1 / beta_x.sqrt()).repeat(1, 1, x.size(2), x.size(3))
        # beta_x.size() torch.Size([4, 65, 128, 128])
        x = self.net_x(torch.cat([x, beta_x], dim=1))

        # Solve D
        if self.net_d is not None:
            d = self.solve_ls(x.unsqueeze(1), d, y.unsqueeze(2), alpha_d, reg)


            beda_d = (1 / beta_d.sqrt()).repeat(1, 1, d.size(3), d.size(4))
            size_d = [d.size(1), d.size(2)]
            d = d.view(d.size(0), d.size(1) * d.size(2), d.size(3), d.size(4))

            d = self.net_d(torch.cat([d, beda_d], dim=1))  #

            d = d.view(d.size(0), size_d[0], size_d[1], d.size(2), d.size(3))

        return x, d

    def rfft_xd(self, x: Tensor, d: Tensor):
        X = torch.rfft(x, 2).unsqueeze(1)
        D = p2o(d, x.shape[-2:])

        return X, D


class SoftThreshold(nn.Module):
    def __init__(self, size, init_threshold=1e-3):
        super(SoftThreshold, self).__init__()
        self.threshold = nn.Parameter(init_threshold * torch.ones(1,size,1,1))

    def forward(self, x):
        mask1 = (x > self.threshold).float()
        mask2 = (x < -self.threshold).float()
        out = mask1.float() * (x - self.threshold)
        out += mask2.float() * (x + self.threshold)
        return out


class BodyNet_m(nn.Module):
    def __init__(self, nums_size, init_threshold):
        super(BodyNet_m, self).__init__()


        self.solve_fft = SolveFFT()

        self.solve_ls = SolveLS_G()

        self.soft_thrink = SoftThreshold(nums_size, init_threshold)

    def forward(self, x: Tensor, d: Tensor, y: Tensor, Y: Tensor,
                alpha_x: Tensor, alpha_d: float, reg: float):
        """
            x: N, C_in, H, W
            d: N, C_out, C_in, d_size, d_size
            Y: N, C_out, 1, H, W, 2
            y: N, C_out, H, W
            alpha/beta: 1, 1, 1, 1
            reg: float
        """
        # Solve X
        X, D = self.rfft_xd(x, d)
        size_x = np.array(list(x.shape[-2:]))

        x = self.solve_fft(X, D, Y, alpha_x, size_x)  #求解x第一步

        x = self.soft_thrink(x)
        # Solve D

        d = self.solve_ls(x.unsqueeze(1), d, y.unsqueeze(2), alpha_d, reg)  #求解d的第一步

        return x, d

    def rfft_xd(self, x: Tensor, d: Tensor):
        X = torch.rfft(x, 2).unsqueeze(1)
        D = p2o(d, x.shape[-2:])

        return X, D

#  least square solvers
class SolveLS_G(nn.Module):
    def __init__(self):
        super(SolveLS_G, self).__init__()

        self.cholesky_solve = CholeskySolve.apply

    def forward(self, x, d, y, alpha, reg):
        """
            x: N, 1, C_in, H, W
            d: N, C_out, C_in, d_size, d_size
            y: N, C_out, 1, H, W
            alpha: N, 1, 1, 1
            reg: float
        """


        C_in = x.shape[2]
        d_size = d.shape[-1]

        xtx_raw = self.cal_xtx(x, d_size)  # N, C_in, C_in, d_size, d_size #

        xtx_unfold = F.unfold(
            xtx_raw.view(
                xtx_raw.size(0) * xtx_raw.size(1), xtx_raw.size(2),
                xtx_raw.size(3), xtx_raw.size(4)), d_size)



        xtx_unfold = xtx_unfold.view(xtx_raw.size(0), xtx_raw.size(1),
                                     xtx_unfold.size(1), xtx_unfold.size(2))



        xtx = xtx_unfold.view(xtx_unfold.size(0), xtx_unfold.size(1),
                              xtx_unfold.size(1), -1, xtx_unfold.size(3))



        xtx.copy_(xtx[:, :, :, torch.arange(xtx.size(3) - 1, -1, -1), ...])
        xtx = xtx.view(xtx.size(0), -1, xtx.size(-1))  # TODO



        index = torch.arange(
            (C_in * d_size)**2).view(C_in, C_in, d_size,
                                     d_size).permute(0, 2, 3, 1).reshape(-1)
        xtx.copy_(xtx[:, index, :])  # TODO
        xtx = xtx.view(xtx.size(0), d_size**2 * C_in, -1)




        xty = self.cal_xty(x, y, d_size)
        xty = xty.reshape(xty.size(0), xty.size(1), -1).permute(0, 2, 1)

        # reg

        xtx[:, range(len(xtx[0])), range(len(
            xtx[0]))] = xtx[:, range(len(xtx[0])),
                            range(len(xtx[0]))] # + alpha.squeeze(-1).squeeze(-1)    # XTX+alpha*I



        # solve
        try:
            d = self.cholesky_solve(xtx, xty).view(d.size(0), C_in, d_size,
                                                   d_size, d.size(1)).permute(
                                                       0, 4, 1, 2, 3)
        except RuntimeError:
            pass

        return d

    def cal_xtx(self, x, d_size):
        """
            x: N, 1, C_in, H, W
            d_size: kernel (d) size
        """
        padding = d_size - 1
        xtx = conv3d(x,
                     x.view(x.size(0), x.size(2), 1, 1, x.size(3), x.size(4)),
                     padding,
                     sample_wise=True)

        return xtx

    def cal_xty(self, x, y, d_size):
        """
            x: N, 1, C_in, H, W
            d_size: kernel (d) size
            y: N, C_out, 1, H, W
        """
        padding = (d_size - 1) // 2

        xty = conv3d(x, y.unsqueeze(3), padding, sample_wise=True)
        return xty



class NetX(nn.Module):
    def __init__(self,
                 in_nc: int = 65,
                 nc_x: List[int] = [64, 128, 256, 512],
                 nb: int = 4):
        super(NetX, self).__init__()

        self.m_down1 = B.sequential(
            *[
                B.ResBlock(in_nc, in_nc, bias=False, mode='CRC')
                for _ in range(nb)
            ], B.downsample_strideconv(in_nc, nc_x[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(
            *[
                B.ResBlock(nc_x[1], nc_x[1], bias=False, mode='CRC')
                for _ in range(nb)
            ], B.downsample_strideconv(nc_x[1], nc_x[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(
            *[
                B.ResBlock(nc_x[2], nc_x[2], bias=False, mode='CRC')
                for _ in range(nb)
            ], B.downsample_strideconv(nc_x[2], nc_x[3], bias=False, mode='2'))

        self.m_body = B.sequential(*[
            B.ResBlock(nc_x[-1], nc_x[-1], bias=False, mode='CRC')
            for _ in range(nb)
        ])

        self.m_up3 = B.sequential(
            B.upsample_convtranspose(nc_x[3], nc_x[2], bias=False, mode='2'),
            *[
                B.ResBlock(nc_x[2], nc_x[2], bias=False, mode='CRC')
                for _ in range(nb)
            ])
        self.m_up2 = B.sequential(
            B.upsample_convtranspose(nc_x[2], nc_x[1], bias=False, mode='2'),
            *[
                B.ResBlock(nc_x[1], nc_x[1], bias=False, mode='CRC')
                for _ in range(nb)
            ])
        self.m_up1 = B.sequential(
            B.upsample_convtranspose(nc_x[1], nc_x[0], bias=False, mode='2'),
            *[
                B.ResBlock(nc_x[0], nc_x[0], bias=False, mode='CRC')
                for _ in range(nb)
            ])

        self.m_tail = B.conv(nc_x[0], nc_x[0], bias=False, mode='C')

    def forward(self, x):
        x1 = x
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1[:, :-1, :, :])
        return x


class SolveFFT(nn.Module):
    def __init__(self):
        super(SolveFFT, self).__init__()

    def forward(self, X: Tensor, D: Tensor, Y: Tensor, alpha: Tensor,
                x_size: np.ndarray):
        """
            X: N, 1, C_in, H, W, 2
            D: N, C_out, C_in, H, W, 2
            Y: N, C_out, 1, H, W, 2
            alpha: N, 1, 1, 1
        """
        alpha = alpha.unsqueeze(-1).unsqueeze(-1) / X.size(2)

        _D = cconj(D)
        Z = cmul(Y, D) + alpha * X

        factor1 = Z / alpha

        numerator = cmul(_D, Z).sum(2, keepdim=True)
        denominator = csum(alpha * cmul(_D, D).sum(2, keepdim=True),
                           alpha.squeeze(-1)**2)
        factor2 = cmul(D, cdiv(numerator, denominator))
        X = (factor1 - factor2).mean(1)

        return torch.irfft(X, 2, signal_sizes=list(x_size))


class NetD(nn.Module):
    def __init__(self, nc_d: List[int] = [16], out_nc: int = 1):
        super(NetD, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(out_nc * nc_d[0] + 1, out_nc * nc_d[0], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_nc * nc_d[0], out_nc * nc_d[0], 3, padding=1))
        self.mlp2 = nn.Sequential(
            nn.Conv2d(out_nc * nc_d[0], out_nc * nc_d[0], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_nc * nc_d[0], out_nc * nc_d[0], 3, padding=1))
        self.mlp3 = nn.Sequential(
            nn.Conv2d(out_nc * nc_d[0], out_nc * nc_d[0], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_nc * nc_d[0], out_nc * nc_d[0], 3, padding=1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = x
        x = self.relu(self.mlp(x))
        x = self.relu(self.mlp2(x))
        x = self.mlp3(x) + x1[:, :-1, :, :]
        return x


class CholeskySolve(autograd.Function):
    @staticmethod
    def forward(ctx, Q, P):
        L = torch.cholesky(Q)
        D = torch.cholesky_solve(P, L)  # D = Q-1 @ P
        ctx.save_for_backward(L, D)
        return D

    @staticmethod
    def backward(ctx, dLdD):
        L, D = ctx.saved_tensors
        dLdP = torch.cholesky_solve(dLdD, L)
        dLdQ = -dLdP.matmul(D.transpose(-2, -1))

        return dLdQ, dLdP

#  least square solvers
class SolveLS(nn.Module):
    def __init__(self):
        super(SolveLS, self).__init__()

        self.cholesky_solve = CholeskySolve.apply

    def forward(self, x, d, y, alpha, reg):
        """
            x: N, 1, C_in, H, W
            d: N, C_out, C_in, d_size, d_size
            y: N, C_out, 1, H, W
            alpha: N, 1, 1, 1
            reg: float
        """
        C_in = x.shape[2]
        d_size = d.shape[-1]

        xtx_raw = self.cal_xtx(x, d_size)  # N, C_in, C_in, d_size, d_size
        xtx_unfold = F.unfold(
            xtx_raw.view(
                xtx_raw.size(0) * xtx_raw.size(1), xtx_raw.size(2),
                xtx_raw.size(3), xtx_raw.size(4)), d_size)
        xtx_unfold = xtx_unfold.view(xtx_raw.size(0), xtx_raw.size(1),
                                     xtx_unfold.size(1), xtx_unfold.size(2))

        xtx = xtx_unfold.view(xtx_unfold.size(0), xtx_unfold.size(1),
                              xtx_unfold.size(1), -1, xtx_unfold.size(3))
        xtx.copy_(xtx[:, :, :, torch.arange(xtx.size(3) - 1, -1, -1), ...])
        xtx = xtx.view(xtx.size(0), -1, xtx.size(-1))  # TODO
        index = torch.arange(
            (C_in * d_size)**2).view(C_in, C_in, d_size,
                                     d_size).permute(0, 2, 3, 1).reshape(-1)
        xtx.copy_(xtx[:, index, :])  # TODO
        xtx = xtx.view(xtx.size(0), d_size**2 * C_in, -1)

        xty = self.cal_xty(x, y, d_size)
        xty = xty.reshape(xty.size(0), xty.size(1), -1).permute(0, 2, 1)

        # reg
        alpha = alpha * x.size(3) * x.size(4) * reg / (d_size**2 * d.size(2))
        xtx[:, range(len(xtx[0])), range(len(
            xtx[0]))] = xtx[:, range(len(xtx[0])),
                            range(len(xtx[0]))] + alpha.squeeze(-1).squeeze(-1)


        xty += alpha.squeeze(-1) * d.reshape(d.size(0), d.size(1), -1).permute(
            0, 2, 1)

        # solve
        try:
            d = self.cholesky_solve(xtx, xty).view(d.size(0), C_in, d_size,
                                                   d_size, d.size(1)).permute(
                                                       0, 4, 1, 2, 3)
        except RuntimeError:
            pass

        return d

    def cal_xtx(self, x, d_size):
        """
            x: N, 1, C_in, H, W
            d_size: kernel (d) size
        """
        padding = d_size - 1
        xtx = conv3d(x,
                     x.view(x.size(0), x.size(2), 1, 1, x.size(3), x.size(4)),
                     padding,
                     sample_wise=True)

        return xtx

    def cal_xty(self, x, y, d_size):
        """
            x: N, 1, C_in, H, W
            d_size: kernel (d) size
            y: N, C_out, 1, H, W
        """
        padding = (d_size - 1) // 2

        xty = conv3d(x, y.unsqueeze(3), padding, sample_wise=True)
        return xty


class TailNet(nn.Module):
    def __init__(self):
        super(TailNet, self).__init__()

    def forward(self, x, d,sample_wise):

        y = conv2d(F.pad(x, [
            (d.size(-1) - 1) // 2,
        ] * 4, mode='circular'),
                   d,
                   sample_wise=sample_wise)

        return y

class HyPaNet(nn.Module):
    def __init__(
            self,
            in_nc: int = 1,
            nc: int = 256,
            out_nc: int = 8,
    ):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_nc, nc, 1, padding=0, bias=True), nn.Sigmoid(),
            nn.Conv2d(nc, out_nc, 1, padding=0, bias=True), nn.Softplus())

    def forward(self, x: Tensor):
        x = (x - 0.098) / 0.0566
        x = self.mlp(x) + 1e-6
        return x




class xNon_local(nn.Module):
    def __init__(self, inchannel):
        super(xNon_local, self).__init__()

        self.inchannel = inchannel

        self.quary_conv = nn.Conv2d(in_channels=self.inchannel, out_channels=self.inchannel, kernel_size=1,
                                    )

        self.key_conv = nn.Conv2d(in_channels=self.inchannel, out_channels=self.inchannel, kernel_size=1,
                                 )

        self.value_conv = nn.Conv2d(in_channels=self.inchannel, out_channels=self.inchannel, kernel_size=1,
                                    )

        self.sigma_conv = nn.Conv2d(in_channels=self.inchannel, out_channels=self.inchannel, kernel_size=1,
                                    )

        self.out_conv = nn.Conv2d(in_channels=self.inchannel, out_channels=self.inchannel, kernel_size=1,
                                 )

        self.cat_conv = nn.Conv2d(in_channels=self.inchannel, out_channels=self.inchannel, kernel_size=1)

        self.conv_x = nn.Conv2d(in_channels= self.inchannel, out_channels= self.inchannel//2 ,kernel_size=1)
        self.conv_m = nn.Conv2d(in_channels= self.inchannel, out_channels= self.inchannel//2 ,kernel_size=1)

        self.w1 = nn.Parameter(torch.rand(1), requires_grad=True)
    def forward(self, x, m):
        n = x.size(0)

        m1 = self.conv_m(m)
        x1 = self.conv_x(x)
        xmcat = self.cat_conv(torch.cat([m1, x1], dim=1))

        value_x = self.value_conv(xmcat).view(n, self.inchannel, -1)

        quary_x = self.quary_conv(xmcat).view(n, self.inchannel, -1)
        quary_x = quary_x.permute(0, 2, 1) # hw c

        sigma_x = self.sigma_conv(xmcat).view(n, self.inchannel, -1)
        sigma_x = sigma_x.permute(0, 2, 1) # hw c

        key_x = self.key_conv(xmcat).view(n, self.inchannel, -1) #c hw

        # computer similar

        pairwisr_weight = torch.matmul(key_x, self.w1 * quary_x + (1 - self.w1) * sigma_x) # 多核的计算相似性

        pairwisr_weight = pairwisr_weight.softmax(dim=-1)

        out = torch.matmul(pairwisr_weight, value_x)

        out = out.reshape(n, self.inchannel, x.size(2), x.size(3))

        out = self.out_conv(out)

        output = out + x


        return output



class DCDicL(nn.Module):
    def __init__(self,
                 n_iter: int = 1,
                 in_nc: int = 1,
                 nc_x: List[int] = [64, 128, 256, 512],
                 out_nc: int = 1,
                 nb: int = 1,
                 d_size: int = 5,
                 **kargs):
        super(DCDicL, self).__init__()

        self.head = HeadNet(in_nc, nc_x, out_nc, d_size)
        self.headm = HeadNetM(in_nc, nc_x, out_nc, d_size)

        self.body = BodyNet(in_nc=nc_x[0] + 1,
                            nc_x=nc_x,
                            nc_d=nc_x,
                            out_nc=out_nc,
                            nb=nb)


        self.body_m = BodyNet_m(nums_size=64,init_threshold=1e-3)

        self.xnlb = xNon_local(inchannel=64)

        self.tail = TailNet()

        self.hypa_list: nn.ModuleList = nn.ModuleList()
        for _ in range(n_iter):
            self.hypa_list.append(HyPaNet(in_nc=1, out_nc=4))

        self.n_iter = n_iter

    def forward(self, y, sigma_y, m):
        # padding
        h_y, w_y = y.size()[-2:]
        paddingBottom = int(ceil(h_y / 8) * 8 - h_y)
        paddingRight = int(ceil(w_y / 8) * 8 - w_y)
        y = F.pad(y, [0, paddingRight, 0, paddingBottom], mode='circular')

        # prepare Y
        Y = torch.rfft(y, 2)
        Y = Y.unsqueeze(2)

        # head_net d=0
        y_x, y_d = self.head(y, sigma_y)

        h_m, w_m = m.size()[-2:]
        paddingBottom = int(ceil(h_m / 8) * 8 - h_m)
        paddingRight = int(ceil(w_m / 8) * 8 - w_m)
        m = F.pad(m, [0, paddingRight, 0, paddingBottom], mode='circular')

        # prepare Y
        M = torch.rfft(m, 2)
        M = M.unsqueeze(2)

        # head_net d=0
        sigma_m = torch.zeros(sigma_y.size()).cuda()
        m_x, m_d = self.head(m, sigma_m)


        alpha_m_x = torch.tensor([0.7]).cuda()
        alpha_m_d = torch.tensor([0.8]).cuda()


        pred = None
        preds = []

        for i in range(self.n_iter):
            hypas_y = self.hypa_list[i](sigma_y)
            alpha_y_x = hypas_y[:, 0].unsqueeze(-1)
            beta_y_x = hypas_y[:, 1].unsqueeze(-1)
            alpha_y_d = hypas_y[:, 2].unsqueeze(-1)
            beta_y_d = hypas_y[:, 3].unsqueeze(-1)



            y_x, y_d = self.body(y_x, y_d, y, Y, alpha_y_x, beta_y_x, alpha_y_d, beta_y_d, 0.001)


            m_x, m_d = self.body_m(m_x, m_d, m, M, alpha_m_x, alpha_m_d,
                                   0.001)


            y_x = self.xnlb(y_x, m_x)


            dx = self.tail(y_x, y_d, True)
            dx = dx[..., :h_y, :w_y]
            pred = dx
            preds.append(pred)



        if self.training:
            return preds, y_d
        else:
            return pred, y_d

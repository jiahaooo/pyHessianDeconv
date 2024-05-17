"""
You, Y-L., and Mostafa Kaveh. "Fourth-order partial differential equations for noise removal." IEEE Transactions on Image Processing 9.10 (2000): 1723-1730.
Goldstein, Tom, and Stanley Osher. "The split Bregman method for L1-regularized problems." SIAM journal on imaging sciences 2.2 (2009): 323-343.
"""

import torch

from math import sqrt

from common import shrink, fft3d, ifft3d, forward_diff, back_diff, conv, psf2otf

device = torch.device('cuda')


def hessian(g, otf=None, psf=None, para_fidelity=400.0, para_relaxation=1.0, para_smooth_space=1.0, para_smooth_timepoint=2.0, iter_Bregman=100, tol=1e-12):
    # ----------------------------------------
    # data pre process
    # ----------------------------------------
    if para_relaxation <= 0 or (para_smooth_space < 0 and para_smooth_timepoint < 0):
        return g

    g = g.float().to(device)
    gshape = g.shape
    g = g.squeeze()

    if len(g.shape) == 2:
        g = g.unsqueeze(0)
        para_smooth_timepoint = 0
    assert len(g.shape) == 3, "input image doesnot have two or three vaild dims"
    if otf is None and psf is not None:
        if not isinstance(psf, torch.Tensor): psf = torch.from_numpy(psf)
        assert len(psf.shape) == 2
        psf = psf.float()
        psf /= psf.sum()
        otf = psf2otf(psf.unsqueeze(0).unsqueeze(0), g.shape[-2:]).to(device).squeeze()
    else:
        otf = torch.ones_like(g)

    mean_g = torch.mean(g, dim=[1, 2], keepdim=True)
    if mean_g.min() > 1e-8:
        mean_g = mean_g / g.mean()
    else:
        mean_g = torch.ones_like(mean_g)
    g /= mean_g

    gmax = torch.max(g)
    g /= gmax
    g[g < 0] = 0

    # ----------------------------------------
    # initialize
    # ----------------------------------------
    denominator = 0
    b_tt = 0
    b_th = 0
    b_tw = 0
    if para_smooth_timepoint > 0:
        filter_tt = torch.Tensor([1, -2, 1]).reshape(3, 1, 1).to(device)
        filter_th = torch.Tensor([[1, -1], [-1, 1]]).reshape(2, 2, 1).to(device)
        filter_tw = torch.Tensor([[1, -1], [-1, 1]]).reshape(2, 1, 2).to(device)
        denominator += para_smooth_timepoint * torch.abs(fft3d(filter_tt, s=g.shape)) ** 2
        denominator += 2 * sqrt(para_smooth_timepoint * para_smooth_space) * torch.abs(fft3d(filter_th, s=g.shape)) ** 2
        denominator += 2 * sqrt(para_smooth_timepoint * para_smooth_space) * torch.abs(fft3d(filter_tw, s=g.shape)) ** 2
        b_tt = torch.zeros_like(g)
        b_th = torch.zeros_like(g)
        b_tw = torch.zeros_like(g)

    filter_hh = torch.Tensor([1, -2, 1]).reshape(1, 3, 1).to(device)
    filter_ww = torch.Tensor([1, -2, 1]).reshape(1, 1, 3).to(device)
    filter_hw = torch.Tensor([[1, -1], [-1, 1]]).reshape(1, 2, 2).to(device)
    denominator += para_smooth_space * torch.abs(fft3d(filter_hh, s=g.shape)) ** 2
    denominator += para_smooth_space * torch.abs(fft3d(filter_ww, s=g.shape)) ** 2
    denominator += 2 * para_smooth_space * torch.abs(fft3d(filter_hw, s=g.shape)) ** 2
    b_hh = torch.zeros_like(g)
    b_ww = torch.zeros_like(g)
    b_hw = torch.zeros_like(g)
    denominator += para_fidelity / para_relaxation * torch.abs(otf) ** 2

    hg = torch.real(ifft3d(fft3d(g) * torch.conj(otf)))

    f_last = torch.zeros_like(g)

    # ----------------------------------------
    # iteration
    # ----------------------------------------
    numerator = None
    f = None
    for idx in range(iter_Bregman):
        if idx == 0:
            f = g.clone()
        else:
            f = torch.real(ifft3d(fft3d(numerator) / denominator))

            if (f_last - f).sum() ** 2 / f.numel() < tol: break
            f_last = f.clone()

        numerator = para_fidelity / para_relaxation * hg

        if para_smooth_timepoint > 0:
            # u = conv(f, filter_tt)
            u = back_diff(forward_diff(f, 't'), 't')
            d = shrink(u + b_tt, 1 / para_relaxation)
            b_tt += u - d
            # numerator += para_smooth_timepoint * conv(d - b_tt, filter_tt, conj=True, back_pad=False)
            numerator += para_smooth_timepoint * back_diff(forward_diff(d - b_tt, 't'), 't')  # [1, -2, 1]^* = [1, -2 ,1]

        if para_smooth_timepoint * para_smooth_space > 0:
            # u = conv(f, filter_th)
            u = forward_diff(forward_diff(f, 't'), 'h')
            d = shrink(u + b_th, 1 / para_relaxation)
            b_th += u - d
            # numerator += 2 * sqrt(para_smooth_timepoint) * conv(d - b_th, filter_th, conj=True, back_pad=False)
            numerator += 2 * sqrt(para_smooth_space * para_smooth_timepoint) * back_diff(back_diff(d - b_th, 't'), 'h')

            # u = conv(f, filter_tw)
            u = forward_diff(forward_diff(f, 't'), 'w')
            d = shrink(u + b_tw, 1 / para_relaxation)
            b_tw += u - d
            # numerator += 2 * sqrt(para_smooth_timepoint) * conv(d - b_tw, filter_tw, conj=True, back_pad=False)
            numerator += 2 * sqrt(para_smooth_space * para_smooth_timepoint) * back_diff(back_diff(d - b_tw, 't'), 'w')

        if para_smooth_space > 0:
            # u = conv(f, filter_hh)
            u = back_diff(forward_diff(f, 'h'), 'h')
            d = shrink(u + b_hh, 1 / para_relaxation)
            b_hh += u - d
            # numerator += conv(d - b_hh, filter_hh, conj=True, back_pad=False)
            numerator += para_smooth_space * back_diff(forward_diff(d - b_hh, 'h'), 'h')

            # u = conv(f, filter_ww)
            u = back_diff(forward_diff(f, 'w'), 'w')
            d = shrink(u + b_ww, 1 / para_relaxation)
            b_ww += u - d
            # numerator += conv(d - b_ww, filter_ww, conj=True, back_pad=False)
            numerator += para_smooth_space * back_diff(forward_diff(d - b_ww, 'w'), 'w')

            # u = conv(f, filter_hw)
            u = forward_diff(forward_diff(f, 'h'), 'w')
            d = shrink(u + b_hw, 1 / para_relaxation)
            b_hw += u - d
            # numerator += 2 * conv(d - b_hw, filter_hw, conj=True, back_pad=False)
            numerator += 2 * para_smooth_space * back_diff(back_diff(d - b_hw, 'h'), 'w')

    # ----------------------------------------
    # data post process
    # ----------------------------------------
    f = torch.where(torch.isnan(f), torch.full_like(f, 0), f)
    f[f < 0] = 0
    f *= gmax
    f *= mean_g

    return f.reshape(gshape)

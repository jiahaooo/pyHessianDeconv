"""
Wang, Yilun, et al. "A new alternating minimization algorithm for total variation image reconstruction." SIAM Journal on Imaging Sciences 1.3 (2008): 248-272.
Goldstein, Tom, and Stanley Osher. "The split Bregman method for L1-regularized problems." SIAM journal on imaging sciences 2.2 (2009): 323-343.
"""

import numpy as np
import torch

from common import shrink, fft3d, ifft3d, forward_diff, back_diff, conv, psf2otf

device = torch.device('cuda')


def tv(g, otf=None, psf=None, para_fidelity=400.0, para_relaxation=1.0, para_smooth_space=1.0, para_smooth_timepoint=2.0, iter_Bregman=100, tol=1e-12):
    # ----------------------------------------
    # data pre process
    # ----------------------------------------
    assert para_smooth_space > 0 or para_smooth_timepoint > 0
    if psf is not None or otf is not None: assert para_smooth_space > 0

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

    # ----------------------------------------
    # initialize
    # ----------------------------------------
    denominator = 0
    filter_t = 0
    b_t = 0
    if para_smooth_timepoint > 0:
        filter_t = torch.Tensor([1, -1]).reshape(2, 1, 1).to(device)
        denominator += para_smooth_timepoint * torch.abs(fft3d(filter_t, s=g.shape)) ** 2
        b_t = torch.zeros_like(g)

    filter_h = torch.Tensor([1, -1]).reshape(1, 2, 1).to(device)
    filter_w = torch.Tensor([1, -1]).reshape(1, 1, 2).to(device)
    denominator += para_smooth_space * torch.abs(fft3d(filter_h, s=g.shape)) ** 2
    denominator += para_smooth_space * torch.abs(fft3d(filter_w, s=g.shape)) ** 2
    b_h = torch.zeros_like(g)
    b_w = torch.zeros_like(g)

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
            # u = conv(f, filter_t)
            u_t = back_diff(f, 't')
            d_t = shrink(u_t + b_t, 1 / para_relaxation)
            b_t += u_t - d_t
            # numerator += para_smooth_timepoint * conv(d_t - b_t, filter_t, conj=True, back_pad=False)
            numerator += - para_smooth_timepoint * forward_diff(d_t - b_t, 't')

        if para_smooth_space > 0:
            # u_h = conv(f, filter_h)
            u_h = back_diff(f, 'h')
            # u_w = conv(f, filter_w)
            u_w = back_diff(f, 'w')

            # d_h = shrink(u_h + b_h, 1 / para_relaxation)
            # d_w = shrink(u_w + b_w, 1 / para_relaxation)
            d_h, d_w = shrink([u_h + b_h, u_w + b_w], 1 / para_relaxation)

            b_h += u_h - d_h
            b_w += u_w - d_w

            # numerator += conv(d_h - b_h, filter_h, conj=True, back_pad=False)
            numerator += - forward_diff(d_h - b_h, 'h')

            # numerator += conv(d_w - b_w, filter_w, conj=True, back_pad=False)
            numerator += - forward_diff(d_w - b_w, 'w')

    # ----------------------------------------
    # data post process
    # ----------------------------------------
    f = torch.where(torch.isnan(f), torch.full_like(f, 0), f)
    f[f < 0] = 0
    f *= gmax
    f *= mean_g

    return f.reshape(gshape)
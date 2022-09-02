import numpy as np
import torch

from common import shrink, fft3d, ifft3d, forward_diff, back_diff, conv, psf2otf

device = torch.device('cuda')


def tikhonov(g, otf=None, psf=None, para_fidelity=400.0, para_smooth_space=1.0, para_smooth_timepoint=2.0, iter_Bregman=100, tol=1e-12):
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
    filter_laplace = torch.Tensor([[1, 1.5, 1], [1.5, -10, 1.5], [1, 1.5, 1]]).reshape(1, 3, 3).to(device)
    denominator = para_smooth_space * torch.abs(fft3d(filter_laplace, s=g.shape)) ** 2 + para_fidelity * torch.abs(otf) ** 2

    if para_smooth_timepoint > 0:
        filter_tt = torch.Tensor([1, -2, 1]).reshape(3, 1, 1).to(device)
        denominator += para_smooth_timepoint * torch.abs(fft3d(filter_tt, s=g.shape)) ** 2

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

        numerator = para_fidelity * hg


    # ----------------------------------------
    # data post process
    # ----------------------------------------
    f = torch.where(torch.isnan(f), torch.full_like(f, 0), f)
    f[f < 0] = 0
    f *= gmax
    f *= mean_g

    return f.reshape(gshape)
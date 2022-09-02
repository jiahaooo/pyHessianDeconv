import numpy as np
import torch
from torch.nn.functional import conv3d, conv2d, pad
from cv2 import getGaussianKernel

device = torch.device('cuda')


def fft3d(x, s=None):
    if s is not None:
        return torch.fft.fftn(x, s=s, dim=[-3, -2, -1])
    else:
        return torch.fft.fftn(x, dim=[-3, -2, -1])


def ifft3d(x, s=None):
    if s is not None:
        return torch.fft.ifftn(x, s=s, dim=[-3, -2, -1])
    else:
        return torch.fft.ifftn(x, dim=[-3, -2, -1])


def back_diff(data, dim):
    if len(data.shape) == 3:
        if dim == 't': return pad((data[1:, :, :] - data[:-1, :, :]).unsqueeze(0).unsqueeze(0), (0, 0, 0, 0, 1, 0)).squeeze(1).squeeze(0)
        if dim == 'h': return pad((data[:, 1:, :] - data[:, :-1, :]).unsqueeze(0).unsqueeze(0), (0, 0, 1, 0, 0, 0)).squeeze(1).squeeze(0)
        if dim == 'w': return pad((data[:, :, 1:] - data[:, :, :-1]).unsqueeze(0).unsqueeze(0), (1, 0, 0, 0, 0, 0)).squeeze(1).squeeze(0)
    elif len(data.shape) == 2:
        if dim == 'h': return pad((data[1:, :] - data[:-1, :]).unsqueeze(0).unsqueeze(0), (0, 0, 1, 0)).squeeze(1).squeeze(0)
        if dim == 'w': return pad((data[:, 1:] - data[:, :-1]).unsqueeze(0).unsqueeze(0), (1, 0, 0, 0)).squeeze(1).squeeze(0)
    else:
        raise NotImplementedError


def forward_diff(data, dim):
    if len(data.shape) == 3:
        if dim == 't': return pad((data[1:, :, :] - data[:-1, :, :]).unsqueeze(0).unsqueeze(0), (0, 0, 0, 0, 0, 1)).squeeze(1).squeeze(0)
        if dim == 'h': return pad((data[:, 1:, :] - data[:, :-1, :]).unsqueeze(0).unsqueeze(0), (0, 0, 0, 1, 0, 0)).squeeze(1).squeeze(0)
        if dim == 'w': return pad((data[:, :, 1:] - data[:, :, :-1]).unsqueeze(0).unsqueeze(0), (0, 1, 0, 0, 0, 0)).squeeze(1).squeeze(0)
    elif len(data.shape) == 2:
        if dim == 'h': return pad((data[1:, :] - data[:-1, :]).unsqueeze(0).unsqueeze(0), (0, 0, 0, 1)).squeeze(1).squeeze(0)
        if dim == 'w': return pad((data[:, 1:] - data[:, :-1]).unsqueeze(0).unsqueeze(0), (0, 1, 0, 0)).squeeze(1).squeeze(0)
    else:
        raise NotImplementedError


def shrink(x, gamma):
    if (isinstance(x, list) or isinstance(x, tuple)) and len(x) >= 2:
        qu = torch.abs(x[0]) ** 2
        for idx in range(1, len(x)):
            qu += torch.abs(x[idx]) ** 2
        norm_ = torch.sqrt(qu + 1e-30)
        temp = torch.nn.ReLU()(norm_ - gamma) / norm_
        return [x[idx] * temp for idx in range(len(x))]
    else:
        return torch.sign(x) * torch.nn.ReLU()(torch.abs(x) - gamma)


def conv(img, kernel, conj=False, back_pad=True):
    ndim = len(img.shape)
    if ndim == 2:
        [H, W] = kernel.shape
        H_, W_ = H - 1, W - 1
        if back_pad:
            img_ = pad(img.unsqueeze(0).unsqueeze(0), [W_ - W_ // 2, W_ // 2, H_ - H_ // 2, H_ // 2], mode='replicate')  # circular replicate
        else:
            img_ = pad(img.unsqueeze(0).unsqueeze(0), [W_ // 2, W_ - W_ // 2, H_ // 2, H_ - H_ // 2], mode='replicate')  # circular replicate
        if conj:
            kernel_ = kernel.unsqueeze(0).unsqueeze(0)
        else:
            kernel_ = torch.flip(kernel, dims=[0, 1]).unsqueeze(0).unsqueeze(0)
        return conv2d(img_, kernel_, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1).squeeze(0).squeeze(0)
    elif ndim == 3:
        [T, H, W] = kernel.shape
        T_, H_, W_ = T - 1, H - 1, W - 1
        if back_pad:
            img_ = pad(img.unsqueeze(0).unsqueeze(0), [W_ - W_ // 2, W_ // 2, H_ - H_ // 2, H_ // 2, T_ - T_ // 2, T_ // 2], mode='replicate')
        else:
            img_ = pad(img.unsqueeze(0).unsqueeze(0), [W_ // 2, W_ - W_ // 2, H_ // 2, H_ - H_ // 2, T_ // 2, T_ - T_ // 2], mode='replicate')
        if conj:
            kernel_ = kernel.unsqueeze(0).unsqueeze(0)
        else:
            kernel_ = torch.flip(kernel, dims=[0, 1, 2]).unsqueeze(0).unsqueeze(0)
        return conv3d(img_, kernel_, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1).squeeze(0).squeeze(0)
    else:
        raise NotImplementedError


def psf2otf(psf, shape):
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., :psf.shape[-2], :psf.shape[-1]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[-2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=-2 + axis)
    otf = torch.fft.fftn(otf, dim=(-2, -1))
    return otf


def make_gaussian_kernal(shape, sigma):
    [height, width] = list(shape)
    [height_sigma, width_sigma] = list(sigma)
    h = torch.from_numpy(getGaussianKernel(height, height_sigma))
    w = torch.from_numpy(getGaussianKernel(width, width_sigma))
    kernel = torch.mm(h, w.t()).float()
    kernel /= kernel.sum()
    return kernel
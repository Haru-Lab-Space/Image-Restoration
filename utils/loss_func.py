import torch
import torch.nn
from torchmetrics.functional import structural_similarity_index_measure as tmf_ssim
from torchmetrics.functional import peak_signal_noise_ratio as tmf_psnr


def mse(output, target):
    squared_diff = torch.square(output - target)
    mse = torch.mean(squared_diff)
    return mse.item()


def ssim(output, target):
    ssim_score = tmf_ssim(output, target)
    return ssim_score.item()


def L1(output, target):
    abs_diff = torch.abs(output - target)
    l1_loss = torch.mean(abs_diff)
    return l1_loss.item()


def psnr(output, target):
    psnr_score = tmf_psnr(output, target)
    return psnr_score.item()

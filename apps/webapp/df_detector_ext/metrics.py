import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image

def compute_psnr_ssim(orig, edited):
    psnr = peak_signal_noise_ratio(orig, edited, data_range=255)
    ssim = structural_similarity(orig, edited, channel_axis=2)
    return psnr, ssim

def save_diff_heatmap(orig, edited, save_path):
    diff = cv2.absdiff(orig, edited)
    heatmap = cv2.applyColorMap(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(edited, 0.6, heatmap, 0.4, 0)
    Image.fromarray(overlay).save(save_path)

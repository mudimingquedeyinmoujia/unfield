import torch
import pyredner
import h5py
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
from testfiles.rednertest.redner_03_test import model

# target: differentiable rendering for 3dmm fitting
target = pyredner.imread('../../datas/testdatas/target.png').to(pyredner.get_device())
imshow(torch.pow(target, 1.0 / 2.2).cpu())

cam_pos = torch.tensor([-0.2697, -5.7891, 373.9277], requires_grad=True)
cam_look_at = torch.tensor([-0.2697, -5.7891, 54.7918], requires_grad=True)
shape_coeffs = torch.zeros(199, device=pyredner.get_device(), requires_grad=True)
color_coeffs = torch.zeros(199, device=pyredner.get_device(), requires_grad=True)
ambient_color = torch.ones(3, device=pyredner.get_device(), requires_grad=True)
dir_light_intensity = torch.zeros(3, device=pyredner.get_device(), requires_grad=True)

# Use two different optimizers for different learning rates
optimizer = torch.optim.Adam([shape_coeffs, color_coeffs, ambient_color, dir_light_intensity], lr=0.1)
cam_optimizer = torch.optim.Adam([cam_pos, cam_look_at], lr=0.5)

imgs, losses = [], []
# Run 500 Adam iterations
num_iters = 500
for t in range(num_iters):
    optimizer.zero_grad()
    cam_optimizer.zero_grad()
    img = model(cam_pos, cam_look_at, shape_coeffs, color_coeffs, ambient_color, dir_light_intensity)
    # Compute the loss function. Here it is L2 plus a regularization term to avoid coefficients to be too far from zero.
    # Both img and target are in linear color space, so no gamma correction is needed.
    loss = (img - target).pow(2).mean()
    loss = loss + 0.0001 * shape_coeffs.pow(2).mean() + 0.001 * color_coeffs.pow(2).mean()
    loss.backward()
    optimizer.step()
    cam_optimizer.step()
    ambient_color.data.clamp_(0.0)
    dir_light_intensity.data.clamp_(0.0)
    # Plot the loss
    losses.append(loss.data.item())
    # Only store images every 10th iterations
    if t % 100 == 0:
        imgs.append(torch.pow(img.data, 1.0 / 2.2).cpu())  # Record the Gamma corrected image
        f, (ax_loss, ax_diff_img, ax_img) = plt.subplots(1, 3)
        ax_loss.plot(range(len(losses)), losses, label='loss')
        ax_loss.legend()
        ax_diff_img.imshow((img - target).pow(2).sum(dim=2).data.cpu())
        ax_img.imshow(torch.pow(img.data.cpu(), 1.0 / 2.2))
        plt.show()

import torch
import pyredner
import h5py
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np

# target: load the bfm model and fit it
with h5py.File(r'../../datas/testdatas/model2017-1_bfm_nomouth.h5', 'r') as hf:
    shape_mean = torch.tensor(hf['shape/model/mean'], device=pyredner.get_device())
    shape_basis = torch.tensor(hf['shape/model/pcaBasis'], device=pyredner.get_device())
    triangle_list = torch.tensor(hf['shape/representer/cells'], device=pyredner.get_device())
    color_mean = torch.tensor(hf['color/model/mean'], device=pyredner.get_device())
    color_basis = torch.tensor(hf['color/model/pcaBasis'], device=pyredner.get_device())

indices = triangle_list.permute(1, 0).contiguous()  # trans [3,ntri] to [ntri,3]


def model(cam_pos, cam_look_at, shape_coeffs, color_coeffs, ambient_color, dir_light_intensity):
    vertices = (shape_mean + shape_basis @ shape_coeffs).view(-1, 3)  # tensor [53149,3] 90,94,131 value  0,-4,55 mean
    normals = pyredner.compute_vertex_normal(vertices, indices)  # ver normal tensor
    colors = (color_mean + color_basis @ color_coeffs).view(-1, 3)  # color tensor 0-1
    # print(max(vertices[:, 0]), max(vertices[:, 1]), max(vertices[:, 2]))
    # print(torch.mean(vertices[:, 0]), torch.mean(vertices[:, 1]), torch.mean(vertices[:, 2]))
    m = pyredner.Material(use_vertex_color=True)  # an object
    obj = pyredner.Object(vertices=vertices, indices=indices, normals=normals, material=m, colors=colors)
    cam = pyredner.Camera(position=cam_pos,
                          look_at=cam_look_at,  # Center of the vertices
                          up=torch.tensor([0.0, 1.0, 0.0]),
                          fov=torch.tensor([45.0]),
                          resolution=(256, 256))
    scene = pyredner.Scene(camera=cam, objects=[obj])
    ambient_light = pyredner.AmbientLight(ambient_color)  # ambient
    dir_light = pyredner.DirectionalLight(torch.tensor([0.0, 0.0, -1.0]), dir_light_intensity)
    img = pyredner.render_deferred(scene=scene, lights=[ambient_light, dir_light])
    return img


if __name__ == '__main__':
    cam_pos = torch.tensor([-0.2697, -5.7891, 373.9277])
    cam_look_at = torch.tensor([-0.2697, -5.7891, 54.7918])
    img = model(cam_pos, cam_look_at, torch.rand(199, device=pyredner.get_device()),
                torch.rand(199, device=pyredner.get_device()), torch.ones(3), torch.zeros(3))
    imshow(torch.pow(img, 1.0 / 2.2).cpu())
    plt.show()

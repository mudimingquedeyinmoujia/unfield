import torch
import pyredner
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

# target: pose estimation
objects = pyredner.load_obj('../../datas/testdatas/teapot/teapot.obj', return_objects=True)
camera = pyredner.automatic_camera_placement(objects, resolution=(512, 512))

# Obtain the teapot vertices we want to apply the transformation on.
vertices = []
for obj in objects:
    vertices.append(obj.vertices.clone())

# Compute the center of the teapot
center = torch.mean(torch.cat(vertices), 0)  # 按照行进行求平均


def model(translation, euler_angles):
    # Get the rotation matrix from Euler angles
    rotation_matrix = pyredner.gen_rotate_matrix(euler_angles)
    # Shift the vertices to the center, apply rotation matrix,
    # shift back to the original space, then apply the translation.
    for obj, v in zip(objects, vertices):
        obj.vertices = (v - center) @ torch.t(rotation_matrix) + center + translation
    # Assemble the 3D scene.
    scene = pyredner.Scene(camera=camera, objects=objects)
    # Render the scene.
    img = pyredner.render_albedo(scene)
    return img


target_translation = torch.tensor([15.0, -4.0, 8.0], device=pyredner.get_device())
target_euler_angles = torch.tensor([0.3, 0.4, 0.2], device=pyredner.get_device())

target = model(target_translation, target_euler_angles).data
imshow(torch.pow(target, 1.0 / 2.2).cpu())
plt.show()
# Set requires_grad=True since we want to optimize them later
translation = torch.tensor([10.0, -10.0, 10.0], device=pyredner.get_device(), requires_grad=True)
euler_angles = torch.tensor([0.1, -0.1, 0.1], device=pyredner.get_device(), requires_grad=True)
init = model(translation, euler_angles)
# Visualize the initial guess
# imshow(torch.pow(init.data, 1.0/2.2).cpu()) # add .data to stop PyTorch from complaining
# Need to gamma compress the image for displaying.
# imshow(torch.pow(target, 1.0 / 2.2).cpu())

t_optimizer = torch.optim.Adam([translation], lr=0.5)
r_optimizer = torch.optim.Adam([euler_angles], lr=0.01)

# plt.figure()
imgs, losses = [], []
# Run 80 Adam iterations
num_iters = 80
for t in range(num_iters + 1):
    t_optimizer.zero_grad()
    r_optimizer.zero_grad()
    img = model(translation, euler_angles)
    # Compute the loss function. Here it is L2.
    # Both img and target are in linear color space, so no gamma correction is needed.
    loss = (img - target).pow(2).mean()  # target是真实值，所以使用了.data不让torch跟踪梯度
    loss.backward()
    t_optimizer.step()
    r_optimizer.step()
    # Plot the loss

    losses.append(loss.data.item())
    imgs.append(torch.pow(img.data, 1.0 / 2.2).cpu())  # Record the Gamma corrected image
    if t % 20 == 0:
        f, (ax_loss, ax_img) = plt.subplots(1, 2)
        ax_loss.plot(range(len(losses)), losses, label='loss')
        ax_loss.legend()
        ax_img.imshow((img - target).pow(2).sum(axis=2).data.cpu())
        plt.show()
        print(translation.data, euler_angles.data)
        print(vertices)
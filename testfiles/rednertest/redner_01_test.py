import pyredner
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# target: render an image
objects = pyredner.load_obj('../../datas/testdatas/teapot/teapot.obj', return_objects=True)
# 这个壶有两个组件，一个盖子，一个壶身

camera = pyredner.automatic_camera_placement(objects, resolution=(512, 512))

scene = pyredner.Scene(camera = camera, objects = objects)

img = pyredner.render_albedo(scene)
# Need to gamma compress the image for displaying.
imshow(torch.pow(img, 1.0/2.2).cpu())
# plt.imshow(img.cpu())
plt.show()
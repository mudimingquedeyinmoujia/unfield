import pyredner
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import json
import os
import numpy as np, trimesh
from facescape.toolkit.src.facescape_bm import facescape_bm
from skimage import io
import math
from idea_01.utils import *
from torch.utils.data import Dataset, DataLoader
from facescape.toolkit.src.renderer import render_cvcam
from facescape.toolkit.src.utility import show_img_arr


# render one object
def show_one():
    obj_path = '../datas/facescape/facescape_trainset_001_100/1/models_reg/2_smile.obj'
    objects = pyredner.load_obj(obj_path, return_objects=True)
    camera = pyredner.automatic_camera_placement(objects, resolution=(512, 512))
    print(camera.position)
    print(camera.up)
    print(camera.fov)
    print(camera.clip_near)
    print(camera.resolution)
    print(camera.viewport)
    print(camera.intrinsic_mat)
    print(camera.cam_to_world)
    print(camera.distortion_params)
    print(camera.camera_type)

    camera.position[2] = -camera.position[2]

    camera2 = pyredner.Camera(position=camera.position, look_at=camera.look_at, up=camera.up, fov=torch.Tensor([60]),
                              resolution=(256, 256))
    print(camera2.intrinsic_mat)
    camera2.fov = torch.Tensor([45])
    print(camera2.intrinsic_mat)
    print(camera2.clip_near)
    scene = pyredner.Scene(camera=camera2, objects=objects)

    img = pyredner.render_albedo(scene)
    # Need to gamma compress the image for displaying.
    imshow(torch.pow(img, 1.0 / 2.2).cpu())
    # plt.imshow(img.cpu())
    plt.show()


# test obj color
def show_color():
    obj_path = '../datas/facescape/facescape_trainset_001_100/3/models_reg/3_mouth_stretch.obj'
    objects = pyredner.load_obj(obj_path, return_objects=True)
    mip256 = objects[0].material.diffuse_reflectance.mipmap[4]
    imshow(torch.pow(mip256, 1.0 / 2.2).cpu())
    plt.show()

    return


def show_texture():
    tex_path = '../datas/testdatas/target.png'
    tex_img = io.imread(tex_path)
    texture = pyredner.Texture(texels=torch.Tensor(tex_img))
    return


# show object info such as mean vertices
# vertices:26317,faces:52261
def show_info():
    # obj_path = '../datas/facescape/facescape_trainset_001_100/1/models_reg/2_smile.obj'
    obj_path = '../datas/facescape/facescape_trainset_001_100/3/models_reg/3_mouth_stretch.obj'
    objects = pyredner.load_obj(obj_path, return_objects=True)
    print('objs length: ', len(objects))
    ver = objects[0].vertices
    inds = objects[0].indices
    print(len(ver), len(inds))
    print(torch.max(ver[:, 0]))
    print(torch.min(ver[:, 0]))
    print(torch.max(ver[:, 1]))
    print(torch.min(ver[:, 1]))
    print(torch.max(ver[:, 2]))
    print(torch.min(ver[:, 2]))
    print(torch.mean(ver, dim=0))


# render obj use offered cam param
def show_generated(data_dir, obj_path):
    data_name = data_dir.split('/')[-1]
    json_name = data_name + '_params.json'
    with open(os.path.join(data_dir, json_name), 'r') as f:
        params = json.load(f)
    img_nums = len(params) // 5
    for i in range(img_nums):
        position = params['%d_position' % i]
        look_at = params['%d_look_at' % i]
        up = params['%d_up' % i]
        fov = params['%d_fov' % i]
        resolution = params['%d_resolution' % i]
        camera = pyredner.Camera(position=torch.Tensor(position), look_at=look_at, up=up,
                                 fov=torch.Tensor([fov]))
        camera.resolution = resolution
        objects = pyredner.load_obj(obj_path, return_objects=True)
        scene = pyredner.Scene(camera=camera, objects=objects)
        img = pyredner.render_albedo(scene)
        imshow(torch.pow(img.cpu(), 1.0 / 2.2))
        plt.show()


def fill_colors(vertices, r=200, g=200, b=200):
    r, g, b = r / 255., g / 255., b / 255.
    colors = np.ones(vertices.shape)
    colors[:, 0] = colors[:, 0] * r
    colors[:, 1] = colors[:, 1] * g
    colors[:, 2] = colors[:, 2] * b
    return colors


# face index start from 1
def adapt_uv(vertices, uvs, vertices_ind, uv_ind):
    """
    given facescape obj, generate uv list about vertices
    :param vertices:
    :param uvs:
    :param vertices_ind:
    :param uv_ind:
    :return: [nver,2]
    """
    uv_list = [[] for i in range(vertices.shape[0])]
    ver_ind_fla = vertices_ind.flatten()
    uv_ind_fla = uv_ind.flatten()
    for i in range(len(ver_ind_fla)):
        ver_ind = ver_ind_fla[i] - 1
        adapt_uv_ind = uv_ind_fla[i] - 1
        if len(uv_list[ver_ind]) == 0:
            uv_list[ver_ind].append(uvs[adapt_uv_ind])

    uv_arr = np.array(uv_list).squeeze()
    return uv_arr


# g buffer show
def pre_obj_show():
    np.random.seed(1000)

    model = facescape_bm("../datas/facescape/facescape_bilinear_model_v1_6/facescape_bm_v1.6_847_50_52_id_front.npz")
    # vertices: 26278
    # texcoords: 26364
    # indices: 52183
    # indices uv: 52183

    # create random identity vector
    random_id_vec = np.random.normal(model.id_mean, np.sqrt(model.id_var))

    # create random expression vector
    exp_vec = np.zeros(52)
    exp_vec[0] = 1

    # generate and save full head mesh
    mesh_full = model.gen_full(random_id_vec, exp_vec)

    # redner导出obj的时候认为是索引从0开始，但是facescape的索引是从1开始
    full_face = pyredner.Object(vertices=torch.Tensor(mesh_full.vertices),
                                indices=torch.tensor(mesh_full.faces_v - 1, dtype=torch.int32),
                                uvs=torch.Tensor(mesh_full.texcoords),
                                uv_indices=torch.tensor(mesh_full.faces_vt - 1, dtype=torch.int32),
                                material=pyredner.Material(use_vertex_color=True),
                                colors=torch.Tensor(fill_colors(mesh_full.vertices, 104, 151, 187)))
    # ### render
    camera = pyredner.automatic_camera_placement([full_face], resolution=(512, 512))
    camera.position[2] = -camera.position[2]
    scene = pyredner.Scene(camera=camera, objects=[full_face])

    img = pyredner.render_g_buffer(scene=scene, channels=[pyredner.channels.position, pyredner.channels.shading_normal,
                                                          pyredner.channels.diffuse_reflectance])

    # pyredner.save_obj(full_face,'./demo_output/redner_save_init_sub.obj')

    # ### render g buffer
    pos = img[:, :, :3]
    normal = img[:, :, 3:6]
    albedo = img[:, :, 6:9]
    plt.figure()
    pos_vis = (pos - pos.min()) / (pos.max() - pos.min())
    imshow(pos_vis.cpu())
    plt.figure()
    normal_vis = (normal - normal.min()) / (normal.max() - normal.min())
    imshow(normal_vis.cpu())
    plt.figure()
    imshow(torch.pow(albedo, 1.0 / 2.2).cpu())
    plt.show()

    # ### normal rendering
    # img = pyredner.render_albedo(scene)
    # Need to gamma compress the image for displaying.
    # imshow(torch.pow(img, 1.0 / 2.2).cpu())
    # plt.imshow(img.cpu())
    # plt.show()


def get_pre_obj():
    """
    bilinear model random generate mesh without color for pre training
    :return: obj object in facescape(numpy) not redner
    """
    np.random.seed(1000)

    model = facescape_bm("../datas/facescape/facescape_bilinear_model_v1_6/facescape_bm_v1.6_847_50_52_id_front.npz")
    # vertices: 26278
    # texcoords: 26364
    # indices: 52183
    # indices uv: 52183

    random_id_vec = np.random.normal(model.id_mean, np.sqrt(model.id_var))

    exp_vec = np.zeros(52)
    exp_vec[0] = 1

    # generate and save full head mesh
    mesh_full = model.gen_full(random_id_vec, exp_vec)

    # redner导出obj的时候认为是索引从0开始，但是facescape的索引是从1开始
    full_face = pyredner.Object(vertices=torch.Tensor(mesh_full.vertices),
                                indices=torch.tensor(mesh_full.faces_v - 1, dtype=torch.int32),
                                uvs=torch.Tensor(mesh_full.texcoords),
                                uv_indices=torch.tensor(mesh_full.faces_vt - 1, dtype=torch.int32),
                                material=pyredner.Material(use_vertex_color=True),
                                colors=torch.Tensor(fill_colors(mesh_full.vertices, 104, 151, 187)))

    return mesh_full


def get_pre_tex1(resolution=256):
    """
    read texture directly, don't need pow
    :param wanted resolution
    :return: reso img (numpy), max reso img (numpy)
    """
    tex_path = '../datas/facescape/facescape_trainset_001_100/2/models_reg/2_smile.jpg'  # 4096
    tex_img = io.imread(tex_path) / 255.
    max_reso = len(tex_img)
    temp_reso = resolution
    ind = int(math.log(max_reso / temp_reso, 2))
    texture = pyredner.Texture(texels=torch.Tensor(tex_img))

    return texture.mipmap[ind].numpy(), texture.mipmap[0].numpy()


def get_pre_tex2(resolution=256):
    """
    read obj and then give texture, need pow
    :param resolution:
    :return:
    """
    obj_path = '../datas/facescape/facescape_trainset_001_100/2/models_reg/2_smile.obj'
    obj = pyredner.load_obj(obj_path, return_objects=True)[0]
    max_reso = 4096
    temp_reso = resolution
    ind = int(math.log(max_reso / temp_reso, 2))
    texture = obj.material.diffuse_reflectance
    return texture.mipmap[ind].cpu().numpy(), texture.mipmap[0].cpu().numpy()


def render_mesh(vertices, uvs, vertices_ind, uv_ind):
    """
    preview mesh without color by generating g buffer
    :param vertices:
    :param uvs:
    :param vertices_ind: start from 1
    :param uv_ind: start from 1
    :return:
    """
    vertices = vertices.cpu()
    uvs = uvs.cpu()
    obj = pyredner.Object(vertices=torch.Tensor(vertices),
                          indices=torch.tensor(vertices_ind - 1, dtype=torch.int32),
                          uvs=torch.Tensor(uvs),
                          uv_indices=torch.tensor(uv_ind - 1, dtype=torch.int32),
                          material=pyredner.Material(use_vertex_color=True),
                          colors=torch.Tensor(fill_colors(vertices, 104, 151, 187)))

    camera = pyredner.automatic_camera_placement([obj], resolution=(512, 512))
    camera.position[2] = -camera.position[2]
    scene = pyredner.Scene(camera=camera, objects=[obj])

    img = pyredner.render_g_buffer(scene=scene, channels=[pyredner.channels.position, pyredner.channels.shading_normal,
                                                          pyredner.channels.diffuse_reflectance])

    # pyredner.save_obj(full_face,'./demo_output/redner_save_init_sub.obj')

    # ### render g buffer
    pos = img[:, :, :3]
    normal = img[:, :, 3:6]
    albedo = img[:, :, 6:9]
    plt.figure()
    pos_vis = (pos - pos.min()) / (pos.max() - pos.min())
    imshow(pos_vis.cpu())
    plt.figure()
    normal_vis = (normal - normal.min()) / (normal.max() - normal.min())
    imshow(normal_vis.cpu())
    plt.figure()
    imshow(torch.pow(albedo, 1.0 / 2.2).cpu())
    plt.show()


def render_obj(vertices, uvs, triangles, colors, h=512, w=512):
    vertices = vertices.cpu()
    uvs = uvs.cpu()
    colors = colors.cpu()
    obj = pyredner.Object(vertices=torch.Tensor(vertices),
                          indices=torch.tensor(triangles - 1, dtype=torch.int32),
                          uvs=torch.Tensor(uvs),
                          uv_indices=torch.tensor(triangles - 1, dtype=torch.int32),
                          material=pyredner.Material(use_vertex_color=True),
                          colors=torch.Tensor(colors))
    camera = pyredner.automatic_camera_placement([obj], resolution=(h, w))
    camera.position[2] = -camera.position[2]
    scene = pyredner.Scene(camera=camera, objects=[obj])
    rend_img = pyredner.render_albedo(scene)
    # Need to gamma compress the image for displaying.
    imshow(torch.pow(rend_img, 1.0 / 2.2).cpu())
    # plt.imshow(rend_img.cpu())
    plt.show()


def render_obj_tex(vertices, uvs, triangles, colors, h=512, w=512):
    vertices = vertices.cpu()
    uvs = uvs.cpu()
    colors = colors.cpu()
    # here must flip the texture because of redner's implement
    tex_img = render_texture(uvs.numpy(), triangles, colors.numpy(), h, w)
    tex_img_f = np.flipud(tex_img).copy()
    obj = pyredner.Object(vertices=torch.Tensor(vertices),
                          indices=torch.tensor(triangles - 1, dtype=torch.int32),
                          uvs=torch.Tensor(uvs),
                          uv_indices=torch.tensor(triangles - 1, dtype=torch.int32),
                          material=pyredner.Material(
                              diffuse_reflectance=pyredner.Texture(texels=torch.Tensor(tex_img_f)))
                          )
    camera = pyredner.automatic_camera_placement([obj], resolution=(h, w))
    camera.position[2] = -camera.position[2]
    scene = pyredner.Scene(camera=camera, objects=[obj])
    rend_img = pyredner.render_albedo(scene)
    # Need to gamma compress the image for displaying.
    imshow(torch.pow(rend_img, 1.0 / 2.2).cpu())
    # plt.imshow(rend_img.cpu())
    plt.show()


def mesh2obj(vertices, uvs, vertices_ind, uv_ind, name):
    """
    save mesh to .obj file without color
    :param vertices: tensor vertices [nver,3]
    :param uvs: tensor uvcoords [nver,2]
    :param vertices_ind: tensor triangles [ntri,3] start from 1
    :param uv_ind: tensor uvtriangles [ntri,3] start from 1
    :param name: obj file name
    :return: save .obj file to ./demo_output
    """
    vertices = vertices.cpu()
    uvs = uvs.cpu()
    obj = pyredner.Object(vertices=torch.Tensor(vertices),
                          indices=torch.tensor(vertices_ind - 1, dtype=torch.int32),
                          uvs=torch.Tensor(uvs),
                          uv_indices=torch.tensor(uv_ind - 1, dtype=torch.int32),
                          material=pyredner.Material(diffuse_reflectance=torch.Tensor([104, 151, 187]))
                          )

    save_path = os.path.join('./demo_output', name + '.obj')
    pyredner.save_obj(obj, save_path)
    print('save done: ' + save_path)


# given texture(need pow) and mesh save it to obj ok!
def mesh2obj_tex(vertices, uvs, vertices_ind, uv_ind, tex_img, path='./demo_output', name='demo'):
    vertices = vertices.cpu()
    uvs = uvs.cpu()
    obj = pyredner.Object(vertices=torch.Tensor(vertices),
                          indices=torch.tensor(vertices_ind - 1, dtype=torch.int32),
                          uvs=torch.Tensor(uvs),
                          uv_indices=torch.tensor(uv_ind - 1, dtype=torch.int32),
                          material=pyredner.Material(diffuse_reflectance=pyredner.Texture(texels=torch.Tensor(tex_img)))
                          )
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, name + '.obj')

    pyredner.save_obj(obj, save_path, flip_tex_coords=False)
    print('save done: ' + save_path)


class DataManager(Dataset):
    def __init__(self, data_dir):
        super(DataManager, self).__init__()
        self.dir_name = data_dir
        self.data_name = data_dir.split('/')[-1]
        self.json_name = self.data_name + '_params.json'
        self.camconf = []
        self.imgs = []
        self.imgs_ten = []
        self.img_nums = 0

    def read_data(self):
        with open(os.path.join(self.dir_name, self.json_name), 'r') as f:
            params = json.load(f)
        self.img_nums = len(params) // 5
        for i in range(self.img_nums):
            position = params['%d_position' % i]
            look_at = params['%d_look_at' % i]
            up = params['%d_up' % i]
            fov = params['%d_fov' % i]
            resolution = params['%d_resolution' % i]
            self.camconf.append({'position': position, 'look_at': look_at, 'up': up,
                                 'fov': fov, 'resolution': resolution})
            img_path = os.path.join(self.dir_name, self.data_name + '_{:03d}.jpg'.format(i))
            self.imgs.append(img_path)

    def load_data(self):
        for i in range(self.img_nums):
            target = pyredner.imread(self.imgs[i]).to(pyredner.get_device())
            self.imgs_ten.append(target)

    def __getitem__(self, index):
        return self.imgs_ten[index], self.camconf[index]

    def __len__(self):
        return len(self.imgs)


def getDataLoader(dir, batchsize=10, shuffle=True, num_worker=0):
    data_manager = DataManager(dir)
    data_manager.read_data()
    data_manager.load_data()
    data_loader = DataLoader(data_manager, batch_size=batchsize, shuffle=shuffle, num_workers=num_worker)
    return data_loader


if __name__ == '__main__':
    # show_info()
    # gt = get_pre_obj()
    # pre_obj_show()

    # eval 3: redner imwrite
    i_path = '../datas/facescape/generate_test/6_4_anger/6_4_anger_000.jpg'
    img = pyredner.imread(i_path) # 内部使用未进行gamma矫正
    imshow(img)
    plt.show()
    img_pow=torch.pow(img,1.0/2.2).cpu() # 使用gamma矫正后
    img_powback=torch.pow(img_pow,2.2/1.0) # gamma逆矫正
    # imshow(torch.pow(img, 1.0 / 2.2).cpu())
    imshow(img_pow)
    plt.show()
    imshow(img_powback)
    plt.show()
    img_np = io.imread(i_path)
    img_ten=torch.Tensor(img_np)
    imshow(img_np)
    plt.show()

    # eval 2: evaluate save obj with color
    # img256, img4096 = get_pre_tex2(256)
    # gt_obj = get_pre_obj()
    # # # facescape object
    # x = adapt_uv(gt_obj.vertices, gt_obj.texcoords, gt_obj.faces_v, gt_obj.faces_vt)  # [26278,2]
    # col = get_color_with_tex(x, img4096)
    # # rend_tex=render_texture(x,gt_obj.faces_v,col,512,512)
    # # imshow(rend_tex)
    # # plt.show()
    # render_obj(torch.Tensor(gt_obj.vertices), torch.Tensor(x), gt_obj.faces_v, torch.Tensor(col))
    # render_obj_tex(torch.Tensor(gt_obj.vertices), torch.Tensor(x), gt_obj.faces_v, torch.Tensor(col))
    # # mesh2obj_tex(torch.Tensor(gt_obj.vertices),torch.Tensor(gt_obj.texcoords),torch.Tensor(gt_obj.faces_v),torch.Tensor(gt_obj.faces_vt),img256,'objtex4')

    # # eval 1: evaluate the param of camera is right
    # obj_path='../datas/facescape/facescape_trainset_001_100/2/models_reg/2_smile.obj'
    # data_dir='../datas/facescape/generate/2_2_smile'
    # show_generated(data_dir,obj_path)

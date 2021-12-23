import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from skimage import io
from idea_01.data import *
from idea_01.render import *

uv_mask_path = '../datas/maskdata/uv_mask_256.bmp'
uv_mask = io.imread(uv_mask_path)


def loss_curve(loss_list):
    epochs_list = np.arange(len(loss_list)) + 1

    plt.plot(epochs_list, loss_list, label="loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc=0, ncol=1)  # 参数：loc设置显示的位置，0是自适应；ncol设置显示的列数

    plt.show()

# only for one iter
def loss_list_curve(loss_list,ind=0):
    # [epoch,iters,batchsize]
    one_list=[]
    for epoch in range(len(loss_list)):
        one_list.append(loss_list[epoch][0][ind])

    loss_curve(one_list)

# 获取uv坐标网格中心点坐标,[h*w,2],从上至下
def get_uvcoords(H, W):
    i, j = np.meshgrid(np.linspace(0, W - 1, W), np.linspace(0, H - 1, H))
    dx = 1 / W
    dy = 1 / H
    for x in range(W):
        i[:, x] = dx / 2 + dx * x
    for y in range(H):
        j[y, :] = dy / 2 + dy * y

    j = np.flipud(j)
    uvs = np.stack((i, j), axis=2).reshape(H * W, 2)
    return uvs


# 这里的第一个索引为1！
def get_uvs_triangles(uv_h, uv_w):
    triangles = []
    for i in range(uv_h):
        for j in range(uv_w):
            pa = i * uv_w + j
            pb = i * uv_w + j + 1
            pc = (i - 1) * uv_w + j
            pd = (i + 1) * uv_w + j + 1
            if (i > 0) & (i < uv_h - 1) & (j < uv_w - 1):
                triangles.append([pa, pc, pb])
                triangles.append([pa, pb, pc])
                triangles.append([pa, pb, pd])
                triangles.append([pa, pd, pb])

    triangles = np.array(triangles)
    return triangles + 1


def ifavail(ind):
    if ind < 0 or ind >= len(uv_mask.flatten()):
        return False
    uv_mask_flat = uv_mask.flatten()
    if uv_mask_flat[ind] == 0:
        return False
    else:
        return True


def get_masked_mesh(H, W):
    """
    given h and w generate vertices list and corresponding tri list by mask
    :param H:
    :param W:
    :return: [nuv,2],[ntri,3]
    """
    uvs = get_uvcoords(H, W)
    pixlist = [{'u': uvs[i][0], 'v': uvs[i][1], 'avail': False, 'verind': 0} for i in range(len(uvs))]
    ind_cnt = 1
    for i in range(len(pixlist)):
        if ifavail(i) is True:
            pixlist[i]['avail'] = True
            pixlist[i]['verind'] = ind_cnt
            ind_cnt = ind_cnt + 1

    tri_list = []
    for i in range(H):
        for j in range(W - 1):
            ind = i * W + j
            if pixlist[ind]['avail'] is False:
                continue
            else:
                pa = i * W + j
                pb = i * W + j + 1
                pc = (i - 1) * W + j
                pd = (i + 1) * W + j + 1
                if ifavail(pa) & ifavail(pb) & ifavail(pc):
                    tri_list.append([pixlist[pa]['verind'], pixlist[pb]['verind'], pixlist[pc]['verind']])
                    tri_list.append([pixlist[pa]['verind'], pixlist[pc]['verind'], pixlist[pb]['verind']])
                if ifavail(pa) & ifavail(pb) & ifavail(pd):
                    tri_list.append([pixlist[pa]['verind'], pixlist[pb]['verind'], pixlist[pd]['verind']])
                    tri_list.append([pixlist[pa]['verind'], pixlist[pd]['verind'], pixlist[pb]['verind']])

    ver_list = []
    for i in range(len(pixlist)):
        if pixlist[i]['avail'] is True:
            ver_list.append([pixlist[i]['u'], pixlist[i]['v']])

    vers = np.array(ver_list)
    triangles = np.array(tri_list)

    return vers, triangles


# !!! not use bilinear operation
def get_color_with_tex(uvs, img):
    """
    get corresponding color from img, no bilinear since the img reso is 4096
    :param uvs: [nuvs,2]
    :param img: [4096,4096,3]
    :return: [nuvs,3] which is the corresponding color of uvs
    """
    reso = len(img)
    color_list = []
    for i in range(len(uvs)):
        u = uvs[i][0]
        v = uvs[i][1]
        uu = int(np.floor(u * reso))
        vv = int(np.floor(v * reso))
        color_list.append(img[reso - 1 - vv][uu])
    return np.array(color_list)


def process_uv(uv_coords, uv_h=256, uv_w=256):
    uv_coords = uv_coords.copy()
    uv_coords[:, 0] = uv_coords[:, 0] * (uv_w - 1)
    uv_coords[:, 1] = uv_coords[:, 1] * (uv_h - 1)
    uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1))))  # add z
    return uv_coords


def render_texture(uvs, triangles, uvs_color, h, w):
    uvs_ver = process_uv(uvs, h, w)
    img = render_colors(uvs_ver, triangles - 1, uvs_color, h, w)
    return img


if __name__ == '__main__':
    vertices_uv, triangles = get_masked_mesh(256, 256)
    # z = np.ones(len(vertices_uv)).reshape(1, len(vertices_uv))
    # vertices = np.insert(vertices_uv, 2, z, axis=1)
    # mesh2obj(torch.Tensor(vertices), torch.Tensor(vertices_uv), torch.Tensor(triangles), torch.Tensor(triangles),
    #          'uv_generate01')
    img256, img4096 = get_pre_tex1()
    vertices_color = get_color_with_tex(vertices_uv, img4096)
    img = render_texture(vertices_uv, triangles, vertices_color, 512, 512)
    imshow(img)
    plt.show()

    # print(get_uvs_triangles(256, 256))

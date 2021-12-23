from idea_01.config import config
import numpy as np
import torch
import torch.nn as nn
import os, sys
import matplotlib
import matplotlib.pyplot as plt
import time as mytime
from tqdm import tqdm, trange
from idea_01.data import *
from idea_01.utils import *
from idea_01.model import *
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H_eval = 256
W_eval = 256
epochs = 100
iter_single = 1


def main():
    kwargs_train, kwargs_test, start, optimizer, loss_list = create_UVShapeField(config, device)
    kwargs_train2, kwargs_test2, start2, optimizer2, loss_list2 = create_UVTexField(config, device)
    x_eval, tri_eval = get_masked_mesh(H_eval, W_eval)
    x_eval = torch.Tensor(x_eval).to(device)
    tri_eval_ten = torch.tensor(tri_eval - 1, dtype=torch.int32).to(device)
    print('1. loading pretrained ckpts')
    ckpts = [os.path.join(config.savedir, expname_shape, f) for f in
             sorted(os.listdir(os.path.join(config.savedir, expname_shape)))
             if config.ckpts in f]
    if len(ckpts) == 1:
        print('find shape checkpoint: ', ckpts[0])
    else:
        print('not find shape checkpoint')

    if len(ckpts) > 0:
        ckpt_path = ckpts[0]
        print('reloading shape from ', ckpt_path)
        ckpt = torch.load(ckpt_path)
        kwargs_train['network'].load_state_dict(ckpt['network_fn_state_dict'])

    ckpts = [os.path.join(config.savedir, expname_tex, f) for f in
             sorted(os.listdir(os.path.join(config.savedir, expname_tex)))
             if config.ckptt in f]
    if len(ckpts) == 1:
        print('find tex checkpoint: ', ckpts[0])
    else:
        print('not find tex checkpoint')

    if len(ckpts) > 0:
        ckpt_path = ckpts[0]
        print('reloading tex from ', ckpt_path)
        ckpt = torch.load(ckpt_path)
        kwargs_train2['network'].load_state_dict(ckpt['network_fn_state_dict'])

    grad_vars_shape = list(kwargs_train['network'].parameters())
    grad_vars_tex = list(kwargs_train2['network'].parameters())

    optimizer_shape = torch.optim.Adam(grad_vars_shape, lr=1e-5)
    optimizer_tex = torch.optim.Adam(grad_vars_tex, lr=1e-5)

    #####
    # print('test : rending init img')
    # # preview init output
    # with torch.no_grad():
    #     pred_vertices=kwargs_train['network_query_fn'](x_eval) #[nver,3] x,y,z
    #     pred_vertices=pred_vertices*200.
    #     pred_colors=kwargs_train2['network_query_fn'](x_eval) #[nver,3] r,g,b
    #     obj=pyredner.Object(vertices=pred_vertices,indices=tri_eval_ten,uvs=x_eval,
    #                         uv_indices=tri_eval_ten,material=pyredner.Material(use_vertex_color=True),
    #                         colors=pred_colors)
    #     camera = pyredner.automatic_camera_placement([obj], resolution=(512, 512))
    #     camera.position[2] = -camera.position[2]
    #     scene = pyredner.Scene(camera=camera, objects=[obj])
    #     rend_img = pyredner.render_albedo(scene)
    #     # Need to gamma compress the image for displaying.
    #     imshow(torch.pow(rend_img, 1.0 / 2.2).cpu())
    #     # plt.imshow(rend_img.cpu())
    #     plt.show()

    print('2. preparing data')
    data_dir_train = '../datas/facescape/generate/6_4_anger'
    data_dir_test = '../datas/facescape/generate_test/6_4_anger'
    train_iter = getDataLoader(data_dir_train, batchsize=10)
    test_iter = getDataLoader(data_dir_test, batchsize=20, shuffle=False)

    print('3. start training')

    img_test_list_epoch = []
    loss_test_list_epoch = []
    for epoch in trange(1, epochs + 1):
        print('epoch {} start'.format(epoch))
        iter_cnt = 0
        for imgs, confs in train_iter:
            iter_cnt = iter_cnt + 1
            print('iter {} start'.format(iter_cnt))

            for i in range(len(imgs)):
                print('start single img {}'.format(i))
                # show target img
                target = imgs[i]
                # imshow(torch.pow(target, 1.0 / 2.2).cpu())
                # plt.show()
                # prepare camera
                position = torch.Tensor([confs['position'][0][i], confs['position'][1][i], confs['position'][2][i]])
                look_at = torch.Tensor([confs['look_at'][0][i], confs['look_at'][1][i], confs['look_at'][2][i]])
                up = torch.Tensor([confs['up'][0][i], confs['up'][1][i], confs['up'][2][i]])
                fov = confs['fov'][i].item()
                resolution = (confs['resolution'][0][i].item(), confs['resolution'][1][i].item())
                camera = pyredner.Camera(position=position, look_at=look_at,
                                         up=up,
                                         fov=torch.Tensor([fov]))
                camera.resolution = resolution
                # single_imglist, single_losses = [], [] # for every img, after iter show results
                for t in range(1, iter_single + 1):
                    optimizer_shape.zero_grad()
                    optimizer_tex.zero_grad()
                    # rend img use the same camera
                    pred_vertices = kwargs_train['network_query_fn'](x_eval)  # [nver,3] x,y,z
                    pred_vertices = pred_vertices * 200.
                    pred_colors = kwargs_train2['network_query_fn'](x_eval)  # [nver,3] r,g,b
                    obj = pyredner.Object(vertices=pred_vertices, indices=tri_eval_ten, uvs=x_eval,
                                          uv_indices=tri_eval_ten, material=pyredner.Material(use_vertex_color=True),
                                          colors=pred_colors)
                    scene = pyredner.Scene(camera=camera, objects=[obj])
                    rend_img = pyredner.render_albedo(scene)
                    # Need to gamma compress the image for displaying.
                    # imshow(torch.pow(rend_img.detach(), 1.0 / 2.2).cpu())
                    # # plt.imshow(rend_img.cpu())
                    # plt.show()

                    # Compute the loss function. Here it is L2 plus a regularization term to avoid coefficients to be too far from zero.
                    # Both img and target are in linear color space, so no gamma correction is needed.
                    loss = (rend_img - target).pow(2).mean()
                    loss.backward()
                    optimizer_shape.step()
                    optimizer_tex.step()

                    # Plot the loss
                    # losses.append(loss.data.item())
                    # Only store images every 10th iterations
                    if t % iter_single == 0:
                        #     single_losses.append(loss.data.item())
                        #     single_img=torch.pow(rend_img.data, 1.0 / 2.2).cpu()
                        #     diff_img=(rend_img - target).pow(2).sum(dim=2).data.cpu()
                        #     single_imglist.append([single_img,diff_img])
                        print(
                            ' done after epoch{}, iter{} of img {}, loss is {}'.format(epoch, iter_cnt, i,
                                                                                       loss.data.item()))
                    ####### show
                    #     # f, (ax_loss, ax_diff_img, ax_img) = plt.subplots(1, 3)
                    #     # imglist.append(torch.pow(rend_img.data, 1.0 / 2.2).cpu())  # Record the Gamma corrected image
                    #     # ax_loss.plot(range(len(losses)), losses, label='loss')
                    #     # ax_loss.legend()
                    #     # ax_diff_img.imshow((rend_img - target).pow(2).sum(dim=2).data.cpu())
                    #     # ax_img.imshow(torch.pow(rend_img.data.cpu(), 1.0 / 2.2))
                    #     # plt.show()

            print('iter {} done'.format(iter_cnt))

        print('epoch {} done, start to evaluate'.format(epoch))

        img_test_list = []
        loss_test_list = []
        with torch.no_grad():
            for imgs, confs in test_iter:
                loss_test_list_temp = []  # [batchsize]
                img_test_list_temp = []  # [batchsize,3]
                for i in range(len(imgs)):
                    target = imgs[i]
                    position = torch.Tensor([confs['position'][0][i], confs['position'][1][i], confs['position'][2][i]])
                    look_at = torch.Tensor([confs['look_at'][0][i], confs['look_at'][1][i], confs['look_at'][2][i]])
                    up = torch.Tensor([confs['up'][0][i], confs['up'][1][i], confs['up'][2][i]])
                    fov = confs['fov'][i].item()
                    resolution = (confs['resolution'][0][i].item(), confs['resolution'][1][i].item())
                    camera = pyredner.Camera(position=position, look_at=look_at,
                                             up=up,
                                             fov=torch.Tensor([fov]))
                    camera.resolution = resolution
                    pred_vertices = kwargs_train['network_query_fn'](x_eval)  # [nver,3] x,y,z
                    pred_vertices = pred_vertices * 200.
                    pred_colors = kwargs_train2['network_query_fn'](x_eval)  # [nver,3] r,g,b
                    obj = pyredner.Object(vertices=pred_vertices, indices=tri_eval_ten, uvs=x_eval,
                                          uv_indices=tri_eval_ten, material=pyredner.Material(use_vertex_color=True),
                                          colors=pred_colors)
                    scene = pyredner.Scene(camera=camera, objects=[obj])
                    rend_img = pyredner.render_albedo(scene)
                    loss = (rend_img - target).pow(2).mean()

                    loss_test_list_temp.append(loss.data.item())
                    single_img = rend_img.data.cpu()
                    diff_img = (rend_img - target).pow(2).sum(dim=2).data.cpu()
                    gt_img = target.data.cpu()
                    img_test_list_temp.append([single_img, diff_img, gt_img])

                loss_test_list.append(loss_test_list_temp)  # [iter_cnt,batchsize]
                img_test_list.append(img_test_list_temp)  # [iter_cnt,batchsize,3]

            img_test_list_epoch.append(img_test_list)  # [epoch,iter_cnt,batchsize]
            loss_test_list_epoch.append(loss_test_list)  # [epoch,iter_cnt,batchsize,3]
            print('evaluate ok, show results')

            # for j in range(len(img_test_list)):
            #     f, (ax_diff_img, ax_img, ax_gt) = plt.subplots(1, 3)
            #     ax_diff_img.imshow(img_test_list[j][1])
            #     ax_img.imshow(img_test_list[j][0])
            #     ax_gt.imshow(img_test_list[j][2])
            #     plt.show()

        print('evaluate done, save ckpt')
        os.makedirs(os.path.join(config.savedir, expname_all), exist_ok=True)
        now = mytime.localtime()
        nowt = mytime.strftime("%Y-%m-%d-%H_%M_%S_", now)
        path = os.path.join(config.savedir, expname_all, nowt + '{:06d}.tar'.format(epoch))
        torch.save({
            'global_step': epoch,
            'network_fn_state_dict_shape': kwargs_train['network'].state_dict(),
            'network_fn_state_dict_tex': kwargs_train2['network'].state_dict(),
            'optimizer_state_dict_shape': optimizer_shape.state_dict(),
            'optimizer_state_dict_tex': optimizer_tex.state_dict(),
            'loss_list': loss_test_list_epoch,
            'conf': config,
        }, path)
        print('ckpt save ok, save imgs evaluated')
        ckpt_img_dir = os.path.join(config.savedir, expname_all, nowt + '{:06d}_imgs'.format(epoch))
        os.makedirs(ckpt_img_dir, exist_ok=True)
        for a in range(len(img_test_list_epoch[epoch - 1])):
            # a is the iter_cnt (may be 0)
            for b in range(len(img_test_list_epoch[epoch - 1][a])):
                # b is the batchsize
                diff_img_name = 'epoch{}_iter{}_no{}_diff.jpg'.format(epoch, a + 1, b)
                rend_img_name = 'epoch{}_iter{}_no{}_rend.jpg'.format(epoch, a + 1, b)
                gt_img_name = 'epoch{}_iter{}_no{}_gt.jpg'.format(epoch, a + 1, b)
                diff_img_path = os.path.join(ckpt_img_dir, diff_img_name)
                rend_img_path = os.path.join(ckpt_img_dir, rend_img_name)
                gt_img_path = os.path.join(ckpt_img_dir, gt_img_name)
                pyredner.imwrite(img_test_list_epoch[epoch - 1][a][b][1], diff_img_path)
                pyredner.imwrite(img_test_list_epoch[epoch - 1][a][b][0], rend_img_path)
                pyredner.imwrite(img_test_list_epoch[epoch - 1][a][b][2], gt_img_path)
        print(loss_test_list_epoch)


def main_save():
    kwargs_train, kwargs_test, start, optimizer, loss_list = create_UVShapeField(config, device)
    kwargs_train2, kwargs_test2, start2, optimizer2, loss_list2 = create_UVTexField(config, device)
    x_eval, tri_eval = get_masked_mesh(H_eval, W_eval)
    x_eval = torch.Tensor(x_eval).to(device)
    tri_eval_ten = torch.tensor(tri_eval - 1, dtype=torch.int32).to(device)
    print('loading saved checkpoints both shape and tex')
    ckpts = [os.path.join(config.savedir, expname_all, f) for f in
             sorted(os.listdir(os.path.join(config.savedir, expname_all)))
             if config.ckpta in f]
    if len(ckpts) == 1:
        print('find checkpoint both of shape and tex: ', ckpts[0])
    else:
        print('not find checkpoint')

    if len(ckpts) > 0:
        ckpt_path = ckpts[0]
        print('reloading ckpt both shape and tex from ', ckpt_path)
        ckpt = torch.load(ckpt_path)
        kwargs_train['network'].load_state_dict(ckpt['network_fn_state_dict_shape'])
        kwargs_train2['network'].load_state_dict(ckpt['network_fn_state_dict_tex'])
        loss_list_all = ckpt['loss_list']
        ckpt_conf = ckpt['conf']

    print('loadding done, start to visualize loss curve')
    loss_list_curve(loss_list_all)

    with torch.no_grad():
        pred_vertices = kwargs_train['network_query_fn'](x_eval)
        pred_vertices = pred_vertices * 200
        pred_colors = kwargs_train2['network_query_fn'](x_eval)
        texture_img = render_texture(x_eval.detach().cpu().numpy(), tri_eval, pred_colors.detach().cpu().numpy(), 512,
                                     512)
        imshow(np.power(texture_img, 1.0 / 2.2))
        plt.show()
        print('rending obj')
        render_obj(pred_vertices, x_eval, tri_eval, pred_colors)
        render_mesh(pred_vertices,x_eval,tri_eval,tri_eval)
        # print('save to .obj file')
        # obj_name=config.ckpta.split('.tar')[0]
        # mesh2obj_tex(pred_vertices,x_eval,tri_eval,tri_eval,texture_img,path=config.objdir,name=obj_name)

if __name__ == '__main__':
    if config.mode == 0:
        main()

    if config.mode == 1:
        main_save()

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


def main_pre():
    kwargs_train, kwargs_test, start, optimizer, loss_list = create_UVShapeField(config, device)
    kwargs_train2, kwargs_test2, start2, optimizer2, loss_list2 = create_UVTexField(config, device)

    global_step = start
    global_step2=start2

    # evaluate the final results
    ## not use mask
    # x_eval = get_uvcoords(H_eval, W_eval)
    # tri_eval = get_uvs_triangles(H_eval, W_eval)
    ## use mask
    x_eval,tri_eval=get_masked_mesh(H_eval,W_eval)

    x_eval = torch.Tensor(x_eval).to(device)

    gt_obj = get_pre_obj()
    gt_tex512,gt_tex4096=get_pre_tex2(512)

    # facescape object
    x = adapt_uv(gt_obj.vertices, gt_obj.texcoords, gt_obj.faces_v, gt_obj.faces_vt)  # [26278,2]
    label = gt_obj.vertices / 200. # [26278,3] scale to [-1,1]
    label2=get_color_with_tex(x,gt_tex4096)


    x = torch.Tensor(x).to(device)
    label = torch.Tensor(label).to(device)
    label2 = torch.Tensor(label2).to(device)

    if config.premode==0:
        print('mode 0: shape pretraining')
        for i in trange(global_step + 1, 50000 + 1):
            y_pred = kwargs_train['network_query_fn'](x)
            optimizer.zero_grad()
            loss = mesh2mse(y_pred, label, x, False)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            if i % 50 == 0:
                print('epoch {}, loss {}'.format(i, loss.item()))

            if i % config.saveepoch == 0:
                now = mytime.localtime()
                nowt = mytime.strftime("%Y-%m-%d-%H_%M_%S_", now)
                path = os.path.join(config.savedir, expname_shape, nowt + '{:06d}.tar'.format(i))
                torch.save({
                    'global_step': i,
                    'network_fn_state_dict': kwargs_train['network'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_list': loss_list,
                    'conf': config,
                }, path)
                print('Saved checkpoints at ', path)

                # after save then evaluate:
                with torch.no_grad():
                    y_pred_eval = kwargs_train['network_query_fn'](x_eval)
                    y_pred = kwargs_train['network_query_fn'](x)
                    render_mesh(y_pred*200, x, gt_obj.faces_v, gt_obj.faces_v)
                    render_mesh(y_pred_eval*200, x_eval, tri_eval, tri_eval)

    if config.premode==1:
        print('mode 1: color pretraining')
        for i in trange(global_step2 + 1, 50000 + 1):
            y_pred2 = kwargs_train2['network_query_fn'](x)
            optimizer2.zero_grad()
            loss = color2mse(y_pred2, label2, x, False)
            loss.backward()
            optimizer2.step()
            loss_list2.append(loss.item())
            if i % 50 == 0:
                print('epoch {}, loss {}'.format(i, loss.item()))

            if i % config.saveepoch == 0:
                now = mytime.localtime()
                nowt = mytime.strftime("%Y-%m-%d-%H_%M_%S_", now)
                path = os.path.join(config.savedir, expname_tex, nowt + '{:06d}.tar'.format(i))
                torch.save({
                    'global_step': i,
                    'network_fn_state_dict': kwargs_train2['network'].state_dict(),
                    'optimizer_state_dict': optimizer2.state_dict(),
                    'loss_list': loss_list2,
                    'conf': config,
                }, path)
                print('Saved checkpoints at ', path)

                # after save then evaluate:
                with torch.no_grad():
                    loss_curve(loss_list2)
                    y_pred2_eval = kwargs_train2['network_query_fn'](x_eval)
                    y_pred2 = kwargs_train2['network_query_fn'](x)
                    imshow(gt_tex512)
                    plt.show()
                    img_eval=render_texture(x_eval.cpu().numpy(),tri_eval,y_pred2_eval.cpu().numpy(),512,512)
                    img_gt=render_texture(x.cpu().numpy(),gt_obj.faces_v,y_pred2.cpu().numpy(),512,512)
                    imshow(img_eval)
                    plt.show()
                    imshow(img_gt)
                    plt.show()

    if config.premode==2:
        print('mode 2: shape preview by loading model')
        ckpts = []
        ckpts = [os.path.join(config.savedir, expname_shape, f) for f in
                 sorted(os.listdir(os.path.join(config.savedir, expname_shape)))
                 if config.ckpts in f]
        if len(ckpts)==1:
            print("find checkpoint: ", ckpts[0])
        else:
            print('not find ckpt path')

        # reload weights
        if len(ckpts) > 0:
            ckpt_path = ckpts[0]
            print('reloading from ', ckpt_path)
            ckpt = torch.load(ckpt_path)
            start = ckpt['global_step']
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            kwargs_train['network'].load_state_dict(ckpt['network_fn_state_dict'])
            loss_list = ckpt['loss_list']

            with torch.no_grad():
                y_pred_eval = kwargs_train['network_query_fn'](x_eval)
                y_pred = kwargs_train['network_query_fn'](x)
                # render_mesh(y_pred * 200, x, gt_obj.faces_v, gt_obj.faces_v)
                # render_mesh(y_pred_eval * 200, x_eval, tri_eval, tri_eval)
                mesh2obj(y_pred * 200, x, gt_obj.faces_v, gt_obj.faces_v,config.ckpts.replace('.tar','')+'_gt')
                mesh2obj(y_pred_eval * 200, x_eval, tri_eval, tri_eval,config.ckpts.replace('.tar','')+'_eval')

    if config.premode==3:
        print('mode 3: tex preview by loading model')
        ckpts = []
        ckpts = [os.path.join(config.savedir, expname_tex, f) for f in
                 sorted(os.listdir(os.path.join(config.savedir, expname_tex)))
                 if config.ckptt in f]
        if len(ckpts) == 1:
            print("find checkpoint: ", ckpts[0])
        else:
            print('not find ckpt path')

        # reload weights
        if len(ckpts) > 0:
            ckpt_path = ckpts[0]
            print('reloading from ', ckpt_path)
            ckpt = torch.load(ckpt_path)
            start2 = ckpt['global_step']
            optimizer2.load_state_dict(ckpt['optimizer_state_dict'])
            kwargs_train2['network'].load_state_dict(ckpt['network_fn_state_dict'])
            loss_list2 = ckpt['loss_list']

            with torch.no_grad():
                y_pred2_eval = kwargs_train2['network_query_fn'](x_eval)
                y_pred2 = kwargs_train2['network_query_fn'](x)
                imshow(gt_tex512)
                plt.show()
                img_eval=render_texture(x_eval.cpu().numpy(),tri_eval,y_pred2_eval.cpu().numpy(),512,512)
                img_gt=render_texture(x.cpu().numpy(),gt_obj.faces_v,y_pred2.cpu().numpy(),512,512)
                imshow(img_eval)
                plt.show()
                imshow(img_gt)
                plt.show()


if __name__ == '__main__':
    main_pre()

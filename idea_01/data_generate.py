import pyredner
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import json
import numpy as np
import random
import os

def sample_sphere(n_samples=200, rho=500):
    samples = []
    cnt = 0
    while cnt < n_samples:
        theta = np.pi * random.random()
        phi = 2 * np.pi * random.random()
        x=rho*np.sin(theta)*np.cos(phi)
        y=rho*np.sin(theta)*np.sin(phi)
        z=rho*np.cos(theta)
        if y<-rho*0.2:
            continue
        if z<0:
            if random.random()<0.8:
                continue
        samples.append([x,y,z])
        cnt=cnt+1

    return np.array(samples)
# 5 camera params
def generate_img(id,obj_path,save_path,n_sample=200):
    obj_name=obj_path.split('/')[-1].split('.obj')[0]
    save_dir=str(id)+'_'+obj_name
    save_dir_path=os.path.join(save_path,save_dir)
    os.makedirs(save_dir_path,exist_ok=True)
    config_name=save_dir+'_params.json'
    sam=sample_sphere(n_sample)
    objects=pyredner.load_obj(obj_path,return_objects=True)
    camera = pyredner.Camera(position=torch.Tensor([0,0,0]),look_at=torch.Tensor([0,0,0]),up=torch.Tensor([0,1,0]),fov=torch.Tensor([45]))
    camera.resolution=(512,512)
    camera_config={}
    for i in range(sam.shape[0]):
        camera.position=torch.Tensor(sam[i])
        camera_config.update({'{}_position'.format(i):camera.position.cpu().numpy().tolist()})
        camera_config.update({'{}_look_at'.format(i):camera.look_at.cpu().numpy().tolist()})
        camera_config.update({'{}_up'.format(i):camera.up.cpu().numpy().tolist()})
        camera_config.update({'{}_fov'.format(i):camera.fov.cpu().item()})
        camera_config.update({'{}_resolution'.format(i):camera.resolution})
        scene=pyredner.Scene(camera=camera,objects=objects)
        img=pyredner.render_albedo(scene).cpu()
        img_name=save_dir+'_{:03d}.jpg'.format(i)
        img_path=os.path.join(save_dir_path,img_name)
        pyredner.imwrite(img,img_path)
        # imshow(torch.pow(img,1.0/2.2))
        # plt.show()

    json_path=os.path.join(save_dir_path,config_name)

    with open(json_path,'w') as file:
        json.dump(camera_config,file)

if __name__ == '__main__':
    ## generate train
    # id=6
    # obj_path='../datas/facescape/facescape_trainset_001_100/6/models_reg/4_anger.obj'
    # save_path='../datas/facescape/generate'
    # n=100
    # generate_img(id,obj_path,save_path,n)
    ## generate test
    id=6
    obj_path='../datas/facescape/facescape_trainset_001_100/6/models_reg/4_anger.obj'
    save_path='../datas/facescape/generate_test'
    n=20
    generate_img(id,obj_path,save_path,n)
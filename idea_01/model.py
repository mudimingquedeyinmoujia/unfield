import torch.nn as nn
import torch
import torch.nn.functional as F
from skimage import io
import numpy as np
import os

expname_shape = 'shapenet'
expname_tex = 'texnet'
expname_all='allnet'


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:  # true
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']  # nerf:9/3 num_freqs-1,
        N_freqs = self.kwargs['num_freqs']  # nerf:10/4 位置编码的数量

        if self.kwargs['log_sampling']:  # true
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)  # [2^0,2^1,...,2^max_freq] 位置编码数量
        else:  # 如果不按照log采样那就是均匀采样
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns  # sin(2^0 x) cos(2^0 x) sin(2^1 x) cos(2^1 x) ...
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)  # 按照最后一维拼接，拼接后维度不变


def get_embedder(multires, i=0):
    """
    获取位置编码函数与位置编码后的维度,没有使用pi
    :param multires: 位置编码频率，从0开始计算，依次是2^0,2^1,...,2^(multires-1) 共 multires 个
    :param i: 0代表默认位置编码，-1代表不进行位置编码
    :return: example: x,y,z --> x,y,z,sin(x),sin(y),sin(z),cos(x),cos(y),cos(z),sin(2^1x),sin(2^1y),sin(2^1z),cos(2^1x),
    cos(2^1y),cos(x^1z),... ///  输出的对于每个batch的维度
    """
    if i == -1:
        return nn.Identity(), 2

    embed_kwargs = {
        'include_input': True,  # 输出的编码是否包含输入的特征本身
        'input_dims': 2,  # 最好每个batch维度是一维，默认一维有三个分量
        'max_freq_log2': multires - 1,  # 最大的位置编码频率
        'num_freqs': multires,  # 位置编码频率数量
        'log_sampling': True,  # 默认按照2^0,2^1...2^n采样,否则在[0,2^n]均匀采样
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class UVShapeField(nn.Module):
    def __init__(self, D=8, W=256, input_ch=2, output_ch=3):
        """
        (u,v)-->(x,y,z) in local coordinates
        """
        super(UVShapeField, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.architecture = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) for i in range(D - 1)])
        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        h = x
        for i, l in enumerate(self.architecture):
            h = self.architecture[i](h)
            h = F.relu(h)

        output = self.output_linear(h)
        # ten_pi = torch.tensor(np.pi)
        # output = torch.atan(output) * 2 / ten_pi # use arctan to map to [-1,1]
        output = torch.tanh(output)  # map to [-1,1]

        return output


class UVTexField(nn.Module):
    def __init__(self, D=8, W=256, input_ch=2, output_ch=3):
        """
        (u,v)-->(r,g,b) in for texture search
        """
        super(UVTexField, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.architecture = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) for i in range(D - 1)])
        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        h = x
        for i, l in enumerate(self.architecture):
            h = self.architecture[i](h)
            h = F.relu(h)

        output = self.output_linear(h)
        # ten_pi = torch.tensor(np.pi)
        # output = torch.atan(output) * 2 / ten_pi # use arctan to map to [-1,1]
        output = torch.sigmoid(output)  # map to [0,1]

        return output


def run_network(inputs, model, embed_fn):
    return model(embed_fn(inputs))


def create_UVShapeField(config, device):
    input_ch = 2
    output_ch = 3
    depth = config.depths
    width = config.widths
    multires = config.muls
    i_embed = config.embeds
    embed_fn, input_ch = get_embedder(multires, i_embed)
    model = UVShapeField(depth, width, input_ch, output_ch).to(device)
    # to optimize
    grad_vars = list(model.parameters())
    network_query_fn = lambda inputs, model=model, embed_fn=embed_fn: run_network(inputs, model, embed_fn)
    optimizer = torch.optim.Adam(params=grad_vars, lr=config.lrates, betas=(0.9, 0.999))

    start = 0
    loss_list = []

    # load check points
    os.makedirs(os.path.join(config.savedir, expname_shape), exist_ok=True)
    ckpts = []
    ckpts = [os.path.join(config.savedir, expname_shape, f) for f in
             sorted(os.listdir(os.path.join(config.savedir, expname_shape)))
             if 'tar' in f]
    print("shape checkpoints nums: ", len(ckpts))

    # reload weights
    if len(ckpts) > 0 and config.reload:
        ckpt_path = ckpts[-1]
        print('reloading from ', ckpt_path)
        ckpt = torch.load(ckpt_path)
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        model.load_state_dict(ckpt['network_fn_state_dict'])
        loss_list = ckpt['loss_list']

    kwargs_train = {
        'network_query_fn': network_query_fn,
        'network': model,
    }
    kwargs_test = {k: kwargs_train[k] for k in kwargs_train}

    return kwargs_train, kwargs_test, start, optimizer, loss_list


def create_UVTexField(config, device):
    input_ch = 2
    output_ch = 3
    depth = config.deptht
    width = config.widtht
    multires = config.mult
    i_embed = config.embedt
    embed_fn, input_ch = get_embedder(multires, i_embed)
    model = UVTexField(depth, width, input_ch, output_ch).to(device)
    # to optimize
    grad_vars = list(model.parameters())
    network_query_fn = lambda inputs, model=model, embed_fn=embed_fn: run_network(inputs, model, embed_fn)
    optimizer = torch.optim.Adam(params=grad_vars, lr=config.lratet, betas=(0.9, 0.999))

    start = 0
    loss_list = []

    # load check points
    os.makedirs(os.path.join(config.savedir, expname_tex), exist_ok=True)
    ckpts = []
    ckpts = [os.path.join(config.savedir, expname_tex, f) for f in
             sorted(os.listdir(os.path.join(config.savedir, expname_tex)))
             if 'tar' in f]
    print("tex checkpoints nums: ", len(ckpts))

    # reload weights
    if len(ckpts) > 0 and config.reload:
        ckpt_path = ckpts[-1]
        print('reloading from ', ckpt_path)
        ckpt = torch.load(ckpt_path)
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        model.load_state_dict(ckpt['network_fn_state_dict'])
        loss_list = ckpt['loss_list']

    kwargs_train = {
        'network_query_fn': network_query_fn,
        'network': model,
    }
    kwargs_test = {k: kwargs_train[k] for k in kwargs_train}

    return kwargs_train, kwargs_test, start, optimizer, loss_list


def mesh2mse(y_pred, label, x, mask=False):
    if mask is False:
        return torch.mean((y_pred - label) ** 2)
    else:
        return torch.mean((y_pred - label) ** 2)


def color2mse(y_pred, label, x, mask=False):
    loss=nn.L1Loss()
    if mask is False:
        return loss(y_pred,label)
    else:
        return loss(y_pred,label)

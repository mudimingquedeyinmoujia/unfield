import argparse
import ast

parser = argparse.ArgumentParser(description='Unfield')

##### for pre train
# mode 0: 使用facescape生成人脸形状进行shape的预训练，保存预训练模型与评估
# mode 1: 使用facescape的纹理图进行texture的预训练，保存预训练模型与评估
# mode 2: 加载预训练的形状模型进行生成预览
# mode 3: 加载预训练的纹理模型进行生成预览
parser.add_argument('-mo', '--premode', default=2, type=int, help="this is the pre training programme running mode")

# if use mode 2 or 3 to load ckpt
parser.add_argument('-cks', '--ckpts', default='2021-12-08-20_35_25_003000.tar', type=str, help="mode 2 ckpt shape name")
parser.add_argument('-ckt', '--ckptt', default='2021-12-08-11_03_28_003000.tar', type=str, help="mode 3 ckpt tex name")

# ckpt save epoch and eval epoch
parser.add_argument('-sep', '--saveepoch', default=1000, type=int, help="save model epoch")

# net work shape
parser.add_argument('-ds', '--depths', default=8, type=int, help="layers of shape net")
parser.add_argument('-ws', '--widths', default=256, type=int, help="width of shape net")
parser.add_argument('-es', '--embeds', default=0, type=int, help="if embedding of shape net")
parser.add_argument('-ms', '--muls', default=4, type=int, help="embedding multires of shape net")
parser.add_argument('-ls', '--lrates', default=1e-3, type=float, help="save dir of model")

# net work tex
parser.add_argument('-dt', '--deptht', default=8, type=int, help="layers of tex net")
parser.add_argument('-wt', '--widtht', default=256, type=int, help="width of tex net")
parser.add_argument('-et', '--embedt', default=0, type=int, help="if embedding of tex net")
parser.add_argument('-mt', '--mult', default=4, type=int, help="embedding multires of tex net")
parser.add_argument('-lt', '--lratet', default=1e-3, type=float, help="save dir of model")

# if continue to train
parser.add_argument('-re', '--reload', default=False, type=bool, help="if continue training")
# ckpt save dir (both shape and tex)
parser.add_argument('-sdr', '--savedir', default='./ckpts', type=str, help="save dir of model")




##### for reconstruction
# mode 0: 三维重建框架，会加载预训练模型
# mode 1: 假设模型已经训练好了，也就是网络表示这个mesh，那么将它可视化并保存至obj
parser.add_argument('-m', '--mode', default=1, type=int, help="this is the programme running mode")
parser.add_argument('-cka', '--ckpta', default='2021-12-12-20_11_27_000100.tar', type=str, help="mode 1 ckpt all name")
parser.add_argument('-od', '--objdir', default='./demo_output/final_results', type=str, help="mode 1 save model")





config, _ = parser.parse_known_args()

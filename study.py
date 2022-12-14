import torch
from vit_pytorch import ViT
from sklearn.metrics import cohen_kappa_score
# 创建ViT模型实例
v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 5,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)
# 随机化一个图像输入
img1 = torch.randn(1, 3, 256, 256)
img2 = torch.randn(1, 3, 256, 256)
# 获取输
y_pred = [1,1,1,1,1,6]
y_true = [1,1,1,1,1,5]

print(y_pred,y_true)
kappa = cohen_kappa_score(y_pred,y_true,weights = 'quadratic')
print(kappa)
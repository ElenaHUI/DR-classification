
# 准备数据集
from datetime import datetime
from torch import nn
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter
import os
import torch
from vit_pytorch import ViT, SimpleViT
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import time

# 创建ViT模型实例
#     v = ViT(
#         image_size = 224,
#         patch_size = 16,
#         num_classes = 5,
#         dim = 768,#线性投射
#         depth = 6,
#         heads = 12,
#         mlp_dim = 1024,
#         dropout = 0.1,
#         emb_dropout = 0.1
# )
def get_vit_model():
    v = SimpleViT(
        image_size = 512,
        patch_size = 32,
        num_classes = 5,
        dim = 256,
        depth = 2,
        heads = 4,
        mlp_dim = 128
    )
    return v

# # 随机化一个图像输入
# img = torch.randn(1, 3, 256, 256)
# # 获取输出
# preds = v(img) # (1, 1000)
# print(preds)
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total

# 数据集路径
class MyData(Dataset):
    def __init__(self, root_dir, lable_dir):
        self.root_dir = root_dir
        self.label_dir = lable_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        tensor_trans = torchvision.transforms.ToTensor()
        tensor_img = tensor_trans(img)
        tensor_resize = transforms.Resize((512, 512))
        tensor_img = tensor_resize(tensor_img)
        lable =  self.label_dir
        if(lable=='0'):lable=0
        if (lable == '1'): lable = 1
        if (lable == '2'): lable = 2
        if (lable == '3'): lable = 3
        if (lable == '4'): lable = 4
        lable = torch.tensor(lable)
        return tensor_img, lable
    def __len__(self):
        return len(self.img_path)

# 测试集中第一张数据
# writer = SummaryWriter("dataloader")
# step = 0
# for data in train_loader:
#     imgs, targets = data
#     writer.add_images("train data", imgs, step)
#     step = step + 1
# writer.close()
# 创建ViT模型实例
def train(net, train_data, valid_data, num_epochs, optimizer, criterion):
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:
            im = im.to(device)  # (bs, 3, h, w)
            # label = label.to(device)  # (bs, h, w)
            # forward
            output = net(im)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += get_acc(output, label)
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                im = im.to(device)  # (bs, 3, h, w)
                label = label.to(device)  # (bs, h, w)
                output = net(im)
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
            epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                    % (epoch, train_loss / len(train_data),
                       train_acc / len(train_data), valid_loss / len(valid_data),
                       valid_acc / len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)

def main():
    root_dir = "devidedata/train"
    lable_dir_0 = "0"
    lable_dir_1 = "1"
    lable_dir_2 = "2"
    lable_dir_3 = "3"
    lable_dir_4 = "4"
    dataset_0 = MyData(root_dir, lable_dir_0)
    dataset_1 = MyData(root_dir, lable_dir_1)
    dataset_2 = MyData(root_dir, lable_dir_2)
    dataset_3 = MyData(root_dir, lable_dir_3)
    dataset_4 = MyData(root_dir, lable_dir_4)
    train_dataset = dataset_0 + dataset_1 + dataset_2 + dataset_3 + dataset_4
    train_loader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True, num_workers = 0,
                              drop_last = False)
    # test_loader = train_loader
    # 使用预训练模型
    model = get_vit_model()
    # 查看总参数及训练参数
    total_params = sum(p.numel() for p in model.parameters())
    print('总参数个数:{}'.format(total_params))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()  # 损失函数
    # 只需要优化最后一层参数
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, weight_decay = 1e-3, momentum = 0.9)  # 优化器
    # train
    train(model, train_loader, train_loader, 20, optimizer, criterion)
    torch.save(model, "vit1")

if __name__ == '__main__':
    main()

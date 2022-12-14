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
from torch.nn import Conv2d, Sequential
from torch.utils.tensorboard import SummaryWriter
import os
import torch
from  sklearn.metrics import  f1_score
from vit_pytorch import ViT, SimpleViT
from sklearn.metrics import cohen_kappa_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import time
train_loss_figure = []
train_acc_figure = []
valid_acc_figure = []
valid_loss_figure = []
kappa_figure = []
f1_figure = []
def get_vit_model():
    v = Sequential(nn.Conv2d(3, 96, (11, 11), (4, 4)),  # in_channels, out_channels, kernel_size, stride, padding
                   nn.ReLU(),
                   nn.MaxPool2d(3, 2),  # kernel_size, stride
                   nn.Conv2d(96, 256, (5, 5), (1, 1), 2),
                   nn.ReLU(),
                   nn.MaxPool2d(3, 2),
                   nn.Conv2d(256, 384, (3, 3), (1, 1), 1),
                   nn.ReLU(),
                   nn.Conv2d(384, 384, (3, 3), (1, 1), 1),
                   nn.ReLU(),
                   nn.Conv2d(384, 256, (3, 3), (1, 1), 1),
                   nn.ReLU(),
                   nn.MaxPool2d(3, 2),
                   SimpleViT(image_size = 13,
                       channels = 256,
                       patch_size = 1,
                       num_classes = 5,
                       dim = 256,
                       depth = 2,
                       heads = 4,
                       mlp_dim = 128))

    return v

def picture():
    plt.figure(figsize = (14, 14), dpi = 100)
    x = range(0,300,1)

    plt.subplot(3, 2, 1)
    plt.ylim([0, 1])
    plt.plot(x, kappa_figure)
    plt.title("kappa")  # 设置标题
    plt.xlabel("epoch")  # 设置x轴标注
    plt.ylabel("kappa")  # 设置y轴标注

    plt.subplot(3, 2, 2)
    plt.plot(x, f1_figure)
    plt.ylim([0.4,1])
    plt.title("f1")  # 设置标题
    plt.xlabel("epoch")  # 设置x轴标注
    plt.ylabel("f1")  # 设置y轴标注

    plt.subplot(3, 2, 3)
    plt.plot(x, train_acc_figure)
    plt.ylim([0.4, 1])
    plt.title("Train accuracy")  # 设置标题
    plt.xlabel("epoch")  # 设置x轴标注
    plt.ylabel("accuracy")  # 设置y轴标注

    plt.subplot(3, 2, 4)
    plt.plot(x, train_loss_figure)
    plt.ylim([0.8, 1.6])
    plt.title("Train Loss")  # 设置标题
    plt.xlabel("epoch")  # 设置x轴标注
    plt.ylabel("loss")  # 设置y轴标注

    plt.subplot(3, 2, 5)
    plt.plot(x,valid_acc_figure)
    plt.ylim([0.4, 0.8])
    plt.title("Valid Accuracy")  # 设置标题
    plt.xlabel("epoch")  # 设置x轴标注
    plt.ylabel("accuracy")  # 设置y轴标注

    plt.subplot(3, 2, 6)
    plt.plot(x, valid_loss_figure)
    plt.ylim([0.8, 1.6])
    plt.title("Valid Loss")  # 设置标题
    plt.xlabel("epoch")  # 设置x轴标注
    plt.ylabel("loss")  # 设置y轴标注
    plt.savefig("subplot.png")
    plt.show()
def get_acc(output, label):
    total = output.shape[0]
    _ , pred_label = output.max(1)
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
        tensor_resize = transforms.Resize((227, 227))
        tensor_img = tensor_resize(img)
        tensor_img = tensor_trans(tensor_img)
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

# 创建ViT模型实例
def train(net, train_data, valid_data, num_epochs, optimizer, criterion):
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        y_true = []
        y_pred = []
        net = net.train()
        for im, label in train_data:
            im = im.to(device)  # (bs, 3, h, w)
            label = label.to(device)  # (bs, h, w)
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
                _, pred_label = output.max(1)
                y_pred.extend(pred_label.tolist())
                y_true.extend(label.tolist())
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
            valid_acc_figure.append(valid_acc / len(valid_data))
            valid_loss_figure.append( valid_loss / len(valid_data))
            train_acc_figure.append(train_acc / len(train_data))
            train_loss_figure.append( train_loss/len(train_data))
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
        labels = [0,1,2,3,4]
        kappa = cohen_kappa_score(y_true,y_pred,weights = 'quadratic')
        kappa_figure.append(kappa)
        f1 = f1_score(y_true,y_pred, labels=labels, average="micro")
        f1_figure.append(f1)
        print(epoch_str + time_str)
        print("kappa:",kappa,"f1:",f1)
def get_data(root_dir):
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
    dataset_sum =dataset_0 + dataset_1 + dataset_2 + dataset_3 + dataset_4
    return dataset_sum
def main():
    train_dir = "jupyter/download/train"
    val_dir="jupyter/download/val"
    train_dataset = get_data(train_dir)
    train_loader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True, num_workers = 0,
                              drop_last = False)
    val_dataset = get_data((val_dir))
    val_loader = DataLoader(dataset = val_dataset, batch_size = 64, shuffle = True, num_workers = 0,
                              drop_last = False)
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
    train(model, train_loader, val_loader,300, optimizer, criterion)
    torch.save\
        (model, "vit1")
    picture()


if __name__ == '__main__':
    main()
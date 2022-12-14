# 准备数据集
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from datetime import datetime
import torchvision.transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
from sklearn.metrics import cohen_kappa_score
from  sklearn.metrics import  f1_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loss_figure = []
train_acc_figure = []
valid_acc_figure = []
valid_loss_figure = []
kappa_figure = []
f1_figure = []
class LeNet(nn.Module):  # 继承于nn.Module这个父类
    def __init__(self):  # 初始化网络结构
        super(LeNet, self).__init__()  # 多继承需用到super函数
        self.conv1 = nn.Conv2d(3, 16, (5,5))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, (5,5))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):  # 正向传播过程
        x = F.relu(self.conv1(x))  # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)  # output(16, 14, 14)
        x = F.relu(self.conv2(x))  # output(32, 10, 10)
        x = self.pool2(x)  # output(32, 5, 5)
        x = x.view(-1, 32 * 5 * 5)  # output(32*5*5)
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(10)
        return x
def get_vit_model():
    x = LeNet()
    return x


# # 随机化一个图像输入
# img = torch.randn(1, 3, 256, 256)
# # 获取输出
# preds = v(img) # (1, 1000)
# print(preds)
def get_acc(output, label):
    total = output.shape[0]
    _ , pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total
def picture():
    plt.figure(figsize = (14, 14), dpi = 100)
    x = range(0,100,1)

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
        tensor_resize = transforms.Resize((32, 32))
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
    train_dir = "devidedata/train"
    val_dir="devidedata/val"
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
    train(model, train_loader, val_loader,100, optimizer, criterion)
    torch.save\
        (model, "le1")
    picture()


if __name__ == '__main__':
    main()
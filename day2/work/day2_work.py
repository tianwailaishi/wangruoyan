# 完整的模型训练套路(以CIFAR10为例)
import time

import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./dataset_chen",
                                          train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10(root="./dataset_chen",
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度{train_data_size}")
print(f"测试数据集的长度{test_data_size}")

# 加载数据集
train_loader = DataLoader(train_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)


# 创建网络模型
class alex(nn.Module):
    def __init__(self):
        super(alex, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        y = self.model(x)

        return y


chen = alex()
if torch.cuda.is_available():
    chen = chen.cuda()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
# learning_rate = 1e-2 相当于(10)^(-2)
learning_rate = 0.01
optim = torch.optim.SGD(chen.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
total_train_step = 0  # 记录训练的次数
total_test_step = 0  # 记录测试的次数
epoch = 20  # 训练的轮数

# 添加tensorboard
writer = SummaryWriter("../../logs_train")

# 添加开始时间
start_time = time.time()

for i in range(epoch):
    print(f"-----第{i + 1}轮训练开始-----")
    # 训练步骤
    for data in train_loader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = chen(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optim.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optim.step()

        total_train_step += 1
        if total_train_step % 500 == 0:
            print(f"第{total_train_step}的训练的loss:{loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤（以测试数据上的正确率来评估模型）
    total_test_loss = 0.0
    # 整体正确个数
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = chen(imgs)
            # 损失
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            # 正确率
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"整体测试集上的loss:{total_test_loss}")
    print(f"整体测试集上的正确率：{total_accuracy / test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy, total_test_step)
    total_test_step += 1

# 保存每一轮训练模型
torch.save(chen, f"model_save_alex\\chen_alex.pth")
print("模型已保存")

writer.close()


'''
-----第20轮训练开始-----
第15000的训练的loss:0.7844412922859192
第15500的训练的loss:0.8393945693969727
整体测试集上的loss:174.18701827526093
整体测试集上的正确率：0.6263999938964844
'''
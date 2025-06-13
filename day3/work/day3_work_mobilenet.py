# 完整的模型训练套路(以CIFAR10为例)
import time

import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import mobilenet_v2  # 导入MobileNetV2模型

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../dataset_chen",
                                          train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10(root="../dataset_chen",
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

# 创建MobileNetV2模型
chen = mobilenet_v2(pretrained=False, num_classes=10)  # 使用MobileNetV2

# 关键修改：调整第一层卷积的步长以适应32x32输入
# 原模型stride=2会导致尺寸过小，改为stride=1保持特征图尺寸
chen.features[0][0].stride = (1, 1)

if torch.cuda.is_available():
    chen = chen.cuda()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 0.01
optim = torch.optim.SGD(chen.parameters(), lr=learning_rate)

# 设置训练网络参数
total_train_step = 0
total_test_step = 0
epoch = 5

# 添加tensorboard
writer = SummaryWriter("../../logs_train")

start_time = time.time()

for i in range(epoch):
    print(f"-----第{i + 1}轮训练开始-----")

    # 训练步骤
    chen.train()  # 设置训练模式
    for data in train_loader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()

        outputs = chen(imgs)
        loss = loss_fn(outputs, targets)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_train_step += 1
        if total_train_step % 500 == 0:
            print(f"第{total_train_step}步的训练loss: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤
    chen.eval()  # 设置评估模式
    total_test_loss = 0.0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()

            outputs = chen(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()

            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = total_accuracy / test_data_size

    print(f"测试集loss: {avg_test_loss}")
    print(f"测试集正确率: {test_accuracy:.4f}")

    writer.add_scalar("test_loss", avg_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", test_accuracy, total_test_step)
    total_test_step += 1

# 保存模型
torch.save(chen.state_dict(), f"model_save/mobilenet.pth")
print("模型已保存")

end_time = time.time()
print(f"训练耗时: {end_time - start_time:.2f}秒")

writer.close()


'''
-----第30轮训练-----
第15000的训练的loss:1.3930941820144653
第15500的训练的loss:1.4757148027420044
整体测试集上的loss:1.5575771066033917
整体测试集上的正确率：0.8557
'''
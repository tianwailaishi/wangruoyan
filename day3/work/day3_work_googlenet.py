# 完整的模型训练套路(使用GoogleNet模型)
import time
import torch
import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
print(f"训练数据集的长度: {train_data_size}")
print(f"测试数据集的长度: {test_data_size}")

# 加载数据集
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)


# 定义基础卷积模块
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# 定义Inception模块
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        # 1x1卷积分支
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        # 1x1卷积 + 3x3卷积分支
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        # 1x1卷积 + 5x5卷积分支
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1),
            BasicConv2d(ch5x5, ch5x5, kernel_size=3, padding=1)  # 两个3x3卷积代替5x5卷积
        )

        # 3x3池化 + 1x1卷积分支
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


# 定义GoogleNet模型
class GoogleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogleNet, self).__init__()

        # 初始层（针对32x32图像调整）
        self.conv1 = BasicConv2d(3, 64, kernel_size=3, stride=1, padding=1)  # 保持32x32
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)  # 32x32 -> 16x16

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)  # 16x16 -> 8x8

        # Inception模块堆叠
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)  # 8x8 -> 4x4

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)  # 4x4 -> 2x2

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        # 平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)  # 1024 = 384 + 384 + 128 + 128

    def forward(self, x):
        # 初始层
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        # Inception模块堆叠
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        # 分类器
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


# 创建GoogleNet模型
def create_googlenet():
    return GoogleNet(num_classes=10)


# 创建模型
chen = create_googlenet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chen = chen.to(device)

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 0.01
optim = torch.optim.SGD(chen.parameters(), lr=learning_rate, momentum=0.9, weight_decay=4e-4)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.1)

# 设置训练参数
total_train_step = 0
total_test_step = 0
epoch = 50

# 添加tensorboard
writer = SummaryWriter("../../logs_train")

# 添加开始时间
start_time = time.time()

# 训练循环
for i in range(epoch):
    print(f"-----第 {i + 1} 轮训练开始-----")
    chen.train()  # 设置为训练模式

    # 训练步骤
    train_loss = 0.0
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = chen(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optim.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optim.step()

        train_loss += loss.item()
        total_train_step += 1

        if batch_idx % 100 == 0:
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 更新学习率
    scheduler.step()

    # 测试步骤
    chen.eval()  # 设置为评估模式
    total_test_loss = 0.0
    total_correct = 0

    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = chen(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == targets).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    avg_test_loss = total_test_loss / len(test_loader)
    accuracy = total_correct / test_data_size

    print(f"训练集平均Loss: {avg_train_loss:.4f}")
    print(f"测试集Loss: {avg_test_loss:.4f}")
    print(f"测试集正确率: {accuracy:.4f}")

    writer.add_scalar("train_avg_loss", avg_train_loss, i)
    writer.add_scalar("test_loss", avg_test_loss, i)
    writer.add_scalar("test_accuracy", accuracy, i)
    total_test_step += 1

    # 保存模型
    if (i + 1) % 10 == 0 or i == epoch - 1:
        torch.save({
            'epoch': i + 1,
            'model_state_dict': chen.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': avg_train_loss,
        }, f"model_save/googlenet_{i + 1}.pth")
        print(f"模型已保存: googlenet_{i + 1}.pth")

# 计算总时间
end_time = time.time()
total_time = end_time - start_time
print(f"训练完成! 总耗时: {total_time // 60:.0f}分 {total_time % 60:.0f}秒")

writer.close()

'''
-----第 50 轮训练开始-----
训练集平均Loss: 0.0013
测试集Loss: 0.6036
测试集正确率: 0.8717
'''

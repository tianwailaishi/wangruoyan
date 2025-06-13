# 完整的模型训练套路(使用ResNet模型)
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


# 定义基本残差块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# 定义完整的ResNet模型
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # 初始卷积层（针对CIFAR10的32x32图像调整）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个残差层
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# 创建ResNet-18模型
def create_resnet():
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)


# 创建模型
chen = create_resnet()
if torch.cuda.is_available():
    chen = chen.cuda()


# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 0.1
optim = torch.optim.SGD(chen.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[30, 60], gamma=0.1)

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

        train_loss += loss.item()
        total_train_step += 1

        if total_train_step % 500 == 0:
            print(f"训练次数: {total_train_step}, Loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 更新学习率
    scheduler.step()

    # 测试步骤
    chen.eval()  # 设置为评估模式
    total_test_loss = 0.0
    total_correct = 0

    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()

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
            'loss': loss.item(),
        }, f"model_save/resnet18_cifar10_{i + 1}.pth")
        print(f"模型已保存: resnet18_cifar10_{i + 1}.pth")

# 计算总时间
end_time = time.time()
total_time = end_time - start_time
print(f"训练完成! 总耗时: {total_time // 60:.0f}分 {total_time % 60:.0f}秒")

writer.close()


'''
-----第 50 轮训练-----
训练次数: 38500, Loss: 0.0082
训练次数: 39000, Loss: 0.0427
训练集平均Loss: 0.0458
测试集Loss: 0.7369
测试集正确率: 0.8147
'''
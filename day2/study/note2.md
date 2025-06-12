# DAY2 Note
## 深度学习基础

### 1.完整的深度学习训练套路
#### (1)数据准备‌：
- 数据收集‌：收集足够数量的、与任务相关的数据，这些数据可以来自公开数据集或自定义数据。
- 数据清洗‌：处理噪声和异常值，提高数据质量。
‌- 数据预处理‌：包括归一化、标准化、缩放等，以适应模型的需要。
‌- 数据划分‌：将数据划分为训练集、验证集和测试集，分别用于模型训练、验证和测试‌
#### (2)模型构建‌：
- 选择模型架构‌：根据任务需求和数据特点选择合适的模型架构，如卷积神经网络（CNN）、循环神经网络（RNN）等。
- 定义网络层‌：包括输入层、隐藏层和输出层。
- 设置超参数‌：如学习率、批量大小、迭代次数等，这些参数对模型训练效果有重要影响‌
#### (3)‌模型训练‌：
- 加载数据‌：将训练集和验证集加载到模型中。
- ‌定义损失函数‌：选择合适的损失函数，如交叉熵损失或均方误差损失。
- 选择优化器‌：如随机梯度下降（SGD）或Adam等，用于更新模型参数。
- ‌训练过程‌：通过迭代训练集，不断更新模型参数，同时使用验证集防止过拟合‌
#### (4)模型评估与调优‌：
- 模型评估‌：使用测试集评估模型性能，计算准确率、召回率等指标。
- 模型调优‌：根据评估结果调整模型架构、优化器或损失函数，以提高模型性能‌
#### (5)实际应用‌：
- 将训练好的模型应用到实际场景中，解决具体问题‌
- 训练一定是两次循环
- 欠拟合：训练训练数据集表现不好，验证表现不好
- 过拟合：训练数据训练过程表现得很好，在我得验证过程表现不好
### 2. 卷积神经网络(CNN)
`import torch`

`import torch.nn.functional as F`

`input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])`
                      
`kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])`
                       
`print(input.shape)`

`print(kernel.shape)`

`input = torch.reshape(input,(1,1,5,5))`

`kernel = torch.reshape(kernel,(1,1,3,3))`

`print(input.shape)`

`print(kernel.shape)`

`output = F.conv2d(input=input,weight=kernel,stride=1)`

`print(output)`

`output2 = F.conv2d(input=input,weight=kernel,stride=2)`

`print(output2)`

`output3 = F.conv2d(input=input,weight=kernel,stride=1,padding=1)`

`print(output3)`

### 3. tensorboard使用
- 使用之前安装一下tensorboard

- 这段代码的作用只是为了拿到我的conv_logs里面的文件

- 使用tensorboard命令打开`tensorboard --logdir=conv_logs`

### 4. 池化层
- 池化有最大池化和平均池化


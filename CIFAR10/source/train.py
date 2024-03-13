import torch
import torchvision
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from model import *

# 搞数据集
training_sets = torchvision.datasets.CIFAR10("../CIFAR10", train=True, transform=ToTensor(), download=True)
testing_sets = torchvision.datasets.CIFAR10("../CIFAR10", train=False, transform=ToTensor(), download=True)

# 用dataloader加载数据集
training_data = DataLoader(training_sets, batch_size=64)
testing_data = DataLoader(testing_sets, batch_size=64)

model = torch.load("../model/model1.pth")
# model = Cifar10Model()
model = model.cuda()
print("模型加载")

# 训练集上的训练次数
training_epoch = 10
# 损失函数
loss_fun = nn.CrossEntropyLoss()
loss_fun = loss_fun.cuda()
# 梯度优化
learning_rate = 1e-3
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model.train()

# 计数器
train_times = 1
train_rounds = 1
accurate_times = 0

for epoch in range(training_epoch):  # 训练集上的重复训练
    for data in training_data:
        img, target = data
        img = img.cuda()
        target = target.cuda()
        # 正向传播
        forward = model(img)
        # 计算损失
        loss = loss_fun(forward, target)
        # 反向传播
        # 先置零梯度
        optimizer.zero_grad()
        loss.backward()
        # 梯度下降
        optimizer.step()

        if train_times % 100 == 0:
            print("第{}轮，第{}次训练, loss为{}".format(train_rounds, train_times, loss))
        train_times = train_times + 1
    # 以下是验证
    with torch.no_grad():
        accurate_times = 0
        for data in testing_data:
            img, target = data
            img = img.cuda()
            target = target.cuda()

            output = model(img)
            # 这里好像是用向量在统计正确的个数
            accurate_times = accurate_times + (output.argmax(1) == target).sum()
        print("第{}轮正确率{}".format(train_rounds, accurate_times / len(testing_sets)))

    train_rounds = train_rounds + 1

torch.save(model, "../model/model1.pth")
print("模型保存")

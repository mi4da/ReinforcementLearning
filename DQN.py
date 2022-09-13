import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import sys
sys.path.append("FlappyBird/")
import random
import numpy as np
from collections import deque
import game
import os

# 定义系列常数，epsilon为每周期随机出动作的概率
GAME = "bird" # game name
ACTIONS = 2 #numbers of efficient actions
GAMMA = 0.99 # futural 衰减率 in reinforcement learning
# OBSERVE = 1e4 # TODO：训练之前的时间步，需要先观察10000 frames
OBSERVE = 1e3 # TODO：训练之前的时间步，需要先观察10000 frames
EXPLORE = 1e4 # 退火所需时间步，means epsilon going lower slowly.
INITIAL_EPSILON = 1e-1 # 开始e
FINAL_EPSILON = 1e-4 # 最终e
REPLAY_MEMORY = 5e4 # 最多记忆多少帧训练数据
BATCH = 32
FRAME_PER_ACTION = 1 # 每隔多少时间完成一次有效动作的输出
use_cuda = torch.cuda.is_available()

# Creating a multi layer CNN with 4 frames inputs and the output is Q value of each possible action.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # may 3D conv be better.
        # attention is the best chose.
        # filters(channels) = 32, kernel size = 8, stride = 4, padding = 2
        self.conv1 = nn.Conv2d(4,32,8,4,padding = 2)
        # Pooling层，窗口2*2
        self.pool = nn.MaxPool2d(2, 2)
        # filters(channels) = 64, kernel size = 4, stride = 2, padding = 1
        self.conv2 = nn.Conv2d(32, 64, 4, 2, padding=1)
        # filters(channels) = 64, kernel size = 3, stride = 1, padding = 1
        self.conv3 = nn.Conv2d(64, 64, 3, 1, padding=1)

        # last with mlps
        self.fc_sz = 1600
        self.fc1 = nn.Linear(self.fc_sz, 256)
        self.fc2 = nn.Linear(256, ACTIONS)

    def forward(self, x):
        # batch = [32,4,80,80]
        x = self.conv1(x)
        # [bz, 32, 20, 20]
        x = F.relu(x)
        # [bz, 32, 10, 10]
        x = self.pool(x)
        # [bz, 64, 5, 5]
        x = F.relu(self.conv2(x))
        # [bz, 65, 5, 5]
        x = F.relu(self.conv3(x))
        # transfer the x into 1600 dimensional vectors.
        x = x.view(-1, self.fc_sz)
        x = F.relu(self.fc1(x))
        readout = self.fc2(x)
        return readout, x

    def init(self):
        # initiate all weights
        self.conv1.weight.data = torch.abs(.01 * torch.randn(self.conv1.weight.size()))
        self.conv2.weight.data = torch.abs(.01 * torch.randn(self.conv2.weight.size()))
        self.conv3.weight.data = torch.abs(.01 * torch.randn(self.conv3.weight.size()))
        self.fc1.weight.data = torch.abs(0.01 * torch.randn(self.fc1.weight.size()))
        self.fc2.weight.data = torch.abs(0.01 * torch.randn(self.fc2.weight.size()))
        self.conv1.bias.data = torch.ones(self.conv1.bias.size()) * 0.01
        self.conv2.bias.data = torch.ones(self.conv2.bias.size()) * 0.01
        self.conv3.bias.data = torch.ones(self.conv3.bias.size()) * 0.01
        self.fc1.bias.data = torch.ones(self.fc1.bias.size()) * 0.01
        self.fc2.bias.data = torch.ones(self.fc2.bias.size()) * 0.01

# initial a net
net = Net()
# weight initiative
net.init()
# gpu setting
net = net.cuda() if use_cuda else net
# def the loss as MSE
criterion = nn.MSELoss().cuda() if use_cuda else nn.MSELoss()
# def the optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-6)

# starting a game process
game_state = game.GameState()

# 使用双端队列存储学习样本
D = deque()

# 80*80的游戏湖面，进行灰度图像二值化处理
do_nothing = np.zeros(ACTIONS)
do_nothing[0] = 1
x_t, r_0, terminal = game_state.frame_step(do_nothing)
x_t = cv2.cvtColor(cv2.resize(x_t, (80,80)), cv2.COLOR_BGR2GRAY)# 转换大小以及灰度
ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)# TODO: 不知道第一个返回值是啥意思？总之就是个二值化函数

# 将游戏画面叠加成4frames传入
s_t = np.stack(4*(x_t,), axis=0)

# 设置出视的epsilon(采取随即行动的概率)，并准备训练
epsilon = INITIAL_EPSILON
t = 0
"""
DQN-action-value based learning with neural network.
该算法分为三个阶段：

1、按照Epsilon贪婪算法采取一次行动；
2、将选择好的行动输入给游戏引擎，得到下一帧的状态，并生成本帧的训练数据
3、开始训练：
"""
# 记录每轮平均得分的容器
scores =[]
all_turn_scores = []
while "flappy bird" != "angry bird":# 进入游戏循环
    ####首先按照贪婪策略选择行动####
    s = torch.from_numpy(s_t).type(torch.FloatTensor)# 转换类型
    s = s.cuda() if use_cuda else s
    s = s.view(-1, s.size()[0], s.size()[1], s.size()[2]) # TODO: 重构张量，-1代表自己推断维度
    # 获取当前时刻的游戏画面，
    readout, h_fc1 = net(s)
    # net的输出为readout， 选择每一个行动的预期Q值
    readout = readout.cpu() if use_cuda else readout
    # readout是一个二维向量，分别对应每一个动作的预期Q值
    readout_t = readout.data.numpy()[0]

    # 按照epsilon贪婪策略产生小鸟的行动，即以epsilong的概率随即输出行动
    # 或者以1-epsilon的概率按照预期输出最大的Q值来行动
    a_t = np.zeros([ACTIONS])
    action_index = 0
    if t % FRAME_PER_ACTION == 0:
        # 如果当前帧可以行动
        if random.random() <= epsilon:
            # 产生随即行动
            action_index = random.randrange(ACTIONS)
        else:
            # 选择神经网络判断的预期Q最大的行动
            action_index = np.argmax(readout_t)
        a_t[action_index] = 1
    else:
        a_t[0] = 1 #do nothing

    # 对epsilon进行模拟退火
    if epsilon > FINAL_EPSILON and t > OBSERVE:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

    #### 将选择好的行动输入游戏引擎 ####
    x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
    # 返回的x_t1_colored为游戏画面，r_t为本轮的得分，terminal为游戏在本轮是否已经结束

    # 记录每一步的成绩
    scores.append(r_t)
    if terminal:
        # 当游戏结束，计算一下本轮成绩，并将总成绩存储到all_turn_scores中
        all_turn_scores.append(sum(scores))
        scores = []

    # 对游戏画面做预处理
    x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored,(80,80)),
                        cv2.COLOR_BGR2GRAY)
    ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
    x_t1 = np.reshape(x_t1, (1,80,80))# TODO: reshape和resize有啥区别来着，可以复习一下
    # 将当前frame与前三帧合并作为Agent获得的环境反馈结果
    s_t1 = np.append(x_t1, s_t[:3, :, :], axis=0)# TODO：也就是新的环境状态还有只有一个新的状态？其他三个都是老的？
    # 生成一个训练数据，分别将本帧的输入画面s_t,本帧的行动a_t，
    # 得到的环境回报r_t,环境被转换的新状态s_t1,游戏是否结束存到D中
    D.append((s_t, a_t, r_t, s_t1, terminal))
    if len(D) > REPLAY_MEMORY:
        # 如果D中的元素已满，则扔掉最老的训练数据
        D.popleft() # TODO: 和pop()有什么区别呢

    #### 当运行周期超过一定次数后开始使用TD算法训练神经网络 ####
    if t > OBSERVE:
        # 从D中随机采样出一个batch的训练数据
        # TODO 我一直好奇，这样随机采样的数据不是不具备时序性吗？
        minibatch = random.sample(D, BATCH)
        optimizer.zero_grad() # 清空梯度

        # 将这个batch中的s变量都分别存放到列表中
        # TODO：这种写法真是太糟糕了，四个循环，直接用数组分块不是更快吗
        s_j_batch = [d[0] for d in minibatch]
        a_batch = [d[1] for d in minibatch]
        r_batch = [d[2] for d in minibatch]
        s_j1_batch = [d[3] for d in minibatch]

        # 根据s_j1_batch, 神经网络给出预估的未来Q值
        s = torch.tensor(np.array(s_j1_batch,dtype=float),dtype=torch.float)
        s = s.cuda() if use_cuda else s
        readout, h_fc1 = net(s)
        readout = readout.cpu() if use_cuda else readout
        readout_j1_batch = readout.data.numpy()
        # readout_j1_batch 中存储了一个minibatch中所有未来一步的Q预估值
        # 根据Q的预估值，反馈r，以及游戏是否结束，更新等待训练的目标函数值
        y_batch = []
        for i in range(0, len(minibatch)):
            terminal = minibatch[i][4]
            # 当游戏结束的时候，则用当前的r作为目标，否则就用下一状态的预估Q+当前的r
            if terminal:
                y_batch.append(r_batch[i])
            else:
                y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
        # 开始梯度跟新
        y = torch.tensor(y_batch, dtype=torch.float, requires_grad=True)
        a = torch.tensor(a_batch, dtype=torch.float, requires_grad=True)
        s = torch.tensor(np.array(s_j_batch, dtype=float), dtype=torch.float, requires_grad=True)
        if use_cuda:
            y = y.cuda()
            s = s.cuda()
            a = a.cuda()
        # 计算s_j_batch的Q -> qt
        readout, h_fc1 = net(s)
        readout_action = readout.mul(a).sum(1)# TODO：为啥这里要和a做点乘？qt sum(axis=1) 求数组行和
        # 计算qt与yt的差作为损失函数
        loss = criterion(readout_action, y)
        loss.backward()
        optimizer.step()
        if t % 1000 == 0:
            print("损失函数：", loss)

    # 状态更新，时间步+1
    s_t = s_t1
    t += 1

    # 每隔 10000 次循环，存储一下网络,在本机上测试由于权限问题无法写入
    # if t % 100 == 0:
    #     # with open('saving_nets/' + GAME + '-dqn' + str(t) + '.txt',mode="w") as f:
    #     file_path = 'saving_nets/' + GAME + '-dqn' + str(t) + '.txt'
    #     if not os.path.exists(file_path):
    #         os.makedirs(file_path)
    #         torch.save(net, file_path)

    # 状态信息的转化，基本分为Observe，explore和train三个阶段
    # Observe没有训练，explore开始训练，并且开始模拟退火，train模拟退火结束
    state = ""
    if t <= OBSERVE:
        state = "observe"
    elif t > OBSERVE and t <= OBSERVE + EXPLORE:
        state = "explore"
    else:
        state = "train"

    # 打印当前运行的一些基本数据，分别输出到屏幕以及log文件中
    if t % 1000 == 0:
        sss = "时间步 {}/ 状态 {}/ Epsilon {:.2f}/ 行动 {}/ 奖励 {}/ Q_MAX {:e}/ 轮得分 {:.2f}".format(
            t, state, epsilon, action_index, r_t, np.max(readout_t), np.mean(all_turn_scores[-1000:]))
        print(sss)
        f = open('log_file.txt', 'a')
        f.write(sss + '\n')
        f.close()
    # write info to files







# -*- coding: utf-8 -*-
"""
@author: Chen Zhonghao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义一个残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels=256):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

class PriorHead(nn.Module):
    def __init__(self, in_channel, hidden_channel=2, output_size=3*3):
        super(PriorHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, hidden_channel, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(hidden_channel)
        self.linear1 = nn.Linear(hidden_channel*output_size, output_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        return out

# 定义整个网络
class ResNet(nn.Module):
    def __init__(self, num_blocks, in_channels=256, output_size=3**2):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(1, in_channels, kernel_size=3, padding=1)  # 棋盘可以看作单通道的图像
        self.bn = nn.BatchNorm2d(in_channels)
        self.res_blocks = self._make_layer(num_blocks, in_channels)
        self.prior_head = PriorHead(in_channels, 2, output_size)
        # 可以根据需要添加更多的层

    def _make_layer(self, num_blocks, in_channels):
        layers = []
        for i in range(num_blocks):
            layers.append(ResidualBlock(in_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.res_blocks(out)
        out = self.prior_head(out)
        return out

    def infer(self, x):
        with torch.no_grad():
            out = self.conv(x)
            out = self.bn(out)
            out = F.relu(out)
            out = self.res_blocks(out)
            out = self.prior_head(out)
            out = torch.softmax(out, dim=1)
        return out

    def predict_move(self, x):
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        out = self.infer(x)
        available_moves = x.reshape(1, -1) == 0
        move = int(torch.argmax(torch.softmax(out, dim=1) * available_moves))
        return move


class MovePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(10*torch.exp(x))
        out = torch.relu(out)
        out = self.fc2(out)
        # out = torch.sigmoid(out)
        return out


if __name__ == '__main__':
    # 棋盘参数
    board_size = 3
    win_condition = 3
    # 实例化网络
    num_blocks = 1  # 根据你的需求设置残差块的数量
    net = ResNet(num_blocks, 256)
    net.load_state_dict(torch.load('ckpts/33_iter100/MovePredictor_4999.pth'))
    net.eval()
    x = torch.tensor([
        [1, 0, -1],
        [0, 1, 1],
        [0, 0, -1]
    ], dtype=torch.float32).reshape(1, 1, 3, 3)
    print(net.predict_move(x))
    # out = net(x)
    # print(torch.softmax(out, dim=1))
    # available_moves = x.reshape(1, -1) == 0
    # print(int(torch.argmax(torch.softmax(out, dim=1)*available_moves)))

    # # 使用训练的模型作为白方进行对局，黑方随机下，统计胜率
    # stats = {}
    # for _ in range(1000):
    #     board_state = BoardState(board_size, board_size, np.zeros((board_size, board_size)), -1, win_condition)
    #     while not board_state.is_terminal():
    #         random_move = random.choice(board_state.get_legal_moves())
    #         board_state.do_move(random_move)
    #         if board_state.is_terminal():
    #             break
    #         x = torch.tensor(board_state.board, dtype=torch.float32).reshape(1, 1, 3, 3)
    #         available_moves = x.reshape(1, -1) == 0
    #         ai_prediction = net(x)
    #         ai_selection = int(torch.argmax(torch.softmax(ai_prediction, dim=1)*available_moves))
    #         ai_move = (ai_selection//board_size, ai_selection%board_size)
    #         board_state.do_move(ai_move)
    #     stats.setdefault(int(board_state.is_terminal()), 0)
    #     stats[int(board_state.is_terminal())] += 1
    # print('黑棋胜率', stats[1] / sum(stats.values()))
    # print('白棋胜率', stats[-1] / sum(stats.values()))
    # print('和棋概率', stats[2] / sum(stats.values()))

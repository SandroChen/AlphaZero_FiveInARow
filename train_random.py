# -*- coding: utf-8 -*-
"""
@author: Chen Zhonghao
"""

import torch
import numpy as np
import random
import tqdm
random.seed(1234)
from copy import deepcopy
from game import Gomoku
from mcts import Node, mcts
from models import ResNet
from torch.utils.tensorboard import SummaryWriter

# 由一个player自我对弈，产生对局，这些棋局用于训练神经网络；
# 每个iteration都会产生一个新的player，这个player会与best player对弈，如果胜率高于55%，则替换之；


if __name__ == '__main__':
    device = 'cpu'
    # 棋盘参数
    board_size = 6
    win_condition = 4
    black = 1
    white = -1
    # mcts参数
    n_iters = 100
    # model = MovePredictor(board_size**2, 128, board_size**2).to(device)
    model = ResNet(1, 256, board_size**2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter('runs/move_predictor')
    # 训练参数
    n_epoch = 20000
    save_epoch = 1000
    i_train_step = 0
    i_step = 0
    batch_size = 8
    batch_inputs = []
    batch_gt = []
    # player是一个mct，进行自我对弈，产生对局，用于训练。
    # 但为了防止每次都选择最佳节点，因而导致只选择一种走法、忽略其他节点，所以黑棋随机走，白棋则选用最佳节点
    root = Node(white, None, Gomoku(np.zeros((board_size, board_size)), white, win_condition))
    # 统计胜率
    stats = {}
    # 一个epoch走完一整盘棋
    for epoch in tqdm.tqdm(range(n_epoch)):
        # 初始化棋盘
        player = 1
        node = root
        node_parents = {}
        while not node.game.is_terminal():
            # 复制一份棋盘，不改变node节点自身的棋盘状态
            board_state = deepcopy(node.game)
            # print(f'board: \n{board_state.board}')
            # 黑棋选择随机走
            random_move = random.choice(board_state.get_legal_moves())
            board_state.do_move(random_move)
            # node = update_by_board(node, board_state.board, random_move)
            node = node.select_with_move(random_move)
            random_node = node
            # print(f'random_move: {node.move}')
            if node.game.is_terminal():
                break
            # 白棋根据mcts选择下一步，并将下一步的节点当作根节点，暂时忽略上层节点，backpropagation只回传至当前根节点。
            node_parents[node] = node.parent
            node.parent = None
            prior_board = deepcopy(node.game.board)
            node = mcts(node, n_iters)
            best_move = node.move
            # print(f', board: \n{prior_board}, \nbest_move: {best_move}')
            # 更新模型
            i_step += 1
            batch_inputs.append(prior_board)
            # move_gt = torch.tensor([board_size * best_move[0] + best_move[1]], dtype=torch.long).to(device)
            batch_gt.append(board_size * best_move[0] + best_move[1])
            if i_step != 0 and i_step % batch_size == 0:
                i_train_step += 1
                input_tensor = torch.tensor(batch_inputs, dtype=torch.float32).reshape(batch_size, 1, board_size, board_size).to(device)
                move_predicted = model(input_tensor).to(device)
                move_gt = torch.tensor(batch_gt, dtype=torch.long).to(device)
                loss = criterion(move_predicted, move_gt)
                loss.backward()
                print(f'epoch: {epoch}, step: {i_train_step}, loss:{loss}')
                writer.add_scalar('traning loss', loss.item(), i_train_step)
                optimizer.step()
                optimizer.zero_grad()
                # 清空记忆
                batch_inputs = []
                batch_gt = []
        # 重新将子节点连接至原父节点
        for x in node_parents:
            x.parent = node_parents[x]
        del node_parents
        stats.setdefault(node.game.is_terminal(), 0)
        stats[node.game.is_terminal()] += 1
        # print('黑棋首步走某步之后，白棋的胜率：')
        # for child in random_node.children:
        #     print(child.move, child.visit_count, child.mean_action_value, child.uct())
        # 保存模型
        if epoch % save_epoch == 0:
            torch.save(model.state_dict(), f'ckpts/MovePredictor_{epoch}.pth')

    # 训练结束时也保存
    torch.save(model.cpu().state_dict(), f'ckpts/MovePredictor_{epoch}.pth')


    print('胜率统计')
    for k in stats:
        print(f'赢家: {k}, 次数: {stats[k]}, 胜率: {stats[k] / sum(stats.values())}')

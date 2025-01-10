# -*- coding: utf-8 -*-
"""
@author: Chen Zhonghao
"""

import math
import random
import numpy as np
import torch
from copy import deepcopy
from operator import itemgetter
from game import Game
from models import ResNet


class Node:
    def __init__(self, parent, game, move=None):
        self.parent: Node = parent
        self.total_action_value = 0
        self.visit_count = 0
        self.game: Game = game
        self.children = []
        self.move = move

    def get_leaves(self, level):
        if self.children:
            leaves = []
            for child in self.children:
                leaves.extend(child.get_leaves(level+1))
            return leaves
        return [(self, level)]

    def select(self, use_network=False, network=None):
        if len(self.children) == 0:
            return self
        max_ucb = -math.inf
        selected_children = []
        if use_network and network is not None:
            network: ResNet
            p_list = network.infer(torch.tensor(self.game.board, dtype=torch.float32).unsqueeze(0).unsqueeze(0)).squeeze().tolist()
        else:
            p_list = [1 for _ in range(len(self.children))]
        for i, child in enumerate(self.children):
            value = child.get_value(p=p_list[i])
            if value > max_ucb:
                max_ucb = value
                selected_children = [child]
            elif value == max_ucb:
                selected_children.append(child)
        selected_child = random.choice(selected_children)
        return selected_child.select(use_network, network)

    def expansion(self):
        if self.children or self.game.is_terminal():
            return
        for move in self.game.get_legal_moves():
            game: Game = deepcopy(self.game)
            game.do_move(move)
            child = Node(self, game, move)
            self.children.append(child)

    def rollout(self):
        # start monte carlo simulation, totally random
        game = deepcopy(self.game)
        while not game.is_terminal():
            legal_moves = game.get_legal_moves()
            if len(legal_moves) > 0:
                random_move = random.choice(legal_moves)
                game.do_move(random_move)
            else:
                break
        if self.game.current_player == game.is_terminal():
            action_value = 1
        # 和棋只加0.5, 这可以使和棋在当前节点和父节点对胜率的影响一致。
        elif self.game.current_player == -game.is_terminal():
            action_value = -1
        else:
            action_value = 0
        return action_value

    def backpropagation(self, action_value):
        # update
        self.total_action_value += action_value
        self.visit_count += 1
        # backpropagation
        if self.parent is not None:
            self.parent.backpropagation(-action_value)

    def puct(self, c=5, p=1):
        """
        :param c: exploration constant
        :param p: prior winning probability from neural network, set to 1 if not using network
        """
        return self.mean_action_value + c * p * math.sqrt(self.parent.visit_count) / (self.visit_count + 1)

    def get_value(self, c=5, p=1):
        return self.puct(c, p) + self.mean_action_value

    @property
    def mean_action_value(self):
        return self.total_action_value / (self.visit_count + 1)

    def select_best_node(self):
        try:
            best_node = max([(child, child.mean_action_value) for child in self.children], key=itemgetter(1))[0]
        except:
            best_node = None
        finally:
            return best_node

    def select_random_node(self):
        return random.choice(self.children)

    def select_with_move(self, move):
        game = deepcopy(self.game)
        game.do_move(move)
        if len(self.children) > 0:
            child: Node
            for child in self.children:
                if np.array_equal(child.game.board, game.board):
                    return child
        new_child = Node(self, game)
        self.children.append(new_child)
        return new_child

    def __str__(self):
        return f"board:{self.game.board}\nvisit_count:{self.visit_count}\nmean_action_value:{self.mean_action_value}"


class Player:
    def __init__(self, network: ResNet):
        self.network = network

    def play(self, game: Game, n_iter: int):
        root = Node(None, game)
        mcts_node = mcts(root, n_iter, use_network=True, network=self.network)
        return mcts_node.move


# monte carlo tree search
def mcts(root: Node, n_iters: int, random_node=False, use_network=False, network=None) -> Node:
    for _ in range(n_iters):
        leaf = root.select(use_network, network)
        leaf.expansion()
        value = leaf.rollout()
        leaf.backpropagation(value)
    if random_node:
        return root.select_random_node()
    else:
        return root.select_best_node()


def test_func():
    from tqdm import tqdm
    from game import Gomoku
    # 棋盘参数
    board_size = 3
    win_condition = 3
    stats = {}
    stats_chunks = [{}]
    win_steps = []
    root = Node(-1, None, Gomoku(np.zeros((board_size, board_size)), -1, win_condition), move=None)
    for i in tqdm(range(5000)):
        node = root
        board_state = deepcopy(root.game)
        node_parents = {}
        while not board_state.is_terminal():
            random_move = random.choice(board_state.get_legal_moves())
            board_state.do_move(random_move)
            # node = update_by_board(node, board_state.board, random_move)
            node = node.select_with_move(random_move)
            if board_state.is_terminal():
                break
            # 根据mcts选择下一步，并将下一步的节点当作根节点，暂时忽略上层节点，backpropagation只回传至当前根节点。
            node_parents[node] = node.parent
            node.parent = None
            print(node)
            node = mcts(node, 100)
            ai_move = node.move
            board_state.do_move(ai_move)
        # 重新将子节点连接至原父节点
        # for node in node_parents:
        #     node.parent = node_parents[node]
        del node_parents
        stats.setdefault(int(board_state.is_terminal()), 0)
        stats[int(board_state.is_terminal())] += 1
        stats_chunks[-1].setdefault(int(board_state.is_terminal()), 0)
        stats_chunks[-1][int(board_state.is_terminal())] += 1
        if int(board_state.is_terminal()) == -1:
            win_steps.append(i)
        if (i+1) % 100 == 0:
            stats_chunks.append({})
    print('黑棋胜率', stats.get(1, 0) / sum(stats.values()))
    print('白棋胜率', stats.get(-1, 0) / sum(stats.values()))
    print('和棋概率', stats.get(2, 0) / sum(stats.values()))

    import matplotlib.pyplot as plt
    while {} in stats_chunks:
        stats_chunks.remove({})
    plt.plot([chunk.get(1, 0)/sum(chunk.values()) for chunk in stats_chunks])
    plt.show()
    plt.plot([chunk.get(-1, 0) / sum(chunk.values()) for chunk in stats_chunks])
    plt.show()

    # plt.scatter(win_steps, [1 for _ in win_steps], s=1)
    # plt.show()
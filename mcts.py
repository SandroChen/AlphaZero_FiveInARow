# -*- coding: utf-8 -*-
"""
@author: Chen Zhonghao
"""

import math
import random
import numpy as np
import hashlib
from copy import deepcopy
from operator import itemgetter


def get_hash(board):
    return hashlib.sha256(board).hexdigest()


class Node:
    def __init__(self, player, parent, board_state, move=None):
        self.player = player
        self.parent: Node = parent
        self.total_action_value = 0
        self.visit_count = 0
        self.board_state: BoardState = board_state
        self.children = []
        self.move = move

    def get_leaves(self, level):
        if self.children:
            leaves = []
            for child in self.children:
                leaves.extend(child.get_leaves(level+1))
            return leaves
        return [(self, level)]

    def select(self):
        if len(self.children) == 0:
            return self
        max_ucb = -math.inf
        selected_children = []
        for child in self.children:
            value = child.uct() + child.mean_action_value
            if value > max_ucb:
                max_ucb = value
                selected_children = [child]
            elif value == max_ucb:
                selected_children.append(child)
        selected_child = random.choice(selected_children)
        return selected_child.select()

    # def select(self):
    #     if len(self.children) == 0:
    #         return self
    #     # 获取所有叶子节点，并附带层数信息
    #     leaves = self.get_leaves(0)
    #
    #     # 找到最浅的层数
    #     min_level = min(level for _, level in leaves)
    #
    #     # 过滤出所有最浅层的叶子节点
    #     shallowest_leaves = [node for node, level in leaves if level == min_level]
    #
    #     selected_leaves = []
    #     max_ucb = 0
    #     for i, leaf in enumerate(shallowest_leaves):
    #         # 根据ucb选取叶子节点时，不应选择对当前player有利的节点，这相当于把对手想成一个蠢货，而非高手。
    #         # 因此在计算ucb时，无需对player与当前节点不一致的叶子节点，进行win_rate=1-win_rate。
    #         leaf_ucb = leaf.ucb()
    #         if leaf_ucb > max_ucb:
    #             selected_leaves = [leaf]
    #             max_ucb = leaf_ucb
    #         elif leaf_ucb == max_ucb:
    #             selected_leaves.append(leaf)
    #
    #     selected_leaf = random.choice(selected_leaves)
    #     return selected_leaf

    def expansion(self):
        if self.children or self.board_state.is_terminal():
            return
        for move in self.board_state.get_legal_moves():
            board_state: BoardState = deepcopy(self.board_state)
            board_state.do_move(move)
            child = Node(-self.player, self, board_state, move)
            self.children.append(child)

    def rollout(self):
        # start monte carlo simulation, totally random
        board_state = deepcopy(self.board_state)
        while not board_state.is_terminal():
            legal_moves = board_state.get_legal_moves()
            if len(legal_moves) > 0:
                random_move = random.choice(legal_moves)
                board_state.do_move(random_move)
            else:
                break
        if self.player == board_state.is_terminal():
            action_value = 1
        # 和棋只加0.5, 这可以使和棋在当前节点和父节点对胜率的影响一致。
        elif self.player == -board_state.is_terminal():
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

    def uct(self, c=5):
        """
        :param c: exploration constant, 由于n=simulation_times对计算会产生影响，或许应该考虑c = c * math.sqrt(n/log(n))
        """
        # modified_constant = math.sqrt(simulation_times/math.log(simulation_times))
        return self.mean_action_value + c * math.sqrt(math.log(self.parent.visit_count) / (self.visit_count + 1))

    @property
    def mean_action_value(self):
        return self.total_action_value / (self.visit_count + 1)

    def select_best_node(self):
        return max([(child, child.mean_action_value) for child in self.children], key=itemgetter(1))[0]

    def select_random_node(self):
        return random.choice(self.children)

    def select_with_move(self, move):
        board_state = deepcopy(self.board_state)
        board_state.do_move(move)
        if len(self.children) > 0:
            child: Node
            for child in self.children:
                if np.array_equal(child.board_state.board, board_state.board):
                    return child
        new_child = Node(-self.player, self, board_state)
        self.children.append(new_child)
        return new_child

    def __str__(self):
        return f"board:{node.board_state.board}\nvisit_count:{node.visit_count}\nmean_action_value:{self.mean_action_value}"


class BoardState:
    def __init__(self, board=None, current_player=1, win_condition=3, move=None):
        self.board: np.ndarray = np.array(board)
        self.current_player = current_player
        self.win_condition = win_condition
        if move is None:
            self.move = [0, 0]
        else:
            self.move = move

    def do_move(self, move):
        """
        :param move: (row, col, player)
        """
        self.current_player = -self.current_player
        self.board[move[0]][move[1]] = self.current_player
        self.move = move

    def get_legal_moves(self):
        return np.argwhere(self.board == 0)

    def check_in_a_row(self, line):
        """
        win_condition: the number of consecutive stones to win
        """
        if line is None or len(line) == 0 or line[0] == 0:
            return 0
        for i in range(1, self.win_condition):
            if line[i] != line[0]:
                return 0
        return line[0]

    def is_terminal(self):
        """
        win_condition: the number of consecutive stones to win
        """
        # win_condition = self.win_condition
        # size = len(self.board)
        # # 检查行和列
        # for i in range(size):
        #     for j in range(size - (win_condition-1)):
        #         if winner := self.check_in_a_row(self.board[i][j:j + win_condition], win_condition):
        #             return winner
        #         if winner := self.check_in_a_row([self.board[k][i] for k in range(j, j + win_condition)], win_condition):
        #             return winner
        # # 检查主对角线
        # for i in range(size - (win_condition-1)):
        #     for j in range(size - (win_condition-1)):
        #         if winner := self.check_in_a_row([self.board[i + k][j + k] for k in range(win_condition)], win_condition):
        #             return winner
        # # 检查副对角线
        # for i in range(size - (win_condition-1)):
        #     for j in range((win_condition-1), size):
        #         if winner := self.check_in_a_row([self.board[i + k][j - k] for k in range(win_condition)], win_condition):
        #             return winner
        # if len(self.get_legal_moves()) == 0:
        #     return 2
        # return 0

        # M = len(self.board)
        # rowcum = np.cumsum(self.board, 0)
        # colcum = np.cumsum(self.board, 1)
        # rowsum = rowcum[self.win_condition-1:, :] - np.vstack((np.zeros(M), rowcum[:M - self.win_condition, :]))
        # colsum = colcum[:, self.win_condition-1:] - np.hstack((np.zeros((M, 1)), colcum[:, :M - self.win_condition]))
        # diag_tl = np.array([
        #     self.board[i:i + self.win_condition, j:j + self.win_condition]
        #     for i in range(M - (self.win_condition-1))
        #     for j in range(M - (self.win_condition-1))
        # ])
        # diag_sum_tl = diag_tl.trace(axis1=1, axis2=2)
        # diag_sum_tr = diag_tl[:, ::-1].trace(axis1=1, axis2=2)
        # player_one_wins = self.win_condition in rowsum
        # player_one_wins += self.win_condition in colsum
        # player_one_wins += self.win_condition in diag_sum_tl
        # player_one_wins += self.win_condition in diag_sum_tr
        # if player_one_wins:
        #     return 1
        # player_two_wins = -self.win_condition in rowsum
        # player_two_wins += -self.win_condition in colsum
        # player_two_wins += -self.win_condition in diag_sum_tl
        # player_two_wins += -self.win_condition in diag_sum_tr
        # if player_two_wins:
        #     return -1
        # if np.all(self.board != 0):
        #     return 2
        # return 0

        x, y = self.move[0], self.move[1]
        for i in range(max(0, x-self.win_condition+1), min(x+1, len(self.board)-self.win_condition+1)):
            sum_value = sum(self.board[i:i+self.win_condition, y])
            if sum_value == 1*self.win_condition:
                return 1
            elif sum_value == -1*self.win_condition:
                return -1
        for j in range(max(0, y-self.win_condition+1), min(y+1, len(self.board)-self.win_condition+1)):
            sum_value = sum(self.board[x, j:j+self.win_condition])
            if sum_value == 1*self.win_condition:
                return 1
            elif sum_value == -1*self.win_condition:
                return -1
        diag_tl = self.board.diagonal(offset=abs(x-y))
        diag_index = min(x, y)
        for k in range(max(0, diag_index-self.win_condition+1), min(diag_index+1, len(self.board)-self.win_condition+1)):
            sum_value = sum(diag_tl[k:k+self.win_condition])
            if sum_value == 1*self.win_condition:
                return 1
            elif sum_value == -1*self.win_condition:
                return -1
        diag_tr = self.board[::-1].diagonal(offset=abs(len(self.board)-1-x-y))
        diag_index = min(len(self.board)+1-x, y)
        for k in range(max(0, diag_index-self.win_condition+1), min(diag_index+1, len(self.board)-self.win_condition+1)):
            sum_value = sum(diag_tr[k:k+self.win_condition])
            if sum_value == 1*self.win_condition:
                return 1
            elif sum_value == -1*self.win_condition:
                return -1
        if len(self.get_legal_moves()) == 0:
            return 2
        return 0


# monte carlo tree search
def mcts(root: Node, n_iters: int, random_node=False) -> Node:
    for _ in range(n_iters):
        leaf = root.select()
        leaf.expansion()
        value = leaf.rollout()
        leaf.backpropagation(value)
    # for child in root.children:children
    #     print(child.move, child.visit_count)
    if random_node:
        return root.select_random_node()
    else:
        return root.select_best_node()


if __name__ == '__main__':
    from tqdm import tqdm

    # 棋盘参数
    board_size = 3
    win_condition = 3
    stats = {}
    stats_chunks = [{}]
    win_steps = []
    root = Node(-1, None, BoardState(np.zeros((board_size, board_size)), -1, win_condition), move=None)
    for i in tqdm(range(5000)):
        node = root
        board_state = deepcopy(root.board_state)
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
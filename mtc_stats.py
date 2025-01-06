# -*- coding: utf-8 -*-

import random
import numpy as np
from mcts import BoardState
import tqdm
#
# random.seed(1234)

if __name__ == '__main__':
    times = 10000
    stats = {}
    player = 1
    p1_win = 0
    p2_win = 0
    for _ in tqdm.tqdm(range(times)):
        board = BoardState(np.zeros((3, 3)), current_player=player)
        first_move = (1,1,1)
        board.do_move(first_move)
        # second_move = (1,2,-1)
        # board.do_move(second_move)
        while not board.is_terminal():
            legal_moves = board.get_legal_moves()
            random_move = random.choice(legal_moves)
            board.do_move(random_move)
        if board.is_terminal() != -player:
            stats.setdefault(board.history[1], 0)
            stats[board.history[1]] += 1
        if board.is_terminal() == 1:
            p1_win += 1
        elif board.is_terminal() == -1:
            p2_win += 1
    print(stats)
    print(max(stats, key=stats.get))
    print(p1_win, p2_win)
    print(p1_win/times, p2_win/times)

# -*- coding: utf-8 -*-

import numpy as np
import pygame
import torch
import argparse

from mcts import Node, BoardState, mcts
from models import ResNet


def update_by_human(node: Node, move):
    # if len(node.children) > 0:
    #     child: Node
    #     for child in node.children:
    #         if np.array_equal(child.board_state.board, node.board_state.board):
    #             return child
    #
    # new_node = Node(node.board_state.current_player, node, node.board_state)
    # node.children.append(new_node)
    # return new_node
    node = node.select_with_move(move)
    return node

def update_by_ai(node, pure_network: bool = False, model: ResNet = None, device='cpu'):
    # 可复用的节点，而非每次重建
    if pure_network and model is not None:
        model.to(device)
        board = node.board_state.board
        out = model.infer(torch.tensor(board, dtype=torch.float32).reshape(1, 1, len(board), len(board[0])))
        n_row = out // len(board)
        n_col = out % len(board)
        move = (n_row, n_col)
        best_node = node.select_with_move(move)
    else:
        best_node: Node = mcts(node, 1000)
    return best_node


def draw_board(screen, size):
    screen.fill((230, 185, 70))
    for x in range(size):
        pygame.draw.line(screen, [0, 0, 0], [25 + 50 * x, 25], [25 + 50 * x, size*50-25], 1)
        pygame.draw.line(screen, [0, 0, 0], [25, 25 + 50 * x], [size*50-25, 25 + 50 * x], 1)
    pygame.display.update()


def update_board(screen, state):
    indices = np.where(state != 0)
    for (row, col) in list(zip(indices[0], indices[1])):
        if state[row][col] == 1:
            pygame.draw.circle(screen, [0, 0, 0], [25 + 50 * col, 25 + 50 * row], 15, 15)
        elif state[row][col] == -1:
            pygame.draw.circle(screen, [255, 255, 255], [25 + 50 * col, 25 + 50 * row], 15, 15)
    pygame.display.update()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', type=int, default=3, help='board size')
    parser.add_argument('-n', '--number', type=int, default=3, help='winning condition. e.g.n=5 means five in a row')
    parser.add_argument('-p', '--player', type=int, default=1, help='human player.1 means black, -1 means white')
    parser.add_argument('-m', '--method', type=int, default=0, help='0:pure mcts;1:pure network')
    parser.add_argument('-c', '--ckpt', default='', help='checkpoint of pretrained model')
    parser.add_argument('-d', '--device', default='cpu', help='device to run pretrained model')

    # board parameters
    args = parser.parse_args()
    M = args.size
    win_condition = args.number
    black = 1
    white = -1
    players = [black, white]
    players.remove(args.player)
    human_player = args.player
    ai_player = players[0]
    pc_step = 0
    pure_network = (args.method == 1)
    prior_network = None

    # load ai model
    if args.method == 1:
        device = 'cpu'
        prior_network = ResNet(1, 256, M**2)
        prior_network.load_state_dict(torch.load(args.ckpt))
        prior_network.to(device)
        prior_network.eval()

    # front-end parameters
    pygame.init()
    screen = pygame.display.set_mode((50*M, 50*M))
    pygame.display.set_caption('Five-in-a-Row')
    draw_board(screen, M)
    pygame.display.update()

    root = Node(white, None, BoardState(np.zeros((M, M)), current_player=white, win_condition=win_condition))
    if ai_player == 1:
        pc_step += 1
        root = update_by_ai(root, pure_network=pure_network, model=prior_network)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            update_board(screen, root.board_state.board)
            if event.type == pygame.MOUSEBUTTONDOWN:
                (x, y) = event.pos
                col = round((x - 25) / 50)
                row = round((y - 25) / 50)
                if root.board_state.board[row][col] != 0:
                    continue
                move = (row, col)
                root = root.select_with_move(move)
                update_board(screen, root.board_state.board)
                done = root.board_state.is_terminal()
                if done:
                    break
                else:
                    pc_step += 1
                    root = update_by_ai(root, pure_network=pure_network, model=prior_network)


        if root.board_state.is_terminal():
            myfont = pygame.font.Font(None, 40)
            text = ""
            if root.board_state.is_terminal() == human_player:
                text = "Human player wins!"
            elif root.board_state.is_terminal() == ai_player:
                text = "Computer player wins!"
            elif root.board_state.is_terminal() == 2:
                text = "Nobody wins!"
            textImage = myfont.render(text, True, (255, 255, 255))
            screen.blit(textImage, (int(M*50/2)-len(text)*M, int(M*50/2)-20))
            pygame.display.update()


if __name__ == '__main__':
    main()


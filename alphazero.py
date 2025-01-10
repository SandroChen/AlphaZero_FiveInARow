from models import ResNet
from game import Game, Gomoku
from mcts import Player
from copy import deepcopy
import numpy as np
import tqdm


game_dicts = {
    'gomoku': Gomoku
}


def evaluator(network1: ResNet, network2: ResNet, init_game: Game, num=30, n_iter=300, threshold=0.55):
    # assume that there are only 2 players, and player1 is the default best player. First player is 1 in the programme.
    winner_stats = {network1: 0, network2: 0}
    for networks in [[network1, network2], [network1, network2]]:
        players = [
            Player(networks[0]),
            Player(networks[1])
        ]
        # player mapping
        network_mapping = {1: networks[0], -1: networks[1]}
        print('evaluating...')
        for _ in tqdm.tqdm(range(num // 2)):
            # init board
            game = deepcopy(init_game)
            i = 0
            while True:
                move = players[i%2].play(game, n_iter)
                game.do_move(move)
                print(game.game_state)
                if game.is_terminal():
                    break
                i += 1
            print('\nwinner: ', game.is_terminal())
            print('*****************')
            if game.is_terminal() in [1, -1]:
                winner_stats[networks[game.is_terminal()]] += 1
    print(winner_stats.values())
    if winner_stats[network2] / (winner_stats[network1] + winner_stats[network1]) > threshold:
        return network2
    else:
        return network1


def selfplay(network: ResNet, init_game: Game, num=1, n_iter=300):
    new_data_batch = []
    for i in range(num):
        game = deepcopy(init_game)
        player = Player(network)
        while not game.is_terminal():
            new_data = [game.game_state]
            move = player.play(game, n_iter=n_iter)
            game.do_move(move)
            new_data.append(move)
            new_data_batch.append(new_data)
            print(new_data)
    return new_data_batch


def test_evaluator():
    board_size = 6
    best_net = ResNet(1, output_size=board_size**2)
    current_net = ResNet(1, output_size=board_size**2)
    game = Gomoku(np.zeros((6, 6), dtype=np.int16), -1, 4)
    best_player = evaluator(best_net, current_net, game)
    return best_player


def test_selfplay():
    board_size = 6
    best_net = ResNet(1, output_size=board_size ** 2)
    game = Gomoku(np.zeros((6, 6), dtype=np.int16), -1, 4)
    selfplay_data = []
    new_data = selfplay(best_net, game, 3)
    selfplay_data.extend(new_data)


test_selfplay()
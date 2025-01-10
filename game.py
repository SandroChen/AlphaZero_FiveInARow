import numpy as np

class Game:
    def __init__(self, game_state, current_player, **kwargs):
        self.game_state = game_state
        self.current_player = current_player
        ...

    def do_move(self, move):
        ...

    def get_legal_moves(self) -> np.ndarray:
        ...

    def is_terminal(self) -> int:
        ...


class Gomoku(Game):
    def __init__(self, board: np.ndarray, current_player=-1, win_condition=3, move=None):
        super().__init__(game_state=board, current_player=current_player)
        self.game_state = self.board = board
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
        if type(move) is int:
            move = [move // self.board.shape[0], move % self.board.shape[1]]
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


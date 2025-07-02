import numpy as np
import random
from itertools import product

class P2():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2)
                       for k in range(2) for l in range(2)]
        self.board = board
        self.available_pieces = available_pieces

    def select_piece(self):
        safe_pieces = []
        for piece in self.available_pieces:
            if not self.opponent_can_win_with(piece):
                safe_pieces.append(piece)
        return random.choice(safe_pieces if safe_pieces else self.available_pieces)

    def place_piece(self, selected_piece):
        for row, col in product(range(4), range(4)):
            if self.board[row][col] == 0:
                temp_board = self.copy_board()
                temp_board[row][col] = self.pieces.index(selected_piece) + 1
                if self.check_win(temp_board):
                    return (row, col)

        best_score = -float('inf')
        best_move = None

        for row, col in product(range(4), range(4)):
            if self.board[row][col] == 0:
                temp_board = self.copy_board()
                temp_board[row][col] = self.pieces.index(selected_piece) + 1
                score = self.evaluate_board(temp_board)
                if score > best_score:
                    best_score = score
                    best_move = (row, col)

        return best_move if best_move else random.choice(
            [(r, c) for r in range(4) for c in range(4) if self.board[r][c] == 0]
        )

    def opponent_can_win_with(self, piece):
        for row, col in product(range(4), range(4)):
            if self.board[row][col] == 0:
                temp_board = self.copy_board()
                temp_board[row][col] = self.pieces.index(piece) + 1
                if self.check_win(temp_board):
                    return True
        return False

    def evaluate_board(self, board):
        score = 0
        for i in range(4):
            row = [board[i][j] for j in range(4)]
            col = [board[j][i] for j in range(4)]
            score += self.line_score(row) + self.line_score(col)

        diag1 = [board[i][i] for i in range(4)]
        diag2 = [board[i][3 - i] for i in range(4)]
        score += self.line_score(diag1) + self.line_score(diag2)
        score += self.subgrid_score(board)
        return score

    def line_score(self, line):
        if 0 in line:
            chars = [self.pieces[idx - 1] for idx in line if idx != 0]
            return sum(len(set(attr)) == 1 for attr in zip(*chars)) * 10
        return 0

    def subgrid_score(self, board):
        score = 0
        for r in range(3):
            for c in range(3):
                subgrid = [board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]]
                if 0 in subgrid:
                    chars = [self.pieces[idx - 1] for idx in subgrid if idx != 0]
                    if len(chars) >= 2:
                        score += sum(len(set(attr)) == 1 for attr in zip(*chars)) * 5
        return score

    def check_win(self, board):
        def check_line(line):
            if 0 in line:
                return False
            chars = np.array([self.pieces[idx - 1] for idx in line])
            return any(len(set(chars[:, i])) == 1 for i in range(4))

        for i in range(4):
            if check_line([board[i][j] for j in range(4)]) or check_line([board[j][i] for j in range(4)]):
                return True

        if check_line([board[i][i] for i in range(4)]) or check_line([board[i][3 - i] for i in range(4)]):
            return True

        for r in range(3):
            for c in range(3):
                block = [board[r][c], board[r][c+1], board[r+1][c], board[r+1][c+1]]
                if 0 not in block:
                    chars = [self.pieces[i - 1] for i in block]
                    for i in range(4):
                        if len(set(char[i] for char in chars)) == 1:
                            return True
        return False

    def copy_board(self):
        return np.copy(self.board)
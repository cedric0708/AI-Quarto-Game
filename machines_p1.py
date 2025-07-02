import numpy as np
import random
from itertools import product
import copy
import time
import math

class MCTSNode:
    def __init__(self, board, available_pieces, parent=None, piece=None, position=None):
        self.board = board
        self.available_pieces = available_pieces
        self.parent = parent
        self.piece = piece
        self.position = position
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_pieces = available_pieces.copy()
        self.untried_positions = [(i, j) for i in range(4) for j in range(4) if board[i][j] == 0]

    def add_child(self, piece, position, piece_to_index):
        child_board = copy.deepcopy(self.board)
        child_board[position[0]][position[1]] = piece_to_index[piece]
        child_pieces = self.available_pieces.copy()
        child_pieces.remove(piece)
        child = MCTSNode(child_board, child_pieces, self, piece, position)
        self.untried_pieces.remove(piece)
        self.untried_positions.remove(position)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result

    def get_ucb1(self):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + math.sqrt(2 * math.log(self.parent.visits) / self.visits)

class P1:
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2)
                       for k in range(2) for l in range(2)]
        self.piece_to_index = {p: i+1 for i, p in enumerate(self.pieces)}
        self.board = board
        self.available_pieces = available_pieces
        self.start_time = time.time()

    def select_piece(self):
        safe_pieces = [p for p in self.available_pieces if not self.opponent_can_win_with(p)]
        candidates = safe_pieces if safe_pieces else self.available_pieces

        if len(self.available_pieces) > 12:
            return self.mcts_select_piece(candidates)

        best_piece = None
        best_score = float('-inf')
        for piece in candidates:
            score = 0
            for row, col in self.get_available_moves(self.board):
                temp_board = copy.deepcopy(self.board)
                temp_board[row][col] = self.piece_to_index[piece]
                temp_avail = candidates.copy()
                temp_avail.remove(piece)
                score += self.minimax(temp_board, temp_avail, depth=4 if len(candidates) > 6 else 6, alpha=-float('inf'), beta=float('inf'), maximizing=False)
            if score > best_score:
                best_score = score
                best_piece = piece
        return best_piece

    def mcts_select_piece(self, candidates):
        root = MCTSNode(self.board.copy(), candidates.copy())
        end_time = time.time() + 1.0
        while time.time() < end_time:
            node = self.mcts_select(root)
            if node.untried_pieces and node.untried_positions:
                node = node.add_child(random.choice(node.untried_pieces), random.choice(node.untried_positions), self.piece_to_index)
            result = self.mcts_simulate_game(node.board, node.available_pieces)
            self.mcts_backpropagate(node, result)
        best_child = max(root.children, key=lambda c: c.visits, default=None)
        return best_child.piece if best_child else random.choice(candidates)

    def place_piece(self, selected_piece):
        available_locs = self.get_available_moves(self.board)
        if np.count_nonzero(self.board == 0) >= 14:
            for r, c in [(1, 1), (1, 2), (2, 1), (2, 2)]:
                if self.board[r][c] == 0:
                    return (r, c)

        if len(self.available_pieces) > 12:
            return self.mcts_place_piece(selected_piece)

        best_score = -float('inf')
        best_move = None
        for r, c in available_locs:
            temp_board = copy.deepcopy(self.board)
            temp_board[r][c] = self.piece_to_index[selected_piece]
            if self.check_win(temp_board):
                return (r, c)
            score = self.minimax(temp_board, self.available_pieces, depth=4 if len(self.available_pieces) > 6 else 6, alpha=-float('inf'), beta=float('inf'), maximizing=False)
            score += self.evaluate_board(temp_board)
            if (r, c) in [(1, 1), (1, 2), (2, 1), (2, 2)]:
                score += 3
            if score > best_score:
                best_score = score
                best_move = (r, c)
        return best_move if best_move else random.choice(available_locs)

    def mcts_place_piece(self, selected_piece):
        root = MCTSNode(self.board.copy(), self.available_pieces.copy())
        end_time = time.time() + 1.0
        while time.time() < end_time:
            node = self.mcts_select(root)
            if node.untried_pieces and node.untried_positions:
                node = node.add_child(selected_piece, random.choice(node.untried_positions), self.piece_to_index)
            result = self.mcts_simulate_game(node.board, node.available_pieces)
            self.mcts_backpropagate(node, result)
        best_child = max(root.children, key=lambda c: c.visits, default=None)
        return best_child.position if best_child else random.choice(self.get_available_moves(self.board))

    def mcts_select(self, node):
        while node.untried_pieces == [] and node.untried_positions == [] and node.children != []:
            node = max(node.children, key=lambda c: c.get_ucb1())
        return node

    def mcts_backpropagate(self, node, result):
        while node is not None:
            node.update(result)
            node = node.parent

    def mcts_simulate_game(self, board, available):
        b = copy.deepcopy(board)
        avail = available.copy()
        while avail:
            p = random.choice(avail)
            avail.remove(p)
            locs = self.get_available_moves(b)
            if not locs:
                break
            b[random.choice(locs)] = self.piece_to_index[p]
            if self.check_win(b):
                return 1
        return 0

    def get_available_moves(self, board):
        return [(i, j) for i in range(4) for j in range(4) if board[i][j] == 0]

    def opponent_can_win_with(self, piece):
        idx = self.piece_to_index[piece]
        for r, c in product(range(4), range(4)):
            if self.board[r][c] == 0:
                temp = copy.deepcopy(self.board)
                temp[r][c] = idx
                if self.check_win(temp):
                    return True
        return False

    def minimax(self, board, available_pieces, depth, alpha, beta, maximizing):
        if depth == 0 or self.check_win(board):
            return self.evaluate_board(board)

        moves = self.get_available_moves(board)
        if maximizing:
            max_eval = -float('inf')
            for r, c in moves:
                temp = copy.deepcopy(board)
                temp[r][c] = 1
                eval = self.minimax(temp, available_pieces, depth-1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for r, c in moves:
                temp = copy.deepcopy(board)
                temp[r][c] = 1
                eval = self.minimax(temp, available_pieces, depth-1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate_board(self, board):
        score = 0
        for i in range(4):
            score += self.line_score([board[i][j] for j in range(4)])
            score += self.line_score([board[j][i] for j in range(4)])
        score += self.line_score([board[i][i] for i in range(4)])
        score += self.line_score([board[i][3-i] for i in range(4)])
        score += self.subgrid_score(board)
        return score

    def line_score(self, line):
        filled = [idx for idx in line if idx != 0]
        if not filled:
            return 0
        chars = [self.pieces[idx - 1] for idx in filled]
        attr_sets = list(zip(*chars))
        full_match = sum(len(set(attr)) == 1 for attr in attr_sets)
        three_match = sum(len(set(attr)) == 2 and attr.count(attr[0]) == 3 for attr in attr_sets)
        two_match = sum(len(set(attr)) == 3 and attr.count(attr[0]) == 2 for attr in attr_sets)
        return full_match * 20 + three_match * 5 + two_match * 2

    def subgrid_score(self, board):
        score = 0
        for r in range(3):
            for c in range(3):
                block = [board[r+i][c+j] for i in range(2) for j in range(2)]
                filled = [idx for idx in block if idx != 0]
                if len(filled) < 2:
                    continue
                chars = [self.pieces[i - 1] for i in filled]
                attr_sets = zip(*chars)
                for attr in attr_sets:
                    unique = set(attr)
                    if len(unique) == 1:
                        score += 10
                    elif len(unique) == 2 and list(attr).count(attr[0]) == 3:
                        score += 3
                    elif len(unique) == 3 and list(attr).count(attr[0]) == 2:
                        score += 1
        return score

    def check_win(self, board):
        def win_line(line):
            if 0 in line:
                return False
            chars = [self.pieces[i - 1] for i in line]
            return any(len(set(attr)) == 1 for attr in zip(*chars))

        for i in range(4):
            if win_line([board[i][j] for j in range(4)]) or win_line([board[j][i] for j in range(4)]):
                return True
        if win_line([board[i][i] for i in range(4)]) or win_line([board[i][3-i] for i in range(4)]):
            return True
        for r in range(3):
            for c in range(3):
                block = [board[r+i][c+j] for i in range(2) for j in range(2)]
                if 0 not in block:
                    chars = [self.pieces[i - 1] for i in block]
                    for i in range(4):
                        if len(set(char[i] for char in chars)) == 1:
                            return True
        return False

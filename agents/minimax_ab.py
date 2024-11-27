from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("Minimax_with_alpha_beta_pruning")
class Minimax_with_alpha_beta_pruning(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """
    
    def minimax(self, chess_board, depth, maximizing_player, player, opponent, alpha, beta, start_time):
        """
        Minimax algorithm with alpha-beta pruning.
        """
        # Time limit check
        if time.time() - start_time > 1.9:
            raise TimeoutError
        # Endgame or depth limit check
        is_endgame, p1_score, p2_score = check_endgame(chess_board, player, opponent)
        if depth == 0 or is_endgame:
            return self.evaluate_board(chess_board, player, opponent), None

        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            valid_moves = get_valid_moves(chess_board, player)
            if not valid_moves:
                # Pass move
                eval_score, _ = self.minimax(chess_board, depth - 1, False, player, opponent, alpha, beta, start_time)
                return eval_score, None
            for move in valid_moves:
                new_board = chess_board.copy()
                execute_move(new_board, move, player)
                eval_score, _ = self.minimax(new_board, depth - 1, False, player, opponent, alpha, beta, start_time)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            valid_moves = get_valid_moves(chess_board, opponent)
            if not valid_moves:
                # Pass move
                eval_score, _ = self.minimax(chess_board, depth - 1, True, player, opponent, alpha, beta, start_time)
                return eval_score, None
            for move in valid_moves:
                new_board = chess_board.copy()
                execute_move(new_board, move, opponent)
                eval_score, _ = self.minimax(new_board, depth - 1, True, player, opponent, alpha, beta, start_time)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval, best_move
        
    def step(self, chess_board, player, opponent): 
        """
        Decide the best move for the AI agent.
        """
        start_time = time.time()
        time_limit = 1.9  # Time limit in seconds
        depth = 1  # Start from depth 1
        best_move = None

        try:
            while True:
                # Check time limit before each iteration
                if time.time() - start_time > time_limit:
                    break
                eval_score, move = self.minimax(chess_board, depth, True, player, opponent, float('-inf'), float('inf'), start_time)
                if move is not None:
                    best_move = move
                depth += 1
        except TimeoutError:
            pass  # Stop when time limit is reached

        if best_move is None:
            return random_move(chess_board, player)

        return best_move

    def evaluate_board(self, board, color, opponent):
        """
        Evaluate the board state based on multiple factors.
        """
        # Corner positions are highly valuable
        corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
        corner_score = sum(1 for corner in corners if board[corner] == color) * 10
        corner_penalty = sum(1 for corner in corners if board[corner] == 3 - color) * -10

        # Mobility: the number of moves the opponent can make
        opponent_moves = len(get_valid_moves(board, 3 - color))
        mobility_score = -opponent_moves

        # Combine scores
        total_score = corner_score + corner_penalty + mobility_score
        return total_score

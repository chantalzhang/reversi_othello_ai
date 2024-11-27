from agents.agent import Agent
from store import register_agent
import time
import numpy as np
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("MiniMaxAgent")
class MiniMax(Agent):
    """
    A simple MiniMax agent for Reversi/Othello.
    """

    def minimax(self, board, depth, maximizing_player, player, opponent, start_time):
        """
        Perform a minimax search to the specified depth.

        Parameters:
        - board: 2D numpy array representing the current board state.
        - depth: Current depth limit for the minimax search.
        - maximizing_player: True if it's the maximizing player's turn, False otherwise.
        - player: Integer representing the AI's color.
        - opponent: Integer representing the opponent's color.
        - start_time: The start time of the decision-making process.

        Returns:
        - Tuple (score, move) where score is the evaluated score of the board and move is the best move.
        """
        # Time limit check
        if time.time() - start_time > 1.9:
            raise TimeoutError  # Timeout reached, stop further computation

        # Check endgame or depth limit
        is_endgame, _, _ = check_endgame(board, player, opponent)
        if depth == 0 or is_endgame:
            return self.evaluate_board(board, player, opponent), None

        valid_moves = get_valid_moves(board, player if maximizing_player else opponent)

        # No valid moves: opponent gets a turn (pass case)
        if not valid_moves:
            next_player = not maximizing_player  # Switch turns
            return self.minimax(board, depth - 1, next_player, player, opponent, start_time)

        best_score = float('-inf') if maximizing_player else float('inf')
        best_move = None

        for move in valid_moves:
            new_board = board.copy()
            execute_move(new_board, move, player if maximizing_player else opponent)

            eval_score, _ = self.minimax(new_board, depth - 1, not maximizing_player, player, opponent, start_time)

            if maximizing_player and eval_score > best_score:
                best_score = eval_score
                best_move = move
            elif not maximizing_player and eval_score < best_score:
                best_score = eval_score
                best_move = move

        return best_score, best_move

    def step(self, chess_board, player, opponent):
        """
        Decide the best move for the AI agent.

        Parameters:
        - chess_board: 2D numpy array representing the current board state.
        - player: Integer representing the AI's color.
        - opponent: Integer representing the opponent's color.

        Returns:
        - Tuple (row, column) where the AI decides to place its piece.
        """
        start_time = time.time()
        depth = 1  # Start with depth 1
        best_move = None

        try:
            while True:
                _, move = self.minimax(chess_board, depth, True, player, opponent, start_time)
                if move is not None:
                    best_move = move
                depth += 1  # Increase search depth
        except TimeoutError:
            pass  # Stop computation when time limit is reached

        # If no move was found, fall back to a random move
        if best_move is None:
            return random_move(chess_board, player)

        return best_move

    def evaluate_board(self, board, color, opponent):
        """
        Evaluate the board state based on mobility and corner ownership.

        Parameters:
        - board: 2D numpy array representing the board state.
        - color: Integer representing the AI's color.
        - opponent: Integer representing the opponent's color.

        Returns:
        - int: The heuristic score of the board for the current player.
        """
        corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
        corner_score = sum(1 for corner in corners if board[corner] == color) * 10
        corner_penalty = sum(1 for corner in corners if board[corner] == opponent) * -10

        # Mobility: the number of moves available to the opponent
        opponent_moves = len(get_valid_moves(board, opponent))
        mobility_score = -opponent_moves

        return corner_score + corner_penalty + mobility_score

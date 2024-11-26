from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("minimax_with_gpt_greedy_corners")
class minimax_with_gpt_greedy_corners(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """
    

    def minimax(self, chess_board, depth, maximizing_player, player, opponent, alpha, beta, start_time):
            """
            Minimax algorithm with alpha-beta pruning.

            Parameters
            ----------
            chess_board : numpy array
                The current state of the board.
            depth : int
                The depth limit for the search.
            maximizing_player : bool
                True if the current move is for the maximizing player.
            player : int
                The AI player's number.
            opponent : int
                The opponent's player number.
            alpha : float
                The alpha value for pruning.
            beta : float
                The beta value for pruning.
            start_time : float
                The starting time of the search.

            Returns
            -------
            tuple
                (score, move) where score is the evaluated score and move is the best move found.
            """
            # Time limit check
            if time.time() - start_time > 1.9:
                return self.evaluate_board(chess_board, player, opponent), None
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

        Parameters
        ----------
        chess_board : numpy array
            The current state of the board.
        player : int
            The AI player's number.
        opponent : int
            The opponent's player number.

        Returns
        -------
        tuple
            The position (row, column) where the AI wants to place the next disc.
        """
        start_time = time.time()
        time_limit = 1.9  # Time limit in seconds
        depth = 3  # Initial search depth
        best_move = None

        # Iterative deepening to adjust depth dynamically
        while True:
            time_taken = time.time() - start_time
            if time_taken > time_limit:
                break
            eval_score, move = self.minimax(chess_board, depth, True, player, opponent, float('-inf'), float('inf'), start_time)
            if move is not None:
                best_move = move
            depth += 1  # Increase depth for next iteration

        time_taken = time.time() - start_time
        print("My AI's turn took ", time_taken, "seconds.")
        print("Last depth searched: ", depth)   

        if best_move is None:
            # If no best move found, play a random valid move
            return random_move(chess_board, player)

        return best_move


    def evaluate_board(self, board, color, opponent):
            """
            Evaluate the board state based on multiple factors.

            Parameters:
            - board: 2D numpy array representing the game board.
            - color: Integer representing the agent's color (1 for Player 1/Blue, 2 for Player 2/Brown).
            - player_score: Score of the current player.
            - opponent_score: Score of the opponent.

            Returns:
            - int: The evaluated score of the board.
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

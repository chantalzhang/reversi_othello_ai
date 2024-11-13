# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

# python simulator.py --player_1 random_agent --player_2 random_agent

@register_agent("test_agent")
class TestAgent(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(TestAgent, self).__init__()
        self.name = "TestAgent"
    
    def evaluate_board(self, chess_board, player, opponent):
        """
        Evaluate the board from the perspective of the player.

        Parameters
        ----------
        chess_board : numpy array
            The current state of the board.
        player : int
            The player number (1 or 2).
        opponent : int
            The opponent's player number (1 or 2).

        Returns
        -------
        int
            The evaluated score of the board.
        """
        # Initialize score
        score = 0

        # Corners
        corners = [(0,0), (0, chess_board.shape[0]-1), 
                (chess_board.shape[0]-1,0), (chess_board.shape[0]-1, chess_board.shape[0]-1)]
        player_corners = 0
        opponent_corners = 0
        for corner in corners:
            if chess_board[corner[0], corner[1]] == player:
                player_corners += 1
            elif chess_board[corner[0], corner[1]] == opponent:
                opponent_corners += 1
        corner_diff = player_corners - opponent_corners

        # Edges
        player_edges = 0
        opponent_edges = 0
        edge_positions = []

        N = chess_board.shape[0]  # Board size

        # Top and Bottom edges (excluding corners)
        for i in range(1, N - 1):
            edge_positions.append((0, i))       # Top edge
            edge_positions.append((N - 1, i))   # Bottom edge

        # Left and Right edges (excluding corners)
        for i in range(1, N - 1):
            edge_positions.append((i, 0))       # Left edge
            edge_positions.append((i, N - 1))   # Right edge

        for pos in edge_positions:
            if chess_board[pos[0], pos[1]] == player:
                player_edges += 1
            elif chess_board[pos[0], pos[1]] == opponent:
                opponent_edges += 1

        edge_diff = player_edges - opponent_edges

        # Inner tiles bordering the edge (tiles adjacent to edges but not on the edge)
        player_inner_border = 0
        opponent_inner_border = 0
        inner_border_positions = []

        # Top and Bottom inner borders
        for i in range(1, N - 1):
            inner_border_positions.append((1, i))         # Just below top edge
            inner_border_positions.append((N - 2, i))     # Just above bottom edge

        # Left and Right inner borders
        for i in range(1, N - 1):
            inner_border_positions.append((i, 1))         # Right next to left edge
            inner_border_positions.append((i, N - 2))     # Left next to right edge

        for pos in inner_border_positions:
            if chess_board[pos[0], pos[1]] == player:
                player_inner_border += 1
            elif chess_board[pos[0], pos[1]] == opponent:
                opponent_inner_border += 1

        inner_border_diff = player_inner_border - opponent_inner_border

        # Mobility
        player_moves = len(get_valid_moves(chess_board, player))
        opponent_moves = len(get_valid_moves(chess_board, opponent))
        if player_moves + opponent_moves != 0:
            mobility = 100 * (player_moves - opponent_moves) / (player_moves + opponent_moves)
        else:
            mobility = 0

        # Weighting the features
        score = (1000 * corner_diff) + (50 * edge_diff) - (30 * inner_border_diff) + (10 * mobility)

        return score

      
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
        if time.time() - start_time > 1.8:
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
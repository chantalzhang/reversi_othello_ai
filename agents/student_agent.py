# Student agent: Improved agent with requested changes
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import random
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    An improved Othello agent implementing minimax with alpha-beta pruning,
    iterative deepening, transposition tables, and optimized heuristics.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.transposition_table = {}  # For caching board evaluations

    def step(self, chess_board, player, opponent): 
        """
        Decide the best move for the AI agent using iterative deepening and alpha-beta pruning.
        """
        start_time = time.time()
        time_limit = 1.9  # Updated time limit in seconds
        depth = 1  # Start from depth 1
        best_move = None

        try:
            while True:
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time >= time_limit:
                    break
                # Estimate the remaining time
                remaining_time = time_limit - elapsed_time
                # Start timer for depth search
                depth_start_time = current_time
                # Perform minimax search at the current depth
                eval_score, move = self.minimax(chess_board, depth, True, player, opponent,
                                                float('-inf'), float('inf'), start_time, time_limit)
                # Update best move if found
                if move is not None:
                    best_move = move
                # Check time taken for current depth
                depth_time_taken = time.time() - depth_start_time
                # Estimate if there's enough time for the next depth (assuming exponential growth)
                estimated_next_depth_time = depth_time_taken * 3  # Adjust multiplier as needed
                if elapsed_time + estimated_next_depth_time >= time_limit:
                    break
                depth += 1
        except TimeoutError:
            pass  # Time limit reached; exit search

        # If no best move found, play a random valid move or pass if no moves are available
        if best_move is None:
            valid_moves = get_valid_moves(chess_board, player)
            if valid_moves:
                return random.choice(valid_moves)
            else:
                return None  # No valid moves; must pass

        return best_move

    def minimax(self, chess_board, depth, maximizing_player, player, opponent, alpha, beta, start_time, time_limit):
        """
        Minimax algorithm with alpha-beta pruning and move ordering.
        """
        # Time limit check
        if time.time() - start_time >= time_limit:
            raise TimeoutError  # Immediate termination

        # Transposition table lookup
        board_key = (tuple(chess_board.flatten()), depth, maximizing_player)
        if board_key in self.transposition_table:
            return self.transposition_table[board_key]

        # Endgame or depth limit check
        is_endgame, _, _ = check_endgame(chess_board, player, opponent)
        if depth == 0 or is_endgame:
            score = self.evaluate_board(chess_board, player, opponent)
            # Store in transposition table
            self.transposition_table[board_key] = (score, None)
            return score, None

        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            valid_moves = get_valid_moves(chess_board, player)
            if not valid_moves:
                # Pass move
                eval_score, _ = self.minimax(chess_board, depth - 1, False, player, opponent,
                                             alpha, beta, start_time, time_limit)
                # Store in transposition table
                self.transposition_table[board_key] = (eval_score, None)
                return eval_score, None
            # Move ordering
            valid_moves = self.order_moves(chess_board, valid_moves, player, opponent, True)
            for move in valid_moves:
                new_board = np.copy(chess_board)
                execute_move(new_board, move, player)
                eval_score, _ = self.minimax(new_board, depth - 1, False, player, opponent,
                                             alpha, beta, start_time, time_limit)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            # Store in transposition table
            self.transposition_table[board_key] = (max_eval, best_move)
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            valid_moves = get_valid_moves(chess_board, opponent)
            if not valid_moves:
                # Pass move
                eval_score, _ = self.minimax(chess_board, depth - 1, True, player, opponent,
                                             alpha, beta, start_time, time_limit)
                # Store in transposition table
                self.transposition_table[board_key] = (eval_score, None)
                return eval_score, None
            # Move ordering
            valid_moves = self.order_moves(chess_board, valid_moves, opponent, player, False)
            for move in valid_moves:
                new_board = np.copy(chess_board)
                execute_move(new_board, move, opponent)
                eval_score, _ = self.minimax(new_board, depth - 1, True, player, opponent,
                                             alpha, beta, start_time, time_limit)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            # Store in transposition table
            self.transposition_table[board_key] = (min_eval, best_move)
            return min_eval, best_move

    def order_moves(self, board, moves, player, opponent, maximizing_player):
        """
        Order moves to improve alpha-beta pruning efficiency.
        """
        move_scores = []
        for move in moves:
            new_board = np.copy(board)
            execute_move(new_board, move, player)
            score = self.evaluate_board(new_board, player, opponent)
            move_scores.append((score, move))
        # Sort moves based on their heuristic score
        move_scores.sort(reverse=maximizing_player)
        ordered_moves = [move for _, move in move_scores]
        return ordered_moves

    def evaluate_board(self, board, color, opponent):
        """
        Evaluate the board state based on multiple heuristics.
        Positive scores favor the maximizing player.
        """
        # Initialize the score
        score = 0

        # -------- Game Phase Detection --------
        total_squares = board.size
        empty_squares = np.count_nonzero(board == 0)

        # Determine game phase based on the number of empty squares
        if empty_squares > 40:
            game_phase = 'early'
        elif empty_squares > 20:
            game_phase = 'mid'
        else:
            game_phase = 'late'

        board_size = board.shape[0]

        # -------- Heuristic Weights --------
        weights = {
            'corner': 25,
            'n2_corner': 15,
            'adjacent_corner': -20,
            'mobility': 100,
            'potential_mobility': 1,
            'disc_difference': {'early': -5, 'mid': 0, 'late': 5}
        }

        # -------- Corners Heuristic --------
        corners = [
            (0, 0),
            (0, board_size - 1),
            (board_size - 1, 0),
            (board_size - 1, board_size - 1)
        ]
        corner_score = 0
        for corner in corners:
            if board[corner] == color:
                corner_score += weights['corner']
            elif board[corner] == opponent:
                corner_score -= weights['corner']
        score += corner_score

        # -------- n-2 Corners Heuristic (Inner Corners) --------
        n2_corners = [
            (0, 2), (0, board_size - 3),
            (2, 0), (board_size - 3, 0),
            (board_size - 1, 2), (board_size - 1, board_size - 3),
            (2, board_size - 1), (board_size - 3, board_size - 1)
        ]
        n2_corner_score = 0
        for pos in n2_corners:
            if board[pos] == color:
                n2_corner_score += weights['n2_corner']
            elif board[pos] == opponent:
                n2_corner_score -= weights['n2_corner']
        score += n2_corner_score

        # -------- Adjacent to Corners Heuristic (X-squares and C-squares) --------
        # X-squares (diagonally adjacent to corners)
        x_squares = [
            (1, 1),
            (1, board_size - 2),
            (board_size - 2, 1),
            (board_size - 2, board_size - 2)
        ]
        # C-squares (adjacent to corners along edges)
        c_squares = [
            (0, 1), (1, 0),
            (0, board_size - 2), (1, board_size - 1),
            (board_size - 1, 1), (board_size - 2, 0),
            (board_size - 2, board_size - 1), (board_size - 1, board_size - 2)
        ]
        adjacent_corner_score = 0
        for pos in x_squares + c_squares:
            if board[pos] == color:
                adjacent_corner_score += weights['adjacent_corner']  # Penalize our occupation
            elif board[pos] == opponent:
                adjacent_corner_score -= weights['adjacent_corner']  # Reward opponent's occupation
        score += adjacent_corner_score

        # -------- Mobility Heuristic --------
        player_moves = len(get_valid_moves(board, color))
        opponent_moves = len(get_valid_moves(board, opponent))
        mobility_score = 0
        if player_moves + opponent_moves != 0:
            mobility_score = weights['mobility'] * (player_moves - opponent_moves) / (player_moves + opponent_moves)
        score += mobility_score

        # -------- Potential Mobility Heuristic --------
        potential_mobility_score = self.calculate_potential_mobility(board, color, opponent)
        score += weights['potential_mobility'] * potential_mobility_score

        # -------- Disc Difference Heuristic --------
        player_discs = np.count_nonzero(board == color)
        opponent_discs = np.count_nonzero(board == opponent)
        disc_difference = player_discs - opponent_discs
        disc_diff_weight = weights['disc_difference'][game_phase]
        disc_difference_score = disc_diff_weight * disc_difference
        score += disc_difference_score

        return score

    def calculate_potential_mobility(self, board, color, opponent):
        """
        Calculate potential mobility for the player using optimized computation.
        """
        potential_mobility = 0
        opponent_positions = np.argwhere(board == opponent)
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),         (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]
        board_size = board.shape[0]
        empty_neighbors = set()

        # Use opponent positions to find adjacent empty squares
        for x, y in opponent_positions:
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < board_size and 0 <= ny < board_size:
                    if board[nx, ny] == 0:
                        empty_neighbors.add((nx, ny))

        # Potential mobility is the number of empty squares adjacent to opponent discs
        potential_mobility_score = len(empty_neighbors)

        # Repeat for opponent's potential mobility
        player_positions = np.argwhere(board == color)
        opponent_empty_neighbors = set()
        for x, y in player_positions:
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < board_size and 0 <= ny < board_size:
                    if board[nx, ny] == 0:
                        opponent_empty_neighbors.add((nx, ny))

        potential_mobility_score -= len(opponent_empty_neighbors)

        return potential_mobility_score

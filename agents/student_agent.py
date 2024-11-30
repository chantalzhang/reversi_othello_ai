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
        board_size = chess_board.shape[0]

        # Adjust time limit based on board size
        base_time_limit = 1.9
        time_limit = base_time_limit + (board_size - 8) * 0.1  # Adjust as needed

        depth = 1
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
                # Estimate if there's enough time for the next depth
                estimated_next_depth_time = depth_time_taken * 2  # Assuming time doubles each depth
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
        print("depth searched:", depth)
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
            # Use a simplified evaluation function for speed
            score = self.quick_evaluate(new_board, player, opponent)
            move_scores.append((score, move))
        # Sort moves: descending if maximizing, ascending if minimizing
        move_scores.sort(reverse=maximizing_player)
        ordered_moves = [move for _, move in move_scores]
        return ordered_moves

    def quick_evaluate(self, board, player, opponent):
        """
        Simple evaluation function focusing on immediate gains.
        """
        return np.count_nonzero(board == player) - np.count_nonzero(board == opponent)

    def evaluate_board(self, board, color, opponent):
        """
        Evaluate the board state based on multiple heuristics.
        Positive scores favor the maximizing player.
        """
        score = 0
        board_size = board.shape[0]
        total_squares = board.size
        empty_squares = np.count_nonzero(board == 0)
        empty_percentage = empty_squares / total_squares

        # Game phase detection
        if empty_percentage > 0.7:
            game_phase = 'early'
        elif empty_percentage > 0.3:
            game_phase = 'mid'
        else:
            game_phase = 'late'

        size_factor = 8 / board_size
        weights = {
            'corner': 25 * size_factor,
            'n2_corner': 15 * size_factor,
            'adjacent_corner': -20 * size_factor,
            'mobility': 100 * size_factor,
            'potential_mobility': 1 * size_factor,
            'disc_difference': {
                'early': -5 * size_factor,
                'mid': 0,
                'late': 5 * size_factor
            },
            'edge_stability': 10 * size_factor,
            'parity': 5 * size_factor,
            'stability': 15 * size_factor
        }

        # -------- Corners Heuristic --------
        corners = [
            (0, 0), (0, board_size - 1),
            (board_size - 1, 0), (board_size - 1, board_size - 1)
        ]
        corner_score = sum(
            weights['corner'] if board[pos] == color else -weights['corner'] if board[pos] == opponent else 0
            for pos in corners
        )
        score += corner_score

        # -------- n-2 Corners Heuristic (Inner Corners) --------
        n2_corners = [
            (0, 2), (0, board_size - 3),
            (2, 0), (board_size - 3, 0),
            (board_size - 1, 2), (board_size - 1, board_size - 3),
            (2, board_size - 1), (board_size - 3, board_size - 1)
        ]
        n2_corner_score = sum(
            weights['n2_corner'] if board[pos] == color else -weights['n2_corner'] if board[pos] == opponent else 0
            for pos in n2_corners if 0 <= pos[0] < board_size and 0 <= pos[1] < board_size
        )
        score += n2_corner_score

        # -------- Adjacent to Corners Heuristic (X-squares and C-squares) --------
        x_squares = [
            (1, 1),
            (1, board_size - 2),
            (board_size - 2, 1),
            (board_size - 2, board_size - 2)
        ]
        c_squares = [
            (0, 1), (1, 0),
            (0, board_size - 2), (1, board_size - 1),
            (board_size - 1, 1), (board_size - 2, 0),
            (board_size - 2, board_size - 1), (board_size - 1, board_size - 2)
        ]
        adjacent_positions = x_squares + c_squares
        adjacent_corner_score = sum(
            weights['adjacent_corner'] if board[pos] == color else -weights['adjacent_corner'] if board[pos] == opponent else 0
            for pos in adjacent_positions if 0 <= pos[0] < board_size and 0 <= pos[1] < board_size
        )
        score += adjacent_corner_score

        # -------- Edge Stability Heuristic --------
        edge_stability_score = self.edge_stability(board, color) - self.edge_stability(board, opponent)
        score += edge_stability_score * weights['edge_stability']

        # -------- Mobility Heuristic --------
        player_moves = len(get_valid_moves(board, color))
        opponent_moves = len(get_valid_moves(board, opponent))
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

        # -------- Parity Heuristic --------
        parity_score = self.parity(board) * weights['parity']
        score += parity_score

        # -------- Stability Heuristic --------
        stable_disc_count = self.stable_discs(board, color) - self.stable_discs(board, opponent)
        score += stable_disc_count * weights['stability']

        return score

    def edge_stability(self, board, color):
        """
        Calculate the number of stable discs along the edges.
        """
        board_size = board.shape[0]
        stability = 0
        # Check all four edges
        for i in range(board_size):
            # Top edge
            if board[0, i] == color:
                stability += 1
            # Bottom edge
            if board[board_size - 1, i] == color:
                stability += 1
            # Left edge
            if board[i, 0] == color:
                stability += 1
            # Right edge
            if board[i, board_size - 1] == color:
                stability += 1
        return stability

    def parity(self, board):
        """
        Parity heuristic to favor having the last move.
        """
        empty_squares = np.count_nonzero(board == 0)
        return 1 if empty_squares % 2 == 1 else -1

    def calculate_potential_mobility(self, board, color, opponent):
        """
        Calculate potential mobility for the player.
        """
        board_size = board.shape[0]
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),         (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]
        potential_mobility = 0
        for x in range(board_size):
            for y in range(board_size):
                if board[x, y] == 0:
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < board_size and 0 <= ny < board_size:
                            if board[nx, ny] == opponent:
                                potential_mobility += 1
                                break
        # Subtract opponent's potential mobility
        opponent_potential_mobility = 0
        for x in range(board_size):
            for y in range(board_size):
                if board[x, y] == 0:
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < board_size and 0 <= ny < board_size:
                            if board[nx, ny] == color:
                                opponent_potential_mobility += 1
                                break
        return potential_mobility - opponent_potential_mobility

    def stable_discs(self, board, color):
        """
        Count the number of stable discs for a given color.
        """
        board_size = board.shape[0]
        stable = np.zeros_like(board, dtype=bool)
        # Check corners for stable discs
        corners = [(0, 0), (0, board_size - 1),
                   (board_size - 1, 0), (board_size - 1, board_size - 1)]
        for corner in corners:
            if board[corner] == color:
                self.mark_stable_discs(board, stable, corner, color)
        return np.count_nonzero(stable)

    def mark_stable_discs(self, board, stable, position, color):
        """
        Recursively mark stable discs from a corner.
        """
        x, y = position
        board_size = board.shape[0]
        if not (0 <= x < board_size and 0 <= y < board_size):
            return
        if stable[x, y]:
            return
        if board[x, y] != color:
            return
        stable[x, y] = True
        # Check all adjacent positions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < board_size and 0 <= ny < board_size:
                if board[nx, ny] == color:
                    self.mark_stable_discs(board, stable, (nx, ny), color)

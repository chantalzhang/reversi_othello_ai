# Student agent: Improved agent with requested changes
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import random
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent2")
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
        time_limit = base_time_limit 

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


                depth += 1
        except TimeoutError:
            pass  # Time limit reached; exit search

        # If no best move found, play a random valid move or pass if no moves are available
        if best_move is None:
            valid_moves = get_valid_moves(chess_board, player)
            if valid_moves:
                print("do random move")
                return random.choice(valid_moves)
            else:
                return None  # No valid moves; must pass
        print("depth searched:", depth)
        return best_move

    def minimax(self, chess_board, depth, maximizing_player, player, opponent, alpha, beta, start_time, time_limit):
        # Time limit check
        if time.time() - start_time >= time_limit:
            raise TimeoutError  # Immediate termination

        # Transposition table lookup
        board_key = tuple(chess_board.flatten())
        entry = self.transposition_table.get(board_key)
        if entry and entry['depth'] >= depth:
            if entry['flag'] == 'EXACT':
                return entry['value'], entry['best_move']
            elif entry['flag'] == 'LOWERBOUND':
                alpha = max(alpha, entry['value'])
            elif entry['flag'] == 'UPPERBOUND':
                beta = min(beta, entry['value'])
            if alpha >= beta:
                return entry['value'], entry['best_move']

        alpha_original = alpha  # Save original alpha value
        best_move = None

        # Endgame or depth limit check
        is_endgame, _, _ = check_endgame(chess_board, player, opponent)
        if depth == 0 or is_endgame:
            score = self.evaluate_board(chess_board, player, opponent)
            # Store in transposition table
            self.transposition_table[board_key] = {
                'value': score,
                'depth': depth,
                'flag': 'EXACT',
                'best_move': None
            }
            return score, None

        if maximizing_player:
            max_eval = float('-inf')
            valid_moves = get_valid_moves(chess_board, player)
            if not valid_moves:
                # Pass move
                eval_score, _ = self.minimax(chess_board, depth - 1, False, player, opponent,
                                            alpha, beta, start_time, time_limit)
                # Store in transposition table
                self.transposition_table[board_key] = {
                    'value': eval_score,
                    'depth': depth,
                    'flag': 'EXACT',
                    'best_move': None
                }
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
            # Determine flag
            if max_eval <= alpha_original:
                flag = 'UPPERBOUND'
            elif max_eval >= beta:
                flag = 'LOWERBOUND'
            else:
                flag = 'EXACT'
            # Store in transposition table
            self.transposition_table[board_key] = {
                'value': max_eval,
                'depth': depth,
                'flag': flag,
                'best_move': best_move
            }
            return max_eval, best_move
        else:
            min_eval = float('inf')
            valid_moves = get_valid_moves(chess_board, opponent)
            if not valid_moves:
                # Pass move
                eval_score, _ = self.minimax(chess_board, depth - 1, True, player, opponent,
                                            alpha, beta, start_time, time_limit)
                # Store in transposition table
                self.transposition_table[board_key] = {
                    'value': eval_score,
                    'depth': depth,
                    'flag': 'EXACT',
                    'best_move': None
                }
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
            # Determine flag
            if min_eval <= alpha_original:
                flag = 'UPPERBOUND'
            elif min_eval >= beta:
                flag = 'LOWERBOUND'
            else:
                flag = 'EXACT'
            # Store in transposition table
            self.transposition_table[board_key] = {
                'value': min_eval,
                'depth': depth,
                'flag': flag,
                'best_move': best_move
            }
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

    def quick_evaluate(self, board, color, opponent):
        """
        Simplified evaluation function focusing on key heuristics.
        """
        # Simple heuristics like disc difference and mobility
        player_discs = np.count_nonzero(board == color)
        opponent_discs = np.count_nonzero(board == opponent)
        disc_difference = player_discs - opponent_discs

        player_moves = len(get_valid_moves(board, color))
        opponent_moves = len(get_valid_moves(board, opponent))
        mobility = player_moves - opponent_moves

        # Weights can be adjusted as needed
        return disc_difference + mobility



    def evaluate_board(self, board, color, opponent):
        """
        Evaluate the board state based on simplified heuristics.
        Positive scores favor the maximizing player.
        """
        board_size = board.shape[0]
        total_squares = board.size
        empty_squares = np.count_nonzero(board == 0)
        empty_percentage = empty_squares / total_squares

        # Game phase detection
        if empty_percentage > 0.8:
            game_phase = 'early'
        elif empty_percentage > 0.4:
            game_phase = 'mid'
        else:
            game_phase = 'late'

        # Raw importance values for heuristics
        importance = {
            'corner': 1000,
            'adjacent_corner': -15,
            'mobility': 15,
            'disc_difference': {'early': 2, 'mid': 5, 'late': 10},
            'stability': 20,
            'parity': 3
        }

        # Total importance for normalization
        total_importance = (
            importance['corner'] +
            abs(importance['adjacent_corner']) +
            importance['mobility'] +
            importance['disc_difference'][game_phase] +
            importance['stability'] +
            importance['parity']
        )

        # Normalized weights
        weights = {
            'corner': importance['corner'] / total_importance,
            'adjacent_corner': importance['adjacent_corner'] / total_importance,
            'mobility': importance['mobility'] / total_importance,
            'disc_difference': importance['disc_difference'][game_phase] / total_importance,
            'stability': importance['stability'] / total_importance,
            'parity': importance['parity'] / total_importance
        }

        score = 0

        # -------- Corners Heuristic --------
        corners = [
            (0, 0), (0, board_size - 1),
            (board_size - 1, 0), (board_size - 1, board_size - 1)
        ]
        corner_score = 0
        for pos in corners:
            if board[pos] == color:
                corner_score += 1
            elif board[pos] == opponent:
                corner_score -= 1
        score += corner_score * weights['corner']

        # -------- Adjacent to Corners Heuristic (X-squares) --------
        x_squares = [
            (1, 1),
            (1, board_size - 2),
            (board_size - 2, 1),
            (board_size - 2, board_size - 2)
        ]
        adjacent_corner_score = 0
        for pos in x_squares:
            if board[pos] == color:
                adjacent_corner_score += 1
            elif board[pos] == opponent:
                adjacent_corner_score -= 1
        score += adjacent_corner_score * weights['adjacent_corner']

        # -------- Mobility Heuristic --------
        player_moves = len(get_valid_moves(board, color))
        opponent_moves = len(get_valid_moves(board, opponent))
        if player_moves + opponent_moves != 0:
            mobility_score = (player_moves - opponent_moves) / (player_moves + opponent_moves)
            score += mobility_score * weights['mobility']

        # -------- Disc Difference Heuristic --------
        player_discs = np.count_nonzero(board == color)
        opponent_discs = np.count_nonzero(board == opponent)
        disc_difference = (player_discs - opponent_discs) / (player_discs + opponent_discs)
        score += disc_difference * weights['disc_difference']

        # -------- Stability Heuristic --------
        stable_player = self.stable_discs(board, color)
        stable_opponent = self.stable_discs(board, opponent)
        if stable_player + stable_opponent != 0:
            stability_score = (stable_player - stable_opponent) / (stable_player + stable_opponent)
            score += stability_score * weights['stability']

        # -------- Parity Heuristic --------
        parity_score = self.parity(board)
        score += parity_score * weights['parity']

        return score
    def parity(self, board):
        """
        Parity heuristic to favor having the last move.
        """
        empty_squares = np.count_nonzero(board == 0)
        return 1 if empty_squares % 2 == 1 else -1

    def stable_discs(self, board, color):
        """
        Count the number of stable discs for a given color using masks and vectorized operations.
        """
        board_size = board.shape[0]
        board_color = (board == color)
        
        stable = np.zeros_like(board_color, dtype=bool)
        
        # Stability from corners
        corners = [
            (0, 0), (0, board_size - 1),
            (board_size - 1, 0), (board_size - 1, board_size - 1)
        ]
        
        for corner in corners:
            x, y = corner
            if board_color[x, y]:
                self._mark_stable(board_color, stable, x, y)
        
        return np.count_nonzero(stable)

    def _mark_stable(self, board_color, stable, x, y):
        """
        Mark stable discs starting from a corner.
        """
        board_size = board_color.shape[0]
        stack = [(x, y)]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while stack:
            x, y = stack.pop()
            if stable[x, y]:
                continue
            stable[x, y] = True
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < board_size and 0 <= ny < board_size:
                    if board_color[nx, ny] and not stable[nx, ny]:
                        stack.append((nx, ny))

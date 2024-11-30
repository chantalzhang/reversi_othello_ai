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
        super().__init__()
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
        Evaluate the board state based on positional weights.
        Positive scores favor the maximizing player.
        """
        score = 0
        board_size = board.shape[0]
        total_squares = board.size
        empty_squares = np.count_nonzero(board == 0)
        empty_percentage = empty_squares / total_squares

        # Game phase detection (optional)
        if empty_percentage > 0.7:
            game_phase = 'early'
        elif empty_percentage > 0.3:
            game_phase = 'mid'
        else:
            game_phase = 'late'

        # Create a positional weight matrix
        positional_weights = self.create_positional_weights(board_size)

        # Calculate the positional score
        positional_score = 0
        for x in range(board_size):
            for y in range(board_size):
                if board[x, y] == color:
                    positional_score += positional_weights[x, y]
                elif board[x, y] == opponent:
                    positional_score -= positional_weights[x, y]

        score += positional_score

        # Optionally, include other heuristics if desired
        # For example, you can keep the mobility heuristic
        # Uncomment the following code to include mobility

        # -------- Mobility Heuristic --------
        # weights = {'mobility': 100 * (8 / board_size)}
        # player_moves = len(get_valid_moves(board, color))
        # opponent_moves = len(get_valid_moves(board, opponent))
        # if player_moves + opponent_moves != 0:
        #     mobility_score = weights['mobility'] * (player_moves - opponent_moves) / (player_moves + opponent_moves)
        #     score += mobility_score

        return score

    def create_positional_weights(self, board_size):
        """
        Create a positional weight matrix with higher weights for corners and edges,
        and decreasing weights towards the center.
        """
        weights = np.zeros((board_size, board_size))
        max_weight = 100  # Maximum weight for corners
        min_weight = 10   # Minimum weight for center positions

        # Calculate the maximum distance from the edge (for normalization)
        max_distance = board_size // 2

        for x in range(board_size):
            for y in range(board_size):
                # Calculate the minimum distance to the edge
                distance_to_edge = min(x, y, board_size - 1 - x, board_size - 1 - y)

                # Assign weights based on distance to edge
                if distance_to_edge == 0:
                    # Corner
                    weights[x, y] = max_weight
                elif distance_to_edge % 2 == 0:
                    # n-2, n-4, etc.
                    weight = max_weight - (distance_to_edge * 20)
                    weights[x, y] = weight
                else:
                    # Positions adjacent to corners or edges
                    weight = max_weight - (distance_to_edge * 30)
                    weights[x, y] = weight

                # Ensure weights are not below the minimum
                weights[x, y] = max(weights[x, y], min_weight)

        return weights

    # ... [Include any other helper methods if needed] ...

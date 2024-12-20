from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import random
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("actual_student")
class ActualStudentAgent(Agent):
    """
    Reversi Othello agent that uses minimax and heuristic search in an iterative deepening mannerto make decisions. 
    Implements alpha-beta pruning and transposition tables to optimize computation time to search deeper in the given time constraint. 

    """

    def __init__(self):
        super(ActualStudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.transposition_table = {}  # For caching board evaluations
        # to track stats: 
        self.breadth = 0
        self.max_breadth = 0


    def step(self, chess_board, player, opponent): 
        """
        Decide the best move at each step using iterative deepening and alpha-beta pruning.
        """
        start_time = time.time() 
        board_size = chess_board.shape[0]
        max_time = 1.93  # given the time constraint of 2 seconds , account for computation overhead
        depth = 1 
        best_move = None

        try:
            while True:
                elapsed_time = time.time() - start_time
                if elapsed_time >= max_time:
                    print(f"Exceeded time limit: {elapsed_time:.4f} seconds.")
                    break
                
                # check time before starting the next depth 
                self.check_time_limit(start_time, max_time)
                
                # to print out current breadth of search tree 
                valid_moves = get_valid_moves(chess_board, player)
                self.breadth = len(valid_moves)
                
                # Update max_breadth using class variable
                self.max_breadth = max(self.max_breadth, self.breadth)
                
                # begin minimax search at current depth 
                eval_score, move = self.minimax(
                    chess_board, depth, True, player, opponent,
                    float('-inf'), float('inf'), start_time, max_time)
                
                # update best move if found
                if move is not None:
                    best_move = move


                # check if the next depth can be searched within the time limit 
                self.check_time_limit(start_time, max_time)
                
                depth += 1

        except TimeoutError:
            pass

        # if no best move found, play a random valid move or pass if no moves are available
        if best_move is None:
            valid_moves = get_valid_moves(chess_board, player)
            if valid_moves:
                return random.choice(valid_moves)
            else:
                return None  
        
        print(f"My AI's turn took {time.time() - start_time:.4f} seconds. Best move found at depth {depth - 1}. Breadth searched: {self.max_breadth}")

        return best_move

    def minimax(self, chess_board, depth, is_maximizing_player, player, opponent, alpha, beta, start_time, time_limit):
        """
        Minimax algorithm with alpha-beta pruning and move ordering.
        """
        # Check time limit before starting search
        self.check_time_limit(start_time, time_limit)

        # Transposition table lookup for computational efficiency
        board_key = (tuple(chess_board.flatten()), depth, is_maximizing_player)
        if board_key in self.transposition_table:
            return self.transposition_table[board_key]

        # Check if node is a leaf node (endgame)
        is_endgame, _, _ = check_endgame(chess_board, player, opponent)
        if depth == 0 or is_endgame:
            score = self.evaluate_board(chess_board, player, opponent)
            # Store in transposition table
            self.transposition_table[board_key] = (score, None)
            return score, None

        if is_maximizing_player:
            best_val = float('-inf')
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
                self.check_time_limit(start_time, time_limit)  # Check time limit within loop
                new_board = np.copy(chess_board)
                execute_move(new_board, move, player)
                eval_score, _ = self.minimax(new_board, depth - 1, False, player, opponent,
                                            alpha, beta, start_time, time_limit)
                if eval_score > best_val:
                    best_val = eval_score
                    best_move = move
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break  # Beta cutoff for pruning
            # Store in transposition table
            self.transposition_table[board_key] = (best_val, best_move)
            return best_val, best_move
        else:
            best_val = float('inf')
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
                self.check_time_limit(start_time, time_limit)  # Check time limit within loop
                new_board = np.copy(chess_board)
                execute_move(new_board, move, opponent)
                eval_score, _ = self.minimax(new_board, depth - 1, True, player, opponent,
                                            alpha, beta, start_time, time_limit)
                if eval_score < best_val:
                    best_val = eval_score
                    best_move = move
                beta = min(beta, best_val)
                if beta <= alpha:
                    break  # Alpha cutoff for pruning
            # Store in transposition table
            self.transposition_table[board_key] = (best_val, best_move)
            return best_val, best_move


    def order_moves(self, board, moves, player, opponent, maximizing_player):
        """
        Order moves to improve alpha-beta pruning efficiency.
        """
        move_scores = []
        for move in moves:
            new_board = np.copy(board)
            execute_move(new_board, move, player)
            # quicker evaluation function for move ordering estimation
            score = self.quick_evaluate(new_board, player, opponent)
            move_scores.append((score, move))
        # sort moves: descending if maximizing, ascending if minimizing
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
        Positive scores favor the maximizing player whereas negative scores favor the minimizing player.
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

   
        weights = {
            'corner': 1000,
            'n2_corner': 50,
            'adjacent_corner': -20,
            'mobility': 10,
            'potential_mobility': 1,
            'disc_difference': {
                'early': 0,
                'mid': 4,
                'late': 8
            },
            'edge_stability': 10,
            'parity': 5,
            'stability': 5
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

    # --------- Helper functions --------- #
    def check_time_limit(self, start_time, max_time):
        time_taken = time.time() - start_time
        if time_taken >= max_time:
            raise TimeoutError("Exceeded the 2s time limit.")
        return time_taken

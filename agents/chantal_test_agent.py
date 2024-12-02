# Student agent: Improved agent with requested changes
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import random
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("chantal_test_agent")
class ChantalTestAgent(Agent):
    """
    An improved Othello agent implementing minimax with alpha-beta pruning,
    iterative deepening, transposition tables, and optimized heuristics.
    """

    def __init__(self):
        super(ChantalTestAgent, self).__init__()
        self.name = "ChantalTestAgent"
        self.transposition_table = {}  # For caching board evaluations
        self.breadth = 0  # Track breadth of the tree

    def step(self, chess_board, player, opponent): 
        """
        Decide the best move for the AI agent using iterative deepening and alpha-beta pruning with a 2-second max time.
        """
        start_time = time.time()
        board_size = chess_board.shape[0]

        max_time = 2.0 # im trying smtg new hold on 

        depth = 1 # counts each move-response pair but not number of completed depths 
        last_completed_depth = 0 # fully completed depth 
        best_move = None

        try:
            while True:
                elapsed_time = time.time() - start_time
                if elapsed_time >= max_time:
                    break  

                # to print out breadth of tree 
                valid_moves = get_valid_moves(chess_board, player)
                self.breadth = max(self.breadth, len(valid_moves))

                eval_score, move = self.minimax(
                    chess_board, depth, True, player, opponent,
                    float('-inf'), float('inf'), start_time, max_time
                )

                if move is not None:
                    best_move = move  
                
                last_completed_depth = depth

                depth += 1  
        except TimeoutError:
            pass  

        # radnom move if no best move found but prob doesnt happen often bc of mobi score
        if best_move is None:
            valid_moves = get_valid_moves(chess_board, player)
            if valid_moves:
                return random.choice(valid_moves)
            else:
                return None  

        # stats
        print(f"Depth searched: {depth - 1}")
        print(f"Last completed depth: {last_completed_depth}")
        print(f"Breadth searched: {self.breadth}")
        return best_move

    def minimax(self, chess_board, depth, maximizing_player, player, opponent, alpha, beta, start_time, time_limit):
        """
        Minimax algorithm with alpha-beta pruning and move ordering.
        """
        if time.time() - start_time >= time_limit - 0.2:  # we can chaneg this later for now its basically 1.8
            raise TimeoutError  
        
        board_key = (tuple(chess_board.flatten()), depth, maximizing_player)
        if board_key in self.transposition_table:
            return self.transposition_table[board_key]

        is_endgame, _, _ = check_endgame(chess_board, player, opponent)
        if depth == 0 or is_endgame:
            score = self.evaluate_board(chess_board, player, opponent)
            self.transposition_table[board_key] = (score, None)
            return score, None

        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            valid_moves = get_valid_moves(chess_board, player)
            self.breadth = max(self.breadth, len(valid_moves))  # Track maximum breadth
            if not valid_moves:
                eval_score, _ = self.minimax(chess_board, depth - 1, False, player, opponent,
                                             alpha, beta, start_time, time_limit)
                self.transposition_table[board_key] = (eval_score, None)
                return eval_score, None
            
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
                    break

            self.transposition_table[board_key] = (max_eval, best_move)
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            valid_moves = get_valid_moves(chess_board, opponent)
            self.breadth = max(self.breadth, len(valid_moves))  # Track maximum breadth
            if not valid_moves:
                eval_score, _ = self.minimax(chess_board, depth - 1, True, player, opponent,
                                             alpha, beta, start_time, time_limit)
                self.transposition_table[board_key] = (eval_score, None)
                return eval_score, None
           
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
                    break
            self.transposition_table[board_key] = (min_eval, best_move)
            return min_eval, best_move

    def order_moves(self, board, moves, player, opponent, maximizing_player):
        """
        Improved move ordering to enhance alpha-beta pruning.
        """
        move_scores = []
        for move in moves:
            new_board = np.copy(board)
            execute_move(new_board, move, player)
            score = self.quick_evaluate(new_board, player, opponent)  #fast
            move_scores.append((score, move))
        move_scores.sort(reverse=maximizing_player)
        return [move for _, move in move_scores]

    def quick_evaluate(self, board, player, opponent):
        """
        Simple evaluation function focusing on immediate gains.
        """
        return np.count_nonzero(board == player) - np.count_nonzero(board == opponent)

    def evaluate_board(self, board, player, opponent):
        """
        Evaluate the board state based on refined heuristics with inner corners.
        Positive scores favor the maximizing player.
        """
        score = 0
        board_size = board.shape[0]

        total_squares = board.size
        empty_squares = np.count_nonzero(board == 0)
        empty_percentage = empty_squares / total_squares


        # Game phase detection
        if empty_percentage > 0.8:
            game_phase = 'early'
        elif empty_percentage > 0.3:
            game_phase = 'mid'
        else:
            game_phase = 'late'


        # Heuristic weights
        weights = {
            'corner': 100,
            'inner_corner': 50,
            'adjacent_corner': -20,
            'edge_stability': 10,
            'mobility': 10,
            'disc_diff':{
                'early': -1,
                'mid': 0,
                'late': 5
            },
            'stability': 15,
        }

        # -------- Corners Heuristic --------
        corners = [
            (0, 0), (0, board_size - 1),
            (board_size - 1, 0), (board_size - 1, board_size - 1)
        ]
        corner_score = sum(
            weights['corner'] if board[pos] == player else -weights['corner'] if board[pos] == opponent else 0
            for pos in corners
        )
        score += corner_score

        # -------- Inner Corners Heuristic --------
        inner_corners = []
        for size in range(4, board_size, 2):  # Create grids for 4x4, 6x6, ..., up to board_size-2
            offset = (board_size - size) // 2
            inner_corners.extend([
                (offset, offset), (offset, offset + size - 1),
                (offset + size - 1, offset), (offset + size - 1, offset + size - 1)
            ])
        inner_corner_score = sum(
            weights['inner_corner'] if board[pos] == player else -weights['inner_corner'] if board[pos] == opponent else 0
            for pos in inner_corners if 0 <= pos[0] < board_size and 0 <= pos[1] < board_size
        )
        score += inner_corner_score

        # -------- Adjacent to Corners Heuristic --------
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
            weights['adjacent_corner'] if board[pos] == player else -weights['adjacent_corner'] if board[pos] == opponent else 0
            for pos in adjacent_positions if 0 <= pos[0] < board_size and 0 <= pos[1] < board_size
        )
        score += adjacent_corner_score

        # -------- Edge Stability Heuristic --------
        edge_stability = self.edge_stability(board, player) - self.edge_stability(board, opponent)
        score += weights['edge_stability'] * edge_stability

        # -------- Mobility Heuristic --------
        player_moves = len(get_valid_moves(board, player))
        opponent_moves = len(get_valid_moves(board, opponent))
        if player_moves + opponent_moves != 0:
            mobility_score = weights['mobility'] * (player_moves - opponent_moves) / (player_moves + opponent_moves)
            score += mobility_score

        # -------- Disc Parity Heuristic --------
        player_discs = np.count_nonzero(board == player)
        opponent_discs = np.count_nonzero(board == opponent)
        disc_parity = weights['disc_parity'] * (player_discs - opponent_discs)
        score += disc_parity

        # -------- Stability Heuristic --------
        stable_disc_count = self.stable_discs(board, player) - self.stable_discs(board, opponent)
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
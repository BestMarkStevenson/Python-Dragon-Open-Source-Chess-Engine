import pygame
import chess
import chess.polyglot
import random
import math
import os
import time
import sys  # Import sys for clean exit

# Constants
SCREEN_WIDTH = 512
SCREEN_HEIGHT = 512  # Board + space for messages/promotion
SQUARE_SIZE = SCREEN_WIDTH // 8
ANIMATION_SPEED = 25  # Pixels per frame, adjust for faster/slower animation
PROMOTION_POPUP_HEIGHT = SQUARE_SIZE * 1.5
PROMOTION_CHOICE_SIZE = SQUARE_SIZE - 10

# Colors
WHITE_SQUARE_COLOR = (240, 217, 181)  # Light wood
BLACK_SQUARE_COLOR = (181, 136, 99)  # Dark wood
SELECTED_COLOR = (100, 130, 255, 150)  # Semi-transparent blue
LEGAL_MOVE_COLOR = (0, 255, 0, 100)  # Semi-transparent green
POPUP_BG_COLOR = (200, 200, 200)
POPUP_TEXT_COLOR = (0, 0, 0)
BUTTON_COLOR = (70, 130, 180)  # SteelBlue
BUTTON_HOVER_COLOR = (100, 160, 210)
BUTTON_TEXT_COLOR = (255, 255, 255)

# Transposition Table Flags
TT_EXACT = 0
TT_LOWERBOUND = 1  # Value is at least this (score >= beta)
TT_UPPERBOUND = 2  # Value is at most this (score <= alpha)


class SelfLearningChess:
    def __init__(self):
        self.experience = {}
        self.win_count = 0
        self.move_history = {}

        # The depth for the minimax search. Higher depth means stronger play but slower calculation.
        # IMPORTANT!!!, THIS IS THE AMOUNT OF MOVES THE ENGINE THINKS AHEAD, ADJUST IT TO YOUR LIKING
        self.search_depth = 1
        self.piece_values = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
        }
        # Score representing a checkmate. Used for win/loss conditions.
        self.MATE_SCORE = 1000000
        # A threshold near MATE_SCORE to detect impending mates.
        self.MATE_THRESHOLD = self.MATE_SCORE - (self.search_depth * 1000)

        self.piece_square_tables = self._load_piece_square_tables()
        self.opening_book = self._load_opening_book()  # Now loads polyglot book

        self.transposition_table = {}  # Stores previously calculated board evaluations to speed up search.
        self.nodes_searched = 0  # Counter for nodes visited during search, for profiling.
        self.pv_line = []  # Principal Variation line: the sequence of moves considered best by the engine.

        # Stores board states (FEN without counters) for repetition detection.
        self.board_history_for_repetition = []

    def _load_piece_square_tables(self):
        """
        Loads the piece-square tables. These tables assign a score to each piece
        based on its position on the board, encouraging pieces to move to
        more advantageous squares.
        """
        pawn_table = [
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10, -20, -20, 10, 10, 5,
            5, -5, -10, 0, 0, -15, -5, 5,
            0, 0, 0, 20, 20, 0, 0, 0,
            5, 5, 10, 25, 25, 10, 5, 5,
            10, 10, 20, 30, 30, 20, 10, 10,
            50, 50, 50, 50, 50, 50, 50, 50,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
        knight_table = [
            -40, -30, -20, -20, -20, -20, -30, -40,
            -40, -20, 0, 5, 5, 0, -20, -40,
            -20, 0, 10, 15, 15, 10, 0, -20,
            -20, 5, 15, 20, 20, 15, 5, -20,
            -20, 0, 15, 20, 20, 15, 0, -20,
            -20, 5, 10, 15, 15, 10, 5, -20,
            -30, -20, 0, 5, 5, 0, -20, -30,
            -40, -30, -20, -20, -20, -20, -30, -40
        ]
        bishop_table = [
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -10, 0, 5, 10, 10, 5, 0, -10,
            -10, 5, 5, 10, 10, 5, 5, -10,
            -10, 0, 10, 10, 10, 10, 0, -10,
            -10, 10, 10, 10, 10, 10, 10, -10,
            -10, 5, 0, 0, 0, 0, 5, -10,
            -20, -10, -10, -10, -10, -10, -10, -20
        ]
        rook_table = [
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10, 10, 10, 10, 10, 5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            0, 0, 0, 5, 5, 0, 0, 0
        ]
        queen_table = [
            -20, -10, -10, -5, -5, -10, -10, -20,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -10, 0, 5, 5, 5, 5, 0, -10,
            -5, 0, 5, 5, 5, 5, 0, -5,
            0, 0, 5, 5, 5, 5, 0, -5,
            -10, 5, 5, 5, 5, 5, 0, -10,
            -10, 0, 5, 0, 0, 0, 0, -10,
            -20, -10, -10, -5, -5, -10, -10, -20
        ]
        king_opening_table = [
            -30, -30, -35, -50, -50, -40, -30, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -20, -30, -30, -40, -40, -30, -30, -20,
            -10, -20, -20, -20, -20, -20, -20, -10,
            20, 20, 0, 0, 0, 0, 20, 20,
            20, 30, 10, 0, 0, 10, 30, 20
        ]
        king_endgame_table = [
            -50, -40, -30, -20, -20, -30, -40, -50,
            -30, -20, -10, 0, 0, -10, -20, -30,
            -30, -10, 20, 30, 30, 20, -10, -30,
            -30, -10, 30, 40, 40, 30, -10, -30,
            -30, -10, 30, 40, 40, 30, -10, -30,
            -30, -10, 20, 30, 30, 20, -10, -30,
            -30, -30, 0, 0, 0, 0, -30, -30,
            -50, -30, -30, -30, -30, -30, -30, -50
        ]
        return {
            chess.PAWN: pawn_table, chess.KNIGHT: knight_table,
            chess.BISHOP: bishop_table, chess.ROOK: rook_table,
            chess.QUEEN: queen_table,
            'KING_OPENING': king_opening_table,
            'KING_ENDGAME': king_endgame_table
        }

    def _load_opening_book(self):
        """
        Loads the opening book from 'book.bin' using the chess.polyglot library.
        Assumes 'book.bin' is in the same directory as the script.
        """
        book_path = 'book.bin'
        if not os.path.exists(book_path):
            print(f"Warning: Opening book '{book_path}' not found. Engine will not use an opening book.")
            return None  # Return None if the book file is not found

        try:
            # chess.polyglot.open_reader() expects the path to the .bin file
            return chess.polyglot.open_reader(book_path)
        except Exception as e:
            print(f"Error loading opening book '{book_path}': {e}")
            return None

    def _get_piece_square_score(self, piece, square, board):
        if not piece:
            return 0

        table_key = piece.piece_type
        if piece.piece_type == chess.KING:
            num_pieces = len(board.piece_map())
            queen_exists_white = bool(board.pieces(chess.QUEEN, chess.WHITE))
            queen_exists_black = bool(board.pieces(chess.QUEEN, chess.BLACK))
            is_endgame = num_pieces <= 10 or (not queen_exists_white and not queen_exists_black and num_pieces <= 14)

            table_to_use = self.piece_square_tables['KING_ENDGAME'] if is_endgame else self.piece_square_tables[
                'KING_OPENING']
        else:
            table_to_use = self.piece_square_tables.get(table_key)

        if table_to_use:
            idx = square if piece.color == chess.WHITE else chess.square_mirror(square)
            return table_to_use[idx]
        return 0

    def evaluate_board(self, board):
        if board.is_checkmate():
            outcome = board.outcome()
            if outcome and outcome.winner == chess.WHITE: return self.MATE_SCORE
            if outcome and outcome.winner == chess.BLACK: return -self.MATE_SCORE
            # This case should ideally not be reached if outcome.winner is set
            return -self.MATE_SCORE if board.turn == chess.WHITE else self.MATE_SCORE

        # Updated to use is_game_over() for consistency with chess.Board.
        # This will cover stalemate, insufficient material, 50-move rule, and 3-fold repetition.
        if board.is_game_over():
            return 0  # Draw score

        score = 0
        for square_idx in chess.SQUARES:
            piece = board.piece_at(square_idx)
            if piece:
                value = self.piece_values.get(piece.piece_type, 0)
                positional_score = self._get_piece_square_score(piece, square_idx, board)
                piece_total_score = value + positional_score
                if piece.color == chess.WHITE:
                    score += piece_total_score
                else:
                    score -= piece_total_score
        return score

    def _order_moves(self, board, moves, heuristic_best_move=None):
        def get_move_score(move):
            score = 0
            if move == heuristic_best_move:
                score += 1000000  # Prioritize the best move from TT

            if board.is_capture(move):
                victim_piece = board.piece_at(move.to_square)
                if board.is_en_passant(move):
                    victim_val = self.piece_values[chess.PAWN]
                elif victim_piece:
                    victim_val = self.piece_values.get(victim_piece.piece_type, 0)
                else:  # Should not happen for a capture
                    victim_val = 0

                attacker_piece = board.piece_at(move.from_square)
                attacker_val = self.piece_values.get(attacker_piece.piece_type, 1) if attacker_piece else 1
                score += 1000 + victim_val - (attacker_val / 10)  # MVV-LVA

            if board.gives_check(move):
                score += 500

            if move.promotion:
                score += self.piece_values.get(move.promotion, 0) * 2
            return score

        return sorted(moves, key=get_move_score, reverse=True)

    def quiescence_search(self, board, alpha, beta):
        self.nodes_searched += 1

        if board.is_checkmate():
            outcome = board.outcome()
            if outcome and outcome.winner == chess.WHITE: return self.MATE_SCORE
            if outcome and outcome.winner == chess.BLACK: return -self.MATE_SCORE
            return -self.MATE_SCORE if board.turn == chess.WHITE else self.MATE_SCORE
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        stand_pat = self.evaluate_board(board)

        if board.turn == chess.WHITE:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
        else:  # board.turn == chess.BLACK
            if stand_pat <= alpha:
                return alpha
            beta = min(beta, stand_pat)

        # Only consider noisy moves (captures and promotions)
        noisy_moves = [move for move in board.legal_moves if board.is_capture(move) or move.promotion]
        ordered_noisy_moves = self._order_moves(board, noisy_moves)

        if board.turn == chess.WHITE:
            for move in ordered_noisy_moves:
                board.push(move)
                score = self.quiescence_search(board, alpha, beta)
                board.pop()
                if score >= beta:
                    return beta
                alpha = max(alpha, score)
            return alpha
        else:  # Minimizing player
            for move in ordered_noisy_moves:
                board.push(move)
                score = self.quiescence_search(board, alpha, beta)
                board.pop()
                if score <= alpha:
                    return alpha
                beta = min(beta, score)
            return beta

    def minimax(self, board, depth, alpha, beta, maximizing_player, ply_from_root=0):
        self.nodes_searched += 1
        original_alpha = alpha
        # original_beta = beta # Not used in this version for setting TT flag directly from original_beta

        board_hash = chess.polyglot.zobrist_hash(board)
        tt_entry = self.transposition_table.get(board_hash)
        tt_best_move = None

        if tt_entry:
            tt_score, tt_depth, tt_flag, tt_best_move_stored_uci = tt_entry
            if tt_depth >= depth:
                if tt_flag == TT_EXACT:
                    return tt_score
                elif tt_flag == TT_LOWERBOUND:
                    alpha = max(alpha, tt_score)
                elif tt_flag == TT_UPPERBOUND:
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score

            if tt_best_move_stored_uci:
                try:
                    move_obj = chess.Move.from_uci(tt_best_move_stored_uci)
                    if board.is_legal(move_obj):
                        tt_best_move = move_obj
                except ValueError:
                    tt_best_move = None

        if board.is_checkmate():
            outcome = board.outcome()
            if outcome and outcome.winner == chess.WHITE: return self.MATE_SCORE - ply_from_root
            if outcome and outcome.winner == chess.BLACK: return -self.MATE_SCORE + ply_from_root
            return (-self.MATE_SCORE + ply_from_root) if board.turn == chess.WHITE else (
                    self.MATE_SCORE - ply_from_root)

        if board.is_game_over():  # Catches stalemate, insufficient material, 50-move rule, 3-fold repetition
            return 0  # Draw score

        if depth == 0:
            return self.quiescence_search(board, alpha, beta)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return 0  # Stalemate or no legal moves

        # Order moves based on heuristics
        ordered_moves = self._order_moves(board, legal_moves, heuristic_best_move=tt_best_move)

        best_move_found_this_node = None
        # Penalty for repeating a position when the AI is winning
        REPETITION_PENALTY_VALUE_WINNING = 100000

        if maximizing_player:
            max_eval = -float('inf')
            for move in ordered_moves:
                board.push(move)
                fen_no_counters = board.fen().rsplit(' ', 2)[0]
                repetition_count = self.board_history_for_repetition.count(fen_no_counters)

                eval_score = self.minimax(board, depth - 1, alpha, beta, False, ply_from_root + 1)

                # Apply repetition penalty if machine is winning and repeating the position
                if repetition_count >= 1:  # Position has occurred twice before, this would be the third time
                    # Check if the AI (White, maximizing player) is winning in the resulting line
                    # eval_score here is from the perspective of the next player (minimizing player),
                    # so a positive eval_score means it's good for White.
                    if eval_score > 0:  # AI is winning
                        eval_score = -REPETITION_PENALTY_VALUE_WINNING  # Penalize severely to avoid draw

                board.pop()

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move_found_this_node = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break

            flag_to_store = TT_EXACT
            if max_eval <= original_alpha:
                flag_to_store = TT_UPPERBOUND
            elif max_eval >= beta:  # Use beta directly for lower bound check
                flag_to_store = TT_LOWERBOUND

            if not tt_entry or depth >= tt_entry[1] or \
                    (depth == tt_entry[1] and flag_to_store == TT_EXACT and tt_entry[2] != TT_EXACT):
                self.transposition_table[board_hash] = (
                    max_eval, depth, flag_to_store,
                    best_move_found_this_node.uci() if best_move_found_this_node else None)
            return max_eval
        else:  # Minimizing player (AI is Black)
            min_eval = float('inf')
            for move in ordered_moves:
                board.push(move)
                fen_no_counters = board.fen().rsplit(' ', 2)[0]
                repetition_count = self.board_history_for_repetition.count(fen_no_counters)

                eval_score = self.minimax(board, depth - 1, alpha, beta, True, ply_from_root + 1)

                # Apply repetition penalty if machine is winning and repeating the position
                if repetition_count >= 1:  # If this move causes a third repetition of a position
                    # Check if the AI (Black, minimizing player) is winning in the resulting line
                    # eval_score here is from the perspective of the next player (maximizing player),
                    # so a negative eval_score means it's good for Black.
                    if eval_score < 0:  # AI is winning
                        eval_score = REPETITION_PENALTY_VALUE_WINNING  # Penalize severely (large positive value for black is bad)

                board.pop()

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move_found_this_node = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break

            flag_to_store = TT_EXACT
            if min_eval >= beta:  # Use beta directly for lower bound check
                flag_to_store = TT_LOWERBOUND
            elif min_eval <= alpha:
                flag_to_store = TT_UPPERBOUND

            if not tt_entry or depth >= tt_entry[1] or \
                    (depth == tt_entry[1] and flag_to_store == TT_EXACT and tt_entry[2] != TT_EXACT):
                self.transposition_table[board_hash] = (
                    min_eval, depth, flag_to_store,
                    best_move_found_this_node.uci() if best_move_found_this_node else None)
            return min_eval

    def choose_move(self, board):
        self.nodes_searched = 0
        self.pv_line = []
        start_time = time.time()

        legal_moves_initial = list(board.legal_moves)
        if not legal_moves_initial:
            return None

        # Check for immediate mate in one
        for move in legal_moves_initial:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                print(f"AI found mate in one: {move.uci()}")
                return move
            board.pop()

        # Try to use opening book if loaded and if the game is still in the early stages
        # Polyglot book works directly with the board object.
        if self.opening_book and len(board.move_stack) < 20:  # Use book for first 20 half-moves
            try:
                # weighted_choice selects a move from the book based on its weight
                book_move_entry = self.opening_book.weighted_choice(board)
                if book_move_entry:
                    # Access the move directly from the BookEntry object
                    book_move = book_move_entry.move
                    if board.is_legal(book_move):
                        print(f"AI playing from opening book: {book_move.uci()}")
                        return book_move
            except IndexError:
                # No entry found for the current position in the book
                print("No book move found for this position.")
            except Exception as e:
                print(f"Error reading from opening book: {e}")

        best_move_overall = None
        best_score_overall = -float('inf') if board.turn == chess.WHITE else float('inf')
        actual_depth_searched = 0

        # Iterative Deepening
        for current_depth_iter in range(1, self.search_depth + 1):
            actual_depth_searched = current_depth_iter
            alpha = -float('inf')
            beta = float('inf')

            heuristic_root_move = self.pv_line[0] if self.pv_line and self.pv_line[0] in legal_moves_initial else None
            ordered_root_moves = self._order_moves(board, legal_moves_initial, heuristic_best_move=heuristic_root_move)

            current_best_move_this_depth = None

            # Calculate the current material score to assess if machine is winning
            current_eval_of_board = self.evaluate_board(board)

            # Define repetition penalty at the root. This is applied to moves
            # that lead to a repetition if the AI is currently winning.
            REPETITION_PENALTY_AT_ROOT = 1000000  # A very large penalty to strongly discourage drawing in a winning position

            if board.turn == chess.WHITE:  # Maximizing player
                current_max_eval = -float('inf')
                for i, move in enumerate(ordered_root_moves):
                    board.push(move)
                    # Check for repetition of the *resulting* board state against the game's actual history
                    fen_no_counters = board.fen().rsplit(' ', 2)[0]
                    repetition_count = self.board_history_for_repetition.count(fen_no_counters)

                    score = self.minimax(board, current_depth_iter - 1, alpha, beta, False, ply_from_root=1)

                    # Apply penalty at the root level if AI is winning and repeating
                    if repetition_count >= 1:  # If this move causes a third repetition of a position
                        # Check if the AI (White) is winning in the *current* actual game board state
                        if current_eval_of_board > 0:  # White is currently winning (positive score)
                            score = -REPETITION_PENALTY_AT_ROOT  # Penalize severely to avoid draw

                    board.pop()

                    if score > current_max_eval:
                        current_max_eval = score
                        current_best_move_this_depth = move
                    alpha = max(alpha, current_max_eval)
                    # If alpha is greater than or equal to beta, we can prune
                    if alpha >= beta:
                        break  # Prune this branch
                best_score_overall = current_max_eval
                best_move_overall = current_best_move_this_depth
                if best_move_overall: self.pv_line = [best_move_overall]

            else:  # AI is Black, Minimizing player
                current_min_eval = float('inf')
                for i, move in enumerate(ordered_root_moves):
                    board.push(move)
                    fen_no_counters = board.fen().rsplit(' ', 2)[0]
                    repetition_count = self.board_history_for_repetition.count(fen_no_counters)

                    score = self.minimax(board, current_depth_iter - 1, alpha, beta, True, ply_from_root=1)

                    # Apply penalty at the root level if AI is winning and repeating
                    if repetition_count >= 1:  # If this move causes a third repetition of a position
                        # Check if the AI (Black) is winning in the *current* actual game board state
                        if current_eval_of_board < 0:  # Black is currently winning (negative score)
                            score = REPETITION_PENALTY_AT_ROOT  # Penalize severely (large positive value for black is bad)

                    board.pop()

                    if score < current_min_eval:
                        current_min_eval = score
                        current_best_move_this_depth = move
                    beta = min(beta, current_min_eval)
                    # If beta is less than or equal to alpha, we can prune
                    if beta <= alpha:
                        break  # Prune this branch
                best_score_overall = current_min_eval
                best_move_overall = current_best_move_this_depth
                if best_move_overall: self.pv_line = [best_move_overall]

            # If a mate is found at this depth, no need to search deeper
            if abs(best_score_overall) >= self.MATE_THRESHOLD:
                break

        end_time = time.time()

        if not best_move_overall and legal_moves_initial:
            print("AI choosing a random fallback move (no best move found).")
            return random.choice(legal_moves_initial)

        print(
            f"AI chose: {best_move_overall.uci() if best_move_overall else 'None'}, Score: {best_score_overall:.0f}, Depth: {actual_depth_searched}/{self.search_depth}, Nodes: {self.nodes_searched}, Time: {end_time - start_time:.2f}s")
        return best_move_overall


class ChessGUI_Pygame:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Open Source Dragon Engine")
        self.screen_height_actual = SCREEN_HEIGHT
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, self.screen_height_actual))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)
        self.small_font = pygame.font.SysFont("Arial", 18)
        self.large_font = pygame.font.SysFont("Arial", 36)  # For start screen title
        # Removed self.label_font as labels are no longer drawn

        self.board = chess.Board()
        self.engine = SelfLearningChess()
        self.images = {}
        self.load_images()

        self.selected_square = None
        self.legal_moves_for_selected_piece = []
        self.animating_pieces = []
        self.promotion_active = False
        self.promotion_square_from = None
        self.promotion_square_to = None
        self.promotion_choices = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        self.promotion_rects = []
        self.game_over_message = ""

        # New attributes for game start and player color selection
        self.game_started = False
        self.player_color = None  # Will be chess.WHITE or chess.BLACK
        self.pending_logical_move = None  # Initialize to None to prevent AttributeError

        # Initialize the board history for the engine with the starting FEN
        # Store only the relevant FEN parts for repetition detection (stripping halfmove and fullmove counters)
        self.engine.board_history_for_repetition.append(self.board.fen().rsplit(' ', 2)[0])

    def load_images(self):
        pieces_map = {
            'P': 'wp', 'N': 'wn', 'B': 'wb', 'R': 'wr', 'Q': 'wq', 'K': 'wk',
            'p': 'bp', 'n': 'bn', 'b': 'bb', 'r': 'br', 'q': 'bq', 'k': 'bk'
        }
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pieces_base_dir = os.path.join(script_dir, "pieces")

        if not os.path.isdir(pieces_base_dir):
            print(f"Error: 'pieces' directory not found at {pieces_base_dir}")
            # Fallback: create placeholder images if 'pieces' directory is missing
            for symbol in pieces_map.keys():
                fallback_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                fallback_surface.fill((128, 128, 128, 100))
                pygame.draw.rect(fallback_surface, (0, 0, 0), fallback_surface.get_rect(), 1)
                text = self.small_font.render(symbol, True, (0, 0, 0))
                text_rect = text.get_rect(center=fallback_surface.get_rect().center)
                fallback_surface.blit(text, text_rect)
                self.images[symbol] = fallback_surface
            return

        for symbol, name in pieces_map.items():
            path = os.path.join(pieces_base_dir, f"{name}.png")
            try:
                img = pygame.image.load(path).convert_alpha()
                self.images[symbol] = pygame.transform.smoothscale(img, (SQUARE_SIZE, SQUARE_SIZE))
            except pygame.error as e:
                print(f"Error loading image {path}: {e}")
                # Fallback: create a red placeholder if an image is missing
                fallback_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                fallback_surface.fill((255, 0, 0, 100))
                pygame.draw.rect(fallback_surface, (0, 0, 0), fallback_surface.get_rect(), 1)
                text = self.small_font.render(symbol, True, (0, 0, 0))
                text_rect = text.get_rect(center=fallback_surface.get_rect().center)
                fallback_surface.blit(text, text_rect)
                self.images[symbol] = fallback_surface

    def square_to_pixels(self, square):
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        x = file * SQUARE_SIZE
        y = (7 - rank) * SQUARE_SIZE  # Default for White's perspective

        if self.player_color == chess.BLACK:
            # Flip horizontally (files)
            x = (7 - file) * SQUARE_SIZE
            # Flip vertically (ranks)
            y = rank * SQUARE_SIZE

        return (x, y)

    def pixels_to_square(self, x, y):
        if y >= SCREEN_HEIGHT:
            return None

        file = x // SQUARE_SIZE
        rank = 7 - (y // SQUARE_SIZE)  # Default for White's perspective

        if self.player_color == chess.BLACK:
            # Un-flip horizontally
            file = 7 - (x // SQUARE_SIZE)
            # Un-flip vertically
            rank = y // SQUARE_SIZE

        if 0 <= file <= 7 and 0 <= rank <= 7:
            return chess.square(file, rank)
        return None

    def draw_board(self):
        for square_index in chess.SQUARES:
            file = chess.square_file(square_index)
            rank = chess.square_rank(square_index)

            # Determine color based on standard board logic: A1 is dark.
            # (rank + file) % 2 == 0 corresponds to A1, C1, E1, G1, B2, D2, etc.
            # These should be dark squares in a standard setup.
            color = BLACK_SQUARE_COLOR if (rank + file) % 2 == 0 else WHITE_SQUARE_COLOR

            # Get pixel coordinates based on player's perspective
            x, y = self.square_to_pixels(square_index)
            pygame.draw.rect(self.screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))

    def draw_pieces(self):
        # List of squares that are currently involved in an animation and should NOT be drawn statically
        animating_squares = set()
        for anim_info in self.animating_pieces:
            animating_squares.add(anim_info['start_square'])
            # Add the target square of the main piece to ensure no duplicate static drawing
            if anim_info['move']:  # Only if it's the primary move
                animating_squares.add(anim_info['move'].to_square)

        # Handle en passant captured pawn's original square.
        # This piece is removed from the board *logically* when the move is made.
        # During animation, it should disappear immediately.
        # We need to know if the pending logical move *was* an en passant.
        if self.pending_logical_move and self.board.is_en_passant(self.pending_logical_move):
            # The captured pawn is on the rank of the moving pawn's origin, but on the file of the target square.
            # Example: white pawn on e5 captures black pawn on d5 -> black pawn was on d4
            # If white to move: pawn moves from rank 5 to rank 6, captured pawn was on rank 5
            # If black to move: pawn moves from rank 4 to rank 3, captured pawn was on rank 4
            captured_pawn_square = chess.square(
                chess.square_file(self.pending_logical_move.to_square),
                chess.square_rank(self.pending_logical_move.from_square)
            )
            animating_squares.add(captured_pawn_square)

        for square_index in chess.SQUARES:
            # Skip drawing pieces that are currently animating from their starting square
            # or those that are captured (and thus no longer on the board for static drawing)
            # A piece at the target square might also be captured, so don't draw it if it's the target of the pending move.
            if square_index in animating_squares:
                continue
            if self.pending_logical_move and square_index == self.pending_logical_move.to_square and self.board.piece_at(
                    square_index) is not None:
                continue  # Do not draw a captured piece at the destination square prematurely

            piece = self.board.piece_at(square_index)
            if piece:
                symbol = piece.symbol()
                if symbol in self.images:
                    x, y = self.square_to_pixels(square_index)
                    self.screen.blit(self.images[symbol], (x, y))

    def draw_selected_square_and_moves(self):
        if self.selected_square is not None:
            x, y = self.square_to_pixels(self.selected_square)
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            s.fill(SELECTED_COLOR)
            self.screen.blit(s, (x, y))

            for move in self.legal_moves_for_selected_piece:
                target_x, target_y = self.square_to_pixels(move.to_square)
                center_x = target_x + SQUARE_SIZE // 2
                center_y = target_y + SQUARE_SIZE // 2
                if self.board.is_capture(move):
                    # Draw a transparent circle to indicate capture
                    pygame.draw.circle(self.screen, (0, 0, 0, 50), (center_x, center_y), SQUARE_SIZE // 3, 5)
                    pygame.draw.circle(self.screen, LEGAL_MOVE_COLOR, (center_x, center_y), SQUARE_SIZE // 3 - 5)
                else:
                    pygame.draw.circle(self.screen, LEGAL_MOVE_COLOR, (center_x, center_y), SQUARE_SIZE // 6)

    def draw_animating_piece(self):
        for anim_info in self.animating_pieces:
            piece_symbol = anim_info.get('promoted_symbol', anim_info['original_piece_at_start_square'].symbol())

            # Add a safety check just in case piece_symbol somehow becomes None or is not in images
            if piece_symbol is None or piece_symbol not in self.images:
                print(f"Warning: Cannot draw animating piece. Symbol '{piece_symbol}' is invalid or missing image.")
                continue  # Skip drawing this piece

            piece_img = self.images[piece_symbol]
            self.screen.blit(piece_img, anim_info['current_pos'])

    def update_animation(self):
        if not self.animating_pieces:
            return False  # No animation in progress

        all_pieces_reached_destination = True
        SNAP_THRESHOLD = 1.5

        for anim_info in self.animating_pieces:
            curr_x, curr_y = anim_info['current_pos']
            end_px, end_py = anim_info['end_pos_pixels_actual']

            dx = end_px - curr_x
            dy = end_py - curr_y
            distance = math.sqrt(dx * dx + dy * dy)

            if distance > SNAP_THRESHOLD:
                all_pieces_reached_destination = False
                move_amount = min(ANIMATION_SPEED, distance)
                anim_info['current_pos'][0] += (dx / distance) * move_amount
                anim_info['current_pos'][1] += (dy / distance) * move_amount
            else:
                # Snap to final position to prevent floating point inaccuracies
                anim_info['current_pos'][0] = end_px
                anim_info['current_pos'][1] = end_py

        if all_pieces_reached_destination:
            # All animations finished
            if self.pending_logical_move:
                self.board.push(self.pending_logical_move)  # Apply the logical move now
                # Add the board's new FEN to the engine's history after the move is *actually* made.
                # Store only the relevant FEN parts for repetition detection (stripping halfmove and fullmove counters)
                self.engine.board_history_for_repetition.append(self.board.fen().rsplit(' ', 2)[0])
                self.pending_logical_move = None  # Clear pending move

            self.animating_pieces = []  # Clear all animation states
            self.check_game_over()

            # Explicitly clear selection state after animation, just in case of lingering visual bugs
            self.selected_square = None
            self.legal_moves_for_selected_piece = []

            return True  # Animation just finished
        return False  # Animation still in progress

    def start_move_animation(self, move, is_player_move):
        # Get the piece that is *actually* moving
        piece_to_animate_obj = self.board.piece_at(move.from_square)
        if not piece_to_animate_obj:
            # This should ideally not happen if it's a legal move, but as a safeguard
            print(f"Error: No piece found at {move.from_square} for move {move.uci()}")
            return

        # Prepare for animation (this doesn't change the board state yet)
        start_pixels = list(self.square_to_pixels(move.from_square))
        end_pixels = self.square_to_pixels(move.to_square)

        # Store the main moving piece's animation info
        self.animating_pieces.append({
            'start_square': move.from_square,
            'piece_img': self.images[piece_to_animate_obj.symbol()],
            'current_pos': start_pixels,
            'end_pos_pixels_actual': end_pixels,
            'move': move,  # Store the actual move object for logical application later
            'original_piece_at_start_square': piece_to_animate_obj
        })

        # Handle castling: also animate the rook
        if self.board.has_castling_rights(piece_to_animate_obj.color) and piece_to_animate_obj.piece_type == chess.KING:
            if move == chess.Move.from_uci("e1g1") and self.board.piece_at(chess.H1) and self.board.piece_at(
                    chess.H1).piece_type == chess.ROOK:  # White kingside
                rook_start = chess.H1
                rook_end = chess.F1
                rook_piece = self.board.piece_at(rook_start)
                self.animating_pieces.append({
                    'start_square': rook_start,
                    'piece_img': self.images[rook_piece.symbol()],
                    'current_pos': list(self.square_to_pixels(rook_start)),
                    'end_pos_pixels_actual': self.square_to_pixels(rook_end),
                    'move': None,  # This is a secondary animation, no logical move attached
                    'original_piece_at_start_square': rook_piece
                })
            elif move == chess.Move.from_uci("e1c1") and self.board.piece_at(chess.A1) and self.board.piece_at(
                    chess.A1).piece_type == chess.ROOK:  # White queenside
                rook_start = chess.A1
                rook_end = chess.D1
                rook_piece = self.board.piece_at(rook_start)
                self.animating_pieces.append({
                    'start_square': rook_start,
                    'piece_img': self.images[rook_piece.symbol()],
                    'current_pos': list(self.square_to_pixels(rook_start)),
                    'end_pos_pixels_actual': self.square_to_pixels(rook_end),
                    'move': None,
                    'original_piece_at_start_square': rook_piece
                })
            elif move == chess.Move.from_uci("e8g8") and self.board.piece_at(chess.H8) and self.board.piece_at(
                    chess.H8).piece_type == chess.ROOK:  # Black kingside
                rook_start = chess.H8
                rook_end = chess.F8
                rook_piece = self.board.piece_at(rook_start)
                self.animating_pieces.append({
                    'start_square': rook_start,
                    'piece_img': self.images[rook_piece.symbol()],
                    'current_pos': list(self.square_to_pixels(rook_start)),
                    'end_pos_pixels_actual': self.square_to_pixels(rook_end),
                    'move': None,
                    'original_piece_at_start_square': rook_piece
                })
            elif move == chess.Move.from_uci("e8c8") and self.board.piece_at(chess.A8) and self.board.piece_at(
                    chess.A8).piece_type == chess.ROOK:  # Black queenside
                rook_start = chess.A8
                rook_end = chess.D8
                rook_piece = self.board.piece_at(rook_start)
                self.animating_pieces.append({
                    'start_square': rook_start,
                    'piece_img': self.images[rook_piece.symbol()],
                    'current_pos': list(self.square_to_pixels(rook_start)),
                    'end_pos_pixels_actual': self.square_to_pixels(rook_end),
                    'move': None,
                    'original_piece_at_start_square': rook_piece
                })

        # Set the pending logical move. Turn management is now based on board.turn directly.
        self.pending_logical_move = move

    def handle_click(self, pos):
        """
        Handles mouse clicks on the board. Manages piece selection,
        move attempts, and promotion choices.
        """
        # Player can only click if it's their turn, and no animations or game over.
        if self.game_over_message or self.promotion_active or self.animating_pieces or self.board.turn != self.player_color:
            return

        clicked_square = self.pixels_to_square(pos[0], pos[1])

        if clicked_square is None:
            # Click outside board area
            self.selected_square = None
            self.legal_moves_for_selected_piece = []
            return

        piece_at_clicked_square = self.board.piece_at(clicked_square)

        if self.selected_square is None:
            # No piece selected, try to select one
            if piece_at_clicked_square and piece_at_clicked_square.color == self.board.turn:
                self.selected_square = clicked_square
                # Get legal moves for the selected piece
                self.legal_moves_for_selected_piece = [
                    move for move in self.board.legal_moves
                    if move.from_square == self.selected_square
                ]
        else:
            # A piece is already selected, try to make a move
            attempted_move = chess.Move(self.selected_square, clicked_square)

            # Check for promotion first
            is_promotion_move = False
            # Get the piece at the selected square for promotion check
            piece_at_selected_square = self.board.piece_at(self.selected_square)

            # Check if it's a pawn move to the last rank and if the piece is actually a pawn
            if piece_at_selected_square and \
                    piece_at_selected_square.piece_type == chess.PAWN and \
                    chess.square_rank(self.selected_square) == (6 if self.board.turn == chess.WHITE else 1) and \
                    chess.square_rank(clicked_square) == (7 if self.board.turn == chess.WHITE else 0):
                # This is potentially a promotion move. We need to check all legal promotion moves.
                # The `chess.Board` generates moves with promotion (e.g., a7b8q)
                possible_promotion_moves = [
                    move for move in self.legal_moves_for_selected_piece
                    if move.to_square == clicked_square and move.promotion is not None
                ]
                if possible_promotion_moves:
                    self.promotion_active = True
                    self.promotion_square_from = self.selected_square
                    self.promotion_square_to = clicked_square
                    self.draw_promotion_popup()  # Draw the popup to get choice
                    is_promotion_move = True  # Indicate that it's a promotion scenario

            if is_promotion_move:
                # If a promotion popup is active, don't clear selection yet, await choice
                pass
            elif attempted_move in self.legal_moves_for_selected_piece:
                # If it's a regular legal move, perform it
                self.start_move_animation(attempted_move, is_player_move=True)
                self.selected_square = None
                self.legal_moves_for_selected_piece = []
            else:
                # If it's not a legal move for the selected piece, try to re-select
                self.selected_square = None
                self.legal_moves_for_selected_piece = []
                # If the clicked square has a piece of the current player's color, select it
                if piece_at_clicked_square and piece_at_clicked_square.color == self.board.turn:
                    self.selected_square = clicked_square
                    self.legal_moves_for_selected_piece = [
                        move for move in self.board.legal_moves
                        if move.from_square == self.selected_square
                    ]

    def draw_promotion_popup(self):
        if not self.promotion_active:
            return

        # Calculate popup position to be centered
        popup_width = SQUARE_SIZE * len(self.promotion_choices)
        popup_height = PROMOTION_POPUP_HEIGHT
        popup_x = (SCREEN_WIDTH - popup_width) // 2
        popup_y = (SCREEN_HEIGHT - popup_height) // 2

        pygame.draw.rect(self.screen, POPUP_BG_COLOR, (popup_x, popup_y, popup_width, popup_height))
        pygame.draw.rect(self.screen, POPUP_TEXT_COLOR, (popup_x, popup_y, popup_width, popup_height), 3)

        self.promotion_rects = []
        for i, piece_type in enumerate(self.promotion_choices):
            symbol = chess.Piece(piece_type, self.board.turn).symbol()
            piece_img = self.images.get(symbol)
            if piece_img:
                choice_x = popup_x + (i * SQUARE_SIZE) + (SQUARE_SIZE - PROMOTION_CHOICE_SIZE) // 2
                choice_y = popup_y + (popup_height - PROMOTION_CHOICE_SIZE) // 2
                choice_rect = pygame.Rect(choice_x, choice_y, PROMOTION_CHOICE_SIZE, PROMOTION_CHOICE_SIZE)
                self.screen.blit(
                    pygame.transform.smoothscale(piece_img, (PROMOTION_CHOICE_SIZE, PROMOTION_CHOICE_SIZE)),
                    choice_rect)
                self.promotion_rects.append((choice_rect, piece_type))

    def handle_promotion_click(self, pos):
        for rect, piece_type in self.promotion_rects:
            if rect.collidepoint(pos):
                promoted_move = chess.Move(
                    self.promotion_square_from,
                    self.promotion_square_to,
                    promotion=piece_type
                )
                if promoted_move in self.board.legal_moves:
                    self.start_move_animation(promoted_move, is_player_move=True)  # Animation for player's move
                self.promotion_active = False
                self.promotion_rects = []
                self.promotion_square_from = None
                self.promotion_square_to = None
                self.selected_square = None  # Ensure this is explicitly set to None
                self.legal_moves_for_selected_piece = []  # Ensure this is explicitly cleared
                return True
        return False

    def check_game_over(self):
        """
        Checks if the game has ended (checkmate, stalemate, draw by rules).
        Sets the game_over_message accordingly.
        """
        if self.board.is_checkmate():
            winner_color = "White" if self.board.turn == chess.BLACK else "Black"
            self.game_over_message = f"Checkmate! {winner_color} wins!"
        elif self.board.is_stalemate():
            self.game_over_message = "Stalemate! It's a draw."
        elif self.board.is_insufficient_material():
            self.game_over_message = "Insufficient material! It's a draw."
        elif self.board.is_fifty_moves():
            self.game_over_message = "Fifty-move rule! It's a draw."
        elif self.board.is_repetition():  # Threefold repetition
            self.game_over_message = "Threefold repetition! It's a draw."
        # You could also add `board.is_seventyfive_moves()` for a more strict rule

    def display_message(self, message):
        """Displays a message on the screen, typically for game over or turn."""
        text_surface = self.font.render(message, True, POPUP_TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT + PROMOTION_POPUP_HEIGHT // 2))
        self.screen.blit(text_surface, text_rect)

    def draw_game_over_message(self):
        if self.game_over_message:
            text_surface = self.font.render(self.game_over_message, True, POPUP_TEXT_COLOR)
            text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            # Draw a background rectangle for the message
            bg_rect = text_rect.inflate(20, 10)  # Add some padding
            pygame.draw.rect(self.screen, POPUP_BG_COLOR, bg_rect)
            pygame.draw.rect(self.screen, POPUP_TEXT_COLOR, bg_rect, 3)  # Border
            self.screen.blit(text_surface, text_rect)

    def draw_start_screen(self):
        self.screen.fill(POPUP_BG_COLOR)
        title_text = self.large_font.render("Choose Your Side", True, POPUP_TEXT_COLOR)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4))
        self.screen.blit(title_text, title_rect)

        button_width = 200
        button_height = 60
        button_spacing = 20

        # Play as White button
        white_button_rect = pygame.Rect(
            (SCREEN_WIDTH - button_width) // 2,
            SCREEN_HEIGHT // 2 - button_height - button_spacing // 2,
            button_width,
            button_height
        )
        # Play as Black button
        black_button_rect = pygame.Rect(
            (SCREEN_WIDTH - button_width) // 2,
            SCREEN_HEIGHT // 2 + button_spacing // 2,
            button_width,
            button_height
        )

        mouse_pos = pygame.mouse.get_pos()

        # Draw White button
        white_color = BUTTON_HOVER_COLOR if white_button_rect.collidepoint(mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(self.screen, white_color, white_button_rect, border_radius=10)
        white_text = self.font.render("Play as White", True, BUTTON_TEXT_COLOR)
        white_text_rect = white_text.get_rect(center=white_button_rect.center)
        self.screen.blit(white_text, white_text_rect)

        # Draw Black button
        black_color = BUTTON_HOVER_COLOR if black_button_rect.collidepoint(mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(self.screen, black_color, black_button_rect, border_radius=10)
        black_text = self.font.render("Play as Black", True, BUTTON_TEXT_COLOR)
        black_text_rect = black_text.get_rect(center=black_button_rect.center)
        self.screen.blit(black_text, black_text_rect)

        return white_button_rect, black_button_rect

    def run(self):
        """Main game loop."""
        running = True

        # Start screen loop
        while not self.game_started and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    white_button_rect, black_button_rect = self.draw_start_screen()  # Get button rects for collision
                    if white_button_rect.collidepoint(mouse_pos):
                        self.player_color = chess.WHITE
                        self.game_started = True
                    elif black_button_rect.collidepoint(mouse_pos):
                        self.player_color = chess.BLACK
                        self.game_started = True

            self.draw_start_screen()
            pygame.display.flip()
            self.clock.tick(60)

        if not running:  # If user quit from start screen
            pygame.quit()
            sys.exit()

        # Main game loop
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Player can only click if it's their turn, not game over, no animation, no promotion active.
                    if not self.game_over_message and not self.animating_pieces and not self.promotion_active and self.board.turn == self.player_color:
                        self.handle_click(event.pos)
                    # If promotion is active, handle promotion click regardless of turn
                    elif self.promotion_active:
                        self.handle_promotion_click(event.pos)

            # AI's turn to think and make a move
            # AI plays only if it's the AI's turn, and no animation is in progress, and the game isn't over.
            if self.board.turn != self.player_color and not self.animating_pieces and not self.game_over_message:
                self.display_message("AI is thinking...")  # Temporarily display message
                pygame.display.flip()  # Update screen to show "AI is thinking" immediately

                ai_move = self.engine.choose_move(self.board)

                if ai_move:
                    self.start_move_animation(ai_move, is_player_move=False)
                else:
                    # If AI can't find a move (e.g., stalemate or no legal moves possible for some reason)
                    self.check_game_over()
                    if not self.game_over_message:
                        print("AI found no legal moves but game is not over by rules. Game ends in draw.")
                        self.game_over_message = "Draw! AI found no legal moves."  # Set game over message to draw

            # Update animation state
            animation_finished = self.update_animation()
            if animation_finished:
                # After animation, check game over again in case the move resulted in checkmate/draw
                self.check_game_over()

            # Drawing
            self.screen.fill((0, 0, 0))  # Clear screen
            self.draw_board()
            self.draw_selected_square_and_moves()
            self.draw_pieces()  # Draw static pieces
            self.draw_animating_piece()  # Draw animating pieces on top

            if self.promotion_active:
                self.draw_promotion_popup()

            if self.game_over_message:
                self.draw_game_over_message()
            elif self.animating_pieces:  # During animation, show "Moving..."
                self.display_message("Moving...")
            elif self.board.turn == self.player_color:
                self.display_message("Your Turn")
            else:  # AI's turn
                self.display_message("AI's Turn")

            pygame.display.flip()  # Update the full display Surface to the screen
            self.clock.tick(60)  # Cap the frame rate

        pygame.quit()
        sys.exit()  # Ensure clean exit


if __name__ == '__main__':
    game = ChessGUI_Pygame()
    game.run()

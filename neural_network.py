from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum
import copy
import torch
import numpy as np

class PieceType(Enum):
    PAWN = 'P'
    ROOK = 'T'
    KNIGHT = 'C'
    BISHOP = 'F'
    QUEEN = 'Q'
    KING = 'K'

class Color(Enum):
    WHITE = 'white'
    BLACK = 'black'

@dataclass
class Move:
    from_row: int
    from_col: int
    to_row: int
    to_col: int
    piece: str
    captured: Optional[str] = None
    evaluation: float = 0.0

class PositionEncoder:
    @staticmethod
    def board_to_tensor(board: list, device: torch.device) -> torch.Tensor:
        """
        Convert an 8×8 list-of-lists `board` (with piece letters) into
        a 1×12×8×8 one-hot tensor on `device`.
        Channels 0–5 = White P, T, C, F, Q, K
        Channels 6–11 = Black p, t, c, f, q, k
        """
        mapping = {
            'P': 0, 'T': 1, 'C': 2, 'F': 3, 'Q': 4, 'K': 5,
            'p': 6, 't': 7, 'c': 8, 'f': 9, 'q': 10, 'k': 11,
        }
        arr = np.zeros((1, 12, 8, 8), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                idx = mapping.get(board[i][j])
                if idx is not None:
                    arr[0, idx, i, j] = 1.0
        return torch.from_numpy(arr).to(device)

class ChessEngine:
    PIECE_VALUES = {
        'P': 1, 'p': 1,
        'T': 5, 't': 5,
        'C': 3, 'c': 3,
        'F': 3, 'f': 3,
        'Q': 9, 'q': 9,
        'K': 0, 'k': 0
    }
    
    def __init__(self):
        self.board = self._create_initial_board()
        self.turn = Color.WHITE
        self.move_history: List[Move] = []
        
    def _create_initial_board(self) -> List[List[str]]:
        board = [[' ' for _ in range(8)] for _ in range(8)]
        board[7] = ['T', 'C', 'F', 'Q', 'K', 'F', 'C', 'T']
        board[6] = ['P'] * 8
        board[0] = ['t', 'c', 'f', 'q', 'k', 'f', 'c', 't']
        board[1] = ['p'] * 8
        return board
    
    def get_piece(self, row: int, col: int) -> str:
        if not self._is_valid_position(row, col):
            return ' '
        return self.board[row][col]
    
    def _is_valid_position(self, row: int, col: int) -> bool:
        return 0 <= row < 8 and 0 <= col < 8
    
    def _get_piece_color(self, piece: str) -> Optional[Color]:
        if piece == ' ':
            return None
        return Color.WHITE if piece.isupper() else Color.BLACK
    
    def _is_enemy_piece(self, piece: str, color: Color) -> bool:
        piece_color = self._get_piece_color(piece)
        return piece_color is not None and piece_color != color
    
    def _is_friendly_piece(self, piece: str, color: Color) -> bool:
        piece_color = self._get_piece_color(piece)
        return piece_color == color
    
    def get_valid_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        piece = self.get_piece(row, col)
        if piece == ' ':
            return []
        
        color = self._get_piece_color(piece)
        if color != self.turn:
            return []
        
        moves = []
        piece_type = piece.upper()
        
        for to_row in range(8):
            for to_col in range(8):
                if self._is_valid_move(row, col, to_row, to_col):
                    moves.append((to_row, to_col))
        
        return moves
    
    def _is_valid_move(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        if not all(self._is_valid_position(r, c) for r, c in [(from_row, from_col), (to_row, to_col)]):
            return False
        
        piece = self.board[from_row][from_col]
        if piece == ' ':
            return False
        
        color = self._get_piece_color(piece)
        if color != self.turn:
            return False
        
        target_piece = self.board[to_row][to_col]
        if self._is_friendly_piece(target_piece, color):
            return False
        
        if not self._is_valid_piece_move(piece, from_row, from_col, to_row, to_col):
            return False
        
        return not self._would_be_in_check_after_move(from_row, from_col, to_row, to_col, color)
    
    def _is_valid_piece_move(self, piece: str, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        piece_type = piece.upper()
        
        if piece_type == 'P':
            return self._is_valid_pawn_move(piece, from_row, from_col, to_row, to_col)
        elif piece_type == 'T':
            return self._is_valid_rook_move(from_row, from_col, to_row, to_col)
        elif piece_type == 'C':
            return self._is_valid_knight_move(from_row, from_col, to_row, to_col)
        elif piece_type == 'F':
            return self._is_valid_bishop_move(from_row, from_col, to_row, to_col)
        elif piece_type == 'Q':
            return self._is_valid_queen_move(from_row, from_col, to_row, to_col)
        elif piece_type == 'K':
            return self._is_valid_king_move(from_row, from_col, to_row, to_col)
        
        return False
    
    def _is_valid_pawn_move(self, piece: str, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        color = self._get_piece_color(piece)
        direction = -1 if color == Color.WHITE else 1
        start_row = 6 if color == Color.WHITE else 1
        
        if from_col == to_col:
            if to_row == from_row + direction and self.board[to_row][to_col] == ' ':
                return True
            if (from_row == start_row and to_row == from_row + 2 * direction and 
                self.board[from_row + direction][from_col] == ' ' and 
                self.board[to_row][to_col] == ' '):
                return True
        
        if (abs(to_col - from_col) == 1 and to_row == from_row + direction and
            self._is_enemy_piece(self.board[to_row][to_col], color)):
            return True
        
        return False
    
    def _is_valid_rook_move(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        if from_row != to_row and from_col != to_col:
            return False
        return self._is_path_clear(from_row, from_col, to_row, to_col)
    
    def _is_valid_knight_move(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        dr, dc = abs(to_row - from_row), abs(to_col - from_col)
        return (dr, dc) in [(2, 1), (1, 2)]
    
    def _is_valid_bishop_move(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        if abs(to_row - from_row) != abs(to_col - from_col):
            return False
        return self._is_path_clear(from_row, from_col, to_row, to_col)
    
    def _is_valid_queen_move(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        return (self._is_valid_rook_move(from_row, from_col, to_row, to_col) or
                self._is_valid_bishop_move(from_row, from_col, to_row, to_col))
    
    def _is_valid_king_move(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        return abs(to_row - from_row) <= 1 and abs(to_col - from_col) <= 1
    
    def _is_path_clear(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        dr = 0 if to_row == from_row else (1 if to_row > from_row else -1)
        dc = 0 if to_col == from_col else (1 if to_col > from_col else -1)
        
        curr_row, curr_col = from_row + dr, from_col + dc
        
        while (curr_row, curr_col) != (to_row, to_col):
            if self.board[curr_row][curr_col] != ' ':
                return False
            curr_row += dr
            curr_col += dc
        
        return True
    
    def _find_king(self, color: Color) -> Optional[Tuple[int, int]]:
        king_symbol = 'K' if color == Color.WHITE else 'k'
        for row in range(8):
            for col in range(8):
                if self.board[row][col] == king_symbol:
                    return (row, col)
        return None
    
    def _is_in_check(self, color: Color) -> bool:
        king_pos = self._find_king(color)
        if not king_pos:
            return False
        
        king_row, king_col = king_pos
        enemy_color = Color.BLACK if color == Color.WHITE else Color.WHITE
        
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if self._get_piece_color(piece) == enemy_color:
                    if self._is_valid_piece_move(piece, row, col, king_row, king_col):
                        return True
        
        return False
    
    def _would_be_in_check_after_move(self, from_row: int, from_col: int, to_row: int, to_col: int, color: Color) -> bool:
        original_piece = self.board[to_row][to_col]
        moving_piece = self.board[from_row][from_col]
        
        self.board[to_row][to_col] = moving_piece
        self.board[from_row][from_col] = ' '
        
        in_check = self._is_in_check(color)
        
        self.board[from_row][from_col] = moving_piece
        self.board[to_row][to_col] = original_piece
        
        return in_check
    
    def make_move(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        if not self._is_valid_move(from_row, from_col, to_row, to_col):
            return False
        
        piece = self.board[from_row][from_col]
        captured = self.board[to_row][to_col]
        
        self.board[to_row][to_col] = piece
        self.board[from_row][from_col] = ' '
        
        move = Move(from_row, from_col, to_row, to_col, piece, captured if captured != ' ' else None)
        self.move_history.append(move)
        
        self.turn = Color.BLACK if self.turn == Color.WHITE else Color.WHITE
        return True
    
    def is_checkmate(self) -> bool:
        if not self._is_in_check(self.turn):
            return False
        
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if self._get_piece_color(piece) == self.turn:
                    for to_row in range(8):
                        for to_col in range(8):
                            if self._is_valid_move(row, col, to_row, to_col):
                                return False
        return True
    
    def is_stalemate(self) -> bool:
        if self._is_in_check(self.turn):
            return False
        
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if self._get_piece_color(piece) == self.turn:
                    for to_row in range(8):
                        for to_col in range(8):
                            if self._is_valid_move(row, col, to_row, to_col):
                                return False
        return True
    
    def evaluate_position(self) -> float:
        if self.is_checkmate():
            return -1000 if self.turn == Color.WHITE else 1000
        
        if self.is_stalemate():
            return 0
        
        material_score = 0
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece != ' ':
                    value = self.PIECE_VALUES[piece]
                    material_score += value if piece.isupper() else -value
        
        positional_score = self._evaluate_position_bonus()
        check_bonus = 0.5 if self._is_in_check(Color.BLACK) else (-0.5 if self._is_in_check(Color.WHITE) else 0)
        
        return material_score + positional_score + check_bonus
    
    def _evaluate_position_bonus(self) -> float:
        bonus = 0
        
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece != ' ':
                    piece_bonus = self._get_piece_position_bonus(piece, row, col)
                    bonus += piece_bonus if piece.isupper() else -piece_bonus
        
        return bonus * 0.1
    
    def _get_piece_position_bonus(self, piece: str, row: int, col: int) -> float:
        center_bonus = max(0, 4 - abs(3.5 - row) - abs(3.5 - col)) * 0.1
        
        if piece.upper() == 'P':
            advancement_bonus = (6 - row) * 0.1 if piece.isupper() else (row - 1) * 0.1
            return advancement_bonus + center_bonus
        
        return center_bonus
    
    def get_all_valid_moves(self) -> List[Move]:
        moves = []
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if self._get_piece_color(piece) == self.turn:
                    valid_destinations = self.get_valid_moves(row, col)
                    for to_row, to_col in valid_destinations:
                        captured = self.board[to_row][to_col] if self.board[to_row][to_col] != ' ' else None
                        moves.append(Move(row, col, to_row, to_col, piece, captured))
        return moves
    
    def copy(self):
        new_engine = ChessEngine()
        new_engine.board = [row[:] for row in self.board]
        new_engine.turn = self.turn
        new_engine.move_history = self.move_history.copy()
        return new_engine
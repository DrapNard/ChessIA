import numpy as np
import tensorflow as tf
import threading
import time
import random
import copy
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.optimizers import Adam

class ChessAI:
    def __init__(self, game=None):
        self.game = game
        self.model = self._create_model()
        self.opponent_model = self._create_model()
        self.games_played = 0
        self.model_wins = 0
        self.opponent_wins = 0
        self.draws = 0
        self.running = False
        self.thread = None
        self.callback = None
        
    def _create_model(self):
        """Create a neural network model for chess evaluation."""
        model = Sequential([
            Input(shape=(8, 8, 12)),  # 8x8 board with 12 channels (6 piece types x 2 colors)
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(1, activation='tanh')  # Output between -1 and 1 (evaluation score)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def board_to_input(self, board):
        """Convert chess board to neural network input."""
        # Create a 8x8x12 tensor (12 channels for 6 piece types x 2 colors)
        input_tensor = np.zeros((8, 8, 12))
        
        # Mapping from piece symbol to channel index
        piece_to_channel = {
            'P': 0, 'p': 6,  # Pawns
            'T': 1, 't': 7,  # Rooks
            'C': 2, 'c': 8,  # Knights
            'F': 3, 'f': 9,  # Bishops
            'Q': 4, 'q': 10, # Queens
            'K': 5, 'k': 11  # Kings
        }
        
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece != ' ':
                    channel = piece_to_channel.get(piece)
                    input_tensor[row, col, channel] = 1
                    
        return input_tensor.reshape(1, 8, 8, 12)  # Add batch dimension
    
    def evaluate_position(self, board, model):
        """Evaluate a chess position using the neural network."""
        board_input = self.board_to_input(board)
        return model.predict(board_input, verbose=0)[0][0]
    
    def get_best_move(self, game, model, depth=2):
        """Find the best move using minimax with neural network evaluation."""
        best_score = float('-inf')
        best_move = None
        
        # Get all valid moves for the current player
        valid_moves = []
        for from_row in range(8):
            for from_col in range(8):
                piece = game.get_piece(from_row, from_col)
                if piece == ' ':
                    continue
                    
                color = 'white' if piece.isupper() else 'black'
                if color != game.turn:
                    continue
                    
                for to_row in range(8):
                    for to_col in range(8):
                        if game.valid_move(from_row, from_col, to_row, to_col):
                            valid_moves.append((from_row, from_col, to_row, to_col))
        
        # Shuffle moves to add variety
        random.shuffle(valid_moves)
        
        for move in valid_moves:
            from_row, from_col, to_row, to_col = move
            
            # Make a copy of the game to simulate the move
            game_copy = copy.deepcopy(game)
            
            # Make the move
            piece = game_copy.board[from_row][from_col]
            game_copy.board[to_row][to_col] = piece
            game_copy.board[from_row][from_col] = ' '
            game_copy.turn = 'black' if game_copy.turn == 'white' else 'white'
            
            # Evaluate the position after the move
            if depth <= 1:
                score = self.evaluate_position(game_copy.board, model)
                # Negate score for black's perspective
                if game.turn == 'black':
                    score = -score
            else:
                # Recursive minimax for deeper search
                next_best_move = self.get_best_move(game_copy, model, depth-1)
                if next_best_move:
                    _, _, _, _, score = next_best_move
                    score = -score  # Negate score for alternating players
                else:
                    score = 0  # No valid moves (checkmate or stalemate)
            
            if score > best_score:
                best_score = score
                best_move = (from_row, from_col, to_row, to_col, score)
                
        return best_move
    
    def play_game(self):
        """Play a game between the two models."""
        game_copy = copy.deepcopy(self.game)
        game_copy.turn = 'white'  # Reset to white's turn
        
        # Reset the board
        game_copy.board = game_copy.create_board()
        
        moves_count = 0
        max_moves = 200  # Prevent infinite games
        
        while moves_count < max_moves:
            # Determine which model to use based on whose turn it is
            current_model = self.model if game_copy.turn == 'white' else self.opponent_model
            
            # Get the best move
            best_move = self.get_best_move(game_copy, current_model)
            
            if not best_move:
                # No valid moves - checkmate or stalemate
                break
                
            from_row, from_col, to_row, to_col, _ = best_move
            
            # Make the move
            piece = game_copy.board[from_row][from_col]
            game_copy.board[to_row][to_col] = piece
            game_copy.board[from_row][from_col] = ' '
            game_copy.turn = 'black' if game_copy.turn == 'white' else 'white'
            
            moves_count += 1
            
            # Check for checkmate
            if self.is_checkmate(game_copy):
                # The player who just moved won
                winner = 'black' if game_copy.turn == 'white' else 'white'
                if winner == 'white':
                    self.model_wins += 1
                else:
                    self.opponent_wins += 1
                return winner
                
        # If we reach max moves, it's a draw
        self.draws += 1
        return 'draw'
    
    def is_checkmate(self, game):
        """Check if the current position is checkmate."""
        # If the king is in check and there are no valid moves, it's checkmate
        color = game.turn
        
        # Check if king is in check
        if not game.is_king_in_check(color):
            return False
            
        # Check if there are any valid moves
        for from_row in range(8):
            for from_col in range(8):
                piece = game.get_piece(from_row, from_col)
                if piece == ' ':
                    continue
                    
                piece_color = 'white' if piece.isupper() else 'black'
                if piece_color != color:
                    continue
                    
                for to_row in range(8):
                    for to_col in range(8):
                        if game.valid_move(from_row, from_col, to_row, to_col):
                            return False  # Found a valid move, not checkmate
                            
        return True  # No valid moves and king in check, it's checkmate
    
    def merge_models(self):
        """Merge the two models by averaging their weights."""
        model_weights = self.model.get_weights()
        opponent_weights = self.opponent_model.get_weights()
        
        merged_weights = []
        for mw, ow in zip(model_weights, opponent_weights):
            merged_weights.append((mw + ow) / 2.0)
            
        # Create a new model with the merged weights
        merged_model = self._create_model()
        merged_model.set_weights(merged_weights)
        
        # Update both models with the merged weights
        self.model = merged_model
        self.opponent_model = clone_model(merged_model)
        self.opponent_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Add some random variations to the opponent model to encourage exploration
        opponent_weights = self.opponent_model.get_weights()
        for i in range(len(opponent_weights)):
            noise = np.random.normal(0, 0.1, opponent_weights[i].shape)
            opponent_weights[i] += noise
        self.opponent_model.set_weights(opponent_weights)
        
        print("Models merged after 10 games")
        
    def start_training(self, game, callback=None):
        """Start the training process in a separate thread."""
        self.game = game
        self.callback = callback
        self.running = True
        self.thread = threading.Thread(target=self._training_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop_training(self):
        """Stop the training process."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def _training_loop(self):
        """Main training loop that runs in a separate thread."""
        while self.running:
            # Play a game between the two models
            winner = self.play_game()
            self.games_played += 1
            
            print(f"Game {self.games_played}: {winner} wins")
            print(f"Stats - Model wins: {self.model_wins}, Opponent wins: {self.opponent_wins}, Draws: {self.draws}")
            
            # After every 10 games, merge the models
            if self.games_played % 10 == 0:
                self.merge_models()
                
            # Call the callback function if provided
            if self.callback:
                self.callback(self.games_played, self.model_wins, self.opponent_wins, self.draws)
                
            # Small delay to prevent hogging CPU
            time.sleep(0.1)

# Function to create and return a ChessAI instance
def create_chess_ai(game):
    return ChessAI(game)
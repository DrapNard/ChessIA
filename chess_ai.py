import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import threading
import time
import random
import copy
import os
import inspect

# Configure PyTorch to use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    # Enable mixed precision for better performance
    torch.backends.cudnn.benchmark = True
else:
    print("CUDA not available, using CPU")

class ChessNet(nn.Module):
    """Neural network model for chess position evaluation."""
    
    def __init__(self):
        super(ChessNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Convolutional layers with ReLU activation
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.tanh(self.fc3(x))  # Output between -1 and 1
        
        return x

class ChessAI:
    def __init__(self, game=None):
        self.game = game
        self.device = device
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Create models
        self.model = self._load_or_create_model('model')
        self.opponent_model = self._load_or_create_model('opponent_model')
        
        # Optimizers
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.opponent_optimizer = optim.Adam(self.opponent_model.parameters(), lr=0.001)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        self.games_played = 0
        self.model_wins = 0
        self.opponent_wins = 0
        self.draws = 0
        self.running = False
        self.thread = None
        self.callback = None
        
    def _create_model(self):
        """Create a neural network model for chess evaluation."""
        model = ChessNet().to(self.device)
        return model
    
    def _load_or_create_model(self, model_name):
        """Load a model if it exists, otherwise create a new one."""
        model_path = os.path.join(self.model_dir, f'{model_name}.pth')
        model = self._create_model()
        
        if os.path.exists(model_path):
            try:
                print(f"Loading existing model from {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint)
                return model
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Creating new model instead")
                return model
        else:
            print(f"No existing model found at {model_path}, creating new model")
            return model
    
    def save_models(self):
        """Save both models to disk."""
        model_path = os.path.join(self.model_dir, 'model.pth')
        opponent_path = os.path.join(self.model_dir, 'opponent_model.pth')
        
        try:
            torch.save(self.model.state_dict(), model_path)
            torch.save(self.opponent_model.state_dict(), opponent_path)
            print(f"Models saved to {self.model_dir}")
            return True
        except Exception as e:
            print(f"Error saving models: {e}")
            return False
    
    def board_to_input(self, board):
        """Convert chess board to neural network input."""
        # Create a 12x8x8 tensor (12 channels for 6 piece types x 2 colors)
        input_tensor = np.zeros((12, 8, 8))
        
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
                    if channel is not None:
                        input_tensor[channel, row, col] = 1
                    
        # Convert to PyTorch tensor and add batch dimension
        tensor = torch.FloatTensor(input_tensor).unsqueeze(0).to(self.device)
        return tensor
    
    def evaluate_position(self, board, model):
        """Evaluate a chess position using the neural network and additional heuristics."""
        model.eval()
        with torch.no_grad():
            # Base evaluation from neural network
            board_input = self.board_to_input(board)
            base_score = model(board_input).cpu().item()
        
        # Additional rewards/punishments based on game state
        reward = 0
        
        # Check for checkmate opportunities
        game_copy = copy.deepcopy(self.game)
        game_copy.board = board
        
        # Reward for putting opponent in check
        if game_copy.is_king_in_check('black' if game_copy.turn == 'white' else 'white'):
            reward += 0.5
        
        # Reward for material advantage (proportional to piece values)
        piece_values = {
            'P': 1, 'p': 1,    # Pawns
            'T': 5, 't': 5,    # Rooks
            'C': 3, 'c': 3,    # Knights
            'F': 3, 'f': 3,    # Bishops
            'Q': 9, 'q': 9,    # Queens
            'K': 0, 'k': 0     # Kings (not counted for material)
        }
        
        material_score = 0
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece != ' ':
                    if piece.isupper():  # White pieces (positive)
                        material_score += piece_values.get(piece, 0)
                    else:  # Black pieces (negative)
                        material_score -= piece_values.get(piece, 0)
        
        # Normalize material score and add to reward
        material_reward = material_score * 0.1  # Scale down the material advantage
        reward += material_reward
        
        # Combine base evaluation with rewards
        return base_score + reward
    
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
        """Play a game between the two models with enhanced rewards/punishments."""
        game_copy = copy.deepcopy(self.game)
        game_copy.turn = 'white'  # Reset to white's turn
        
        # Reset the board
        game_copy.board = game_copy.create_board()
        
        moves_count = 0
        max_moves = 200  # Prevent infinite games
        
        # Track captured pieces for reward calculation
        white_captures = []
        black_captures = []
        
        while moves_count < max_moves:
            # Determine which model to use based on whose turn it is
            current_model = self.model if game_copy.turn == 'white' else self.opponent_model
            
            # Get the best move
            best_move = self.get_best_move(game_copy, current_model)
            
            if not best_move:
                # No valid moves - checkmate or stalemate
                break
                
            from_row, from_col, to_row, to_col, score = best_move
            
            # Check if this move captures a piece (for reward)
            target_piece = game_copy.board[to_row][to_col]
            if target_piece != ' ':
                if game_copy.turn == 'white':
                    white_captures.append(target_piece)
                else:
                    black_captures.append(target_piece)
            
            # Make the move
            piece = game_copy.board[from_row][from_col]
            game_copy.board[to_row][to_col] = piece
            game_copy.board[from_row][from_col] = ' '
            game_copy.turn = 'black' if game_copy.turn == 'white' else 'white'
            
            # Update the preview if callback is provided
            if self.callback and len(inspect.signature(self.callback).parameters) >= 5:
                self.callback(self.games_played, self.model_wins, self.opponent_wins, self.draws, 
                             from_row, from_col, to_row, to_col, score)
                # Add a small delay to make the visualization visible
                time.sleep(self.speed if hasattr(self, 'speed') else 0.5)
            
            moves_count += 1
            
            # Check for checkmate
            if self.is_checkmate(game_copy):
                # The player who just moved won (the opponent of the current turn)
                winner = 'black' if game_copy.turn == 'white' else 'white'
                
                # Apply rewards/punishments based on game outcome
                if winner == 'white':
                    self.model_wins += 1
                    self._apply_reward(self.model, self.optimizer, 20)  # Reward for winning
                    self._apply_reward(self.opponent_model, self.opponent_optimizer, -30)  # Punishment for losing
                else:
                    self.opponent_wins += 1
                    self._apply_reward(self.opponent_model, self.opponent_optimizer, 20)  # Reward for winning
                    self._apply_reward(self.model, self.optimizer, -30)  # Punishment for losing
                
                # Apply additional rewards for captures
                self._apply_capture_rewards(self.model, self.optimizer, white_captures)
                self._apply_capture_rewards(self.opponent_model, self.opponent_optimizer, black_captures)
                
                return winner
            
            # Check for stalemate (no valid moves but not in check)
            if self._is_stalemate(game_copy):
                self.draws += 1
                # Apply punishment for draw (both models)
                self._apply_reward(self.model, self.optimizer, -40)
                self._apply_reward(self.opponent_model, self.opponent_optimizer, -40)
                return 'draw'
                
        # If we reach max moves, it's a draw
        self.draws += 1
        # Apply punishment for draw (both models)
        self._apply_reward(self.model, self.optimizer, -40)
        self._apply_reward(self.opponent_model, self.opponent_optimizer, -40)
        return 'draw'
    
    def _apply_reward(self, model, optimizer, reward_value):
        """Apply a reward or punishment to a model by adjusting its parameters."""
        model.train()
        
        # Create a dummy target based on the reward
        dummy_input = torch.randn(1, 12, 8, 8).to(self.device)
        current_output = model(dummy_input)
        
        # Create target output based on reward
        target_adjustment = reward_value * 0.001
        target_output = current_output + target_adjustment
        
        # Compute loss and backpropagate
        loss = self.criterion(current_output, target_output.detach())
        
        optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
    
    def _apply_capture_rewards(self, model, optimizer, captures):
        """Apply rewards for captured pieces."""
        piece_values = {
            'p': 1,  # Pawn
            't': 5,  # Rook
            'c': 3,  # Knight
            'f': 3,  # Bishop
            'q': 9,  # Queen
            'k': 0,  # King (shouldn't happen in normal play)
            'P': 1,  # Pawn
            'T': 5,  # Rook
            'C': 3,  # Knight
            'F': 3,  # Bishop
            'Q': 9,  # Queen
            'K': 0   # King (shouldn't happen in normal play)
        }
        
        total_reward = 0
        for piece in captures:
            total_reward += piece_values.get(piece, 0)
        
        # Apply the reward
        if total_reward > 0:
            self._apply_reward(model, optimizer, total_reward)
    
    def _is_stalemate(self, game):
        """Check if the current position is a stalemate (no valid moves but not in check)."""
        color = game.turn
        
        # If king is in check, it's not stalemate
        if game.is_king_in_check(color):
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
                            return False  # Found a valid move, not stalemate
                            
        return True  # No valid moves and king not in check, it's stalemate

    def is_checkmate(self, game):
        """Check if the current position is a checkmate."""
        color = game.turn
        
        # First, check if the king is in check
        if not game.is_king_in_check(color):
            return False
            
        # Then check if there are any valid moves that can get out of check
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
                            # Try the move to see if it gets out of check
                            game_copy = copy.deepcopy(game)
                            
                            # Make the move
                            piece = game_copy.board[from_row][from_col]
                            game_copy.board[to_row][to_col] = piece
                            game_copy.board[from_row][from_col] = ' '
                            
                            # Check if king is still in check after the move
                            if not game_copy.is_king_in_check(color):
                                return False  # Found a move that gets out of check
        
        # No moves found that get out of check, it's checkmate
        return True

    def merge_models(self):
        """Merge the two models by averaging their parameters."""
        # Get state dictionaries
        model_state = self.model.state_dict()
        opponent_state = self.opponent_model.state_dict()
        
        # Create merged state dictionary
        merged_state = {}
        for key in model_state.keys():
            merged_state[key] = (model_state[key] + opponent_state[key]) / 2.0
        
        # Create new models with merged parameters
        new_model = self._create_model()
        new_model.load_state_dict(merged_state)
        
        new_opponent = self._create_model()
        new_opponent.load_state_dict(merged_state)
        
        # Add some random variations to the opponent model to encourage exploration
        with torch.no_grad():
            for param in new_opponent.parameters():
                noise = torch.randn_like(param) * 0.1
                param.add_(noise)
        
        # Update models
        self.model = new_model
        self.opponent_model = new_opponent
        
        # Update optimizers
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.opponent_optimizer = optim.Adam(self.opponent_model.parameters(), lr=0.001)
        
        # Save the merged models
        self.save_models()
        
        print("Models merged successfully")
    
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
            # Save models every 5 games
            elif self.games_played % 5 == 0:
                self.save_models()
                
            # Call the callback function if provided
            if self.callback:
                self.callback(self.games_played, self.model_wins, self.opponent_wins, self.draws)
                
            # Small delay to prevent hogging CPU
            time.sleep(0.1)
    
    def set_training_speed(self, speed):
        """Set the speed of the training visualization."""
        self.speed = speed
    
    def start_training(self, game, callback=None):
        """Start the training process in a separate thread."""
        self.game = game
        self.callback = callback
        self.running = True
        self.speed = 0.5  # Default speed
        self.thread = threading.Thread(target=self._training_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def stop_training(self):
        """Stop the training process."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            # Save models when stopping training
            self.save_models()

# Function to create and return a ChessAI instance
def create_chess_ai(game):
    return ChessAI(game)
import tkinter as tk
from tkinter import messagebox
import copy  # Add this import
import random

# --- Game Logic with Piece-Specific Move Validation ---
class ChessGame:
    def __init__(self):
        self.board = self.create_board()
        self.turn = 'white'  # white starts

    def create_board(self):
        # Using custom notation:
        # White pieces (uppercase): T=Rook, C=Knight, F=Bishop, Q=Queen, K=King, P=Pawn
        # Black pieces (lowercase): corresponding lowercase letters.
        board = [[' ' for _ in range(8)] for _ in range(8)]
        board[7] = ['T', 'C', 'F', 'Q', 'K', 'F', 'C', 'T']
        board[6] = ['P'] * 8
        board[0] = ['t', 'c', 'f', 'q', 'k', 'f', 'c', 't']
        board[1] = ['p'] * 8
        return board

    def get_piece(self, row, col):
        return self.board[row][col]

    def is_empty(self, row, col):
        return self.board[row][col] == ' '

    def is_opponent(self, piece, color):
        """Check if 'piece' belongs to the opponent of 'color'."""
        if piece == ' ':
            return False
        return (color == 'white' and piece.islower()) or (color == 'black' and piece.isupper())

    def valid_move(self, from_row, from_col, to_row, to_col):
        """Check if a move is valid according to piece rules and king safety."""
        # 1) Basic boundary check.
        if not (0 <= from_row < 8 and 0 <= from_col < 8 and 0 <= to_row < 8 and 0 <= to_col < 8):
            return False

        # 2) Must move a piece that exists.
        piece = self.board[from_row][from_col]
        if piece == ' ':
            return False

        # 3) Must match current player's turn.
        color = 'white' if piece.isupper() else 'black'
        if color != self.turn:
            return False

        # 4) Cannot capture your own piece.
        dest_piece = self.board[to_row][to_col]
        if dest_piece != ' ' and (
            (color == 'white' and dest_piece.isupper()) or
            (color == 'black' and dest_piece.islower())
        ):
            return False

        # 5) Validate move by piece type.
        piece_type = piece.upper()  # T, C, F, Q, K, P
        if piece_type == 'P':
            valid = self.valid_move_pawn(from_row, from_col, to_row, to_col, color)
        elif piece_type == 'T':
            valid = self.valid_move_rook(from_row, from_col, to_row, to_col)
        elif piece_type == 'C':
            valid = self.valid_move_knight(from_row, from_col, to_row, to_col)
        elif piece_type == 'F':
            valid = self.valid_move_bishop(from_row, from_col, to_row, to_col)
        elif piece_type == 'Q':
            valid = self.valid_move_queen(from_row, from_col, to_row, to_col)
        elif piece_type == 'K':
            valid = self.valid_move_king(from_row, from_col, to_row, to_col)
        else:
            valid = False

        if not valid:
            return False

        # 6) Simulate the move and check if own king is in check.
        original_piece = self.board[to_row][to_col]
        self.board[to_row][to_col] = piece
        self.board[from_row][from_col] = ' '

        king_in_check = self.is_king_in_check(color)

        # Undo the move.
        self.board[from_row][from_col] = piece
        self.board[to_row][to_col] = original_piece

        return not king_in_check

    def valid_move_pawn(self, from_row, from_col, to_row, to_col, color):
        """Validate pawn moves (no promotion or en passant)."""
        direction = -1 if color == 'white' else 1
        start_row = 6 if color == 'white' else 1

        # Move forward (no capture).
        if from_col == to_col:
            # One square forward.
            if to_row == from_row + direction and self.is_empty(to_row, to_col):
                return True
            # Two squares forward from starting row (and path empty).
            if (
                from_row == start_row and
                to_row == from_row + 2 * direction and
                self.is_empty(from_row + direction, from_col) and
                self.is_empty(to_row, to_col)
            ):
                return True

        # Diagonal capture.
        if abs(to_col - from_col) == 1 and to_row == from_row + direction:
            if not self.is_empty(to_row, to_col) and self.is_opponent(self.board[to_row][to_col], color):
                return True

        return False

    def valid_move_rook(self, from_row, from_col, to_row, to_col):
        """Validate rook movement (straight lines, clear path)."""
        if from_row != to_row and from_col != to_col:
            return False
        return self.is_clear_path(from_row, from_col, to_row, to_col)

    def valid_move_knight(self, from_row, from_col, to_row, to_col):
        """Validate knight movement (L-shapes)."""
        dr = abs(to_row - from_row)
        dc = abs(to_col - from_col)
        return (dr, dc) in [(2, 1), (1, 2)]

    def valid_move_bishop(self, from_row, from_col, to_row, to_col):
        """Validate bishop movement (diagonals, clear path)."""
        if abs(to_row - from_row) != abs(to_col - from_col):
            return False
        return self.is_clear_path(from_row, from_col, to_row, to_col)

    def valid_move_queen(self, from_row, from_col, to_row, to_col):
        """Validate queen movement (rook + bishop)."""
        if from_row == to_row or from_col == to_col:
            return self.is_clear_path(from_row, from_col, to_row, to_col)
        elif abs(to_row - from_row) == abs(to_col - from_col):
            return self.is_clear_path(from_row, from_col, to_row, to_col)
        return False

    def valid_move_king(self, from_row, from_col, to_row, to_col):
        """Validate king movement (1 square any direction)."""
        return abs(to_row - from_row) <= 1 and abs(to_col - from_col) <= 1

    def is_clear_path(self, from_row, from_col, to_row, to_col):
        """Check if path is clear for rook/bishop/queen moves."""
        d_row = 0 if to_row == from_row else (1 if to_row > from_row else -1)
        d_col = 0 if to_col == from_col else (1 if to_col > from_col else -1)
        cur_row = from_row + d_row
        cur_col = from_col + d_col

        while (cur_row, cur_col) != (to_row, to_col):
            if not self.is_empty(cur_row, cur_col):
                return False
            cur_row += d_row
            cur_col += d_col

        return True

    def find_king(self, color):
        """Find the king's position for the given color."""
        king_symbol = 'K' if color == 'white' else 'k'
        for row in range(8):
            for col in range(8):
                if self.board[row][col] == king_symbol:
                    return row, col
        return None

    def is_king_in_check(self, color):
        """Check if 'color' king is in check."""
        king_pos = self.find_king(color)
        if not king_pos:
            return False  # Should not happen in a normal game
        king_row, king_col = king_pos

        opponent = 'black' if color == 'white' else 'white'

        # Temporarily treat the board as if it's the opponent's turn
        # to see if the opponent can capture the king.
        original_turn = self.turn
        self.turn = opponent

        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece == ' ':
                    continue
                # If piece belongs to the opponent, check if it can move to king's position.
                if ((opponent == 'white' and piece.isupper()) or
                    (opponent == 'black' and piece.islower())):
                    if self.valid_move(row, col, king_row, king_col):
                        self.turn = original_turn
                        return True

        # Restore turn and return not in check.
        self.turn = original_turn
        return False

    def make_move(self, from_row, from_col, to_row, to_col):
        """Execute the move if valid; otherwise show error."""
        if self.valid_move(from_row, from_col, to_row, to_col):
            piece = self.board[from_row][from_col]
            self.board[to_row][to_col] = piece
            self.board[from_row][from_col] = ' '
            self.turn = 'black' if self.turn == 'white' else 'white'
        else:
            messagebox.showerror("Invalid Move", "This move is not allowed.")

    def parse_move(self, move_str):
        """Parse a move string (e.g. 'e2e4') into board coordinates."""
        if len(move_str) != 4:
            return None
        cols = {'a': 0, 'b': 1, 'c': 2, 'd': 3,
                'e': 4, 'f': 5, 'g': 6, 'h': 7}
        try:
            from_col = cols[move_str[0]]
            from_row = 8 - int(move_str[1])
            to_col = cols[move_str[2]]
            to_row = 8 - int(move_str[3])
        except (KeyError, ValueError):
            return None
        return (from_row, from_col, to_row, to_col)


# --- GUI with Tkinter ---
class ChessUI:
    SQUARE_SIZE = 64
    BOARD_COLOR_1 = "#F0D9B5"
    BOARD_COLOR_2 = "#B58863"
    PIECE_FONT = ("Helvetica", 24, "bold")

    def __init__(self, root, game):
        self.root = root
        self.game = game
        self.selected = None  # (row, col) of a selected piece
        self.move_entry = None

        self.root.title("Chess Game")
        self.canvas = tk.Canvas(self.root, width=8 * self.SQUARE_SIZE, height=8 * self.SQUARE_SIZE)
        self.canvas.pack(side=tk.LEFT)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Frame for move entry and controls.
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.info_label = tk.Label(control_frame, text="Turn: White", font=("Helvetica", 16))
        self.info_label.pack(pady=10)

        tk.Label(control_frame, text="Enter move (e.g., e2e4):").pack(pady=5)
        self.move_entry = tk.Entry(control_frame, font=("Helvetica", 14))
        self.move_entry.pack(pady=5)
        self.move_entry.bind("<Return>", self.on_move_enter)

        move_btn = tk.Button(control_frame, text="Make Move", command=self.on_move_button)
        move_btn.pack(pady=5)

        self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")
        for row in range(8):
            for col in range(8):
                x1 = col * self.SQUARE_SIZE
                y1 = row * self.SQUARE_SIZE
                x2 = x1 + self.SQUARE_SIZE
                y2 = y1 + self.SQUARE_SIZE
                color = self.BOARD_COLOR_1 if (row + col) % 2 == 0 else self.BOARD_COLOR_2
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")

                piece = self.game.get_piece(row, col)
                if piece != ' ':
                    self.canvas.create_text(
                        x1 + self.SQUARE_SIZE // 2,
                        y1 + self.SQUARE_SIZE // 2,
                        text=piece,
                        font=self.PIECE_FONT
                    )
        self.info_label.config(text=f"Turn: {self.game.turn.capitalize()}")

    def on_canvas_click(self, event):
        col = event.x // self.SQUARE_SIZE
        row = event.y // self.SQUARE_SIZE

        # If no piece is selected, select one if it belongs to the current player.
        if self.selected is None:
            piece = self.game.get_piece(row, col)
            if piece != ' ':
                color = 'white' if piece.isupper() else 'black'
                if color == self.game.turn:
                    self.selected = (row, col)
                    # Highlight the selected square.
                    x1 = col * self.SQUARE_SIZE
                    y1 = row * self.SQUARE_SIZE
                    x2 = x1 + self.SQUARE_SIZE
                    y2 = y1 + self.SQUARE_SIZE
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=3, tags="highlight")
        else:
            # Attempt to move from the selected square to the clicked square.
            from_row, from_col = self.selected
            if self.game.valid_move(from_row, from_col, row, col):
                self.game.make_move(from_row, from_col, row, col)
                self.draw_board()
            else:
                messagebox.showerror("Invalid Move", "This move is not allowed.")
            # Clear selection.
            self.selected = None
            self.canvas.delete("highlight")

    def on_move_enter(self, event):
        self.process_move()

    def on_move_button(self):
        self.process_move()

    def process_move(self):
        move_str = self.move_entry.get().strip()
        parsed = self.game.parse_move(move_str)
        if parsed is None:
            messagebox.showerror("Error", "Invalid move format. Use e2e4 format.")
            return

        from_row, from_col, to_row, to_col = parsed
        if self.game.valid_move(from_row, from_col, to_row, to_col):
            self.game.make_move(from_row, from_col, to_row, to_col)
            self.draw_board()
        else:
            messagebox.showerror("Error", "Invalid move!")
        self.move_entry.delete(0, tk.END)


# --- Neural Network Preview Window ---
class NeuralNetworkPreview:
    def __init__(self, master, chess_ai=None):
        self.window = tk.Toplevel(master)
        self.window.title("Neural Network Preview")
        self.chess_ai = chess_ai
        
        # Main frame
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        label = tk.Label(main_frame, text="Neural Network Training", font=("Helvetica", 18, "bold"))
        label.pack(pady=10)
        
        # Stats frame
        stats_frame = tk.Frame(main_frame)
        stats_frame.pack(fill=tk.X, pady=10)
        
        # Stats labels
        self.games_label = tk.Label(stats_frame, text="Games played: 0", font=("Helvetica", 12))
        self.games_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.model1_label = tk.Label(stats_frame, text="Model 1 wins: 0", font=("Helvetica", 12))
        self.model1_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.model2_label = tk.Label(stats_frame, text="Model 2 wins: 0", font=("Helvetica", 12))
        self.model2_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        
        self.draws_label = tk.Label(stats_frame, text="Draws: 0", font=("Helvetica", 12))
        self.draws_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        
        # Add game preview section
        preview_frame = tk.Frame(main_frame)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        preview_label = tk.Label(preview_frame, text="Game Preview", font=("Helvetica", 14, "bold"))
        preview_label.pack(pady=5)
        
        # Create a mini chessboard for visualization
        self.preview_canvas = tk.Canvas(preview_frame, width=320, height=320, bg="white")
        self.preview_canvas.pack(pady=10)
        
        # Move info
        self.move_info = tk.Label(preview_frame, text="Last move: None", font=("Helvetica", 12))
        self.move_info.pack(pady=5)
        
        self.eval_info = tk.Label(preview_frame, text="Evaluation: 0.0", font=("Helvetica", 12))
        self.eval_info.pack(pady=5)
        
        # Canvas for network visualization
        self.canvas = tk.Canvas(main_frame, width=400, height=200, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Draw neural network visualization
        self.draw_network_visualization()
        
        # Control buttons
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = tk.Button(control_frame, text="Start Training", command=self.start_training)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(control_frame, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.use_ai_button = tk.Button(control_frame, text="Use AI for Next Move", command=self.use_ai_move)
        self.use_ai_button.pack(side=tk.LEFT, padx=5)
        
        # Add speed control
        speed_frame = tk.Frame(main_frame)
        speed_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(speed_frame, text="Training Speed:").pack(side=tk.LEFT, padx=5)
        self.speed_var = tk.DoubleVar(value=0.5)
        self.speed_slider = tk.Scale(speed_frame, from_=0.1, to=2.0, resolution=0.1, 
                                     orient=tk.HORIZONTAL, variable=self.speed_var)
        self.speed_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Reference to the main game
        self.game = None
        self.chess_ui = None
        self.preview_game = ChessGame()  # Create a separate game instance for preview
        
        # Initialize the preview board
        self.draw_preview_board()
    
    def draw_preview_board(self):
        """Draw the preview chessboard."""
        SQUARE_SIZE = 40
        self.preview_canvas.delete("all")
        
        for row in range(8):
            for col in range(8):
                x1 = col * SQUARE_SIZE
                y1 = row * SQUARE_SIZE
                x2 = x1 + SQUARE_SIZE
                y2 = y1 + SQUARE_SIZE
                color = "#F0D9B5" if (row + col) % 2 == 0 else "#B58863"
                self.preview_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
                
                piece = self.preview_game.get_piece(row, col)
                if piece != ' ':
                    self.preview_canvas.create_text(
                        x1 + SQUARE_SIZE // 2,
                        y1 + SQUARE_SIZE // 2,
                        text=piece,
                        font=("Helvetica", 16, "bold")
                    )
    
    def update_preview(self, from_row, from_col, to_row, to_col, evaluation):
        """Update the preview board with the latest move."""
        # Make a copy of the piece before moving
        piece = self.preview_game.board[from_row][from_col]
        
        # Make the move using the game's make_move method instead of direct board manipulation
        self.preview_game.make_move(from_row, from_col, to_row, to_col)
        
        # Update the board display
        self.draw_preview_board()
        
        # Update move info
        cols = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
        from_square = f"{cols[from_col]}{8-from_row}"
        to_square = f"{cols[to_col]}{8-to_row}"
        
        player = "White" if piece.isupper() else "Black"
        self.move_info.config(text=f"Last move: {player} {piece} {from_square} → {to_square}")
        self.eval_info.config(text=f"Evaluation: {evaluation:.2f}")
        
        # Update the neural network visualization with real activations based on the current board
        self.draw_network_visualization()
        
        # Update the window
        self.window.update_idletasks()
    
    def highlight_king_in_check(self):
        """Highlight the king that is in check on the preview board."""
        SQUARE_SIZE = 40
        color = self.preview_game.turn
        king_pos = self.preview_game.find_king(color)
        
        if king_pos:
            row, col = king_pos
            x1 = col * SQUARE_SIZE
            y1 = row * SQUARE_SIZE
            x2 = x1 + SQUARE_SIZE
            y2 = y1 + SQUARE_SIZE
            
            # Draw a red border around the king in check
            self.preview_canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=3, tags="check_highlight")

    def draw_network_visualization(self):
        """Draw a visualization of the neural network with active neurons based on current board."""
        self.canvas.delete("all")
        
        # Draw layers
        layer_spacing = 80
        node_radius = 15
        
        # Generate activation values from the current board state if possible
        if self.chess_ai and hasattr(self.chess_ai, 'model') and hasattr(self, 'preview_game'):
            # Get board representation
            board_input = self.chess_ai.board_to_input(self.preview_game.board)
            
            try:
                # Create a model that outputs intermediate layer activations
                if not hasattr(self, 'activation_model'):
                    base_model = self.chess_ai.model
                    layer_outputs = [layer.output for layer in base_model.layers]
                    self.activation_model = tf.keras.Model(inputs=base_model.input, 
                                                          outputs=layer_outputs)
                
                # Get activations for all layers
                activations = self.activation_model.predict(board_input, verbose=0)
                
                # Extract sample activations for visualization (simplified)
                # For convolutional layers, take average activation across filters
                conv_activations = []
                dense_activations = []
                
                for i, activation in enumerate(activations):
                    if len(activation.shape) == 4:  # Conv layer (batch, height, width, filters)
                        # Take average across spatial dimensions and filters
                        avg_activation = np.mean(activation, axis=(1, 2, 3))
                        conv_activations.append(float(avg_activation[0]))
                    elif len(activation.shape) == 2:  # Dense layer (batch, units)
                        # Take first few neurons
                        dense_layer = activation[0]
                        if len(dense_layer) >= 4:
                            dense_activations.append(dense_layer[:4])
                        else:
                            # If less than 4 neurons, pad with zeros
                            padded = np.pad(dense_layer, (0, max(0, 4-len(dense_layer))))
                            dense_activations.append(padded[:4])
                
                # Use these activations for visualization
                input_activations = [abs(float(a)) for a in conv_activations[:1] * 4]
                if len(dense_activations) >= 3:
                    hidden1_activations = [abs(float(a)) for a in dense_activations[0]]
                    hidden2_activations = [abs(float(a)) for a in dense_activations[1]]
                    hidden3_activations = [abs(float(a)) for a in dense_activations[0]]  # Reuse first dense layer
                else:
                    # Fallback to random if not enough dense layers
                    hidden1_activations = [random.random() for _ in range(4)]
                    hidden2_activations = [random.random() for _ in range(4)]
                    hidden3_activations = [random.random() for _ in range(4)]
                
                # Get the final output (evaluation score)
                output_activation = (float(activations[-1][0][0]) + 1) / 2  # Convert from [-1,1] to [0,1]
            except Exception as e:
                print(f"Error getting activations: {e}")
                # Fallback to random activations
                input_activations = [random.random() for _ in range(4)]
                hidden1_activations = [random.random() for _ in range(4)]
                hidden2_activations = [random.random() for _ in range(4)]
                hidden3_activations = [random.random() for _ in range(4)]
                output_activation = random.random()
        else:
            # Default random activations if no model is available
            input_activations = [random.random() for _ in range(4)]
            hidden1_activations = [random.random() for _ in range(4)]
            hidden2_activations = [random.random() for _ in range(4)]
            hidden3_activations = [random.random() for _ in range(4)]
            output_activation = random.random()
        
        # Helper function to get color based on activation
        def get_color(activation):
            # Convert activation (0-1) to a color from blue (0) to red (1)
            r = int(255 * activation)
            b = int(255 * (1 - activation))
            return f'#{r:02x}00{b:02x}'
        
        # Input layer (8x8 grid)
        input_x = 50
        for i, activation in enumerate(input_activations):
            y = 50 + i * 30
            color = get_color(activation)
            self.canvas.create_oval(input_x-node_radius, y-node_radius, 
                                   input_x+node_radius, y+node_radius, 
                                   fill=color, outline="black")
            # Add activation text
            self.canvas.create_text(input_x, y, text=f"{activation:.1f}", font=("Helvetica", 8, "bold"))
        
        # Hidden layers
        hidden_activations = [hidden1_activations, hidden2_activations, hidden3_activations]
        for layer, layer_activations in enumerate(hidden_activations):
            x = input_x + (layer + 1) * layer_spacing
            for i, activation in enumerate(layer_activations):
                y = 50 + i * 30
                color = get_color(activation)
                self.canvas.create_oval(x-node_radius, y-node_radius, 
                                       x+node_radius, y+node_radius, 
                                       fill=color, outline="black")
                # Add activation text
                self.canvas.create_text(x, y, text=f"{activation:.1f}", font=("Helvetica", 8, "bold"))
                
                # Connect to previous layer
                if layer == 0:  # Connect to input layer
                    prev_x = input_x
                    prev_activations = input_activations
                else:  # Connect to previous hidden layer
                    prev_x = input_x + layer * layer_spacing
                    prev_activations = hidden_activations[layer-1]
                
                for j, prev_activation in enumerate(prev_activations):
                    prev_y = 50 + j * 30
                    # Line thickness based on product of activations
                    line_width = 1 + 2 * activation * prev_activation
                    self.canvas.create_line(prev_x+node_radius, prev_y, 
                                           x-node_radius, y, 
                                           fill="gray", width=line_width, arrow=tk.LAST)
        
        # Output layer (single node)
        output_x = input_x + 4 * layer_spacing
        output_y = 95  # Middle of the canvas
        output_color = get_color(output_activation)
        self.canvas.create_oval(output_x-node_radius, output_y-node_radius, 
                               output_x+node_radius, output_y+node_radius, 
                               fill=output_color, outline="black")
        # Add activation text
        self.canvas.create_text(output_x, output_y, text=f"{output_activation:.1f}", font=("Helvetica", 8, "bold"))
        
        # Connect to last hidden layer
        last_hidden_x = input_x + 3 * layer_spacing
        for i, activation in enumerate(hidden3_activations):
            y = 50 + i * 30
            # Line thickness based on product of activations
            line_width = 1 + 2 * activation * output_activation
            self.canvas.create_line(last_hidden_x+node_radius, y, 
                                   output_x-node_radius, output_y, 
                                   fill="gray", width=line_width, arrow=tk.LAST)
        
        # Add a legend
        legend_y = 180
        self.canvas.create_text(50, legend_y, text="Activation Legend:", anchor="w", font=("Helvetica", 10, "bold"))
        for i in range(5):
            val = i / 4
            color = get_color(val)
            self.canvas.create_rectangle(50 + i*40, legend_y+15, 50 + (i+1)*40, legend_y+25, fill=color, outline="black")
            self.canvas.create_text(50 + i*40 + 20, legend_y+30, text=f"{val:.1f}", font=("Helvetica", 8))

    def update_stats(self, games_played, model1_wins, model2_wins, draws):
        """Update the statistics display."""
        self.games_label.config(text=f"Games played: {games_played}")
        self.model1_label.config(text=f"Model 1 wins: {model1_wins}")
        self.model2_label.config(text=f"Model 2 wins: {model2_wins}")
        self.draws_label.config(text=f"Draws: {draws}")
        
        # Update the window
        self.window.update_idletasks()
    
    def set_game_references(self, game, chess_ui):
        """Set references to the main game and UI."""
        self.game = game
        self.chess_ui = chess_ui
        # Reset the preview game to match the current game state
        self.preview_game = copy.deepcopy(game)
        self.draw_preview_board()
        
        # Update the speed setting in the AI if it exists
        if self.chess_ai:
            self.speed_slider.config(command=lambda val: self.update_speed(float(val)))
    
    def update_speed(self, speed):
        """Update the training speed."""
        if self.chess_ai and hasattr(self.chess_ai, 'set_training_speed'):
            self.chess_ai.set_training_speed(speed)
    
    def start_training(self):
        """Start the neural network training."""
        if self.chess_ai:
            # Pass the update_training function that can handle move updates
            def update_training(games_played, model1_wins, model2_wins, draws, from_row=None, from_col=None, to_row=None, to_col=None, evaluation=0.0):
                self.update_stats(games_played, model1_wins, model2_wins, draws)
                if from_row is not None:
                    self.update_preview(from_row, from_col, to_row, to_col, evaluation)
            
            self.chess_ai.start_training(self.game, update_training)
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
    
    def stop_training(self):
        """Stop the neural network training."""
        if self.chess_ai:
            self.chess_ai.stop_training()
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
    
    def use_ai_move(self):
        """Use the AI to make the next move in the game."""
        if not self.chess_ai or not self.game or not self.chess_ui:
            messagebox.showerror("Error", "AI or game not initialized")
            return
            
        # Get the best move from the AI
        best_move = self.chess_ai.get_best_move(self.game, self.chess_ai.model)
        
        if not best_move:
            messagebox.showinfo("AI Move", "AI couldn't find a valid move")
            return
            
        from_row, from_col, to_row, to_col, score = best_move
        
        # Make the move
        self.game.make_move(from_row, from_col, to_row, to_col)
        
        # Update the UI
        self.chess_ui.draw_board()
        
        # Show the move in chess notation
        cols = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
        from_square = f"{cols[from_col]}{8-from_row}"
        to_square = f"{cols[to_col]}{8-to_row}"
        
        messagebox.showinfo("AI Move", f"AI moved from {from_square} to {to_square}\nEvaluation: {score:.2f}")


# --- Main Application ---
def main():
    # Import the chess_ai module here to avoid circular imports
    import chess_ai
    import sys
    
    # Check if any command-line arguments were provided
    human_vs_ai_mode = len(sys.argv) > 1
    
    root = tk.Tk()
    
    if human_vs_ai_mode:
        # Human vs AI mode
        root.title("Chess Game - Human vs AI")
        
        game = ChessGame()
        app = ChessUI(root, game)
        
        # Create the chess AI
        chess_ai_instance = chess_ai.create_chess_ai(game)
        
        # Create the neural network preview window
        nn_preview = NeuralNetworkPreview(root, chess_ai_instance)
        nn_preview.set_game_references(game, app)
        
        root.mainloop()
    else:
        # Automatic AI training mode
        root.title("Chess AI Training")
        
        game = ChessGame()
        
        # Create a minimal UI for training visualization
        frame = tk.Frame(root, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        title = tk.Label(frame, text="Chess AI Training Mode", font=("Helvetica", 18, "bold"))
        title.pack(pady=10)
        
        # Create the chess AI
        chess_ai_instance = chess_ai.create_chess_ai(game)
        
        # Create the neural network preview window with game visualization
        nn_preview = NeuralNetworkPreview(root, chess_ai_instance)
        nn_preview.set_game_references(game, None)
        
        # Define update function for training stats and game preview
        def update_training(games_played, model1_wins, model2_wins, draws, from_row=None, from_col=None, to_row=None, to_col=None, evaluation=0.0):
            nn_preview.update_stats(games_played, model1_wins, model2_wins, draws)
            if from_row is not None:
                nn_preview.update_preview(from_row, from_col, to_row, to_col, evaluation)
            root.update_idletasks()
        
        # Start training automatically
        chess_ai_instance.start_training(game, update_training)
        
        # Add a stop button
        def stop_training():
            chess_ai_instance.stop_training()
            root.destroy()
        
        stop_button = tk.Button(frame, text="Stop Training", command=stop_training)
        stop_button.pack(pady=10)
        
        root.protocol("WM_DELETE_WINDOW", stop_training)  # Handle window close
        root.mainloop()

if __name__ == '__main__':
    main()

# Modify the _training_loop method in chess_ai.py to call update_preview

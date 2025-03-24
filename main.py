import tkinter as tk
from tkinter import messagebox

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
        
        # Reference to the main game
        self.game = None
        self.chess_ui = None
    
    def set_game_references(self, game, chess_ui):
        """Set references to the game and UI."""
        self.game = game
        self.chess_ui = chess_ui
    
    def draw_network_visualization(self):
        """Draw a simple visualization of the neural network."""
        self.canvas.delete("all")
        
        # Draw layers
        layer_spacing = 80
        node_radius = 15
        
        # Input layer (8x8 grid)
        input_x = 50
        for i in range(4):  # Just show 4 nodes to represent the input layer
            y = 50 + i * 30
            self.canvas.create_oval(input_x-node_radius, y-node_radius, 
                                   input_x+node_radius, y+node_radius, 
                                   fill="lightblue", outline="black")
        
        # Hidden layers
        for layer in range(3):
            x = input_x + (layer + 1) * layer_spacing
            for i in range(4):  # Just show 4 nodes per hidden layer
                y = 50 + i * 30
                self.canvas.create_oval(x-node_radius, y-node_radius, 
                                       x+node_radius, y+node_radius, 
                                       fill="lightgreen", outline="black")
                
                # Connect to previous layer
                if layer == 0:  # Connect to input layer
                    prev_x = input_x
                else:  # Connect to previous hidden layer
                    prev_x = input_x + layer * layer_spacing
                
                for j in range(4):
                    prev_y = 50 + j * 30
                    self.canvas.create_line(prev_x+node_radius, prev_y, 
                                           x-node_radius, y, 
                                           fill="gray", arrow=tk.LAST)
        
        # Output layer (single node)
        output_x = input_x + 4 * layer_spacing
        output_y = 95  # Middle of the canvas
        self.canvas.create_oval(output_x-node_radius, output_y-node_radius, 
                               output_x+node_radius, output_y+node_radius, 
                               fill="pink", outline="black")
        
        # Connect to last hidden layer
        last_hidden_x = input_x + 3 * layer_spacing
        for i in range(4):
            y = 50 + i * 30
            self.canvas.create_line(last_hidden_x+node_radius, y, 
                                   output_x-node_radius, output_y, 
                                   fill="gray", arrow=tk.LAST)
    
    def update_stats(self, games_played, model1_wins, model2_wins, draws):
        """Update the statistics display."""
        self.games_label.config(text=f"Games played: {games_played}")
        self.model1_label.config(text=f"Model 1 wins: {model1_wins}")
        self.model2_label.config(text=f"Model 2 wins: {model2_wins}")
        self.draws_label.config(text=f"Draws: {draws}")
        
        # Update the window
        self.window.update_idletasks()
    
    def start_training(self):
        """Start the neural network training."""
        if self.chess_ai:
            self.chess_ai.start_training(self.game, self.update_stats)
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
    
    root = tk.Tk()
    root.title("Chess Game with Neural Network")
    
    game = ChessGame()
    app = ChessUI(root, game)
    
    # Create the chess AI
    chess_ai_instance = chess_ai.create_chess_ai(game)
    
    # Create the neural network preview window
    nn_preview = NeuralNetworkPreview(root, chess_ai_instance)
    nn_preview.set_game_references(game, app)
    
    root.mainloop()

if __name__ == '__main__':
    main()

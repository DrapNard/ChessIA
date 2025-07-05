#!/usr/bin/env python3
"""
Chess AI - Enhanced Neural Network Chess Engine
===============================================

A complete chess AI system featuring:
- Clean chess engine with full rule validation
- PyTorch-based neural network for position evaluation
- Self-training system with genetic algorithm evolution
- Modern web interface with real-time updates
- Desktop GUI for local play
- Comprehensive API for integration

Usage:
    python main.py                    # Launch desktop GUI
    python main.py --web              # Launch web server
    python main.py --train            # Start training mode
    python main.py --cli              # Command line interface
    python main.py --help             # Show all options

Author: Chess AI Project
Version: 2.0.0
"""

import sys
import os
import argparse
import logging
import signal
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our modules
from chess_engine import ChessEngine, Color
from neural_network import ChessAI
from chess_trainer import ChessTrainer, TrainingConfig
from web_api import ChessAPI

# Optional GUI imports (may not be available in headless environments)
try:
    from chess_gui import ChessGUI
    GUI_AVAILABLE = True
except ImportError as e:
    GUI_AVAILABLE = False
    print(f"GUI not available: {e}")

class CLIChess:
    """Command line interface for chess."""
    
    def __init__(self):
        self.engine = ChessEngine()
        self.ai = ChessAI()
        
    def display_board(self):
        """Display the current board state."""
        print("\n   a b c d e f g h")
        print("  ┌─────────────────┐")
        
        for row in range(8):
            print(f"{8-row} │", end="")
            for col in range(8):
                piece = self.engine.board[row][col]
                if piece == ' ':
                    print(" ·", end="")
                else:
                    symbols = {
                        'P': '♙', 'T': '♖', 'C': '♘', 'F': '♗', 'Q': '♕', 'K': '♔',
                        'p': '♟', 't': '♜', 'c': '♞', 'f': '♝', 'q': '♛', 'k': '♚'
                    }
                    print(f" {symbols.get(piece, piece)}", end="")
            print(f" │ {8-row}")
        
        print("  └─────────────────┘")
        print("   a b c d e f g h\n")
        
        print(f"Turn: {self.engine.turn.value.title()}")
        print(f"Evaluation: {self.engine.evaluate_position():.2f}")
        
        if self.engine._is_in_check(self.engine.turn):
            print("⚠️  Check!")
        
        if self.engine.is_checkmate():
            winner = "Black" if self.engine.turn == Color.WHITE else "White"
            print(f"🏆 Checkmate! {winner} wins!")
            return True
        
        if self.engine.is_stalemate():
            print("🤝 Stalemate! It's a draw!")
            return True
        
        return False
    
    def parse_move(self, move_input: str) -> Optional[tuple]:
        """Parse human move input like 'e2e4' or 'e2 e4'."""
        move_input = move_input.replace(' ', '').lower()
        
        if len(move_input) != 4:
            return None
        
        files = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        
        try:
            from_col = files[move_input[0]]
            from_row = 8 - int(move_input[1])
            to_col = files[move_input[2]]
            to_row = 8 - int(move_input[3])
            
            return (from_row, from_col, to_row, to_col)
        except (KeyError, ValueError):
            return None
    
    def get_move_notation(self, from_row: int, from_col: int, to_row: int, to_col: int) -> str:
        """Convert move to algebraic notation."""
        files = "abcdefgh"
        from_square = f"{files[from_col]}{8-from_row}"
        to_square = f"{files[to_col]}{8-to_row}"
        return f"{from_square}{to_square}"
    
    def run(self):
        """Run the CLI chess game."""
        print("🤖 Chess AI - Command Line Interface")
        print("=" * 40)
        print("Enter moves in format: e2e4 or e2 e4")
        print("Commands: 'ai' for AI move, 'quit' to exit, 'help' for help")
        print()
        
        while True:
            game_over = self.display_board()
            if game_over:
                break
            
            if self.engine.turn == Color.WHITE:
                # Human turn
                while True:
                    try:
                        user_input = input(f"{self.engine.turn.value.title()} to move> ").strip()
                        
                        if user_input.lower() in ['quit', 'exit', 'q']:
                            print("Thanks for playing!")
                            return
                        
                        if user_input.lower() in ['help', 'h']:
                            self.show_help()
                            continue
                        
                        if user_input.lower() == 'ai':
                            # Let AI make the move
                            best_move = self.ai.get_best_move(self.engine, depth=3)
                            if best_move:
                                from_row, from_col, to_row, to_col, eval_score = best_move
                                if self.engine.make_move(from_row, from_col, to_row, to_col):
                                    move_notation = self.get_move_notation(from_row, from_col, to_row, to_col)
                                    print(f"AI plays: {move_notation} (evaluation: {eval_score:.2f})")
                                    break
                            print("AI couldn't find a move!")
                            continue
                        
                        # Parse human move
                        move = self.parse_move(user_input)
                        if not move:
                            print("Invalid format! Use: e2e4")
                            continue
                        
                        from_row, from_col, to_row, to_col = move
                        
                        if self.engine.make_move(from_row, from_col, to_row, to_col):
                            break
                        else:
                            print("Illegal move! Try again.")
                    
                    except KeyboardInterrupt:
                        print("\nThanks for playing!")
                        return
                    except Exception as e:
                        print(f"Error: {e}")
            
            else:
                # AI turn
                print("AI is thinking...")
                best_move = self.ai.get_best_move(self.engine, depth=3)
                
                if best_move:
                    from_row, from_col, to_row, to_col, eval_score = best_move
                    if self.engine.make_move(from_row, from_col, to_row, to_col):
                        move_notation = self.get_move_notation(from_row, from_col, to_row, to_col)
                        print(f"AI plays: {move_notation} (evaluation: {eval_score:.2f})")
                else:
                    print("AI couldn't find a move!")
                    break
    
    def show_help(self):
        """Show help information."""
        print("\n📖 Help:")
        print("• Move format: e2e4 (from e2 to e4)")
        print("• 'ai' - Let AI make your move")
        print("• 'quit' - Exit the game")
        print("• 'help' - Show this help")
        print()

class TrainingMode:
    """Standalone training mode for the neural network."""
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.trainer = ChessTrainer(self.config)
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\n🛑 Stopping training...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start training with progress updates."""
        print("🧠 Chess AI Training Mode")
        print("=" * 40)
        print(f"Configuration:")
        print(f"• Max games: {self.config.max_games}")
        print(f"• Save interval: {self.config.save_interval}")
        print(f"• Training speed: {self.config.training_speed}")
        print(f"• Alpha-beta depth: {self.config.alpha_beta_depth}")
        print()
        
        def progress_callback(games, model1_wins, model2_wins, draws, *args):
            """Print training progress."""
            if games % 10 == 0 or games <= 5:
                win_rate = model1_wins / max(1, games) * 100
                print(f"Game {games:4d} | Model1: {model1_wins:3d} | Model2: {model2_wins:3d} | "
                      f"Draws: {draws:3d} | Win Rate: {win_rate:5.1f}%")
                
                if len(args) >= 5:
                    from_row, from_col, to_row, to_col, evaluation = args[:5]
                    files = "abcdefgh"
                    move = f"{files[from_col]}{8-from_row}{files[to_col]}{8-to_row}"
                    print(f"         Last move: {move} (eval: {evaluation:+.2f})")
        
        self.running = True
        
        try:
            if self.trainer.start_training(progress_callback):
                print("🚀 Training started! Press Ctrl+C to stop.\n")
                
                # Keep the main thread alive
                while self.running and self.trainer.is_training:
                    time.sleep(1)
                    
                    # Print final stats periodically
                    stats = self.trainer.get_stats()
                    if stats.games_played > 0 and stats.games_played % 100 == 0:
                        print(f"\n📊 Training Progress:")
                        print(f"   Games played: {stats.games_played}")
                        print(f"   Training time: {stats.training_time:.1f}s")
                        print(f"   Avg moves/game: {stats.avg_moves_per_game:.1f}")
                        print(f"   Model 1 win rate: {stats.model1_win_rate:.1%}")
                        print()
            else:
                print("❌ Failed to start training!")
                
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted by user")
        except Exception as e:
            print(f"❌ Training error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop training and save models."""
        self.running = False
        if self.trainer:
            success = self.trainer.stop_training()
            if success:
                stats = self.trainer.get_stats()
                print(f"\n📊 Final Training Statistics:")
                print(f"   Total games: {stats.games_played}")
                print(f"   Training time: {stats.training_time:.1f}s")
                print(f"   Model 1 wins: {stats.model1_wins}")
                print(f"   Model 2 wins: {stats.model2_wins}")
                print(f"   Draws: {stats.draws}")
                print(f"   Final win rate: {stats.model1_win_rate:.1%}")
                print("✅ Models saved successfully!")
            else:
                print("⚠️  Warning: Failed to save models properly")

def setup_logging(level: str = "INFO"):
    """Configure logging for the application."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('chess_ai.log')
        ]
    )
    
    # Suppress some verbose libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

def create_directories():
    """Create necessary directories for the application."""
    directories = ['models', 'logs', 'games', 'static/css', 'static/js', 'templates']
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def check_dependencies():
    """Check if all required dependencies are available."""
    dependencies = {
        'torch': 'PyTorch for neural networks',
        'numpy': 'NumPy for numerical computations',
        'flask': 'Flask for web API (optional)',
        'flask_socketio': 'Flask-SocketIO for real-time updates (optional)',
    }
    
    missing = []
    
    for package, description in dependencies.items():
        try:
            __import__(package)
        except ImportError:
            if package in ['flask', 'flask_socketio']:
                print(f"⚠️  Optional dependency missing: {package} - {description}")
            else:
                missing.append(f"{package} - {description}")
    
    if missing:
        print("❌ Missing required dependencies:")
        for dep in missing:
            print(f"   • {dep}")
        print("\nInstall with: pip install torch numpy flask flask-socketio flask-cors marshmallow")
        return False
    
    return True

def main():
    """Main entry point for the Chess AI application."""
    parser = argparse.ArgumentParser(
        description="Chess AI - Advanced Neural Network Chess Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['gui', 'web', 'cli', 'train'],
        default='gui',
        help='Application mode (default: gui)'
    )
    
    parser.add_argument(
        '--web', 
        action='store_const', 
        const='web', 
        dest='mode',
        help='Launch web server mode'
    )
    
    parser.add_argument(
        '--cli', 
        action='store_const', 
        const='cli', 
        dest='mode',
        help='Launch command line interface'
    )
    
    parser.add_argument(
        '--train', 
        action='store_const', 
        const='train', 
        dest='mode',
        help='Start training mode'
    )
    
    parser.add_argument(
        '--host',
        default='localhost',
        help='Host for web server (default: localhost)'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=5000,
        help='Port for web server (default: 5000)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--games',
        type=int,
        default=1000,
        help='Number of games for training mode (default: 1000)'
    )
    
    parser.add_argument(
        '--speed',
        type=float,
        default=0.1,
        help='Training speed (default: 0.1)'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    create_directories()
    
    if not check_dependencies():
        sys.exit(1)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Chess AI v2.0 in {args.mode} mode")
    
    try:
        if args.mode == 'gui':
            if not GUI_AVAILABLE:
                print("❌ GUI mode not available. Try --web or --cli mode instead.")
                print("   Install tkinter: sudo apt-get install python3-tk (Linux)")
                sys.exit(1)
            
            print("🎮 Launching Chess AI GUI...")
            app = ChessGUI()
            app.run()
        
        elif args.mode == 'web':
            print(f"🌐 Starting Chess AI web server on http://{args.host}:{args.port}")
            api = ChessAPI()
            api.run(host=args.host, port=args.port, debug=args.debug)
        
        elif args.mode == 'cli':
            cli = CLIChess()
            cli.run()
        
        elif args.mode == 'train':
            config = TrainingConfig(
                max_games=args.games,
                training_speed=args.speed
            )
            trainer = TrainingMode(config)
            trainer.start()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        print(f"❌ Error: {e}")
        if args.debug:
            raise
        sys.exit(1)

if __name__ == '__main__':
    main()
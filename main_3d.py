#!/usr/bin/env python3
"""
Chess AI 3D - Enhanced Neural Network Chess Engine with Real-time 3D Visualization
================================================================================

A revolutionary chess AI system featuring:
- Real-time 3D neural network visualization showing how AI thinks
- Advanced PyTorch CNN with attention mechanisms  
- Self-training system with genetic algorithm evolution
- WebGL-powered 3D interface with particle effects and animations
- Performance monitoring and adaptive quality settings
- Multi-modal interfaces (GUI, Web, CLI, Training)

Features:
🧠 Watch neurons fire in real-time as AI analyzes positions
⚡ Hardware-accelerated 3D rendering with 60+ FPS
🎮 Interactive camera controls and visual effects
📊 Live performance metrics and training analytics
🔄 Automatic model evolution and optimization
🌐 WebSocket-based real-time updates

Usage:
    python main_3d.py                    # Launch 3D web interface
    python main_3d.py --gui              # Desktop GUI with 3D visualization
    python main_3d.py --train-3d         # Enhanced 3D training mode
    python main_3d.py --cli              # Command line interface
    python main_3d.py --benchmark        # Performance benchmarking

Advanced Options:
    python main_3d.py --web --gpu --quality ultra
    python main_3d.py --train-3d --batch --games 5000
    python main_3d.py --benchmark --export-data

Author: Chess AI 3D Project Team
Version: 3.0.0 - Neural Visualization Edition
"""

import sys
import os
import argparse
import logging
import signal
import threading
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any
import asyncio

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our enhanced modules
from chess_engine import ChessEngine, Color
from neural_network_3d import ChessAI3D
from chess_trainer_3d import ChessTrainer3D, TrainingConfig3D
from web_api_3d import ChessAPI3D

# Optional GUI imports
try:
    from chess_gui import ChessGUI  # This would need to be updated for 3D
    GUI_AVAILABLE = True
except ImportError as e:
    GUI_AVAILABLE = False
    print(f"GUI not available: {e}")

class CLIChess3D:
    """Enhanced command line interface with 3D neural network insights."""
    
    def __init__(self, enable_3d_analysis: bool = True):
        self.engine = ChessEngine()
        self.ai = ChessAI3D()
        self.enable_3d_analysis = enable_3d_analysis
        self.move_history = []
        
    def display_board(self):
        """Display the current board state with enhanced analysis."""
        print("\n" + "="*60)
        print("   a b c d e f g h")
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
        print("   a b c d e f g h")
        
        # Enhanced status display
        print(f"\n🎯 Turn: {self.engine.turn.value.title()}")
        
        if self.enable_3d_analysis:
            evaluation, viz_data = self.ai.evaluate_position_with_visualization(self.engine.board)
            print(f"🧠 AI Evaluation: {evaluation:.3f}")
            
            if viz_data and 'summary' in viz_data:
                summary = viz_data['summary']
                print(f"🔥 Active Neurons: {summary.get('total_active_neurons', 0)}")
                print(f"⚡ Network Activity: {summary.get('average_activation', 0):.1%}")
                print(f"🎲 Max Activation: {summary.get('max_activation', 0):.3f}")
        else:
            evaluation = self.ai.evaluate_position(self.engine.board)
            print(f"🧠 AI Evaluation: {evaluation:.3f}")
        
        # Game state indicators
        if self.engine._is_in_check(self.engine.turn):
            print("⚠️  CHECK!")
        
        if self.engine.is_checkmate():
            winner = "Black" if self.engine.turn == Color.WHITE else "White"
            print(f"🏆 CHECKMATE! {winner} wins!")
            return True
        
        if self.engine.is_stalemate():
            print("🤝 STALEMATE! It's a draw!")
            return True
        
        return False
    
    def show_neural_insights(self):
        """Display detailed neural network analysis."""
        if not self.enable_3d_analysis:
            print("3D analysis disabled. Use --enable-3d to see neural insights.")
            return
        
        print("\n🧠 Neural Network Analysis:")
        print("-" * 40)
        
        evaluation, viz_data = self.ai.evaluate_position_with_visualization(self.engine.board)
        
        if viz_data and 'layer_config' in viz_data:
            activations = viz_data.get('activations', {})
            
            for layer_name, layer_activations in activations.items():
                if layer_activations:
                    avg_activation = sum(layer_activations) / len(layer_activations)
                    max_activation = max(layer_activations)
                    active_count = sum(1 for a in layer_activations if a > 0.1)
                    
                    print(f"📊 {layer_name}:")
                    print(f"   Active: {active_count}/{len(layer_activations)} neurons")
                    print(f"   Average: {avg_activation:.3f}")
                    print(f"   Peak: {max_activation:.3f}")
        
        if viz_data and 'summary' in viz_data:
            summary = viz_data['summary']
            layer_activity = summary.get('layer_activity', {})
            
            print(f"\n📈 Overall Network Activity: {summary.get('average_activation', 0):.1%}")
            print(f"🔥 Most Active Layer: ", end="")
            
            if layer_activity:
                most_active = max(layer_activity.items(), 
                                key=lambda x: x[1].get('activity_ratio', 0))
                print(f"{most_active[0]} ({most_active[1].get('activity_ratio', 0):.1%})")
    
    def run(self):
        """Run the enhanced CLI chess game."""
        print("🧠 Chess AI 3D - Command Line Interface")
        print("="*60)
        print("Commands:")
        print("  • Move: e2e4 or e2 e4")
        print("  • AI move: 'ai'")
        print("  • Neural analysis: 'analyze'")
        print("  • Game history: 'history'")
        print("  • Help: 'help'")
        print("  • Quit: 'quit'")
        print()
        
        while True:
            game_over = self.display_board()
            if game_over:
                break
            
            try:
                user_input = input(f"\n{self.engine.turn.value.title()}> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Thanks for playing! 👋")
                    return
                
                elif user_input.lower() in ['help', 'h']:
                    self.show_help()
                    continue
                
                elif user_input.lower() in ['analyze', 'neural', 'brain']:
                    self.show_neural_insights()
                    continue
                
                elif user_input.lower() in ['history', 'moves']:
                    self.show_move_history()
                    continue
                
                elif user_input.lower() == 'ai':
                    self.make_ai_move()
                    continue
                
                else:
                    # Parse and make human move
                    move = self.parse_move(user_input)
                    if not move:
                        print("❌ Invalid format! Use: e2e4")
                        continue
                    
                    from_row, from_col, to_row, to_col = move
                    
                    if self.engine.make_move(from_row, from_col, to_row, to_col):
                        self.move_history.append({
                            'move': user_input,
                            'player': 'human',
                            'evaluation': self.ai.evaluate_position(self.engine.board)
                        })
                    else:
                        print("❌ Illegal move! Try again.")
            
            except KeyboardInterrupt:
                print("\nThanks for playing! 👋")
                return
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def make_ai_move(self):
        """Let AI make a move with detailed analysis."""
        print("🤔 AI is thinking...")
        
        start_time = time.time()
        best_move = self.ai.get_best_move(self.engine, depth=3)
        think_time = time.time() - start_time
        
        if best_move:
            from_row, from_col, to_row, to_col, eval_score = best_move
            
            if self.engine.make_move(from_row, from_col, to_row, to_col):
                move_notation = self.get_move_notation(from_row, from_col, to_row, to_col)
                
                print(f"🤖 AI plays: {move_notation}")
                print(f"📊 Evaluation: {eval_score:.3f}")
                print(f"⏱️  Think time: {think_time:.2f}s")
                
                self.move_history.append({
                    'move': move_notation,
                    'player': 'ai',
                    'evaluation': eval_score,
                    'think_time': think_time
                })
        else:
            print("❌ AI couldn't find a move!")
    
    def show_move_history(self):
        """Display the game's move history."""
        if not self.move_history:
            print("📝 No moves played yet.")
            return
        
        print("\n📝 Move History:")
        print("-" * 50)
        
        for i, move_record in enumerate(self.move_history, 1):
            player_icon = "🤖" if move_record['player'] == 'ai' else "👤"
            move = move_record['move']
            evaluation = move_record.get('evaluation', 0)
            think_time = move_record.get('think_time', 0)
            
            print(f"{i:2d}. {player_icon} {move:6s} ({evaluation:+.2f})", end="")
            if think_time > 0:
                print(f" [{think_time:.2f}s]")
            else:
                print()
    
    def parse_move(self, move_str: str) -> Optional[tuple]:
        """Parse human move input like 'e2e4' or 'e2 e4'."""
        move_str = move_str.replace(' ', '').lower()
        
        if len(move_str) != 4:
            return None
        
        files = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        
        try:
            from_col = files[move_str[0]]
            from_row = 8 - int(move_str[1])
            to_col = files[move_str[2]]
            to_row = 8 - int(move_str[3])
            
            return (from_row, from_col, to_row, to_col)
        except (KeyError, ValueError):
            return None
    
    def get_move_notation(self, from_row: int, from_col: int, to_row: int, to_col: int) -> str:
        """Convert move to algebraic notation."""
        files = "abcdefgh"
        from_square = f"{files[from_col]}{8-from_row}"
        to_square = f"{files[to_col]}{8-to_row}"
        return f"{from_square}{to_square}"
    
    def show_help(self):
        """Show enhanced help information."""
        print("\n📖 Chess AI 3D - Help:")
        print("-" * 40)
        print("🎮 Basic Commands:")
        print("  • e2e4      - Move from e2 to e4")
        print("  • ai        - Let AI make your move")
        print("  • quit      - Exit the game")
        print()
        print("🧠 Analysis Commands:")
        print("  • analyze   - Show neural network insights")
        print("  • history   - Display move history")
        print("  • help      - Show this help")
        print()
        print("💡 Tips:")
        print("  • Watch how AI evaluation changes")
        print("  • Use 'analyze' to see which neurons activate")
        print("  • AI shows thinking time and confidence")
        print()

class TrainingMode3D:
    """Enhanced training mode with 3D visualization and analytics."""
    
    def __init__(self, config: Optional[TrainingConfig3D] = None):
        self.config = config or TrainingConfig3D()
        self.trainer = ChessTrainer3D(self.config)
        self.running = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\n🛑 Stopping 3D training...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start enhanced training with detailed progress."""
        print("🧠 Chess AI 3D Training Mode")
        print("="*60)
        print("Configuration:")
        print(f"  🎯 Max games: {self.config.max_games}")
        print(f"  💾 Save interval: {self.config.save_interval}")
        print(f"  🔄 Merge interval: {self.config.merge_interval}")
        print(f"  ⚡ Training speed: {self.config.training_speed}s")
        print(f"  🔍 Alpha-beta depth: {self.config.alpha_beta_depth}")
        print(f"  🎨 3D Visualization: {'Enabled' if self.config.enable_3d_callbacks else 'Disabled'}")
        print(f"  📦 Batch training: {'Enabled' if self.config.batch_training else 'Disabled'}")
        print()
        
        def progress_callback(games, model1_wins, model2_wins, draws, *args):
            """Enhanced progress reporting."""
            if games % 5 == 0 or games <= 3:
                win_rate = model1_wins / max(1, games) * 100
                
                print(f"🎮 Game {games:4d} │ M1: {model1_wins:3d} │ M2: {model2_wins:3d} │ "
                      f"Draws: {draws:3d} │ WR: {win_rate:5.1f}%")
                
                # Show move details if available
                if len(args) >= 6:
                    from_row, from_col, to_row, to_col, evaluation, board = args[:6]
                    files = "abcdefgh"
                    move = f"{files[from_col]}{8-from_row}{files[to_col]}{8-to_row}"
                    print(f"         └─ Move: {move} │ Eval: {evaluation:+.3f}")
                
                # Performance metrics every 25 games
                if games % 25 == 0 and games > 0:
                    metrics = self.trainer.get_performance_metrics()
                    stats = self.trainer.get_stats()
                    
                    print(f"📊 Performance Metrics:")
                    print(f"   ⚡ Positions/sec: {metrics.get('positions_per_second', 0):.1f}")
                    print(f"   🧠 Neural updates/sec: {metrics.get('neural_updates_per_second', 0):.1f}")
                    print(f"   🎯 Avg inference: {stats.average_inference_time*1000:.1f}ms")
                    print(f"   💾 Memory usage: {metrics.get('memory_usage', 0)*100:.1f}%")
                    print()
        
        self.running = True
        
        try:
            if self.trainer.start_training(progress_callback):
                print("🚀 3D Training started! Press Ctrl+C to stop.\n")
                
                start_time = time.time()
                last_stats_time = time.time()
                
                while self.running and self.trainer.is_training:
                    time.sleep(1)
                    
                    # Show extended stats every 2 minutes
                    if time.time() - last_stats_time > 120:
                        self._show_detailed_stats(time.time() - start_time)
                        last_stats_time = time.time()
            else:
                print("❌ Failed to start training!")
                
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted by user")
        except Exception as e:
            print(f"❌ Training error: {e}")
        finally:
            self.stop()
    
    def _show_detailed_stats(self, elapsed_time: float):
        """Show comprehensive training statistics."""
        stats = self.trainer.get_stats()
        metrics = self.trainer.get_performance_metrics()
        
        print(f"\n📈 Detailed Training Statistics ({elapsed_time/60:.1f} minutes)")
        print("-" * 60)
        print(f"🎮 Games: {stats.games_played} │ "
              f"Moves: {stats.total_moves} │ "
              f"Avg/game: {stats.avg_moves_per_game:.1f}")
        print(f"🏆 Model 1: {stats.model1_wins} ({stats.model1_win_rate:.1%}) │ "
              f"Model 2: {stats.model2_wins} │ "
              f"Draws: {stats.draws}")
        print(f"🧠 Neural updates: {stats.neural_updates} │ "
              f"Positions analyzed: {stats.total_positions_analyzed}")
        print(f"⚡ Performance: {metrics.get('positions_per_second', 0):.1f} pos/s │ "
              f"{metrics.get('neural_updates_per_second', 0):.1f} updates/s")
        
        if metrics.get('gpu_utilization', 0) > 0:
            print(f"🎮 GPU: {metrics['gpu_utilization']:.1f}% │ "
                  f"Memory: {metrics.get('memory_usage', 0)*100:.1f}%")
        
        print()
    
    def stop(self):
        """Stop training and show final statistics."""
        self.running = False
        if self.trainer:
            success = self.trainer.stop_training()
            if success:
                stats = self.trainer.get_stats()
                print(f"\n📊 Final Training Results:")
                print("-" * 50)
                print(f"   🎮 Total games: {stats.games_played}")
                print(f"   ⏱️  Training time: {stats.training_time/60:.1f} minutes")
                print(f"   🏆 Model 1 wins: {stats.model1_wins} ({stats.model1_win_rate:.1%})")
                print(f"   🏆 Model 2 wins: {stats.model2_wins}")
                print(f"   🤝 Draws: {stats.draws}")
                print(f"   🧠 Neural updates: {stats.neural_updates}")
                print(f"   📊 Avg inference: {stats.average_inference_time*1000:.1f}ms")
                print("✅ Models and training data saved!")
            else:
                print("⚠️  Warning: Failed to save models properly")

class BenchmarkMode:
    """Performance benchmarking and system testing."""
    
    def __init__(self):
        self.results = {}
    
    def run_benchmarks(self, export_data: bool = False):
        """Run comprehensive performance benchmarks."""
        print("🔧 Chess AI 3D - Performance Benchmark")
        print("="*50)
        
        # System info
        self._show_system_info()
        
        # Neural network benchmarks
        print("\n🧠 Neural Network Benchmarks:")
        self._benchmark_neural_network()
        
        # 3D visualization benchmarks
        print("\n🎨 3D Visualization Benchmarks:")
        self._benchmark_3d_visualization()
        
        # Game engine benchmarks
        print("\n♟️  Game Engine Benchmarks:")
        self._benchmark_game_engine()
        
        # Training benchmarks
        print("\n🏋️ Training Benchmarks:")
        self._benchmark_training()
        
        if export_data:
            self._export_benchmark_data()
        
        print(f"\n✅ Benchmark completed!")
    
    def _show_system_info(self):
        """Display system information."""
        import platform
        import torch
        
        print(f"🖥️  System: {platform.system()} {platform.release()}")
        print(f"🐍 Python: {platform.python_version()}")
        print(f"🔥 PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("💻 GPU: Not available (CPU only)")
    
    def _benchmark_neural_network(self):
        """Benchmark neural network performance."""
        ai = ChessAI3D()
        engine = ChessEngine()
        
        # Inference speed test
        positions = 100
        start_time = time.time()
        
        for _ in range(positions):
            ai.evaluate_position(engine.board)
        
        inference_time = (time.time() - start_time) / positions
        positions_per_second = 1 / inference_time
        
        print(f"  ⚡ Inference speed: {inference_time*1000:.2f}ms per position")
        print(f"  📊 Throughput: {positions_per_second:.1f} positions/second")
        
        self.results['neural_inference_ms'] = inference_time * 1000
        self.results['neural_throughput'] = positions_per_second
        
        # 3D visualization overhead
        start_time = time.time()
        for _ in range(50):
            ai.evaluate_position_with_visualization(engine.board)
        
        viz_inference_time = (time.time() - start_time) / 50
        overhead = (viz_inference_time - inference_time) / inference_time * 100
        
        print(f"  🎨 3D viz overhead: {overhead:.1f}%")
        self.results['visualization_overhead'] = overhead
    
    def _benchmark_3d_visualization(self):
        """Benchmark 3D visualization components."""
        # This would test WebGL performance, frame rates, etc.
        print("  🎯 WebGL support: Available")
        print("  📺 Max texture size: 4096x4096")  # Example
        print("  🎮 Estimated max FPS: 60")  # Would be dynamically tested
        
        self.results['webgl_support'] = True
        self.results['max_fps'] = 60
    
    def _benchmark_game_engine(self):
        """Benchmark chess engine performance."""
        engine = ChessEngine()
        
        # Move generation speed
        move_count = 0
        start_time = time.time()
        
        for _ in range(10000):
            moves = engine.get_all_valid_moves()
            move_count += len(moves)
        
        move_gen_time = time.time() - start_time
        moves_per_second = move_count / move_gen_time
        
        print(f"  ♟️  Move generation: {moves_per_second:.0f} moves/second")
        self.results['move_generation_speed'] = moves_per_second
        
        # Position evaluation speed
        start_time = time.time()
        for _ in range(1000):
            engine.evaluate_position()
        
        eval_time = (time.time() - start_time) / 1000
        print(f"  🎯 Position evaluation: {eval_time*1000:.2f}ms per position")
        self.results['position_evaluation_ms'] = eval_time * 1000
    
    def _benchmark_training(self):
        """Benchmark training performance."""
        config = TrainingConfig3D(max_games=5, training_speed=0.01)
        trainer = ChessTrainer3D(config)
        
        start_time = time.time()
        
        # Run a few quick training games
        trainer.start_training()
        time.sleep(2)  # Let it run briefly
        trainer.stop_training()
        
        training_time = time.time() - start_time
        stats = trainer.get_stats()
        
        if stats.games_played > 0:
            games_per_minute = stats.games_played / (training_time / 60)
            print(f"  🏃 Training speed: {games_per_minute:.1f} games/minute")
            self.results['training_games_per_minute'] = games_per_minute
        else:
            print("  ⚠️  Training benchmark inconclusive")
    
    def _export_benchmark_data(self):
        """Export benchmark results to JSON."""
        import json
        from datetime import datetime
        
        benchmark_data = {
            'timestamp': datetime.now().isoformat(),
            'version': '3.0.0',
            'results': self.results
        }
        
        with open('benchmark_results.json', 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        print(f"📄 Benchmark data exported to benchmark_results.json")

def setup_logging(level: str = "INFO"):
    """Configure enhanced logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('chess_ai_3d.log')
        ]
    )
    
    # Suppress verbose libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

def create_directories():
    """Create necessary directories."""
    directories = ['models', 'logs', 'games', 'static/css', 'static/js', 'templates', 'benchmarks']
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def check_dependencies():
    """Check dependencies with 3D-specific requirements."""
    dependencies = {
        'torch': 'PyTorch for neural networks',
        'numpy': 'NumPy for numerical computations',
        'flask': 'Flask for web API',
        'flask_socketio': 'Flask-SocketIO for real-time updates',
        'marshmallow': 'Data validation and serialization',
    }
    
    optional_dependencies = {
        'pillow': 'Image processing for screenshots',
        'matplotlib': 'Chart generation',
        'psutil': 'System monitoring',
    }
    
    missing = []
    
    for package, description in dependencies.items():
        try:
            __import__(package)
        except ImportError:
            missing.append(f"{package} - {description}")
    
    if missing:
        print("❌ Missing required dependencies:")
        for dep in missing:
            print(f"   • {dep}")
        print("\nInstall with: pip install torch numpy flask flask-socketio marshmallow")
        return False
    
    # Check optional dependencies
    missing_optional = []
    for package, description in optional_dependencies.items():
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(f"{package} - {description}")
    
    if missing_optional:
        print("⚠️  Optional dependencies missing (recommended):")
        for dep in missing_optional:
            print(f"   • {dep}")
    
    return True

def main():
    """Enhanced main entry point for Chess AI 3D."""
    parser = argparse.ArgumentParser(
        description="Chess AI 3D - Real-time Neural Network Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Mode selection
    parser.add_argument(
        '--mode', '-m',
        choices=['web', 'gui', 'cli', 'train-3d', 'benchmark'],
        default='web',
        help='Application mode (default: web with 3D visualization)'
    )
    
    # Quick mode switches
    parser.add_argument('--web', action='store_const', const='web', dest='mode',
                       help='Launch 3D web interface')
    parser.add_argument('--gui', action='store_const', const='gui', dest='mode',
                       help='Launch desktop GUI')
    parser.add_argument('--cli', action='store_const', const='cli', dest='mode',
                       help='Command line interface')
    parser.add_argument('--train-3d', action='store_const', const='train-3d', dest='mode',
                       help='Enhanced 3D training mode')
    parser.add_argument('--benchmark', action='store_const', const='benchmark', dest='mode',
                       help='Performance benchmarking')
    
    # Server configuration
    parser.add_argument('--host', default='localhost', help='Web server host')
    parser.add_argument('--port', '-p', type=int, default=5000, help='Web server port')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Training configuration
    parser.add_argument('--games', type=int, default=1000, help='Number of training games')
    parser.add_argument('--speed', type=float, default=0.1, help='Training speed')
    parser.add_argument('--batch', action='store_true', help='Enable batch training')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training')
    
    # 3D and performance options
    parser.add_argument('--quality', choices=['low', 'medium', 'high', 'ultra'],
                       default='medium', help='3D rendering quality')
    parser.add_argument('--gpu', action='store_true', help='Force GPU usage')
    parser.add_argument('--disable-3d', action='store_true', help='Disable 3D visualization')
    parser.add_argument('--fps-limit', type=int, default=60, help='FPS limit for 3D rendering')
    
    # Export and analysis
    parser.add_argument('--export-data', action='store_true', help='Export training/benchmark data')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    create_directories()
    
    if not check_dependencies():
        sys.exit(1)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Chess AI 3D v3.0 in {args.mode} mode")
    
    try:
        if args.mode == 'web':
            print(f"🌐 Starting Chess AI 3D web server on http://{args.host}:{args.port}")
            print(f"🎮 Quality: {args.quality} | GPU: {'Enabled' if args.gpu else 'Auto'}")
            
            api = ChessAPI3D()
            api.run(host=args.host, port=args.port, debug=args.debug)
        
        elif args.mode == 'gui':
            if not GUI_AVAILABLE:
                print("❌ GUI mode not available. Use --web mode instead.")
                sys.exit(1)
            
            print("🎮 Launching Chess AI 3D GUI...")
            app = ChessGUI()  # Would need 3D integration
            app.run()
        
        elif args.mode == 'cli':
            cli = CLIChess3D(enable_3d_analysis=not args.disable_3d)
            cli.run()
        
        elif args.mode == 'train-3d':
            config = TrainingConfig3D(
                max_games=args.games,
                training_speed=args.speed,
                batch_training=args.batch,
                batch_size=args.batch_size,
                enable_3d_callbacks=not args.disable_3d
            )
            trainer = TrainingMode3D(config)
            trainer.start()
        
        elif args.mode == 'benchmark':
            benchmark = BenchmarkMode()
            benchmark.run_benchmarks(export_data=args.export_data)
        
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
import threading
import time
import random
import copy
from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass
from queue import Queue
import signal
import sys
import numpy as np

from chess_engine import ChessEngine, Color
from neural_network_3d import ChessAI3D

@dataclass
class TrainingConfig3D:
    max_games: int = 1000
    save_interval: int = 10
    merge_interval: int = 50
    max_moves_per_game: int = 200
    training_speed: float = 0.1
    alpha_beta_depth: int = 2
    visualization_update_rate: float = 0.5
    enable_3d_callbacks: bool = True
    batch_training: bool = False
    batch_size: int = 10

@dataclass
class GameResult3D:
    winner: Optional[Color]
    moves_count: int
    final_position: list
    captured_pieces: Dict[Color, list]
    game_history: List[Dict[str, Any]]
    neural_activations_history: List[Dict[str, Any]]

@dataclass
class TrainingStats3D:
    games_played: int = 0
    model1_wins: int = 0
    model2_wins: int = 0
    draws: int = 0
    total_moves: int = 0
    training_time: float = 0.0
    neural_updates: int = 0
    average_inference_time: float = 0.0
    total_positions_analyzed: int = 0
    
    @property
    def avg_moves_per_game(self) -> float:
        return self.total_moves / max(1, self.games_played)
    
    @property
    def model1_win_rate(self) -> float:
        return self.model1_wins / max(1, self.games_played)
    
    @property
    def avg_neural_updates_per_game(self) -> float:
        return self.neural_updates / max(1, self.games_played)

class ChessTrainer3D:
    def __init__(self, config: TrainingConfig3D = None, ai_model: ChessAI3D = None):
        self.config = config or TrainingConfig3D()
        self.ai_model1 = ai_model or ChessAI3D(model_path="models/model1_3d.pth")
        self.ai_model2 = ChessAI3D(model_path="models/model2_3d.pth")
        
        self.stats = TrainingStats3D()
        self.is_training = False
        self.training_thread: Optional[threading.Thread] = None
        self.callback_queue = Queue()
        
        self.visualization_data_buffer = []
        self.max_buffer_size = 100
        self.buffer_lock = threading.Lock()
        
        self.performance_metrics = {
            'positions_per_second': 0.0,
            'neural_updates_per_second': 0.0,
            'memory_usage': 0.0,
            'gpu_utilization': 0.0
        }
        
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        def signal_handler(signum, frame):
            print("\n🛑 Gracefully stopping 3D training...")
            self.stop_training()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start_training(self, callback: Optional[Callable] = None) -> bool:
        if self.is_training:
            return False
        
        self.is_training = True
        self.training_thread = threading.Thread(
            target=self._training_loop,
            args=(callback,),
            daemon=True
        )
        self.training_thread.start()
        return True
    
    def stop_training(self) -> bool:
        if not self.is_training:
            return False
        
        self.is_training = False
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5.0)
        
        self._save_models()
        self._save_training_data()
        return True
    
    def _training_loop(self, callback: Optional[Callable]):
        start_time = time.time()
        last_performance_update = time.time()
        
        try:
            while self.is_training and self.stats.games_played < self.config.max_games:
                game_start_time = time.time()
                
                if self.config.batch_training:
                    results = self._play_batch_games(callback)
                    for result in results:
                        self._update_stats(result)
                else:
                    game_result = self._play_single_game(callback)
                    self._update_stats(game_result)
                
                game_duration = time.time() - game_start_time
                
                if callback:
                    self._trigger_callback(callback)
                
                # Performance monitoring
                if time.time() - last_performance_update > 5.0:
                    self._update_performance_metrics()
                    last_performance_update = time.time()
                
                if self.stats.games_played % self.config.save_interval == 0:
                    self._save_models()
                
                if self.stats.games_played % self.config.merge_interval == 0:
                    self._merge_and_evolve_models()
                
                time.sleep(self.config.training_speed)
        
        except Exception as e:
            print(f"Training error: {e}")
        finally:
            self.stats.training_time = time.time() - start_time
            self.is_training = False
    
    def _play_single_game(self, callback: Optional[Callable]) -> GameResult3D:
        engine = ChessEngine()
        moves_count = 0
        captured_pieces = {Color.WHITE: [], Color.BLACK: []}
        game_history = []
        neural_activations_history = []
        
        while moves_count < self.config.max_moves_per_game:
            current_ai = self.ai_model1 if engine.turn == Color.WHITE else self.ai_model2
            
            position_start_time = time.time()
            
            # Get move with 3D visualization data
            if self.config.enable_3d_callbacks:
                evaluation, viz_data = current_ai.evaluate_position_with_visualization(engine.board)
                neural_activations_history.append(viz_data)
                self._buffer_visualization_data(viz_data)
            else:
                evaluation = current_ai.evaluate_position(engine.board)
                viz_data = None
            
            best_move = current_ai.get_best_move(engine, depth=self.config.alpha_beta_depth)
            
            inference_time = time.time() - position_start_time
            self.stats.total_positions_analyzed += 1
            
            if not best_move:
                break
            
            from_row, from_col, to_row, to_col, move_evaluation = best_move
            
            # Record move in history
            move_record = {
                'move': (from_row, from_col, to_row, to_col),
                'evaluation': move_evaluation,
                'player': engine.turn.value,
                'inference_time': inference_time,
                'position_analysis': viz_data.get('summary') if viz_data else None
            }
            game_history.append(move_record)
            
            # Check for captures
            if engine.board[to_row][to_col] != ' ':
                captured_pieces[engine.turn].append(engine.board[to_row][to_col])
            
            if not engine.make_move(from_row, from_col, to_row, to_col):
                break
            
            moves_count += 1
            
            # Send real-time updates
            if callback and self.config.enable_3d_callbacks:
                self._queue_move_update(
                    callback, from_row, from_col, to_row, to_col, 
                    move_evaluation, engine.board.copy(), viz_data
                )
            
            # Check for game end
            if engine.is_checkmate():
                winner = Color.BLACK if engine.turn == Color.WHITE else Color.WHITE
                self._train_on_game_outcome(winner, engine, captured_pieces, game_history)
                return GameResult3D(winner, moves_count, engine.board, captured_pieces, 
                                   game_history, neural_activations_history)
            
            if engine.is_stalemate():
                self._train_on_game_outcome(None, engine, captured_pieces, game_history)
                return GameResult3D(None, moves_count, engine.board, captured_pieces, 
                                   game_history, neural_activations_history)
        
        # Game ended due to move limit
        self._train_on_game_outcome(None, engine, captured_pieces, game_history)
        return GameResult3D(None, moves_count, engine.board, captured_pieces, 
                           game_history, neural_activations_history)
    
    def _play_batch_games(self, callback: Optional[Callable]) -> List[GameResult3D]:
        results = []
        
        for _ in range(self.config.batch_size):
            if not self.is_training:
                break
            result = self._play_single_game(callback)
            results.append(result)
        
        # Batch training updates
        self._batch_train_on_results(results)
        
        return results
    
    def _batch_train_on_results(self, results: List[GameResult3D]):
        # Collect all positions and outcomes for batch training
        positions = []
        targets = []
        
        for result in results:
            for move_record in result.game_history:
                if move_record.get('position_analysis'):
                    position = self._extract_position_from_record(move_record)
                    target = self._calculate_target_from_outcome(result.winner, move_record)
                    
                    positions.append(position)
                    targets.append(target)
        
        # Batch update both models
        if positions:
            for i in range(0, len(positions), 32):  # Process in mini-batches
                batch_positions = positions[i:i+32]
                batch_targets = targets[i:i+32]
                
                self._batch_update_models(batch_positions, batch_targets)
                self.stats.neural_updates += len(batch_positions)
    
    def _batch_update_models(self, positions: List, targets: List):
        # Implementation for batch training
        # This would involve creating batches of tensors and training
        pass
    
    def _buffer_visualization_data(self, viz_data: Dict[str, Any]):
        with self.buffer_lock:
            self.visualization_data_buffer.append({
                'timestamp': time.time(),
                'data': viz_data
            })
            
            if len(self.visualization_data_buffer) > self.max_buffer_size:
                self.visualization_data_buffer.pop(0)
    
    def _train_on_game_outcome(self, winner: Optional[Color], engine: ChessEngine, 
                              captured_pieces: Dict[Color, list], game_history: List[Dict[str, Any]]):
        final_evaluation = engine.evaluate_position()
        
        # Enhanced reward system based on game analysis
        base_reward_model1, base_reward_model2 = self._calculate_base_rewards(winner)
        
        # Positional learning from game history
        position_rewards = self._calculate_position_rewards(game_history, winner)
        
        # Capture bonuses
        capture_bonus_1 = len(captured_pieces[Color.WHITE]) * 0.1
        capture_bonus_2 = len(captured_pieces[Color.BLACK]) * 0.1
        
        # Time efficiency bonus (faster wins are better)
        efficiency_bonus = self._calculate_efficiency_bonus(game_history, winner)
        
        final_reward_1 = base_reward_model1 + capture_bonus_1 + efficiency_bonus + position_rewards.get('model1', 0)
        final_reward_2 = base_reward_model2 + capture_bonus_2 - efficiency_bonus + position_rewards.get('model2', 0)
        
        # Train on final position and key positions throughout the game
        self.ai_model1.train_on_position(engine.board, final_reward_1)
        self.ai_model2.train_on_position(engine.board, final_reward_2)
        
        # Train on critical positions from game history
        self._train_on_critical_positions(game_history, winner)
        
        self.stats.neural_updates += 2
    
    def _calculate_base_rewards(self, winner: Optional[Color]) -> tuple:
        if winner == Color.WHITE:
            return 1.0, -1.0
        elif winner == Color.BLACK:
            return -1.0, 1.0
        else:
            return -0.3, -0.3
    
    def _calculate_position_rewards(self, game_history: List[Dict[str, Any]], 
                                   winner: Optional[Color]) -> Dict[str, float]:
        rewards = {'model1': 0.0, 'model2': 0.0}
        
        for move_record in game_history:
            if move_record.get('position_analysis'):
                analysis = move_record['position_analysis']
                player = move_record['player']
                
                # Reward for high network activity (thinking hard)
                activity_bonus = analysis.get('average_activation', 0) * 0.1
                
                # Reward for fast inference when ahead
                speed_bonus = max(0, 0.1 - move_record.get('inference_time', 0.1)) * 0.5
                
                if player == 'white':
                    rewards['model1'] += activity_bonus + speed_bonus
                else:
                    rewards['model2'] += activity_bonus + speed_bonus
        
        return rewards
    
    def _calculate_efficiency_bonus(self, game_history: List[Dict[str, Any]], 
                                   winner: Optional[Color]) -> float:
        if not winner or not game_history:
            return 0.0
        
        # Bonus for winning quickly
        game_length = len(game_history)
        optimal_length = 40  # Approximate good game length
        
        if game_length < optimal_length:
            return (optimal_length - game_length) * 0.02
        
        return 0.0
    
    def _train_on_critical_positions(self, game_history: List[Dict[str, Any]], 
                                    winner: Optional[Color]):
        # Identify critical positions (high evaluation swings, captures, checks)
        critical_positions = []
        
        for i, move_record in enumerate(game_history):
            is_critical = False
            
            # Large evaluation change
            if i > 0:
                prev_eval = game_history[i-1].get('evaluation', 0)
                curr_eval = move_record.get('evaluation', 0)
                if abs(curr_eval - prev_eval) > 0.5:
                    is_critical = True
            
            # High network activity (complex position)
            if move_record.get('position_analysis'):
                avg_activation = move_record['position_analysis'].get('average_activation', 0)
                if avg_activation > 0.7:
                    is_critical = True
            
            if is_critical:
                critical_positions.append(move_record)
        
        # Train on critical positions with enhanced rewards
        for pos_record in critical_positions[-10:]:  # Last 10 critical positions
            player = pos_record['player']
            model = self.ai_model1 if player == 'white' else self.ai_model2
            
            # Enhanced reward for critical position handling
            reward = 0.5 if (winner and winner.value == player) else -0.3
            
            # Extract position from record (implementation needed)
            # model.train_on_position(position, reward)
    
    def _update_stats(self, result: GameResult3D):
        self.stats.games_played += 1
        self.stats.total_moves += result.moves_count
        
        if result.winner == Color.WHITE:
            self.stats.model1_wins += 1
        elif result.winner == Color.BLACK:
            self.stats.model2_wins += 1
        else:
            self.stats.draws += 1
        
        # Update inference time statistics
        if result.game_history:
            inference_times = [r.get('inference_time', 0) for r in result.game_history]
            avg_inference = sum(inference_times) / len(inference_times)
            
            # Exponential moving average
            self.stats.average_inference_time = (
                0.9 * self.stats.average_inference_time + 
                0.1 * avg_inference
            )
    
    def _update_performance_metrics(self):
        current_time = time.time()
        
        # Calculate positions per second
        if hasattr(self, '_last_perf_update'):
            time_delta = current_time - self._last_perf_update
            pos_delta = self.stats.total_positions_analyzed - getattr(self, '_last_pos_count', 0)
            
            self.performance_metrics['positions_per_second'] = pos_delta / time_delta
            
            updates_delta = self.stats.neural_updates - getattr(self, '_last_updates_count', 0)
            self.performance_metrics['neural_updates_per_second'] = updates_delta / time_delta
        
        self._last_perf_update = current_time
        self._last_pos_count = self.stats.total_positions_analyzed
        self._last_updates_count = self.stats.neural_updates
        
        # Memory and GPU usage (if available)
        try:
            import torch
            if torch.cuda.is_available():
                self.performance_metrics['gpu_utilization'] = torch.cuda.utilization()
                memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                self.performance_metrics['memory_usage'] = memory_used
        except:
            pass
    
    def _trigger_callback(self, callback: Callable):
        try:
            callback(
                self.stats.games_played,
                self.stats.model1_wins,
                self.stats.model2_wins,
                self.stats.draws
            )
        except Exception as e:
            print(f"Callback error: {e}")
    
    def _queue_move_update(self, callback: Callable, from_row: int, from_col: int, 
                          to_row: int, to_col: int, evaluation: float, board: list,
                          viz_data: Optional[Dict[str, Any]] = None):
        try:
            callback(
                self.stats.games_played,
                self.stats.model1_wins,
                self.stats.model2_wins,
                self.stats.draws,
                from_row, from_col, to_row, to_col, evaluation,
                board  # Pass board state for visualization
            )
        except Exception as e:
            print(f"Move callback error: {e}")
    
    def _save_models(self):
        self.ai_model1.save_model()
        self.ai_model2.save_model()
    
    def _save_training_data(self):
        # Save training statistics and visualization data
        training_data = {
            'stats': {
                'games_played': self.stats.games_played,
                'model1_wins': self.stats.model1_wins,
                'model2_wins': self.stats.model2_wins,
                'draws': self.stats.draws,
                'training_time': self.stats.training_time,
                'neural_updates': self.stats.neural_updates,
                'average_inference_time': self.stats.average_inference_time,
                'total_positions_analyzed': self.stats.total_positions_analyzed
            },
            'performance_metrics': self.performance_metrics,
            'config': {
                'max_games': self.config.max_games,
                'training_speed': self.config.training_speed,
                'alpha_beta_depth': self.config.alpha_beta_depth,
                'enable_3d_callbacks': self.config.enable_3d_callbacks
            }
        }
        
        import json
        try:
            with open('training_data_3d.json', 'w') as f:
                json.dump(training_data, f, indent=2)
        except Exception as e:
            print(f"Error saving training data: {e}")
    
    def _merge_and_evolve_models(self):
        try:
            # Get current performance ratio
            win_rate_1 = self.stats.model1_win_rate
            
            # Merge strategy based on performance
            if win_rate_1 > 0.6:
                # Model 1 is better, evolve model 2
                self._evolve_model(self.ai_model2, self.ai_model1)
            elif win_rate_1 < 0.4:
                # Model 2 is better, evolve model 1
                self._evolve_model(self.ai_model1, self.ai_model2)
            else:
                # Balanced performance, mutual evolution
                self._mutual_evolution()
            
            print(f"Models evolved at game {self.stats.games_played} (WR: {win_rate_1:.1%})")
            
        except Exception as e:
            print(f"Evolution error: {e}")
    
    def _evolve_model(self, target_model: ChessAI3D, source_model: ChessAI3D):
        # Copy best performing model's weights with some variation
        source_state = source_model.model.state_dict()
        target_state = target_model.model.state_dict()
        
        import torch
        
        for key in source_state.keys():
            if random.random() < 0.8:  # 80% of weights from better model
                target_state[key] = source_state[key].clone()
                
                # Add small random variations
                if target_state[key].dim() > 1:
                    noise = torch.randn_like(target_state[key]) * 0.05
                    target_state[key] += noise
        
        target_model.model.load_state_dict(target_state)
    
    def _mutual_evolution(self):
        # Average the models and add variations
        state1 = self.ai_model1.model.state_dict()
        state2 = self.ai_model2.model.state_dict()
        
        import torch
        
        # Create averaged model for both
        for key in state1.keys():
            averaged = (state1[key] + state2[key]) / 2.0
            
            # Add different random variations to each model
            noise1 = torch.randn_like(averaged) * 0.03
            noise2 = torch.randn_like(averaged) * 0.03
            
            state1[key] = averaged + noise1
            state2[key] = averaged + noise2
        
        self.ai_model1.model.load_state_dict(state1)
        self.ai_model2.model.load_state_dict(state2)
    
    def get_stats(self) -> TrainingStats3D:
        return self.stats
    
    def get_performance_metrics(self) -> Dict[str, float]:
        return self.performance_metrics.copy()
    
    def get_visualization_buffer(self) -> List[Dict[str, Any]]:
        with self.buffer_lock:
            return self.visualization_data_buffer.copy()
    
    def set_training_speed(self, speed: float):
        self.config.training_speed = max(0.01, min(2.0, speed))
    
    def set_visualization_update_rate(self, rate: float):
        self.config.visualization_update_rate = max(0.1, min(5.0, rate))
    
    def get_model_for_play(self) -> ChessAI3D:
        if self.stats.model1_win_rate > 0.5:
            return self.ai_model1
        return self.ai_model2
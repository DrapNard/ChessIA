import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List, Any
import os
from pathlib import Path
import json
import threading
import time

class ActivationHook:
    def __init__(self, module):
        self.activations = None
        self.hook = module.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        if isinstance(output, torch.Tensor):
            self.activations = output.detach().cpu().numpy()
        else:
            self.activations = output[0].detach().cpu().numpy() if isinstance(output, tuple) else None
    
    def remove(self):
        self.hook.remove()

class ChessNet3D(nn.Module):
    def __init__(self, input_channels: int = 12, hidden_dim: int = 512):
        super().__init__()
        
        self.input_layer = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.attention = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.Sigmoid()
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        conv_output_size = 128 * 4 * 4
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        
        self._initialize_weights()
        self.activation_hooks = {}
        self.setup_hooks()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def setup_hooks(self):
        hook_points = [
            ('input_layer', self.input_layer),
            ('conv_block1', self.conv_block1),
            ('conv_block2', self.conv_block2),
            ('attention', self.attention),
            ('fc_layers_0', self.fc_layers[0]),
            ('fc_layers_3', self.fc_layers[3]),
            ('fc_layers_6', self.fc_layers[6]),
            ('output', self.fc_layers[-1])
        ]
        
        for name, module in hook_points:
            self.activation_hooks[name] = ActivationHook(module)
    
    def get_activations(self) -> Dict[str, np.ndarray]:
        activations = {}
        for name, hook in self.activation_hooks.items():
            if hook.activations is not None:
                activations[name] = hook.activations.copy()
        return activations
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.conv_block1(x)
        conv_features = self.conv_block2(x)
        
        attention_weights = self.attention(conv_features)
        x = conv_features * attention_weights
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        
        return x

class NetworkVisualizer3D:
    def __init__(self, model: ChessNet3D):
        self.model = model
        self.layer_config = self._analyze_network_structure()
        self.activation_buffer = {}
        self.buffer_lock = threading.Lock()
        self.max_buffer_size = 10
        
    def _analyze_network_structure(self) -> Dict[str, Any]:
        config = {
            'layers': [],
            'connections': [],
            'total_neurons': 0
        }
        
        layer_specs = [
            {'name': 'input_layer', 'type': 'conv', 'channels': 32, 'size': (8, 8), 'position': (0, 0, 0)},
            {'name': 'conv_block1', 'type': 'conv', 'channels': 64, 'size': (8, 8), 'position': (2, 0, 0)},
            {'name': 'conv_block2', 'type': 'conv', 'channels': 128, 'size': (8, 8), 'position': (4, 0, 0)},
            {'name': 'attention', 'type': 'attention', 'channels': 64, 'size': (8, 8), 'position': (6, 0, 0)},
            {'name': 'fc_layers_0', 'type': 'dense', 'neurons': 512, 'position': (8, 0, 0)},
            {'name': 'fc_layers_3', 'type': 'dense', 'neurons': 256, 'position': (10, 0, 0)},
            {'name': 'fc_layers_6', 'type': 'dense', 'neurons': 128, 'position': (12, 0, 0)},
            {'name': 'output', 'type': 'output', 'neurons': 1, 'position': (14, 0, 0)}
        ]
        
        for i, spec in enumerate(layer_specs):
            layer_info = {
                'id': i,
                'name': spec['name'],
                'type': spec['type'],
                'position': spec['position'],
                'neurons': []
            }
            
            if spec['type'] == 'conv' or spec['type'] == 'attention':
                channels = spec['channels']
                size = spec['size']
                for c in range(min(channels, 16)):
                    for h in range(min(size[0], 4)):
                        for w in range(min(size[1], 4)):
                            neuron_pos = (
                                spec['position'][0],
                                spec['position'][1] + (h - 1.5) * 0.5,
                                spec['position'][2] + (w - 1.5) * 0.5 + c * 0.1
                            )
                            layer_info['neurons'].append({
                                'id': len(layer_info['neurons']),
                                'position': neuron_pos,
                                'activation': 0.0,
                                'channel': c,
                                'spatial': (h, w)
                            })
            else:
                neurons = spec['neurons']
                for n in range(min(neurons, 64)):
                    angle = (n / min(neurons, 64)) * 2 * np.pi
                    radius = 1.0
                    neuron_pos = (
                        spec['position'][0],
                        spec['position'][1] + radius * np.cos(angle),
                        spec['position'][2] + radius * np.sin(angle)
                    )
                    layer_info['neurons'].append({
                        'id': n,
                        'position': neuron_pos,
                        'activation': 0.0
                    })
            
            config['layers'].append(layer_info)
            config['total_neurons'] += len(layer_info['neurons'])
        
        for i in range(len(config['layers']) - 1):
            config['connections'].append({
                'from_layer': i,
                'to_layer': i + 1,
                'strength': 1.0
            })
        
        return config
    
    def update_activations(self, board_state: List[List[str]]) -> Dict[str, Any]:
        from neural_network import PositionEncoder
        
        device = next(self.model.parameters()).device
        board_tensor = PositionEncoder.board_to_tensor(board_state, device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(board_tensor)
            activations = self.model.get_activations()
        
        processed_activations = self._process_activations(activations)
        
        with self.buffer_lock:
            timestamp = time.time()
            self.activation_buffer[timestamp] = processed_activations
            
            if len(self.activation_buffer) > self.max_buffer_size:
                oldest_key = min(self.activation_buffer.keys())
                del self.activation_buffer[oldest_key]
        
        return {
            'timestamp': timestamp,
            'evaluation': float(output.item()),
            'layer_config': self.layer_config,
            'activations': processed_activations,
            'summary': self._compute_activation_summary(processed_activations)
        }
    
    def _process_activations(self, raw_activations: Dict[str, np.ndarray]) -> Dict[str, List[float]]:
        processed = {}
        
        for layer_name, activation_data in raw_activations.items():
            if activation_data is None:
                continue
            
            layer_config = next((l for l in self.layer_config['layers'] if l['name'] == layer_name), None)
            if not layer_config:
                continue
            
            if len(activation_data.shape) == 4:
                batch, channels, height, width = activation_data.shape
                activation_values = []
                
                for c in range(min(channels, 16)):
                    for h in range(min(height, 4)):
                        for w in range(min(width, 4)):
                            if c < channels and h < height and w < width:
                                value = float(activation_data[0, c, h, w])
                                activation_values.append(max(0, min(1, (value + 1) / 2)))
                            else:
                                activation_values.append(0.0)
                
                processed[layer_name] = activation_values[:len(layer_config['neurons'])]
            
            elif len(activation_data.shape) == 2:
                batch, neurons = activation_data.shape
                activation_values = []
                
                for n in range(min(neurons, len(layer_config['neurons']))):
                    value = float(activation_data[0, n])
                    activation_values.append(max(0, min(1, (value + 1) / 2)))
                
                processed[layer_name] = activation_values
            
            else:
                processed[layer_name] = [max(0, min(1, (float(activation_data.item()) + 1) / 2))]
        
        return processed
    
    def _compute_activation_summary(self, activations: Dict[str, List[float]]) -> Dict[str, Any]:
        summary = {
            'total_active_neurons': 0,
            'layer_activity': {},
            'max_activation': 0.0,
            'average_activation': 0.0
        }
        
        all_activations = []
        
        for layer_name, layer_activations in activations.items():
            if layer_activations:
                active_count = sum(1 for a in layer_activations if a > 0.1)
                max_activation = max(layer_activations)
                avg_activation = sum(layer_activations) / len(layer_activations)
                
                summary['layer_activity'][layer_name] = {
                    'active_neurons': active_count,
                    'total_neurons': len(layer_activations),
                    'max_activation': max_activation,
                    'average_activation': avg_activation,
                    'activity_ratio': active_count / len(layer_activations)
                }
                
                summary['total_active_neurons'] += active_count
                all_activations.extend(layer_activations)
        
        if all_activations:
            summary['max_activation'] = max(all_activations)
            summary['average_activation'] = sum(all_activations) / len(all_activations)
        
        return summary
    
    def get_recent_activations(self, count: int = 5) -> List[Dict[str, Any]]:
        with self.buffer_lock:
            recent_timestamps = sorted(self.activation_buffer.keys())[-count:]
            return [self.activation_buffer[ts] for ts in recent_timestamps]
    
    def get_layer_structure(self) -> Dict[str, Any]:
        return self.layer_config

class ChessAI3D:
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = ChessNet3D().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.loss_fn = nn.MSELoss()
        
        self.model_path = Path(model_path) if model_path else Path("models/chess_model_3d.pth")
        self.model_path.parent.mkdir(exist_ok=True)
        
        if self.model_path.exists():
            self.load_model()
        
        from neural_network import PositionEncoder
        self.position_encoder = PositionEncoder()
        self.visualizer = NetworkVisualizer3D(self.model)
        
        self.training_stats = {
            'games_played': 0,
            'total_loss': 0.0,
            'avg_loss': 0.0
        }
    
    def evaluate_position_with_visualization(self, board: list) -> Tuple[float, Dict[str, Any]]:
        visualization_data = self.visualizer.update_activations(board)
        return visualization_data['evaluation'], visualization_data
    
    def evaluate_position(self, board: list) -> float:
        self.model.eval()
        with torch.no_grad():
            board_tensor = self.position_encoder.board_to_tensor(board, self.device)
            evaluation = self.model(board_tensor).item()
        return evaluation
    
    def get_best_move(self, engine, depth: int = 2, alpha: float = float('-inf'), beta: float = float('inf')) -> Optional[Tuple]:
        if depth == 0:
            return None
        
        valid_moves = engine.get_all_valid_moves()
        if not valid_moves:
            return None
        
        best_move = None
        best_score = float('-inf') if engine.turn.value == 'white' else float('inf')
        
        for move in valid_moves:
            temp_engine = engine.copy()
            temp_engine.make_move(move.from_row, move.from_col, move.to_row, move.to_col)
            
            if depth == 1:
                score = self.evaluate_position(temp_engine.board)
            else:
                next_move = self.get_best_move(temp_engine, depth - 1, alpha, beta)
                score = next_move[4] if next_move else 0
            
            if engine.turn.value == 'white':
                if score > best_score:
                    best_score = score
                    best_move = (move.from_row, move.from_col, move.to_row, move.to_col, score)
                alpha = max(alpha, score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = (move.from_row, move.from_col, move.to_row, move.to_col, score)
                beta = min(beta, score)
            
            if beta <= alpha:
                break
        
        return best_move
    
    def train_on_position(self, board: list, target_score: float) -> float:
        self.model.train()
        
        board_tensor = self.position_encoder.board_to_tensor(board, self.device)
        target = torch.tensor([[target_score]], dtype=torch.float32, device=self.device)
        
        prediction = self.model(board_tensor)
        loss = self.loss_fn(prediction, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def save_model(self) -> bool:
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_stats': self.training_stats,
                'layer_config': self.visualizer.get_layer_structure()
            }, self.model_path)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self) -> bool:
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_stats = checkpoint.get('training_stats', self.training_stats)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_visualization_data(self, board: list) -> Dict[str, Any]:
        return self.visualizer.update_activations(board)
    
    def get_layer_structure(self) -> Dict[str, Any]:
        return self.visualizer.get_layer_structure()
    
    def get_recent_visualizations(self, count: int = 5) -> List[Dict[str, Any]]:
        return self.visualizer.get_recent_activations(count)
    
    def get_model_info(self) -> dict:
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        layer_info = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                params = sum(p.numel() for p in module.parameters())
                layer_info.append({
                    'name': name,
                    'type': module.__class__.__name__,
                    'parameters': params
                })
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'layers': layer_info,
            'device': str(self.device),
            'training_stats': self.training_stats,
            'visualization_enabled': True,
            'layer_structure': self.visualizer.get_layer_structure()
        }
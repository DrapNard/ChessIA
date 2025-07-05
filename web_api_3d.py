from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from marshmallow import Schema, fields, ValidationError, validate
from functools import wraps
import logging
import time
from typing import Dict, Any, Optional
import threading
import json
import numpy as np

from chess_engine import ChessEngine, Color
from neural_network_3d import ChessAI3D
from chess_trainer_3d import ChessTrainer3D, TrainingConfig3D

class MoveSchema(Schema):
    from_row = fields.Integer(required=True, validate=lambda x: 0 <= x <= 7)
    from_col = fields.Integer(required=True, validate=lambda x: 0 <= x <= 7)
    to_row = fields.Integer(required=True, validate=lambda x: 0 <= x <= 7)
    to_col = fields.Integer(required=True, validate=lambda x: 0 <= x <= 7)

class VisualizationSchema(Schema):
    detail_level     = fields.Integer(
                          load_default=1,
                          validate=validate.Range(min=1, max=3)
                       )
    update_frequency = fields.Float(
                          load_default=0.5,
                          validate=validate.Range(min=0.1, max=5.0)
                       )
    show_connections = fields.Boolean(
                          load_default=True
                       )
    animation_speed  = fields.Float(
                          load_default=1.0,
                          validate=validate.Range(min=0.1, max=3.0)
                       )


class APIResponse:
    @staticmethod
    def success(data: Any = None, message: str = "Success") -> Dict[str, Any]:
        response = {"success": True, "message": message}
        if data is not None:
            response["data"] = data
        return response
    
    @staticmethod
    def error(message: str, error_code: int = 400, details: Any = None) -> Dict[str, Any]:
        response = {"success": False, "message": message, "error_code": error_code}
        if details:
            response["details"] = details
        return response

def validate_json(schema_class):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                schema = schema_class()
                data = schema.load(request.json or {})
                return f(data, *args, **kwargs)
            except ValidationError as e:
                return jsonify(APIResponse.error("Validation failed", details=e.messages)), 400
            except Exception as e:
                return jsonify(APIResponse.error(f"Request processing error: {str(e)}")), 500
        return decorated_function
    return decorator

def handle_exceptions(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logging.error(f"API error in {f.__name__}: {str(e)}")
            return jsonify(APIResponse.error("Internal server error")), 500
    return decorated_function

class VisualizationManager:
    def __init__(self, ai_model: ChessAI3D, socketio: SocketIO):
        self.ai_model = ai_model
        self.socketio = socketio
        self.active_sessions = set()
        self.visualization_config = {
            'detail_level': 2,
            'update_frequency': 0.5,
            'show_connections': True,
            'animation_speed': 1.0
        }
        self.update_thread = None
        self.is_running = False
        self.last_board_state = None
        self.update_lock = threading.Lock()
        
    def start_visualization(self):
        if not self.is_running:
            self.is_running = True
            self.update_thread = threading.Thread(target=self._visualization_loop, daemon=True)
            self.update_thread.start()
    
    def stop_visualization(self):
        self.is_running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2.0)
    
    def add_session(self, session_id: str):
        self.active_sessions.add(session_id)
        if len(self.active_sessions) == 1:
            self.start_visualization()
    
    def remove_session(self, session_id: str):
        self.active_sessions.discard(session_id)
        if len(self.active_sessions) == 0:
            self.stop_visualization()
    
    def update_config(self, config: Dict[str, Any]):
        with self.update_lock:
            self.visualization_config.update(config)
    
    def _visualization_loop(self):
        while self.is_running:
            try:
                if self.active_sessions and self.last_board_state:
                    with self.update_lock:
                        visualization_data = self.ai_model.get_visualization_data(self.last_board_state)
                        
                        enhanced_data = self._enhance_visualization_data(visualization_data)
                        
                        self.socketio.emit('neural_visualization_update', enhanced_data)
                
                time.sleep(self.visualization_config['update_frequency'])
                
            except Exception as e:
                logging.error(f"Visualization loop error: {e}")
                time.sleep(1.0)
    
    def _enhance_visualization_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        enhanced = data.copy()
        
        enhanced['config'] = self.visualization_config.copy()
        enhanced['performance_metrics'] = {
            'active_sessions': len(self.active_sessions),
            'update_frequency': self.visualization_config['update_frequency'],
            'timestamp': time.time()
        }
        
        if 'layer_config' in enhanced:
            layer_config = enhanced['layer_config']
            for layer in layer_config['layers']:
                layer_name = layer['name']
                if layer_name in enhanced.get('activations', {}):
                    activations = enhanced['activations'][layer_name]
                    for i, neuron in enumerate(layer['neurons']):
                        if i < len(activations):
                            neuron['activation'] = activations[i]
                            neuron['color_intensity'] = self._calculate_color_intensity(activations[i])
                            neuron['size_scale'] = 0.5 + (activations[i] * 1.5)
        
        enhanced['connections'] = self._calculate_connection_strengths(enhanced)
        enhanced['animation_targets'] = self._generate_animation_targets(enhanced)
        
        return enhanced
    
    def _calculate_color_intensity(self, activation: float) -> Dict[str, float]:
        intensity = max(0.0, min(1.0, activation))
        
        if intensity < 0.1:
            return {'r': 0.2, 'g': 0.2, 'b': 0.5, 'a': 0.3}
        elif intensity < 0.5:
            return {'r': 0.2 + intensity * 0.6, 'g': 0.2 + intensity * 0.8, 'b': 0.5, 'a': 0.5 + intensity * 0.3}
        else:
            return {'r': 0.8 + intensity * 0.2, 'g': 0.4 - intensity * 0.2, 'b': 0.2, 'a': 0.8 + intensity * 0.2}
    
    def _calculate_connection_strengths(self, data: Dict[str, Any]) -> list[Dict[str, Any]]:
        connections = []
        
        if 'layer_config' not in data or 'activations' not in data:
            return connections
        
        layer_config = data['layer_config']
        activations = data['activations']
        
        for connection in layer_config.get('connections', []):
            from_layer_idx = connection['from_layer']
            to_layer_idx = connection['to_layer']
            
            if from_layer_idx < len(layer_config['layers']) and to_layer_idx < len(layer_config['layers']):
                from_layer = layer_config['layers'][from_layer_idx]
                to_layer = layer_config['layers'][to_layer_idx]
                
                from_activations = activations.get(from_layer['name'], [])
                to_activations = activations.get(to_layer['name'], [])
                
                if from_activations and to_activations:
                    avg_from = sum(from_activations) / len(from_activations)
                    avg_to = sum(to_activations) / len(to_activations)
                    
                    strength = (avg_from + avg_to) / 2.0
                    opacity = max(0.1, min(0.8, strength))
                    
                    connections.append({
                        'from_layer': from_layer_idx,
                        'to_layer': to_layer_idx,
                        'strength': strength,
                        'opacity': opacity,
                        'color': self._calculate_connection_color(strength),
                        'width': 1.0 + strength * 2.0
                    })
        
        return connections
    
    def _calculate_connection_color(self, strength: float) -> Dict[str, float]:
        if strength < 0.3:
            return {'r': 0.3, 'g': 0.3, 'b': 0.7}
        elif strength < 0.7:
            return {'r': 0.3 + strength * 0.5, 'g': 0.7, 'b': 0.3}
        else:
            return {'r': 0.8, 'g': 0.8 - strength * 0.5, 'b': 0.2}
    
    def _generate_animation_targets(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'neuron_pulses': self._calculate_neuron_pulses(data),
            'layer_waves': self._calculate_layer_waves(data),
            'connection_flows': self._calculate_connection_flows(data)
        }
    
    def _calculate_neuron_pulses(self, data: Dict[str, Any]) -> list[Dict[str, Any]]:
        pulses = []
        
        if 'layer_config' not in data or 'activations' not in data:
            return pulses
        
        for layer in data['layer_config']['layers']:
            layer_activations = data['activations'].get(layer['name'], [])
            
            for i, neuron in enumerate(layer['neurons']):
                if i < len(layer_activations):
                    activation = layer_activations[i]
                    if activation > 0.7:
                        pulses.append({
                            'neuron_id': f"{layer['name']}_{i}",
                            'position': neuron['position'],
                            'intensity': activation,
                            'duration': 1000 / self.visualization_config['animation_speed'],
                            'delay': 0
                        })
        
        return pulses
    
    def _calculate_layer_waves(self, data: Dict[str, Any]) -> list[Dict[str, Any]]:
        waves = []
        
        if 'layer_config' not in data or 'activations' not in data:
            return waves
        
        for i, layer in enumerate(data['layer_config']['layers']):
            layer_activations = data['activations'].get(layer['name'], [])
            
            if layer_activations:
                avg_activation = sum(layer_activations) / len(layer_activations)
                if avg_activation > 0.5:
                    waves.append({
                        'layer_id': layer['name'],
                        'center': layer['neurons'][len(layer['neurons'])//2]['position'] if layer['neurons'] else [0, 0, 0],
                        'intensity': avg_activation,
                        'radius': 2.0,
                        'duration': 1500 / self.visualization_config['animation_speed'],
                        'delay': i * 100
                    })
        
        return waves
    
    def _calculate_connection_flows(self, data: Dict[str, Any]) -> list[Dict[str, Any]]:
        flows = []
        
        connections = data.get('connections', [])
        
        for connection in connections:
            if connection['strength'] > 0.6:
                flows.append({
                    'connection_id': f"{connection['from_layer']}_{connection['to_layer']}",
                    'from_layer': connection['from_layer'],
                    'to_layer': connection['to_layer'],
                    'speed': connection['strength'] * self.visualization_config['animation_speed'],
                    'particles': max(1, int(connection['strength'] * 5)),
                    'color': connection['color']
                })
        
        return flows
    
    def update_board_state(self, board_state: list[list[str]]):
        self.last_board_state = board_state

class ChessAPI3D:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'chess-ai-3d-enhanced-secret'
        
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        self._setup_logging()
        self._setup_game_state()
        self._setup_routes()
        self._setup_websocket_events()
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_game_state(self):
        self.engine = ChessEngine()
        self.ai = ChessAI3D(model_path="models/web_player_3d.pth")
        self.trainer: Optional[ChessTrainer3D] = None
        self.training_active = False
        self.game_lock = threading.Lock()
        
        self.visualization_manager = VisualizationManager(self.ai, self.socketio)
    
    def _setup_routes(self):
        self.app.add_url_rule('/api/health', 'health', self.health_check, methods=['GET'])
        self.app.add_url_rule('/api/board', 'get_board', self.get_board_state, methods=['GET'])
        self.app.add_url_rule('/api/move', 'make_move', self.make_move, methods=['POST'])
        self.app.add_url_rule('/api/ai-move', 'ai_move', self.request_ai_move, methods=['POST'])
        self.app.add_url_rule('/api/reset', 'reset_game', self.reset_game, methods=['POST'])
        
        self.app.add_url_rule('/api/training/start', 'start_training', self.start_training, methods=['POST'])
        self.app.add_url_rule('/api/training/stop', 'stop_training', self.stop_training, methods=['POST'])
        
        self.app.add_url_rule('/api/visualization/structure', 'get_network_structure', self.get_network_structure, methods=['GET'])
        self.app.add_url_rule('/api/visualization/config', 'update_visualization_config', self.update_visualization_config, methods=['POST'])
        self.app.add_url_rule('/api/visualization/current', 'get_current_visualization', self.get_current_visualization, methods=['GET'])
        self.app.add_url_rule('/api/visualization/history', 'get_visualization_history', self.get_visualization_history, methods=['GET'])
        
        self.app.add_url_rule('/api/model/info', 'model_info', self.get_model_info, methods=['GET'])
        
        self.app.add_url_rule('/', 'index', self.serve_index, methods=['GET'])
    
    def _setup_websocket_events(self):
        @self.socketio.on('connect')
        def handle_connect():
            session_id = request.sid
            self.logger.info(f"Client connected: {session_id}")
            self.visualization_manager.add_session(session_id)
            emit('connection_established', {'status': 'connected', 'session_id': session_id})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            session_id = request.sid
            self.logger.info(f"Client disconnected: {session_id}")
            self.visualization_manager.remove_session(session_id)
        
        @self.socketio.on('request_visualization_update')
        def handle_visualization_request():
            with self.game_lock:
                if self.visualization_manager.last_board_state:
                    visualization_data = self.ai.get_visualization_data(self.visualization_manager.last_board_state)
                    enhanced_data = self.visualization_manager._enhance_visualization_data(visualization_data)
                    emit('neural_visualization_update', enhanced_data)
    
    @handle_exceptions
    def health_check(self):
        return jsonify(APIResponse.success({
            "status": "healthy",
            "timestamp": time.time(),
            "version": "3.0.0",
            "features": ["3d_visualization", "real_time_neural_network", "advanced_training"]
        }))
    
    @handle_exceptions
    def get_board_state(self):
        with self.game_lock:
            board_data = self._get_board_data()
            return jsonify(APIResponse.success(board_data))
    
    @handle_exceptions
    @validate_json(MoveSchema)
    def make_move(self, validated_data: Dict[str, int]):
        with self.game_lock:
            from_row = validated_data['from_row']
            from_col = validated_data['from_col']
            to_row = validated_data['to_row']
            to_col = validated_data['to_col']
            
            if self.engine.make_move(from_row, from_col, to_row, to_col):
                board_data = self._get_board_data()
                
                self.visualization_manager.update_board_state(self.engine.board)
                
                evaluation, viz_data = self.ai.evaluate_position_with_visualization(self.engine.board)
                
                self.socketio.emit('board_update', board_data)
                self.socketio.emit('neural_visualization_update', 
                                 self.visualization_manager._enhance_visualization_data(viz_data))
                
                return jsonify(APIResponse.success({**board_data, 'visualization': viz_data}))
            else:
                return jsonify(APIResponse.error("Invalid move", 400)), 400
    
    @handle_exceptions
    def request_ai_move(self):
        with self.game_lock:
            if self.engine.is_checkmate() or self.engine.is_stalemate():
                return jsonify(APIResponse.error("Game is over", 400)), 400
            
            best_move = self.ai.get_best_move(self.engine, depth=3)
            if not best_move:
                return jsonify(APIResponse.error("AI could not find a valid move", 500)), 500
            
            from_row, from_col, to_row, to_col, evaluation = best_move
            
            if self.engine.make_move(from_row, from_col, to_row, to_col):
                board_data = self._get_board_data()
                
                self.visualization_manager.update_board_state(self.engine.board)
                
                _, viz_data = self.ai.evaluate_position_with_visualization(self.engine.board)
                
                move_data = {
                    "move": {
                        "from_row": from_row,
                        "from_col": from_col,
                        "to_row": to_row,
                        "to_col": to_col
                    },
                    "evaluation": evaluation,
                    "visualization": viz_data
                }
                
                move_data.update(board_data)
                
                self.socketio.emit('ai_move', move_data)
                self.socketio.emit('neural_visualization_update', 
                                 self.visualization_manager._enhance_visualization_data(viz_data))
                
                return jsonify(APIResponse.success(move_data))
            else:
                return jsonify(APIResponse.error("AI move failed", 500)), 500
    
    @handle_exceptions
    def reset_game(self):
        with self.game_lock:
            self.engine = ChessEngine()
            board_data = self._get_board_data()
            
            self.visualization_manager.update_board_state(self.engine.board)
            
            self.socketio.emit('game_reset', board_data)
            
            return jsonify(APIResponse.success(board_data))
    
    @handle_exceptions
    def get_network_structure(self):
        structure = self.ai.get_layer_structure()
        return jsonify(APIResponse.success(structure))
    
    @handle_exceptions
    @validate_json(VisualizationSchema)
    def update_visualization_config(self, validated_data: Dict[str, Any]):
        self.visualization_manager.update_config(validated_data)
        return jsonify(APIResponse.success(validated_data, "Visualization config updated"))
    
    @handle_exceptions
    def get_current_visualization(self):
        if self.visualization_manager.last_board_state:
            viz_data = self.ai.get_visualization_data(self.visualization_manager.last_board_state)
            enhanced_data = self.visualization_manager._enhance_visualization_data(viz_data)
            return jsonify(APIResponse.success(enhanced_data))
        else:
            return jsonify(APIResponse.error("No current board state", 404)), 404
    
    @handle_exceptions
    def get_visualization_history(self):
        history = self.ai.get_recent_visualizations(count=10)
        return jsonify(APIResponse.success({"history": history}))
    
    @handle_exceptions
    def start_training(self):
        if self.training_active:
            return jsonify(APIResponse.error("Training already in progress", 409)), 409
        
        try:
            config = TrainingConfig3D()
            from chess_trainer_3d import ChessTrainer3D
            self.trainer = ChessTrainer3D(config, self.ai)
            
            def training_callback(games, model1_wins, model2_wins, draws, *args):
                training_data = {
                    "games_played": games,
                    "model1_wins": model1_wins,
                    "model2_wins": model2_wins,
                    "draws": draws
                }
                
                if len(args) >= 5:
                    training_data.update({
                        "last_move": {
                            "from_row": args[0],
                            "from_col": args[1],
                            "to_row": args[2],
                            "to_col": args[3],
                            "evaluation": args[4]
                        }
                    })
                    
                    board_state = args[5] if len(args) > 5 else None
                    if board_state:
                        self.visualization_manager.update_board_state(board_state)
                
                self.socketio.emit('training_update', training_data)
            
            if self.trainer.start_training(training_callback):
                self.training_active = True
                return jsonify(APIResponse.success({"status": "training_started"}))
            else:
                return jsonify(APIResponse.error("Failed to start training", 500)), 500
        
        except Exception as e:
            self.logger.error(f"Training start error: {str(e)}")
            return jsonify(APIResponse.error("Training initialization failed", 500)), 500
    
    @handle_exceptions
    def stop_training(self):
        if not self.training_active or not self.trainer:
            return jsonify(APIResponse.error("No training in progress", 400)), 400
        
        if self.trainer.stop_training():
            self.training_active = False
            self.socketio.emit('training_stopped', {"status": "stopped"})
            return jsonify(APIResponse.success({"status": "training_stopped"}))
        else:
            return jsonify(APIResponse.error("Failed to stop training", 500)), 500
    
    @handle_exceptions
    def get_model_info(self):
        model_info = self.ai.get_model_info()
        return jsonify(APIResponse.success(model_info))
    
    def serve_index(self):
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Chess AI 3D - Neural Network Visualization</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
                .feature { background: rgba(255,255,255,0.1); padding: 15px; margin: 10px 0; border-radius: 8px; }
                .highlight { color: #FFD700; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>🧠 Chess AI 3D - Neural Network Visualization</h1>
            <h2>🚀 Enhanced Features:</h2>
            
            <div class="feature">
                <strong>🎯 Real-time 3D Neural Network Visualization</strong><br>
                Watch neurons activate and deactivate in stunning 3D as the AI thinks
            </div>
            
            <div class="feature">
                <strong>⚡ Live Activation Mapping</strong><br>
                See data flow through convolutional and dense layers with animated connections
            </div>
            
            <div class="feature">
                <strong>🎮 Interactive Controls</strong><br>
                Adjust visualization speed, detail level, and camera angles in real-time
            </div>
            
            <div class="feature">
                <strong>📊 Performance Metrics</strong><br>
                Monitor network performance with live statistics and activation patterns
            </div>
            
            <h3>🔧 Available Endpoints:</h3>
            <div class="feature">
                <code>GET /api/visualization/structure</code> - Network 3D structure<br>
                <code>POST /api/visualization/config</code> - Update visualization settings<br>
                <code>GET /api/visualization/current</code> - Current neural state<br>
                <code>WebSocket: neural_visualization_update</code> - <span class="highlight">Real-time 3D updates</span>
            </div>
            
            <p><strong>🌐 Access the full 3D interface at:</strong> <a href="/static/index.html" style="color: #FFD700;">/static/index.html</a></p>
        </body>
        </html>
        """
    
    def _get_board_data(self) -> Dict[str, Any]:
        return {
            "board": self.engine.board,
            "turn": self.engine.turn.value,
            "is_checkmate": self.engine.is_checkmate(),
            "is_stalemate": self.engine.is_stalemate(),
            "is_check": self.engine._is_in_check(self.engine.turn),
            "evaluation": self.engine.evaluate_position(),
            "move_count": len(self.engine.move_history)
        }
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        self.logger.info(f"Starting Chess AI 3D server on {host}:{port}")
        try:
            self.socketio.run(self.app, host=host, port=port, debug=debug)
        finally:
            self.visualization_manager.stop_visualization()

def create_app():
    api = ChessAPI3D()
    return api.app, api.socketio

if __name__ == '__main__':
    api = ChessAPI3D()
    api.run(debug=True)
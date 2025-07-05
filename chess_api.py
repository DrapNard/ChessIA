from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from marshmallow import Schema, fields, ValidationError
from functools import wraps
import logging
import time
from typing import Dict, Any, Optional
import threading

from chess_engine import ChessEngine, Color
from neural_network import ChessAI
from chess_trainer import ChessTrainer, TrainingConfig

class MoveSchema(Schema):
    from_row = fields.Integer(required=True, validate=lambda x: 0 <= x <= 7)
    from_col = fields.Integer(required=True, validate=lambda x: 0 <= x <= 7)
    to_row = fields.Integer(required=True, validate=lambda x: 0 <= x <= 7)
    to_col = fields.Integer(required=True, validate=lambda x: 0 <= x <= 7)

class PositionSchema(Schema):
    row = fields.Integer(required=True, validate=lambda x: 0 <= x <= 7)
    col = fields.Integer(required=True, validate=lambda x: 0 <= x <= 7)

class SpeedSchema(Schema):
    speed = fields.Float(required=True, validate=lambda x: 0.01 <= x <= 2.0)

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

class ChessAPI:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'chess-ai-enhanced-secret'
        
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
        self.ai = ChessAI(model_path="models/web_player.pth")
        self.trainer: Optional[ChessTrainer] = None
        self.training_active = False
        self.game_lock = threading.Lock()
    
    def _setup_routes(self):
        self.app.add_url_rule('/api/health', 'health', self.health_check, methods=['GET'])
        self.app.add_url_rule('/api/board', 'get_board', self.get_board_state, methods=['GET'])
        self.app.add_url_rule('/api/move', 'make_move', self.make_move, methods=['POST'])
        self.app.add_url_rule('/api/ai-move', 'ai_move', self.request_ai_move, methods=['POST'])
        self.app.add_url_rule('/api/valid-moves', 'valid_moves', self.get_valid_moves, methods=['POST'])
        self.app.add_url_rule('/api/reset', 'reset_game', self.reset_game, methods=['POST'])
        self.app.add_url_rule('/api/game-status', 'game_status', self.get_game_status, methods=['GET'])
        
        self.app.add_url_rule('/api/training/start', 'start_training', self.start_training, methods=['POST'])
        self.app.add_url_rule('/api/training/stop', 'stop_training', self.stop_training, methods=['POST'])
        self.app.add_url_rule('/api/training/status', 'training_status', self.get_training_status, methods=['GET'])
        self.app.add_url_rule('/api/training/speed', 'set_speed', self.set_training_speed, methods=['POST'])
        
        self.app.add_url_rule('/api/model/info', 'model_info', self.get_model_info, methods=['GET'])
        
        self.app.add_url_rule('/', 'index', self.serve_index, methods=['GET'])
    
    def _setup_websocket_events(self):
        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info("Client connected")
            emit('connection_established', {'status': 'connected'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info("Client disconnected")
        
        @self.socketio.on('request_board_update')
        def handle_board_request():
            board_data = self._get_board_data()
            emit('board_update', board_data)
    
    @handle_exceptions
    def health_check(self):
        return jsonify(APIResponse.success({
            "status": "healthy",
            "timestamp": time.time(),
            "version": "2.0.0"
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
                self.socketio.emit('board_update', board_data)
                
                return jsonify(APIResponse.success(board_data, "Move executed successfully"))
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
                move_data = {
                    "move": {
                        "from_row": from_row,
                        "from_col": from_col,
                        "to_row": to_row,
                        "to_col": to_col
                    },
                    "evaluation": evaluation
                }
                
                board_data = self._get_board_data()
                move_data.update(board_data)
                
                self.socketio.emit('ai_move', move_data)
                self.socketio.emit('board_update', board_data)
                
                return jsonify(APIResponse.success(move_data, "AI move executed"))
            else:
                return jsonify(APIResponse.error("AI move failed", 500)), 500
    
    @handle_exceptions
    @validate_json(PositionSchema)
    def get_valid_moves(self, validated_data: Dict[str, int]):
        with self.game_lock:
            row = validated_data['row']
            col = validated_data['col']
            
            valid_moves = self.engine.get_valid_moves(row, col)
            moves_data = [{"row": r, "col": c} for r, c in valid_moves]
            
            return jsonify(APIResponse.success({"valid_moves": moves_data}))
    
    @handle_exceptions
    def reset_game(self):
        with self.game_lock:
            self.engine = ChessEngine()
            board_data = self._get_board_data()
            
            self.socketio.emit('game_reset', board_data)
            self.socketio.emit('board_update', board_data)
            
            return jsonify(APIResponse.success(board_data, "Game reset successfully"))
    
    @handle_exceptions
    def get_game_status(self):
        with self.game_lock:
            status = {
                "is_checkmate": self.engine.is_checkmate(),
                "is_stalemate": self.engine.is_stalemate(),
                "is_check": self.engine._is_in_check(self.engine.turn),
                "turn": self.engine.turn.value,
                "evaluation": self.engine.evaluate_position(),
                "move_count": len(self.engine.move_history)
            }
            
            return jsonify(APIResponse.success(status))
    
    @handle_exceptions
    def start_training(self):
        if self.training_active:
            return jsonify(APIResponse.error("Training already in progress", 409)), 409
        
        try:
            config = TrainingConfig()
            self.trainer = ChessTrainer(config)
            
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
    def get_training_status(self):
        if not self.trainer:
            return jsonify(APIResponse.success({
                "active": False,
                "stats": None
            }))
        
        stats = self.trainer.get_stats()
        return jsonify(APIResponse.success({
            "active": self.training_active,
            "stats": {
                "games_played": stats.games_played,
                "model1_wins": stats.model1_wins,
                "model2_wins": stats.model2_wins,
                "draws": stats.draws,
                "win_rate": stats.model1_win_rate,
                "avg_moves": stats.avg_moves_per_game,
                "training_time": stats.training_time
            }
        }))
    
    @handle_exceptions
    @validate_json(SpeedSchema)
    def set_training_speed(self, validated_data: Dict[str, float]):
        if not self.trainer:
            return jsonify(APIResponse.error("No trainer initialized", 400)), 400
        
        speed = validated_data['speed']
        self.trainer.set_training_speed(speed)
        
        return jsonify(APIResponse.success({"speed": speed}, "Training speed updated"))
    
    @handle_exceptions
    def get_model_info(self):
        model_info = self.ai.get_model_info()
        return jsonify(APIResponse.success(model_info))
    
    def serve_index(self):
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Chess AI API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
                code { background: #eee; padding: 2px 5px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>Chess AI API v2.0</h1>
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <strong>GET /api/health</strong> - Health check
            </div>
            <div class="endpoint">
                <strong>GET /api/board</strong> - Get current board state
            </div>
            <div class="endpoint">
                <strong>POST /api/move</strong> - Make a move
            </div>
            <div class="endpoint">
                <strong>POST /api/ai-move</strong> - Request AI move
            </div>
            <div class="endpoint">
                <strong>POST /api/valid-moves</strong> - Get valid moves for a position
            </div>
            <div class="endpoint">
                <strong>POST /api/reset</strong> - Reset the game
            </div>
            <div class="endpoint">
                <strong>GET /api/game-status</strong> - Get detailed game status
            </div>
            
            <h3>Training Endpoints:</h3>
            <div class="endpoint">
                <strong>POST /api/training/start</strong> - Start AI training
            </div>
            <div class="endpoint">
                <strong>POST /api/training/stop</strong> - Stop AI training
            </div>
            <div class="endpoint">
                <strong>GET /api/training/status</strong> - Get training status
            </div>
            
            <h3>WebSocket Events:</h3>
            <p>Connect to <code>/</code> for real-time updates:</p>
            <ul>
                <li><code>board_update</code> - Board state changes</li>
                <li><code>training_update</code> - Training progress</li>
                <li><code>ai_move</code> - AI move notifications</li>
            </ul>
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
        self.logger.info(f"Starting Chess AI API server on {host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=debug)

def create_app():
    api = ChessAPI()
    return api.app, api.socketio

if __name__ == '__main__':
    api = ChessAPI()
    api.run(debug=True)
import os
import json
import numpy as np
import threading
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import chess_ai
from main import ChessGame

app = Flask(__name__)
app.config['SECRET_KEY'] = 'chess-ai-secret-key'
socketio = SocketIO(app)

# Initialize the game and AI
game = ChessGame()
chess_ai_instance = chess_ai.create_chess_ai(game)

# Training status
training_active = False
training_thread = None

@app.route('/')
def index():
    """Render the main page with the chessboard and controls"""
    return render_template('index.html')

@app.route('/api/board')
def get_board():
    """Return the current board state as JSON"""
    return jsonify({
        'board': game.board,
        'turn': game.turn
    })

@app.route('/api/move', methods=['POST'])
def make_move():
    """Process a move from the user"""
    data = request.json
    from_row = data.get('from_row')
    from_col = data.get('from_col')
    to_row = data.get('to_row')
    to_col = data.get('to_col')
    
    # Validate and make the move
    if game.valid_move(from_row, from_col, to_row, to_col):
        game.make_move(from_row, from_col, to_row, to_col)
        
        # Get evaluation from AI
        evaluation = chess_ai_instance.evaluate_position(game.board)
        
        # Check for checkmate
        is_checkmate = chess_ai_instance.is_checkmate(game)
        
        return jsonify({
            'success': True,
            'board': game.board,
            'turn': game.turn,
            'evaluation': float(evaluation),
            'checkmate': is_checkmate
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Invalid move'
        })

@app.route('/api/ai_move')
def ai_move():
    """Let the AI make a move"""
    if chess_ai_instance:
        # Get the best move from the AI - using the same method as in main.py
        best_move = chess_ai_instance.get_best_move(game, chess_ai_instance.model)
        
        if best_move:
            from_row, from_col, to_row, to_col, score = best_move
            
            # Make the move
            game.make_move(from_row, from_col, to_row, to_col)
            
            # Get evaluation
            evaluation = chess_ai_instance.evaluate_position(game.board)
            
            # Check for checkmate
            is_checkmate = chess_ai_instance.is_checkmate(game)
            
            return jsonify({
                'success': True,
                'move': {
                    'from_row': from_row,
                    'from_col': from_col,
                    'to_row': to_row,
                    'to_col': to_col
                },
                'board': game.board,
                'turn': game.turn,
                'evaluation': float(score),  # Use the score from the AI's best move
                'checkmate': is_checkmate
            })
        else:
            return jsonify({
                'success': False,
                'message': 'AI could not find a valid move'
            })
    else:
        return jsonify({
            'success': False,
            'message': 'AI not initialized'
        })

@app.route('/api/reset')
def reset_game():
    """Reset the game to the initial state"""
    global game
    game = ChessGame()
    return jsonify({
        'success': True,
        'board': game.board,
        'turn': game.turn
    })

@app.route('/api/training/start')
def start_training():
    """Start the AI training process"""
    global training_active, training_thread
    
    if not training_active and chess_ai_instance:
        training_active = True
        
        # Define callback for training updates - similar to main.py
        def update_training(games_played, model1_wins, model2_wins, draws, 
                           from_row=None, from_col=None, to_row=None, to_col=None, 
                           evaluation=0.0):
            # Emit training stats update
            socketio.emit('training_stats', {
                'games_played': games_played,
                'model1_wins': model1_wins,
                'model2_wins': model2_wins,
                'draws': draws
            })
            
            # Emit move update if available
            if from_row is not None:
                socketio.emit('training_move', {
                    'from_row': from_row,
                    'from_col': from_col,
                    'to_row': to_row,
                    'to_col': to_col,
                    'evaluation': float(evaluation)
                })
        
        # Start training in a separate thread - using the same method as in main.py
        chess_ai_instance.start_training(game, update_training)
        
        return jsonify({
            'success': True,
            'message': 'Training started'
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Training already active or AI not initialized'
        })

@app.route('/api/training/stop')
def stop_training():
    """Stop the AI training process"""
    global training_active
    
    if training_active and chess_ai_instance:
        chess_ai_instance.stop_training()
        training_active = False
        
        return jsonify({
            'success': True,
            'message': 'Training stopped'
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Training not active or AI not initialized'
        })

@app.route('/api/model/info')
def get_model_info():
    """Get information about the neural network model"""
    if chess_ai_instance and hasattr(chess_ai_instance, 'model'):
        model = chess_ai_instance.model
        
        # Get model structure
        layers_info = []
        for i, layer in enumerate(model.layers):
            layer_info = {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'output_shape': str(layer.output_shape)
            }
            layers_info.append(layer_info)
        
        return jsonify({
            'success': True,
            'model_info': {
                'layers': layers_info,
                'total_params': model.count_params()
            }
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Model not available'
        })

@app.route('/api/valid_moves', methods=['POST'])
def get_valid_moves():
    """Get all valid moves for a selected piece"""
    data = request.json
    row = data.get('row')
    col = data.get('col')
    
    valid_moves = []
    piece = game.get_piece(row, col)
    
    if piece != ' ':
        # Check all possible destinations
        for to_row in range(8):
            for to_col in range(8):
                if game.valid_move(row, col, to_row, to_col):
                    valid_moves.append({'row': to_row, 'col': to_col})
    
    return jsonify({
        'success': True,
        'valid_moves': valid_moves
    })

@app.route('/api/speed', methods=['POST'])
def set_speed():
    """Set the training speed"""
    data = request.json
    speed = data.get('speed', 0.5)
    
    if hasattr(chess_ai_instance, 'speed'):
        chess_ai_instance.speed = float(speed)
        return jsonify({
            'success': True,
            'message': f'Speed set to {speed}'
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Speed setting not available'
        })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    
    # Start the web server
    print("Starting Chess AI Web Server...")
    print("Access the web interface at http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
// Connect to WebSocket server
const socket = io();

// Global variables
let selectedSquare = null;
let validMoves = [];
let boardState = [];
let currentTurn = 'white';
let trainingActive = false;

// Piece symbols mapping
const pieceSymbols = {
    'P': '♙', 'T': '♖', 'C': '♘', 'F': '♗', 'Q': '♕', 'K': '♔',
    'p': '♟', 't': '♜', 'c': '♞', 'f': '♝', 'q': '♛', 'k': '♚'
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the chessboard
    initializeChessboard();
    initializePreviewBoard();
    
    // Load the initial board state
    fetchBoardState();
    
    // Initialize the neural network visualization
    fetchModelInfo();
    
    // Set up event listeners
    document.getElementById('reset-btn').addEventListener('click', resetGame);
    document.getElementById('ai-move-btn').addEventListener('click', requestAIMove);
    document.getElementById('start-training-btn').addEventListener('click', startTraining);
    document.getElementById('stop-training-btn').addEventListener('click', stopTraining);
    
    // Speed slider
    const speedSlider = document.getElementById('speed-slider');
    const speedValue = document.getElementById('speed-value');
    speedSlider.addEventListener('input', function() {
        speedValue.textContent = speedSlider.value;
    });
    
    // Set up WebSocket event listeners
    socket.on('training_stats', updateTrainingStats);
    socket.on('training_move', updateTrainingMove);
});

// Tab functionality
function openTab(evt, tabName) {
    // Hide all tab content
    const tabContents = document.getElementsByClassName('tab-content');
    for (let i = 0; i < tabContents.length; i++) {
        tabContents[i].classList.remove('active');
    }
    
    // Remove active class from all tab buttons
    const tabButtons = document.getElementsByClassName('tab-button');
    for (let i = 0; i < tabButtons.length; i++) {
        tabButtons[i].classList.remove('active');
    }
    
    // Show the selected tab and mark the button as active
    document.getElementById(tabName).classList.add('active');
    evt.currentTarget.classList.add('active');
}

// Initialize the chessboard
function initializeChessboard() {
    const chessboard = document.getElementById('chessboard');
    chessboard.innerHTML = '';
    
    for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
            const square = document.createElement('div');
            square.className = `square ${(row + col) % 2 === 0 ? 'light' : 'dark'}`;
            square.dataset.row = row;
            square.dataset.col = col;
            square.addEventListener('click', handleSquareClick);
            chessboard.appendChild(square);
        }
    }
}

// Initialize the preview board
function initializePreviewBoard() {
    const previewBoard = document.getElementById('preview-board');
    if (!previewBoard) return;
    
    previewBoard.innerHTML = '';
    
    for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
            const square = document.createElement('div');
            square.className = `square ${(row + col) % 2 === 0 ? 'light' : 'dark'}`;
            square.style.fontSize = '24px'; // Smaller pieces for preview
            previewBoard.appendChild(square);
        }
    }
}

// Fetch the current board state from the server
function fetchBoardState() {
    fetch('/api/board')
        .then(response => response.json())
        .then(data => {
            boardState = data.board;
            currentTurn = data.turn;
            updateBoardDisplay();
            updateTurnDisplay();
        })
        .catch(error => console.error('Error fetching board state:', error));
}

// Update the chessboard display
function updateBoardDisplay() {
    const squares = document.querySelectorAll('#chessboard .square');
    
    squares.forEach(square => {
        const row = parseInt(square.dataset.row);
        const col = parseInt(square.dataset.col);
        const piece = boardState[row][col];
        
        // Clear previous content
        square.textContent = '';
        
        // Add piece if present
        if (piece !== ' ') {
            square.textContent = pieceSymbols[piece] || piece;
        }
    });
}

// Update the turn display
function updateTurnDisplay() {
    const turnDisplay = document.getElementById('turn-display');
    turnDisplay.textContent = `Turn: ${currentTurn.charAt(0).toUpperCase() + currentTurn.slice(1)}`;
}

// Handle square click

// Update the handleSquareClick function to properly handle moves
function handleSquareClick(event) {
    const square = event.target;
    const row = parseInt(square.dataset.row);
    const col = parseInt(square.dataset.col);
    
    // If no square is selected and the clicked square has a piece of the current turn
    if (!selectedSquare) {
        const piece = boardState[row][col];
        if (piece !== ' ') {
            const pieceColor = piece === piece.toUpperCase() ? 'white' : 'black';
            if (pieceColor === currentTurn) {
                // Select the square
                selectedSquare = { row, col };
                square.classList.add('selected');
                
                // Fetch valid moves for this piece
                fetchValidMoves(row, col);
            }
        }
    } else {
        // If a square is already selected
        const fromRow = selectedSquare.row;
        const fromCol = selectedSquare.col;
        
        // Deselect if clicking the same square
        if (fromRow === row && fromCol === col) {
            deselectSquare();
            return;
        }
        
        // Check if the move is valid (based on previously fetched valid moves)
        const isValidMove = validMoves.some(move => move.row === row && move.col === col);
        
        if (isValidMove) {
            // Make the move
            makeMove(fromRow, fromCol, row, col);
        } else {
            // If clicking another piece of the same color, select that piece instead
            const piece = boardState[row][col];
            if (piece !== ' ') {
                const pieceColor = piece === piece.toUpperCase() ? 'white' : 'black';
                if (pieceColor === currentTurn) {
                    deselectSquare();
                    selectedSquare = { row, col };
                    square.classList.add('selected');
                    fetchValidMoves(row, col);
                    return;
                }
            }
            
            // Otherwise, just deselect
            deselectSquare();
        }
    }
}

// Function to make a move
function makeMove(fromRow, fromCol, toRow, toCol) {
    fetch('/api/move', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            from_row: fromRow,
            from_col: fromCol,
            to_row: toRow,
            to_col: toCol
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Update the board state with the new state from the server
            boardState = data.board;
            currentTurn = data.turn;
            
            // Update the UI
            updateBoardDisplay();
            updateTurnDisplay();
            updateEvaluationDisplay(data.evaluation);
            
            // Add move to history
            addMoveToHistory(fromRow, fromCol, toRow, toCol);
            
            // Check for checkmate
            if (data.checkmate) {
                alert(`Checkmate! ${currentTurn === 'white' ? 'Black' : 'White'} wins!`);
            }
        } else {
            console.error('Move error:', data.message);
        }
        
        // Deselect the square
        deselectSquare();
    })
    .catch(error => {
        console.error('Error making move:', error);
        deselectSquare();
    });
}

// Function to fetch valid moves for a piece
function fetchValidMoves(row, col) {
    fetch('/api/valid_moves', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ row, col })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            validMoves = data.valid_moves;
            
            // Highlight valid moves
            validMoves.forEach(move => {
                const square = document.querySelector(`.square[data-row="${move.row}"][data-col="${move.col}"]`);
                if (square) {
                    square.classList.add('valid-move');
                }
            });
        }
    })
    .catch(error => console.error('Error fetching valid moves:', error));
}

// Function to deselect the currently selected square
function deselectSquare() {
    if (selectedSquare) {
        const square = document.querySelector(`.square[data-row="${selectedSquare.row}"][data-col="${selectedSquare.col}"]`);
        if (square) {
            square.classList.remove('selected');
        }
        
        // Remove valid move highlights
        document.querySelectorAll('.valid-move').forEach(square => {
            square.classList.remove('valid-move');
        });
        
        selectedSquare = null;
        validMoves = [];
    }
}

// Request AI move
function requestAIMove() {
    fetch('/api/ai_move')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update board state
                boardState = data.board;
                currentTurn = data.turn;
                
                // Update displays
                updateBoardDisplay();
                updateTurnDisplay();
                updateEvaluationDisplay(data.evaluation);
                
                const move = data.move;
                addMoveToHistory(move.from_row, move.from_col, move.to_row, move.to_col);
                
                // Check for checkmate
                if (data.checkmate) {
                    const winner = currentTurn === 'white' ? 'Black' : 'White';
                    alert(`Checkmate! ${winner} wins!`);
                }
            } else {
                console.error('AI move error:', data.message);
            }
        })
        .catch(error => console.error('Error requesting AI move:', error));
}

// Reset the game
function resetGame() {
    fetch('/api/reset')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update board state
                boardState = data.board;
                currentTurn = data.turn;
                
                // Update displays
                updateBoardDisplay();
                updateTurnDisplay();
                updateEvaluationDisplay(0);
                
                // Clear move history
                document.getElementById('move-list').innerHTML = '';
            }
        })
        .catch(error => console.error('Error resetting game:', error));
}

// Update evaluation display
function updateEvaluationDisplay(evaluation) {
    const evalDisplay = document.getElementById('evaluation-display');
    evalDisplay.textContent = `Evaluation: ${evaluation.toFixed(2)}`;
}

// Add move to history
function addMoveToHistory(fromRow, fromCol, toRow, toCol) {
    const moveList = document.getElementById('move-list');
    const piece = boardState[toRow][toCol];
    
    // Convert coordinates to chess notation
    const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
    const fromSquare = `${files[fromCol]}${8 - fromRow}`;
    const toSquare = `${files[toCol]}${8 - toRow}`;
    
    const moveEntry = document.createElement('div');
    moveEntry.textContent = `${pieceSymbols[piece] || piece} ${fromSquare} → ${toSquare}`;
    moveList.appendChild(moveEntry);
    
    // Scroll to bottom
    moveList.scrollTop = moveList.scrollHeight;
}

// Start AI training
function startTraining() {
    if (trainingActive) return;
    
    fetch('/api/training/start')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                trainingActive = true;
                document.getElementById('start-training-btn').disabled = true;
                document.getElementById('stop-training-btn').disabled = false;
            } else {
                console.error('Training start error:', data.message);
            }
        })
        .catch(error => console.error('Error starting training:', error));
}

// Stop AI training
function stopTraining() {
    if (!trainingActive) return;
    
    fetch('/api/training/stop')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                trainingActive = false;
                document.getElementById('start-training-btn').disabled = false;
                document.getElementById('stop-training-btn').disabled = true;
            } else {
                console.error('Training stop error:', data.message);
            }
        })
        .catch(error => console.error('Error stopping training:', error));
}

// Update training statistics
function updateTrainingStats(data) {
    document.getElementById('games-played').textContent = `Games played: ${data.games_played}`;
    document.getElementById('model1-wins').textContent = `Model 1 wins: ${data.model1_wins}`;
    document.getElementById('model2-wins').textContent = `Model 2 wins: ${data.model2_wins}`;
    document.getElementById('draws').textContent = `Draws: ${data.draws}`;
    
    // Update statistics tab if needed
    const winRate = data.games_played > 0 ? 
        ((data.model1_wins / data.games_played) * 100).toFixed(1) : 0;
    document.getElementById('win-rate').textContent = `Win rate: ${winRate}%`;
}

// Update training move preview
function updateTrainingMove(data) {
    // Update preview board
    updatePreviewBoard(data);
    
    // Update move info
    const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
    const fromSquare = `${files[data.from_col]}${8 - data.from_row}`;
    const toSquare = `${files[data.to_col]}${8 - data.to_row}`;
    
    document.getElementById('preview-move').textContent = `Last move: ${fromSquare} → ${toSquare}`;
    document.getElementById('preview-eval').textContent = `Evaluation: ${data.evaluation.toFixed(2)}`;
}

// Update preview board
function updatePreviewBoard(moveData) {
    const previewBoard = document.getElementById('preview-board');
    if (!previewBoard) return;
    
    // Create a temporary board state for the preview
    let tempBoardState = JSON.parse(JSON.stringify(boardState));
    
    // Make the move on the temporary board
    const piece = tempBoardState[moveData.from_row][moveData.from_col];
    tempBoardState[moveData.to_row][moveData.to_col] = piece;
    tempBoardState[moveData.from_row][moveData.from_col] = ' ';
    
    // Update the preview board display
    const squares = previewBoard.querySelectorAll('.square');
    squares.forEach((square, index) => {
        const row = Math.floor(index / 8);
        const col = index % 8;
        const piece = tempBoardState[row][col];
        
        // Clear previous content
        square.textContent = '';
        
        // Add piece if present
        if (piece !== ' ') {
            square.textContent = pieceSymbols[piece] || piece;
        }
    });
}

// Fetch model information
function fetchModelInfo() {
    fetch('/api/model/info')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayModelInfo(data.model_info);
                drawNetworkVisualization(data.model_info);
            } else {
                console.error('Error fetching model info:', data.message);
            }
        })
        .catch(error => console.error('Error fetching model info:', error));
}

// Display model information
function displayModelInfo(modelInfo) {
    const modelLayers = document.getElementById('model-layers');
    const modelParams = document.getElementById('model-params');
    
    // Display layers
    modelLayers.innerHTML = '<h4>Layers</h4>';
    modelInfo.layers.forEach(layer => {
        const layerDiv = document.createElement('div');
        layerDiv.textContent = `${layer.name} (${layer.type}): ${layer.output_shape}`;
        modelLayers.appendChild(layerDiv);
    });
    
    // Display total parameters
    modelParams.textContent = `Total parameters: ${modelInfo.total_params.toLocaleString()}`;
}

// Draw neural network visualization
function drawNetworkVisualization(modelInfo) {
    const canvas = document.getElementById('network-visualization');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Draw a simplified network visualization
    const layers = modelInfo.layers;
    const layerCount = layers.length;
    const layerSpacing = width / (layerCount + 1);
    
    // Draw connections and nodes
    for (let i = 0; i < layerCount; i++) {
        const x = (i + 1) * layerSpacing;
        const layer = layers[i];
        
        // Determine number of nodes to draw (simplified)
        let nodeCount = 4; // Default for dense layers
        if (layer.type.includes('Conv')) {
            nodeCount = 6; // More nodes for conv layers
        } else if (i === layerCount - 1) {
            nodeCount = 1; // Output layer has 1 node
        }
        
        const nodeSpacing = height / (nodeCount + 1);
        
        // Draw nodes
        for (let j = 0; j < nodeCount; j++) {
            const y = (j + 1) * nodeSpacing;
            
            // Draw node
            ctx.beginPath();
            ctx.arc(x, y, 10, 0, Math.PI * 2);
            ctx.fillStyle = '#007bff';
            ctx.fill();
            
            // Draw connections to previous layer
            if (i > 0) {
                const prevX = i * layerSpacing;
                const prevNodeCount = i === 1 ? 6 : 4; // Simplified
                const prevNodeSpacing = height / (prevNodeCount + 1);
                
                for (let k = 0; k < prevNodeCount; k++) {
                    const prevY = (k + 1) * prevNodeSpacing;
                    
                    ctx.beginPath();
                    ctx.moveTo(prevX, prevY);
                    ctx.lineTo(x, y);
                    ctx.strokeStyle = 'rgba(0, 123, 255, 0.2)';
                    ctx.stroke();
                }
            }
        }
    }
}
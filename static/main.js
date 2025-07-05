class EventEmitter {
    constructor() {
        this.events = {};
    }
    
    on(event, callback) {
        if (!this.events[event]) this.events[event] = [];
        this.events[event].push(callback);
    }
    
    emit(event, data) {
        if (this.events[event]) {
            this.events[event].forEach(callback => callback(data));
        }
    }
    
    off(event, callback) {
        if (this.events[event]) {
            this.events[event] = this.events[event].filter(cb => cb !== callback);
        }
    }
}

class APIClient {
    constructor(baseUrl = '/api') {
        this.baseUrl = baseUrl;
        this.requestInterceptors = [];
        this.responseInterceptors = [];
    }
    
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };
        
        if (config.body && typeof config.body === 'object') {
            config.body = JSON.stringify(config.body);
        }
        
        try {
            const response = await fetch(url, config);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.message || `HTTP ${response.status}`);
            }
            
            return data;
        } catch (error) {
            console.error(`API request failed: ${endpoint}`, error);
            throw error;
        }
    }
    
    get(endpoint) {
        return this.request(endpoint, { method: 'GET' });
    }
    
    post(endpoint, body) {
        return this.request(endpoint, { method: 'POST', body });
    }
    
    put(endpoint, body) {
        return this.request(endpoint, { method: 'PUT', body });
    }
    
    delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    }
}

class GameState extends EventEmitter {
    constructor() {
        super();
        this.board = [];
        this.turn = 'white';
        this.isCheckmate = false;
        this.isStalemate = false;
        this.isCheck = false;
        this.evaluation = 0;
        this.moveCount = 0;
        this.selectedSquare = null;
        this.validMoves = [];
        this.moveHistory = [];
    }
    
    updateBoard(boardData) {
        const changed = JSON.stringify(this.board) !== JSON.stringify(boardData.board);
        
        Object.assign(this, boardData);
        
        if (changed) {
            this.emit('boardChanged', this);
        }
        this.emit('stateChanged', this);
    }
    
    selectSquare(row, col) {
        this.selectedSquare = { row, col };
        this.emit('squareSelected', { row, col });
    }
    
    clearSelection() {
        this.selectedSquare = null;
        this.validMoves = [];
        this.emit('selectionCleared');
    }
    
    setValidMoves(moves) {
        this.validMoves = moves;
        this.emit('validMovesChanged', moves);
    }
    
    addMoveToHistory(move) {
        this.moveHistory.push(move);
        this.emit('moveAdded', move);
    }
    
    reset() {
        this.board = [];
        this.turn = 'white';
        this.isCheckmate = false;
        this.isStalemate = false;
        this.isCheck = false;
        this.evaluation = 0;
        this.moveCount = 0;
        this.selectedSquare = null;
        this.validMoves = [];
        this.moveHistory = [];
        this.emit('gameReset');
    }
}

class ChessBoard {
    constructor(container, gameState, api) {
        this.container = container;
        this.gameState = gameState;
        this.api = api;
        this.squares = new Map();
        
        this.pieceSymbols = {
            'P': '♙', 'T': '♖', 'C': '♘', 'F': '♗', 'Q': '♕', 'K': '♔',
            'p': '♟', 't': '♜', 'c': '♞', 'f': '♝', 'q': '♛', 'k': '♚'
        };
        
        this.init();
        this.bindEvents();
    }
    
    init() {
        this.container.className = 'chess-board';
        this.container.innerHTML = '';
        
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const square = document.createElement('div');
                square.className = `square ${(row + col) % 2 === 0 ? 'light' : 'dark'}`;
                square.dataset.row = row;
                square.dataset.col = col;
                
                square.addEventListener('click', () => this.handleSquareClick(row, col));
                
                this.container.appendChild(square);
                this.squares.set(`${row}-${col}`, square);
            }
        }
    }
    
    bindEvents() {
        this.gameState.on('boardChanged', () => this.updatePieces());
        this.gameState.on('squareSelected', ({ row, col }) => this.highlightSquare(row, col));
        this.gameState.on('selectionCleared', () => this.clearHighlights());
        this.gameState.on('validMovesChanged', (moves) => this.highlightValidMoves(moves));
    }
    
    async handleSquareClick(row, col) {
        try {
            if (!this.gameState.selectedSquare) {
                await this.selectSquare(row, col);
            } else {
                const { row: fromRow, col: fromCol } = this.gameState.selectedSquare;
                
                if (fromRow === row && fromCol === col) {
                    this.gameState.clearSelection();
                } else if (this.isValidMove(row, col)) {
                    await this.makeMove(fromRow, fromCol, row, col);
                } else {
                    await this.selectSquare(row, col);
                }
            }
        } catch (error) {
            console.error('Square click error:', error);
            this.gameState.clearSelection();
        }
    }
    
    async selectSquare(row, col) {
        const piece = this.gameState.board[row]?.[col];
        if (!piece || piece === ' ') return;
        
        const pieceColor = piece === piece.toUpperCase() ? 'white' : 'black';
        if (pieceColor !== this.gameState.turn) return;
        
        const response = await this.api.post('/valid-moves', { row, col });
        
        this.gameState.selectSquare(row, col);
        this.gameState.setValidMoves(response.data.valid_moves);
    }
    
    async makeMove(fromRow, fromCol, toRow, toCol) {
        const response = await this.api.post('/move', {
            from_row: fromRow,
            from_col: fromCol,
            to_row: toRow,
            to_col: toCol
        });
        
        this.gameState.updateBoard(response.data);
        this.gameState.clearSelection();
        
        this.addMoveToHistory(fromRow, fromCol, toRow, toCol);
        
        if (response.data.isCheckmate) {
            this.showGameOver('Checkmate!', response.data.turn === 'white' ? 'Black wins!' : 'White wins!');
        } else if (response.data.isStalemate) {
            this.showGameOver('Stalemate!', 'Draw!');
        }
    }
    
    isValidMove(row, col) {
        return this.gameState.validMoves.some(move => move.row === row && move.col === col);
    }
    
    updatePieces() {
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const square = this.squares.get(`${row}-${col}`);
                const piece = this.gameState.board[row]?.[col] || ' ';
                
                square.textContent = this.pieceSymbols[piece] || '';
                
                if (this.gameState.isCheck && piece.toLowerCase() === 'k') {
                    const pieceColor = piece === piece.toUpperCase() ? 'white' : 'black';
                    if (pieceColor === this.gameState.turn) {
                        square.classList.add('check');
                    }
                } else {
                    square.classList.remove('check');
                }
            }
        }
    }
    
    highlightSquare(row, col) {
        this.clearHighlights();
        const square = this.squares.get(`${row}-${col}`);
        square.classList.add('selected');
    }
    
    highlightValidMoves(moves) {
        moves.forEach(({ row, col }) => {
            const square = this.squares.get(`${row}-${col}`);
            square.classList.add('valid-move');
        });
    }
    
    clearHighlights() {
        this.squares.forEach(square => {
            square.classList.remove('selected', 'valid-move');
        });
    }
    
    addMoveToHistory(fromRow, fromCol, toRow, toCol) {
        const files = 'abcdefgh';
        const piece = this.gameState.board[toRow][toCol];
        const fromSquare = `${files[fromCol]}${8 - fromRow}`;
        const toSquare = `${files[toCol]}${8 - toRow}`;
        
        const move = {
            piece: this.pieceSymbols[piece] || piece,
            from: fromSquare,
            to: toSquare,
            notation: `${fromSquare}→${toSquare}`,
            evaluation: this.gameState.evaluation
        };
        
        this.gameState.addMoveToHistory(move);
    }
    
    showGameOver(title, message) {
        setTimeout(() => {
            alert(`${title} ${message}`);
        }, 100);
    }
}

class GameControls {
    constructor(container, gameState, api) {
        this.container = container;
        this.gameState = gameState;
        this.api = api;
        
        this.init();
        this.bindEvents();
    }
    
    init() {
        this.container.innerHTML = `
            <div class="controls-section">
                <button id="reset-btn" class="btn btn-primary">New Game</button>
                <button id="ai-move-btn" class="btn btn-secondary">AI Move</button>
            </div>
            
            <div class="game-info">
                <div id="turn-display" class="info-item">Turn: White</div>
                <div id="evaluation-display" class="info-item">Evaluation: 0.00</div>
                <div id="status-display" class="info-item">Ready</div>
            </div>
            
            <div class="move-history">
                <h3>Move History</h3>
                <div id="move-list" class="move-list"></div>
            </div>
        `;
        
        this.elements = {
            resetBtn: this.container.querySelector('#reset-btn'),
            aiMoveBtn: this.container.querySelector('#ai-move-btn'),
            turnDisplay: this.container.querySelector('#turn-display'),
            evaluationDisplay: this.container.querySelector('#evaluation-display'),
            statusDisplay: this.container.querySelector('#status-display'),
            moveList: this.container.querySelector('#move-list')
        };
        
        this.elements.resetBtn.addEventListener('click', () => this.resetGame());
        this.elements.aiMoveBtn.addEventListener('click', () => this.requestAiMove());
    }
    
    bindEvents() {
        this.gameState.on('stateChanged', () => this.updateDisplay());
        this.gameState.on('moveAdded', (move) => this.addMoveToList(move));
        this.gameState.on('gameReset', () => this.clearMoveHistory());
    }
    
    updateDisplay() {
        this.elements.turnDisplay.textContent = `Turn: ${this.gameState.turn.charAt(0).toUpperCase() + this.gameState.turn.slice(1)}`;
        this.elements.evaluationDisplay.textContent = `Evaluation: ${this.gameState.evaluation.toFixed(2)}`;
        
        let status = 'Playing';
        if (this.gameState.isCheckmate) {
            status = 'Checkmate';
        } else if (this.gameState.isStalemate) {
            status = 'Stalemate';
        } else if (this.gameState.isCheck) {
            status = 'Check';
        }
        
        this.elements.statusDisplay.textContent = `Status: ${status}`;
        
        const gameOver = this.gameState.isCheckmate || this.gameState.isStalemate;
        this.elements.aiMoveBtn.disabled = gameOver;
    }
    
    async resetGame() {
        try {
            await this.api.post('/reset');
            this.gameState.reset();
            
            const response = await this.api.get('/board');
            this.gameState.updateBoard(response.data);
        } catch (error) {
            console.error('Reset game error:', error);
        }
    }
    
    async requestAiMove() {
        if (this.gameState.isCheckmate || this.gameState.isStalemate) return;
        
        try {
            this.elements.aiMoveBtn.disabled = true;
            this.elements.aiMoveBtn.textContent = 'Thinking...';
            
            const response = await this.api.post('/ai-move');
            this.gameState.updateBoard(response.data);
            
            const { move } = response.data;
            this.addMoveToHistory(move.from_row, move.from_col, move.to_row, move.to_col, response.data.evaluation);
            
        } catch (error) {
            console.error('AI move error:', error);
        } finally {
            this.elements.aiMoveBtn.disabled = false;
            this.elements.aiMoveBtn.textContent = 'AI Move';
        }
    }
    
    addMoveToHistory(fromRow, fromCol, toRow, toCol, evaluation) {
        const files = 'abcdefgh';
        const piece = this.gameState.board[toRow][toCol];
        const fromSquare = `${files[fromCol]}${8 - fromRow}`;
        const toSquare = `${files[toCol]}${8 - toRow}`;
        
        const move = {
            piece,
            from: fromSquare,
            to: toSquare,
            notation: `${fromSquare}→${toSquare}`,
            evaluation: evaluation || 0
        };
        
        this.gameState.addMoveToHistory(move);
    }
    
    addMoveToList(move) {
        const moveElement = document.createElement('div');
        moveElement.className = 'move-item';
        moveElement.innerHTML = `
            <span class="move-notation">${move.notation}</span>
            <span class="move-eval">${move.evaluation.toFixed(2)}</span>
        `;
        
        this.elements.moveList.appendChild(moveElement);
        this.elements.moveList.scrollTop = this.elements.moveList.scrollHeight;
    }
    
    clearMoveHistory() {
        this.elements.moveList.innerHTML = '';
    }
}

class TrainingPanel {
    constructor(container, api, socketManager) {
        this.container = container;
        this.api = api;
        this.socketManager = socketManager;
        this.isTraining = false;
        
        this.init();
        this.bindEvents();
    }
    
    init() {
        this.container.innerHTML = `
            <div class="training-controls">
                <button id="start-training-btn" class="btn btn-primary">Start Training</button>
                <button id="stop-training-btn" class="btn btn-danger" disabled>Stop Training</button>
                
                <div class="speed-control">
                    <label for="speed-slider">Speed:</label>
                    <input type="range" id="speed-slider" min="0.01" max="2" step="0.01" value="0.1">
                    <span id="speed-value">0.1</span>
                </div>
            </div>
            
            <div class="training-stats">
                <h3>Training Statistics</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <span class="stat-label">Games:</span>
                        <span id="games-played" class="stat-value">0</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Model 1 Wins:</span>
                        <span id="model1-wins" class="stat-value">0</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Model 2 Wins:</span>
                        <span id="model2-wins" class="stat-value">0</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Draws:</span>
                        <span id="draws" class="stat-value">0</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Win Rate:</span>
                        <span id="win-rate" class="stat-value">0%</span>
                    </div>
                </div>
            </div>
            
            <div class="training-preview">
                <h3>Live Training</h3>
                <div id="preview-board" class="preview-board"></div>
                <div id="last-move" class="last-move">No moves yet</div>
            </div>
        `;
        
        this.elements = {
            startBtn: this.container.querySelector('#start-training-btn'),
            stopBtn: this.container.querySelector('#stop-training-btn'),
            speedSlider: this.container.querySelector('#speed-slider'),
            speedValue: this.container.querySelector('#speed-value'),
            gamesPlayed: this.container.querySelector('#games-played'),
            model1Wins: this.container.querySelector('#model1-wins'),
            model2Wins: this.container.querySelector('#model2-wins'),
            draws: this.container.querySelector('#draws'),
            winRate: this.container.querySelector('#win-rate'),
            previewBoard: this.container.querySelector('#preview-board'),
            lastMove: this.container.querySelector('#last-move')
        };
        
        this.elements.startBtn.addEventListener('click', () => this.startTraining());
        this.elements.stopBtn.addEventListener('click', () => this.stopTraining());
        this.elements.speedSlider.addEventListener('input', () => this.updateSpeed());
        
        this.initPreviewBoard();
    }
    
    initPreviewBoard() {
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const square = document.createElement('div');
                square.className = `preview-square ${(row + col) % 2 === 0 ? 'light' : 'dark'}`;
                this.elements.previewBoard.appendChild(square);
            }
        }
    }
    
    bindEvents() {
        this.socketManager.on('training_update', (data) => this.updateTrainingStats(data));
    }
    
    async startTraining() {
        try {
            await this.api.post('/training/start');
            this.isTraining = true;
            this.updateTrainingButtons();
        } catch (error) {
            console.error('Start training error:', error);
        }
    }
    
    async stopTraining() {
        try {
            await this.api.post('/training/stop');
            this.isTraining = false;
            this.updateTrainingButtons();
        } catch (error) {
            console.error('Stop training error:', error);
        }
    }
    
    async updateSpeed() {
        const speed = parseFloat(this.elements.speedSlider.value);
        this.elements.speedValue.textContent = speed.toFixed(2);
        
        try {
            await this.api.post('/training/speed', { speed });
        } catch (error) {
            console.error('Update speed error:', error);
        }
    }
    
    updateTrainingButtons() {
        this.elements.startBtn.disabled = this.isTraining;
        this.elements.stopBtn.disabled = !this.isTraining;
    }
    
    updateTrainingStats(data) {
        this.elements.gamesPlayed.textContent = data.games_played;
        this.elements.model1Wins.textContent = data.model1_wins;
        this.elements.model2Wins.textContent = data.model2_wins;
        this.elements.draws.textContent = data.draws;
        
        const winRate = data.games_played > 0 ? (data.model1_wins / data.games_played * 100).toFixed(1) : 0;
        this.elements.winRate.textContent = `${winRate}%`;
        
        if (data.last_move) {
            this.updatePreviewMove(data.last_move);
        }
    }
    
    updatePreviewMove(move) {
        const files = 'abcdefgh';
        const fromSquare = `${files[move.from_col]}${8 - move.from_row}`;
        const toSquare = `${files[move.to_col]}${8 - move.to_row}`;
        
        this.elements.lastMove.textContent = `${fromSquare} → ${toSquare} (${move.evaluation.toFixed(2)})`;
    }
}

class SocketManager extends EventEmitter {
    constructor() {
        super();
        this.socket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
    }
    
    connect() {
        try {
            this.socket = io();
            
            this.socket.on('connect', () => {
                console.log('Connected to server');
                this.reconnectAttempts = 0;
                this.emit('connected');
            });
            
            this.socket.on('disconnect', () => {
                console.log('Disconnected from server');
                this.emit('disconnected');
                this.attemptReconnect();
            });
            
            this.socket.on('board_update', (data) => this.emit('board_update', data));
            this.socket.on('training_update', (data) => this.emit('training_update', data));
            this.socket.on('ai_move', (data) => this.emit('ai_move', data));
            this.socket.on('game_reset', (data) => this.emit('game_reset', data));
            
        } catch (error) {
            console.error('Socket connection error:', error);
        }
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            setTimeout(() => {
                console.log(`Reconnection attempt ${this.reconnectAttempts}`);
                this.connect();
            }, this.reconnectDelay * this.reconnectAttempts);
        }
    }
    
    disconnect() {
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
        }
    }
}

class ChessApp {
    constructor() {
        this.api = new APIClient();
        this.gameState = new GameState();
        this.socketManager = new SocketManager();
        
        this.init();
    }
    
    async init() {
        this.initComponents();
        this.bindGlobalEvents();
        await this.loadInitialState();
        this.socketManager.connect();
    }
    
    initComponents() {
        this.chessBoard = new ChessBoard(
            document.getElementById('chess-board'),
            this.gameState,
            this.api
        );
        
        this.gameControls = new GameControls(
            document.getElementById('game-controls'),
            this.gameState,
            this.api
        );
        
        this.trainingPanel = new TrainingPanel(
            document.getElementById('training-panel'),
            this.api,
            this.socketManager
        );
    }
    
    bindGlobalEvents() {
        this.socketManager.on('board_update', (data) => {
            this.gameState.updateBoard(data);
        });
        
        this.socketManager.on('game_reset', (data) => {
            this.gameState.updateBoard(data);
        });
        
        window.addEventListener('beforeunload', () => {
            this.socketManager.disconnect();
        });
        
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.gameState.clearSelection();
            }
        });
    }
    
    async loadInitialState() {
        try {
            const response = await this.api.get('/board');
            this.gameState.updateBoard(response.data);
        } catch (error) {
            console.error('Failed to load initial state:', error);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.chessApp = new ChessApp();
});
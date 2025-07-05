class StateManager extends EventEmitter {
    constructor() {
        super();
        this.state = {
            game: {
                board: [],
                turn: 'white',
                isCheckmate: false,
                isStalemate: false,
                isCheck: false,
                evaluation: 0,
                moveCount: 0
            },
            neural: {
                isVisualizationActive: false,
                config: {
                    detail_level: 2,
                    update_frequency: 0.5,
                    show_connections: true,
                    animation_speed: 1.0,
                    quality: 'medium',
                    auto_rotate: false
                },
                structure: null,
                activations: null,
                performance: {
                    fps: 60,
                    updateRate: 0,
                    networkLatency: 0
                }
            },
            training: {
                isActive: false,
                stats: {
                    games_played: 0,
                    model1_wins: 0,
                    model2_wins: 0,
                    draws: 0
                }
            },
            ui: {
                activeTab: 'game',
                selectedSquare: null,
                validMoves: [],
                isLoading: false
            }
        };
        
        this.subscribers = new Map();
        this.middleware = [];
    }

    addMiddleware(fn) {
        this.middleware.push(fn);
    }

    subscribe(path, callback) {
        if (!this.subscribers.has(path)) {
            this.subscribers.set(path, new Set());
        }
        this.subscribers.get(path).add(callback);
        
        return () => {
            const callbacks = this.subscribers.get(path);
            if (callbacks) {
                callbacks.delete(callback);
            }
        };
    }

    getState(path = null) {
        if (!path) return this.state;
        return path.split('.').reduce((obj, key) => obj?.[key], this.state);
    }

    setState(path, value, silent = false) {
        const keys = path.split('.');
        const lastKey = keys.pop();
        const target = keys.reduce((obj, key) => {
            if (!obj[key]) obj[key] = {};
            return obj[key];
        }, this.state);
        
        const oldValue = target[lastKey];
        target[lastKey] = value;
        
        this.middleware.forEach(middleware => {
            middleware(path, value, oldValue);
        });
        
        if (!silent) {
            this.notifySubscribers(path, value, oldValue);
        }
    }

    notifySubscribers(path, value, oldValue) {
        const pathParts = path.split('.');
        
        for (let i = 0; i <= pathParts.length; i++) {
            const currentPath = pathParts.slice(0, i).join('.');
            const callbacks = this.subscribers.get(currentPath || '*');
            
            if (callbacks) {
                callbacks.forEach(callback => {
                    try {
                        callback(value, oldValue, path);
                    } catch (error) {
                        console.error('State subscriber error:', error);
                    }
                });
            }
        }
    }
}

class NetworkPerformanceMonitor {
    constructor() {
        this.metrics = {
            requestTimes: [],
            errorCount: 0,
            successCount: 0,
            bytesTransferred: 0,
            lastUpdate: Date.now()
        };
        
        this.windowSize = 100;
    }

    recordRequest(duration, success, bytes = 0) {
        this.metrics.requestTimes.push({
            duration,
            timestamp: Date.now()
        });
        
        if (this.metrics.requestTimes.length > this.windowSize) {
            this.metrics.requestTimes.shift();
        }
        
        if (success) {
            this.metrics.successCount++;
        } else {
            this.metrics.errorCount++;
        }
        
        this.metrics.bytesTransferred += bytes;
        this.metrics.lastUpdate = Date.now();
    }

    getAverageLatency() {
        if (this.metrics.requestTimes.length === 0) return 0;
        
        const recent = this.metrics.requestTimes.filter(
            r => Date.now() - r.timestamp < 30000
        );
        
        return recent.reduce((sum, r) => sum + r.duration, 0) / recent.length;
    }

    getSuccessRate() {
        const total = this.metrics.successCount + this.metrics.errorCount;
        return total === 0 ? 100 : (this.metrics.successCount / total) * 100;
    }

    getBandwidthUsage() {
        const timeWindow = 60000;
        const now = Date.now();
        const cutoff = now - timeWindow;
        
        const recentBytes = this.metrics.requestTimes
            .filter(r => r.timestamp > cutoff)
            .length * (this.metrics.bytesTransferred / this.metrics.requestTimes.length);
        
        return recentBytes / (timeWindow / 1000);
    }
}

class OptimizedAPIClient extends APIClient {
    constructor(baseUrl = '/api') {
        super(baseUrl);
        this.performanceMonitor = new NetworkPerformanceMonitor();
        this.requestCache = new Map();
        this.requestQueue = [];
        this.isProcessingQueue = false;
        this.retryConfig = {
            maxRetries: 3,
            baseDelay: 1000,
            maxDelay: 10000
        };
    }

    async request(endpoint, options = {}) {
        const startTime = Date.now();
        const cacheKey = `${endpoint}_${JSON.stringify(options)}`;
        
        if (options.method === 'GET' && this.requestCache.has(cacheKey)) {
            const cached = this.requestCache.get(cacheKey);
            if (Date.now() - cached.timestamp < 30000) {
                return cached.data;
            }
        }
        
        try {
            const response = await this.retryRequest(endpoint, options);
            const duration = Date.now() - startTime;
            
            this.performanceMonitor.recordRequest(duration, true, 
                JSON.stringify(response).length);
            
            if (options.method === 'GET') {
                this.requestCache.set(cacheKey, {
                    data: response,
                    timestamp: Date.now()
                });
            }
            
            return response;
            
        } catch (error) {
            const duration = Date.now() - startTime;
            this.performanceMonitor.recordRequest(duration, false);
            throw error;
        }
    }

    async retryRequest(endpoint, options, attempt = 1) {
        try {
            return await super.request(endpoint, options);
        } catch (error) {
            if (attempt >= this.retryConfig.maxRetries) {
                throw error;
            }
            
            const delay = Math.min(
                this.retryConfig.baseDelay * Math.pow(2, attempt - 1),
                this.retryConfig.maxDelay
            );
            
            await new Promise(resolve => setTimeout(resolve, delay));
            return this.retryRequest(endpoint, options, attempt + 1);
        }
    }

    getPerformanceMetrics() {
        return {
            averageLatency: this.performanceMonitor.getAverageLatency(),
            successRate: this.performanceMonitor.getSuccessRate(),
            bandwidthUsage: this.performanceMonitor.getBandwidthUsage()
        };
    }

    clearCache() {
        this.requestCache.clear();
    }
}

class EnhancedSocketManager extends SocketManager {
    constructor() {
        super();
        this.messageBuffer = [];
        this.bufferFlushInterval = 50;
        this.compressionEnabled = true;
        this.messageQueue = [];
        this.isConnected = false;
        this.heartbeatInterval = null;
        this.heartbeatTimeout = 30000;
    }

    connect() {
        try {
            this.socket = io({
                transports: ['websocket', 'polling'],
                upgrade: true,
                rememberUpgrade: true,
                compression: this.compressionEnabled
            });
            
            this.socket.on('connect', () => {
                this.isConnected = true;
                this.startHeartbeat();
                this.flushMessageQueue();
                this.emit('connected');
            });
            
            this.socket.on('disconnect', () => {
                this.isConnected = false;
                this.stopHeartbeat();
                this.emit('disconnected');
                this.attemptReconnect();
            });
            
            this.socket.on('neural_visualization_update', (data) => {
                this.bufferMessage('neural_visualization_update', data);
            });
            
            this.socket.on('board_update', (data) => {
                this.emit('board_update', data);
            });
            
            this.socket.on('training_update', (data) => {
                this.emit('training_update', data);
            });
            
        } catch (error) {
            console.error('Socket connection error:', error);
        }
    }

    bufferMessage(event, data) {
        this.messageBuffer.push({ event, data, timestamp: Date.now() });
        
        if (!this.bufferFlushTimer) {
            this.bufferFlushTimer = setTimeout(() => {
                this.flushMessageBuffer();
            }, this.bufferFlushInterval);
        }
    }

    flushMessageBuffer() {
        const messages = this.messageBuffer.splice(0);
        const groupedMessages = new Map();
        
        messages.forEach(({ event, data }) => {
            if (!groupedMessages.has(event)) {
                groupedMessages.set(event, []);
            }
            groupedMessages.get(event).push(data);
        });
        
        groupedMessages.forEach((dataArray, event) => {
            if (event === 'neural_visualization_update' && dataArray.length > 1) {
                this.emit(event, dataArray[dataArray.length - 1]);
            } else {
                dataArray.forEach(data => this.emit(event, data));
            }
        });
        
        this.bufferFlushTimer = null;
    }

    startHeartbeat() {
        this.heartbeatInterval = setInterval(() => {
            if (this.isConnected) {
                this.socket.emit('ping', Date.now());
            }
        }, this.heartbeatTimeout);
    }

    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }

    send(event, data) {
        if (this.isConnected) {
            this.socket.emit(event, data);
        } else {
            this.messageQueue.push({ event, data });
        }
    }

    flushMessageQueue() {
        while (this.messageQueue.length > 0) {
            const { event, data } = this.messageQueue.shift();
            this.socket.emit(event, data);
        }
    }
}

class Neural3DInterface {
    constructor(container, stateManager, controller) {
        this.container = container;
        this.state = stateManager;
        this.controller = controller;
        this.elements = {};
        
        this.init();
    }

    init() {
        this.container.innerHTML = `
            <div class="neural-3d-container">
                <div class="neural-3d-viewport" id="neural-viewport"></div>
                
                <div class="neural-controls-overlay">
                    <div class="control-panel">
                        <div class="control-group">
                            <label>Quality</label>
                            <select id="quality-select">
                                <option value="low">Low</option>
                                <option value="medium" selected>Medium</option>
                                <option value="high">High</option>
                                <option value="auto">Auto</option>
                            </select>
                        </div>
                        
                        <div class="control-group">
                            <label>Animation Speed</label>
                            <input type="range" id="animation-speed" min="0.1" max="3" step="0.1" value="1">
                            <span id="speed-value">1.0x</span>
                        </div>
                        
                        <div class="control-group">
                            <label>Update Rate</label>
                            <input type="range" id="update-rate" min="0.1" max="2" step="0.1" value="0.5">
                            <span id="rate-value">0.5s</span>
                        </div>
                        
                        <div class="control-group checkbox">
                            <label>
                                <input type="checkbox" id="auto-rotate">
                                Auto Rotate
                            </label>
                        </div>
                        
                        <div class="control-group checkbox">
                            <label>
                                <input type="checkbox" id="show-connections" checked>
                                Show Connections
                            </label>
                        </div>
                    </div>
                    
                    <div class="performance-panel">
                        <h4>Performance</h4>
                        <div class="metric">
                            <span>FPS:</span>
                            <span id="fps-value">60</span>
                        </div>
                        <div class="metric">
                            <span>Latency:</span>
                            <span id="latency-value">0ms</span>
                        </div>
                        <div class="metric">
                            <span>Active Neurons:</span>
                            <span id="active-neurons">0</span>
                        </div>
                    </div>
                    
                    <div class="layer-info-panel">
                        <h4>Layer Activity</h4>
                        <div id="layer-activity-list"></div>
                    </div>
                </div>
                
                <div class="neural-info-panel">
                    <div class="info-header">
                        <h3>Neural Network Activity</h3>
                        <button id="toggle-info" class="btn-icon">📊</button>
                    </div>
                    <div class="info-content">
                        <canvas id="activity-chart" width="300" height="150"></canvas>
                        <div class="activity-summary">
                            <div class="summary-item">
                                <span class="label">Total Activations:</span>
                                <span id="total-activations">0</span>
                            </div>
                            <div class="summary-item">
                                <span class="label">Max Activation:</span>
                                <span id="max-activation">0.0</span>
                            </div>
                            <div class="summary-item">
                                <span class="label">Network Load:</span>
                                <span id="network-load">0%</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        this.cacheElements();
        this.bindEvents();
        this.initActivityChart();
    }

    cacheElements() {
        this.elements = {
            viewport: this.container.querySelector('#neural-viewport'),
            qualitySelect: this.container.querySelector('#quality-select'),
            animationSpeed: this.container.querySelector('#animation-speed'),
            speedValue: this.container.querySelector('#speed-value'),
            updateRate: this.container.querySelector('#update-rate'),
            rateValue: this.container.querySelector('#rate-value'),
            autoRotate: this.container.querySelector('#auto-rotate'),
            showConnections: this.container.querySelector('#show-connections'),
            fpsValue: this.container.querySelector('#fps-value'),
            latencyValue: this.container.querySelector('#latency-value'),
            activeNeurons: this.container.querySelector('#active-neurons'),
            layerActivityList: this.container.querySelector('#layer-activity-list'),
            activityChart: this.container.querySelector('#activity-chart'),
            totalActivations: this.container.querySelector('#total-activations'),
            maxActivation: this.container.querySelector('#max-activation'),
            networkLoad: this.container.querySelector('#network-load'),
            toggleInfo: this.container.querySelector('#toggle-info')
        };
    }

    bindEvents() {
        this.elements.qualitySelect.addEventListener('change', (e) => {
            this.controller.setQuality(e.target.value);
            this.state.setState('neural.config.quality', e.target.value);
        });
        
        this.elements.animationSpeed.addEventListener('input', (e) => {
            const speed = parseFloat(e.target.value);
            this.elements.speedValue.textContent = `${speed.toFixed(1)}x`;
            this.updateConfig({ animation_speed: speed });
        });
        
        this.elements.updateRate.addEventListener('input', (e) => {
            const rate = parseFloat(e.target.value);
            this.elements.rateValue.textContent = `${rate.toFixed(1)}s`;
            this.updateConfig({ update_frequency: rate });
        });
        
        this.elements.autoRotate.addEventListener('change', (e) => {
            this.controller.setAutoRotate(e.target.checked);
            this.state.setState('neural.config.auto_rotate', e.target.checked);
        });
        
        this.elements.showConnections.addEventListener('change', (e) => {
            this.updateConfig({ show_connections: e.target.checked });
        });
        
        this.elements.toggleInfo.addEventListener('click', () => {
            this.container.querySelector('.neural-info-panel').classList.toggle('collapsed');
        });
        
        this.state.subscribe('neural.performance', (performance) => {
            this.updatePerformanceDisplay(performance);
        });
        
        this.state.subscribe('neural.activations', (activations) => {
            this.updateActivityDisplay(activations);
        });
    }

    async updateConfig(config) {
        const currentConfig = this.state.getState('neural.config');
        const newConfig = { ...currentConfig, ...config };
        
        this.state.setState('neural.config', newConfig);
        await this.controller.updateConfig(newConfig);
    }

    updatePerformanceDisplay(performance) {
        this.elements.fpsValue.textContent = Math.round(performance.fps || 60);
        this.elements.latencyValue.textContent = `${Math.round(performance.networkLatency || 0)}ms`;
        
        const load = ((performance.fps || 60) / 60) * 100;
        this.elements.networkLoad.textContent = `${Math.round(load)}%`;
    }

    updateActivityDisplay(activationData) {
        if (!activationData || !activationData.summary) return;
        
        const summary = activationData.summary;
        
        this.elements.totalActivations.textContent = summary.total_active_neurons || 0;
        this.elements.maxActivation.textContent = (summary.max_activation || 0).toFixed(2);
        this.elements.activeNeurons.textContent = summary.total_active_neurons || 0;
        
        this.updateLayerActivity(summary.layer_activity || {});
        this.updateActivityChart(summary);
    }

    updateLayerActivity(layerActivity) {
        const list = this.elements.layerActivityList;
        list.innerHTML = '';
        
        Object.entries(layerActivity).forEach(([layerName, activity]) => {
            const item = document.createElement('div');
            item.className = 'layer-activity-item';
            
            const ratio = activity.activity_ratio || 0;
            const percentage = (ratio * 100).toFixed(1);
            
            item.innerHTML = `
                <div class="layer-name">${layerName}</div>
                <div class="layer-bar">
                    <div class="layer-fill" style="width: ${percentage}%"></div>
                </div>
                <div class="layer-value">${percentage}%</div>
            `;
            
            list.appendChild(item);
        });
    }

    initActivityChart() {
        this.chartCtx = this.elements.activityChart.getContext('2d');
        this.chartData = {
            labels: [],
            data: []
        };
        this.maxDataPoints = 50;
    }

    updateActivityChart(summary) {
        const now = new Date().toLocaleTimeString();
        const activity = summary.average_activation || 0;
        
        this.chartData.labels.push(now);
        this.chartData.data.push(activity);
        
        if (this.chartData.labels.length > this.maxDataPoints) {
            this.chartData.labels.shift();
            this.chartData.data.shift();
        }
        
        this.drawChart();
    }

    drawChart() {
        const ctx = this.chartCtx;
        const canvas = this.elements.activityChart;
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        if (this.chartData.data.length < 2) return;
        
        const maxValue = Math.max(...this.chartData.data, 1);
        const stepX = width / (this.chartData.data.length - 1);
        
        ctx.strokeStyle = '#667eea';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        this.chartData.data.forEach((value, index) => {
            const x = index * stepX;
            const y = height - (value / maxValue) * height;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        ctx.fillStyle = 'rgba(102, 126, 234, 0.2)';
        ctx.lineTo(width, height);
        ctx.lineTo(0, height);
        ctx.closePath();
        ctx.fill();
    }

    getViewport() {
        return this.elements.viewport;
    }
}

class EnhancedChessApp {
    constructor() {
        this.state = new StateManager();
        this.api = new OptimizedAPIClient();
        this.socket = new EnhancedSocketManager();
        
        this.components = {
            chessBoard: null,
            gameControls: null,
            trainingPanel: null,
            neural3D: null
        };
        
        this.neural3DController = null;
        this.performanceMonitor = null;
        
        this.setupMiddleware();
        this.init();
    }

    setupMiddleware() {
        this.state.addMiddleware((path, value, oldValue) => {
            if (path.startsWith('neural.config')) {
                this.throttledConfigUpdate();
            }
        });
    }

    throttledConfigUpdate = this.throttle(async () => {
        const config = this.state.getState('neural.config');
        if (this.neural3DController) {
            await this.neural3DController.updateConfig(config);
        }
    }, 500);

    throttle(func, delay) {
        let timeoutId;
        let lastExecTime = 0;
        
        return (...args) => {
            const currentTime = Date.now();
            
            if (currentTime - lastExecTime > delay) {
                func.apply(this, args);
                lastExecTime = currentTime;
            } else {
                clearTimeout(timeoutId);
                timeoutId = setTimeout(() => {
                    func.apply(this, args);
                    lastExecTime = Date.now();
                }, delay - (currentTime - lastExecTime));
            }
        };
    }

    async init() {
        this.showLoadingOverlay();
        
        try {
            await this.initializeComponents();
            await this.initializeNeural3D();
            this.bindGlobalEvents();
            await this.loadInitialState();
            this.socket.connect();
            
            this.startPerformanceMonitoring();
            
        } catch (error) {
            console.error('App initialization failed:', error);
            this.showError('Failed to initialize application');
        } finally {
            this.hideLoadingOverlay();
        }
    }

    async initializeComponents() {
        this.components.chessBoard = new ChessBoard(
            document.getElementById('chess-board'),
            this.state,
            this.api
        );
        
        this.components.gameControls = new GameControls(
            document.getElementById('game-controls'),
            this.state,
            this.api
        );
        
        this.components.trainingPanel = new TrainingPanel(
            document.getElementById('training-panel'),
            this.api,
            this.socket
        );
        
        const neural3DContainer = document.getElementById('neural-3d-container');
        if (neural3DContainer) {
            this.components.neural3D = new Neural3DInterface(
                neural3DContainer,
                this.state,
                null
            );
        }
    }

    async initializeNeural3D() {
        if (!this.components.neural3D) return;
        
        const viewport = this.components.neural3D.getViewport();
        
        try {
            const visualizer = new Neural3DVisualizer(viewport, {
                antialias: true,
                alpha: true,
                powerPreference: "high-performance"
            });
            
            this.neural3DController = new Neural3DController(
                visualizer,
                this.api,
                this.socket
            );
            
            this.components.neural3D.controller = this.neural3DController;
            
            await this.neural3DController.initialize();
            
            this.state.setState('neural.isVisualizationActive', true);
            
        } catch (error) {
            console.error('Failed to initialize 3D visualization:', error);
            this.showError('3D visualization not available');
        }
    }

    bindGlobalEvents() {
        this.socket.on('board_update', (data) => {
            this.state.setState('game', data, false);
        });
        
        this.socket.on('neural_visualization_update', (data) => {
            this.state.setState('neural.activations', data, false);
            
            if (data.performance_metrics) {
                this.state.setState('neural.performance', data.performance_metrics, false);
            }
        });
        
        this.socket.on('training_update', (data) => {
            this.state.setState('training.stats', data, false);
        });
        
        this.socket.on('connected', () => {
            this.updateConnectionStatus(true);
        });
        
        this.socket.on('disconnected', () => {
            this.updateConnectionStatus(false);
        });
        
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });
        
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pause();
            } else {
                this.resume();
            }
        });
        
        window.addEventListener('resize', this.throttle(() => {
            this.handleResize();
        }, 250));
    }

    async loadInitialState() {
        try {
            const boardResponse = await this.api.get('/board');
            this.state.setState('game', boardResponse.data);
            
        } catch (error) {
            console.error('Failed to load initial state:', error);
        }
    }

    startPerformanceMonitoring() {
        this.performanceMonitor = setInterval(() => {
            const apiMetrics = this.api.getPerformanceMetrics();
            
            this.state.setState('neural.performance.networkLatency', apiMetrics.averageLatency);
            this.state.setState('neural.performance.updateRate', apiMetrics.bandwidthUsage);
            
        }, 5000);
    }

    updateConnectionStatus(connected) {
        const indicator = document.getElementById('connection-status');
        if (indicator) {
            indicator.style.display = 'flex';
            indicator.querySelector('.status-indicator').className = 
                `status-indicator ${connected ? '' : 'disconnected'}`;
            indicator.querySelector('.status-text').textContent = 
                connected ? 'Connected' : 'Disconnected';
            
            if (connected) {
                setTimeout(() => {
                    indicator.style.display = 'none';
                }, 3000);
            }
        }
    }

    showLoadingOverlay() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.style.display = 'flex';
        }
    }

    hideLoadingOverlay() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    showNotification(message, type = 'info') {
        const container = document.getElementById('notification-container');
        if (!container) return;
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        container.appendChild(notification);
        
        setTimeout(() => notification.classList.add('show'), 100);
        
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                if (container.contains(notification)) {
                    container.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }

    handleResize() {
        if (this.neural3DController) {
            this.neural3DController.visualizer.handleResize();
        }
    }

    pause() {
        if (this.neural3DController) {
            this.neural3DController.pause();
        }
    }

    resume() {
        if (this.neural3DController) {
            this.neural3DController.resume();
        }
    }

    cleanup() {
        if (this.performanceMonitor) {
            clearInterval(this.performanceMonitor);
        }
        
        if (this.neural3DController) {
            this.neural3DController.dispose();
        }
        
        this.socket.disconnect();
        this.api.clearCache();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    if (typeof THREE !== 'undefined' && typeof io !== 'undefined') {
        window.chessApp = new EnhancedChessApp();
    } else {
        console.error('Required dependencies not loaded');
        document.body.innerHTML = `
            <div style="text-align: center; padding: 50px;">
                <h1>Missing Dependencies</h1>
                <p>Please ensure Three.js and Socket.IO are loaded.</p>
            </div>
        `;
    }
});
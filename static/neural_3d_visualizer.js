class Neural3DVisualizer {
    constructor(container, config = {}) {
        this.container = container;
        this.config = {
            antialias: true,
            alpha: true,
            powerPreference: "high-performance",
            precision: "mediump",
            ...config
        };
        
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        
        this.neurons = new Map();
        this.connections = new Map();
        this.layers = new Map();
        this.animationMixers = [];
        
        this.isInitialized = false;
        this.isAnimating = false;
        this.frameId = null;
        this.clock = new THREE.Clock();
        
        this.performance = {
            fps: 60,
            lastFrame: 0,
            frameCount: 0,
            adaptiveQuality: true
        };
        
        this.geometryCache = new Map();
        this.materialCache = new Map();
        this.updateQueue = [];
        this.batchUpdateTimer = null;
        
        this.initThreeJS();
        this.setupEventListeners();
    }

    initThreeJS() {
        const rect = this.container.getBoundingClientRect();
        
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a0a);
        this.scene.fog = new THREE.Fog(0x0a0a0a, 10, 50);
        
        this.camera = new THREE.PerspectiveCamera(75, rect.width / rect.height, 0.1, 1000);
        this.camera.position.set(15, 10, 15);
        
        this.renderer = new THREE.WebGLRenderer({
            antialias: this.config.antialias,
            alpha: this.config.alpha,
            powerPreference: this.config.powerPreference,
            precision: this.config.precision
        });
        
        this.renderer.setSize(rect.width, rect.height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.2;
        
        this.container.appendChild(this.renderer.domElement);
        
        this.setupLighting();
        this.setupControls();
        this.setupPostProcessing();
        
        this.isInitialized = true;
    }

    setupLighting() {
        const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(20, 20, 20);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.setScalar(2048);
        directionalLight.shadow.camera.near = 0.1;
        directionalLight.shadow.camera.far = 50;
        directionalLight.shadow.camera.left = directionalLight.shadow.camera.bottom = -25;
        directionalLight.shadow.camera.right = directionalLight.shadow.camera.top = 25;
        this.scene.add(directionalLight);
        
        const pointLight = new THREE.PointLight(0x667eea, 0.6, 30);
        pointLight.position.set(10, 5, 10);
        this.scene.add(pointLight);
        
        const hemisphereLight = new THREE.HemisphereLight(0x667eea, 0x764ba2, 0.3);
        this.scene.add(hemisphereLight);
    }

    setupControls() {
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = false;
        this.controls.minDistance = 5;
        this.controls.maxDistance = 100;
        this.controls.maxPolarAngle = Math.PI;
        this.controls.autoRotate = false;
        this.controls.autoRotateSpeed = 0.5;
    }

    setupPostProcessing() {
        this.composer = new THREE.EffectComposer(this.renderer);
        
        const renderPass = new THREE.RenderPass(this.scene, this.camera);
        this.composer.addPass(renderPass);
        
        const bloomPass = new THREE.UnrealBloomPass(
            new THREE.Vector2(this.renderer.domElement.width, this.renderer.domElement.height),
            0.5, 0.4, 0.85
        );
        this.composer.addPass(bloomPass);
        
        const fxaaPass = new THREE.ShaderPass(THREE.FXAAShader);
        fxaaPass.material.uniforms['resolution'].value.x = 1 / this.renderer.domElement.width;
        fxaaPass.material.uniforms['resolution'].value.y = 1 / this.renderer.domElement.height;
        this.composer.addPass(fxaaPass);
    }

    getOrCreateGeometry(type, params) {
        const key = `${type}_${JSON.stringify(params)}`;
        
        if (!this.geometryCache.has(key)) {
            let geometry;
            
            switch(type) {
                case 'sphere':
                    geometry = new THREE.SphereGeometry(params.radius || 0.1, params.widthSegments || 16, params.heightSegments || 12);
                    break;
                case 'cylinder':
                    geometry = new THREE.CylinderGeometry(params.radiusTop || 0.01, params.radiusBottom || 0.01, params.height || 1, params.radialSegments || 8);
                    break;
                case 'box':
                    geometry = new THREE.BoxGeometry(params.width || 1, params.height || 1, params.depth || 1);
                    break;
                default:
                    geometry = new THREE.SphereGeometry(0.1, 16, 12);
            }
            
            this.geometryCache.set(key, geometry);
        }
        
        return this.geometryCache.get(key);
    }

    getOrCreateMaterial(type, params) {
        const key = `${type}_${JSON.stringify(params)}`;
        
        if (!this.materialCache.has(key)) {
            let material;
            
            switch(type) {
                case 'neuron':
                    material = new THREE.MeshPhongMaterial({
                        color: new THREE.Color(params.r || 0.5, params.g || 0.5, params.b || 1.0),
                        transparent: true,
                        opacity: params.a || 0.8,
                        shininess: 100,
                        emissive: new THREE.Color(params.r || 0.5, params.g || 0.5, params.b || 1.0).multiplyScalar(0.1)
                    });
                    break;
                case 'connection':
                    material = new THREE.LineBasicMaterial({
                        color: new THREE.Color(params.r || 0.5, params.g || 0.5, params.b || 0.5),
                        transparent: true,
                        opacity: params.a || 0.3,
                        linewidth: params.width || 1
                    });
                    break;
                case 'layer':
                    material = new THREE.MeshBasicMaterial({
                        color: new THREE.Color(params.r || 0.2, params.g || 0.2, params.b || 0.5),
                        transparent: true,
                        opacity: 0.1,
                        side: THREE.DoubleSide
                    });
                    break;
                default:
                    material = new THREE.MeshBasicMaterial({color: 0x666666});
            }
            
            this.materialCache.set(key, material);
        }
        
        return this.materialCache.get(key);
    }

    createNeuron(neuronData, layerName) {
        const neuronId = `${layerName}_${neuronData.id}`;
        
        if (this.neurons.has(neuronId)) {
            return this.neurons.get(neuronId);
        }
        
        const geometry = this.getOrCreateGeometry('sphere', {radius: 0.15});
        const material = this.getOrCreateMaterial('neuron', {
            r: 0.3, g: 0.3, b: 0.8, a: 0.7
        });
        
        const neuron = new THREE.Mesh(geometry, material.clone());
        neuron.position.set(...neuronData.position);
        neuron.userData = {
            id: neuronId,
            layerName: layerName,
            originalScale: 1.0,
            activation: 0.0,
            targetActivation: 0.0
        };
        
        neuron.castShadow = true;
        neuron.receiveShadow = true;
        
        this.scene.add(neuron);
        this.neurons.set(neuronId, neuron);
        
        return neuron;
    }

    createConnection(fromNeuron, toNeuron, strength = 1.0) {
        const connectionId = `${fromNeuron.userData.id}_${toNeuron.userData.id}`;
        
        if (this.connections.has(connectionId)) {
            return this.connections.get(connectionId);
        }
        
        const points = [
            fromNeuron.position.clone(),
            toNeuron.position.clone()
        ];
        
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = this.getOrCreateMaterial('connection', {
            r: 0.4, g: 0.4, b: 0.8, a: 0.3, width: 1
        });
        
        const connection = new THREE.Line(geometry, material.clone());
        connection.userData = {
            id: connectionId,
            fromNeuron: fromNeuron,
            toNeuron: toNeuron,
            strength: strength,
            targetStrength: strength
        };
        
        this.scene.add(connection);
        this.connections.set(connectionId, connection);
        
        return connection;
    }

    createLayer(layerData) {
        if (this.layers.has(layerData.name)) {
            return this.layers.get(layerData.name);
        }
        
        const neurons = [];
        
        layerData.neurons.forEach(neuronData => {
            const neuron = this.createNeuron(neuronData, layerData.name);
            neurons.push(neuron);
        });
        
        const layer = {
            name: layerData.name,
            type: layerData.type,
            neurons: neurons,
            position: layerData.position
        };
        
        this.layers.set(layerData.name, layer);
        
        return layer;
    }

    updateNetworkStructure(networkData) {
        if (!networkData || !networkData.layers) return;
        
        this.clearScene();
        
        const layers = [];
        
        networkData.layers.forEach(layerData => {
            const layer = this.createLayer(layerData);
            layers.push(layer);
        });
        
        networkData.connections?.forEach(connectionData => {
            if (connectionData.from_layer < layers.length && connectionData.to_layer < layers.length) {
                const fromLayer = layers[connectionData.from_layer];
                const toLayer = layers[connectionData.to_layer];
                
                if (fromLayer.neurons.length > 0 && toLayer.neurons.length > 0) {
                    const maxConnections = Math.min(fromLayer.neurons.length, toLayer.neurons.length, 50);
                    
                    for (let i = 0; i < maxConnections; i += Math.max(1, Math.floor(fromLayer.neurons.length / maxConnections))) {
                        for (let j = 0; j < maxConnections; j += Math.max(1, Math.floor(toLayer.neurons.length / maxConnections))) {
                            if (fromLayer.neurons[i] && toLayer.neurons[j]) {
                                this.createConnection(fromLayer.neurons[i], toLayer.neurons[j], connectionData.strength || 1.0);
                            }
                        }
                    }
                }
            }
        });
    }

    updateActivations(activationData) {
        if (!activationData || !activationData.activations) return;
        
        this.queueUpdate(() => {
            Object.entries(activationData.activations).forEach(([layerName, activations]) => {
                const layer = this.layers.get(layerName);
                if (!layer) return;
                
                layer.neurons.forEach((neuron, index) => {
                    if (index < activations.length) {
                        neuron.userData.targetActivation = Math.max(0, Math.min(1, activations[index]));
                    }
                });
            });
            
            this.updateConnectionStrengths(activationData.connections || []);
            this.triggerAnimations(activationData.animation_targets || {});
        });
    }

    updateConnectionStrengths(connectionData) {
        connectionData.forEach(connData => {
            this.connections.forEach(connection => {
                if (connection.userData.fromNeuron && connection.userData.toNeuron) {
                    const fromActivation = connection.userData.fromNeuron.userData.targetActivation;
                    const toActivation = connection.userData.toNeuron.userData.targetActivation;
                    connection.userData.targetStrength = (fromActivation + toActivation) / 2.0;
                }
            });
        });
    }

    triggerAnimations(animationTargets) {
        if (animationTargets.neuron_pulses) {
            animationTargets.neuron_pulses.forEach(pulse => {
                const neuron = this.neurons.get(pulse.neuron_id);
                if (neuron) {
                    this.animateNeuronPulse(neuron, pulse.intensity, pulse.duration);
                }
            });
        }
        
        if (animationTargets.layer_waves) {
            animationTargets.layer_waves.forEach(wave => {
                const layer = this.layers.get(wave.layer_id);
                if (layer) {
                    this.animateLayerWave(layer, wave.intensity, wave.duration);
                }
            });
        }
        
        if (animationTargets.connection_flows) {
            animationTargets.connection_flows.forEach(flow => {
                this.animateConnectionFlow(flow);
            });
        }
    }

    animateNeuronPulse(neuron, intensity, duration) {
        const originalScale = neuron.userData.originalScale;
        const targetScale = originalScale * (1 + intensity * 0.5);
        
        const scaleUp = new THREE.Tween(neuron.scale)
            .to({x: targetScale, y: targetScale, z: targetScale}, duration * 0.3)
            .easing(THREE.Tween.Easing.Quadratic.Out);
            
        const scaleDown = new THREE.Tween(neuron.scale)
            .to({x: originalScale, y: originalScale, z: originalScale}, duration * 0.7)
            .easing(THREE.Tween.Easing.Quadratic.InOut);
            
        scaleUp.chain(scaleDown);
        scaleUp.start();
    }

    animateLayerWave(layer, intensity, duration) {
        layer.neurons.forEach((neuron, index) => {
            setTimeout(() => {
                this.animateNeuronPulse(neuron, intensity * 0.7, duration * 0.5);
            }, index * 20);
        });
    }

    animateConnectionFlow(flowData) {
        const connectionId = flowData.connection_id;
        const connection = this.connections.get(connectionId);
        
        if (!connection) return;
        
        for (let i = 0; i < flowData.particles; i++) {
            setTimeout(() => {
                this.createFlowParticle(connection, flowData.color, flowData.speed);
            }, i * 100);
        }
    }

    createFlowParticle(connection, color, speed) {
        const geometry = this.getOrCreateGeometry('sphere', {radius: 0.02});
        const material = new THREE.MeshBasicMaterial({
            color: new THREE.Color(color.r, color.g, color.b),
            transparent: true,
            opacity: 0.8
        });
        
        const particle = new THREE.Mesh(geometry, material);
        particle.position.copy(connection.userData.fromNeuron.position);
        
        this.scene.add(particle);
        
        const targetPosition = connection.userData.toNeuron.position.clone();
        const duration = 1000 / speed;
        
        const tween = new THREE.Tween(particle.position)
            .to(targetPosition, duration)
            .easing(THREE.Tween.Easing.Quadratic.InOut)
            .onComplete(() => {
                this.scene.remove(particle);
                geometry.dispose();
                material.dispose();
            });
            
        tween.start();
    }

    queueUpdate(updateFunction) {
        this.updateQueue.push(updateFunction);
        
        if (!this.batchUpdateTimer) {
            this.batchUpdateTimer = setTimeout(() => {
                this.processBatchUpdates();
            }, 16);
        }
    }

    processBatchUpdates() {
        const updates = this.updateQueue.splice(0);
        updates.forEach(update => update());
        this.batchUpdateTimer = null;
    }

    updateNeuronVisuals() {
        this.neurons.forEach(neuron => {
            const userData = neuron.userData;
            const activation = userData.activation;
            const targetActivation = userData.targetActivation;
            
            userData.activation += (targetActivation - activation) * 0.1;
            
            const intensity = Math.max(0, Math.min(1, userData.activation));
            const scale = 0.7 + intensity * 0.6;
            
            neuron.scale.setScalar(scale);
            
            if (neuron.material) {
                const baseColor = intensity < 0.5 
                    ? new THREE.Color(0.2 + intensity * 0.6, 0.2 + intensity * 0.8, 0.8)
                    : new THREE.Color(0.8, 0.8 - intensity * 0.6, 0.2);
                
                neuron.material.color.copy(baseColor);
                neuron.material.emissive.copy(baseColor).multiplyScalar(intensity * 0.3);
                neuron.material.opacity = 0.5 + intensity * 0.5;
            }
        });
    }

    updateConnectionVisuals() {
        this.connections.forEach(connection => {
            const userData = connection.userData;
            const currentStrength = userData.strength;
            const targetStrength = userData.targetStrength;
            
            userData.strength += (targetStrength - currentStrength) * 0.05;
            
            if (connection.material) {
                const alpha = Math.max(0.05, userData.strength * 0.6);
                connection.material.opacity = alpha;
                
                const hue = userData.strength > 0.5 ? 0.3 : 0.6;
                connection.material.color.setHSL(hue, 0.8, 0.5);
            }
        });
    }

    animate() {
        if (!this.isInitialized) return;
        
        this.frameId = requestAnimationFrame(() => this.animate());
        
        const now = performance.now();
        const delta = this.clock.getDelta();
        
        if (this.performance.adaptiveQuality) {
            this.adaptPerformance(now);
        }
        
        this.controls.update();
        
        this.updateNeuronVisuals();
        this.updateConnectionVisuals();
        
        this.animationMixers.forEach(mixer => mixer.update(delta));
        
        THREE.Tween.update();
        
        if (this.composer) {
            this.composer.render();
        } else {
            this.renderer.render(this.scene, this.camera);
        }
        
        this.performance.frameCount++;
    }

    adaptPerformance(now) {
        if (now - this.performance.lastFrame > 1000) {
            this.performance.fps = this.performance.frameCount;
            this.performance.frameCount = 0;
            this.performance.lastFrame = now;
            
            if (this.performance.fps < 30) {
                this.renderer.setPixelRatio(Math.min(window.devicePixelRatio * 0.7, 1));
                this.composer.setSize(
                    this.renderer.domElement.width * 0.8,
                    this.renderer.domElement.height * 0.8
                );
            } else if (this.performance.fps > 55) {
                this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            }
        }
    }

    setupEventListeners() {
        window.addEventListener('resize', () => this.handleResize());
        
        this.container.addEventListener('dblclick', (event) => {
            this.handleDoubleClick(event);
        });
        
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pause();
            } else {
                this.resume();
            }
        });
    }

    handleResize() {
        const rect = this.container.getBoundingClientRect();
        
        this.camera.aspect = rect.width / rect.height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(rect.width, rect.height);
        
        if (this.composer) {
            this.composer.setSize(rect.width, rect.height);
        }
    }

    handleDoubleClick(event) {
        const rect = this.container.getBoundingClientRect();
        const mouse = new THREE.Vector2(
            ((event.clientX - rect.left) / rect.width) * 2 - 1,
            -((event.clientY - rect.top) / rect.height) * 2 + 1
        );
        
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, this.camera);
        
        const intersects = raycaster.intersectObjects([...this.neurons.values()]);
        
        if (intersects.length > 0) {
            const neuron = intersects[0].object;
            this.focusOnNeuron(neuron);
        }
    }

    focusOnNeuron(neuron) {
        const targetPosition = neuron.position.clone();
        targetPosition.add(new THREE.Vector3(2, 2, 2));
        
        const tween = new THREE.Tween(this.camera.position)
            .to(targetPosition, 1000)
            .easing(THREE.Tween.Easing.Quadratic.InOut)
            .onUpdate(() => {
                this.camera.lookAt(neuron.position);
            });
            
        tween.start();
    }

    clearScene() {
        this.neurons.forEach(neuron => {
            this.scene.remove(neuron);
            if (neuron.geometry) neuron.geometry.dispose();
            if (neuron.material) neuron.material.dispose();
        });
        
        this.connections.forEach(connection => {
            this.scene.remove(connection);
            if (connection.geometry) connection.geometry.dispose();
            if (connection.material) connection.material.dispose();
        });
        
        this.neurons.clear();
        this.connections.clear();
        this.layers.clear();
    }

    start() {
        if (!this.isAnimating) {
            this.isAnimating = true;
            this.animate();
        }
    }

    pause() {
        this.isAnimating = false;
        if (this.frameId) {
            cancelAnimationFrame(this.frameId);
            this.frameId = null;
        }
    }

    resume() {
        if (!this.isAnimating) {
            this.start();
        }
    }

    setAutoRotate(enabled) {
        this.controls.autoRotate = enabled;
    }

    setQuality(level) {
        const pixelRatio = level === 'high' ? 2 : level === 'medium' ? 1.5 : 1;
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, pixelRatio));
        
        this.performance.adaptiveQuality = level === 'auto';
    }

    dispose() {
        this.pause();
        
        this.clearScene();
        
        this.geometryCache.forEach(geometry => geometry.dispose());
        this.materialCache.forEach(material => material.dispose());
        
        this.geometryCache.clear();
        this.materialCache.clear();
        
        if (this.composer) {
            this.composer.dispose();
        }
        
        this.renderer.dispose();
        this.controls.dispose();
        
        if (this.container.contains(this.renderer.domElement)) {
            this.container.removeChild(this.renderer.domElement);
        }
    }
}

class Neural3DController {
    constructor(visualizer, apiClient, socketManager) {
        this.visualizer = visualizer;
        this.api = apiClient;
        this.socket = socketManager;
        this.isActive = false;
        this.updateBuffer = [];
        this.bufferTimer = null;
        
        this.bindEvents();
    }

    bindEvents() {
        this.socket.on('neural_visualization_update', (data) => {
            this.bufferUpdate(data);
        });
        
        this.socket.on('board_update', (data) => {
            this.requestVisualizationUpdate();
        });
        
        this.socket.on('ai_move', (data) => {
            if (data.visualization) {
                this.bufferUpdate(data.visualization);
            }
        });
    }

    async initialize() {
        try {
            const structureResponse = await this.api.get('/visualization/structure');
            const structure = structureResponse.data;
            
            this.visualizer.updateNetworkStructure(structure);
            
            const currentResponse = await this.api.get('/visualization/current');
            if (currentResponse.success) {
                this.visualizer.updateActivations(currentResponse.data);
            }
            
            this.visualizer.start();
            this.isActive = true;
            
        } catch (error) {
            console.error('Failed to initialize neural visualization:', error);
        }
    }

    bufferUpdate(data) {
        this.updateBuffer.push(data);
        
        if (!this.bufferTimer) {
            this.bufferTimer = setTimeout(() => {
                this.processBufferedUpdates();
            }, 50);
        }
    }

    processBufferedUpdates() {
        if (this.updateBuffer.length === 0) return;
        
        const latestUpdate = this.updateBuffer[this.updateBuffer.length - 1];
        this.updateBuffer = [];
        this.bufferTimer = null;
        
        this.visualizer.updateActivations(latestUpdate);
    }

    async requestVisualizationUpdate() {
        try {
            const response = await this.api.get('/visualization/current');
            if (response.success) {
                this.visualizer.updateActivations(response.data);
            }
        } catch (error) {
            console.error('Failed to request visualization update:', error);
        }
    }

    async updateConfig(config) {
        try {
            await this.api.post('/visualization/config', config);
        } catch (error) {
            console.error('Failed to update visualization config:', error);
        }
    }

    setQuality(level) {
        this.visualizer.setQuality(level);
    }

    setAutoRotate(enabled) {
        this.visualizer.setAutoRotate(enabled);
    }

    pause() {
        this.visualizer.pause();
        this.isActive = false;
    }

    resume() {
        this.visualizer.resume();
        this.isActive = true;
    }

    dispose() {
        this.pause();
        this.visualizer.dispose();
        
        if (this.bufferTimer) {
            clearTimeout(this.bufferTimer);
        }
    }
}

window.Neural3DVisualizer = Neural3DVisualizer;
window.Neural3DController = Neural3DController;
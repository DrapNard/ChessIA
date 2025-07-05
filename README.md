# Chess AI - Enhanced Neural Network Chess Engine 🤖♟️

Un moteur d'échecs avancé alimenté par des réseaux de neurones PyTorch avec auto-entraînement et interface moderne.

## 🌟 Fonctionnalités

### ♟️ Moteur d'échecs complet
- **Validation complète des règles** : Toutes les règles d'échecs standard
- **Détection d'échec et mat** : Algorithmes optimisés pour les fins de partie
- **Évaluation de position** : Fonction d'évaluation multicritères
- **Interface intuitive** : Sélection/déplacement par clic

### 🧠 Intelligence artificielle
- **Réseau de neurones PyTorch** : Architecture CNN + Dense optimisée
- **Auto-entraînement** : Les modèles s'entraînent en jouant l'un contre l'autre
- **Algorithme génétique** : Évolution et fusion des modèles
- **Recherche Alpha-Beta** : Exploration d'arbre avec élagage

### 🎮 Interfaces multiples
- **GUI Desktop** : Interface Tkinter moderne et responsive
- **Interface Web** : Application web avec WebSocket temps réel
- **CLI** : Interface en ligne de commande pour les puristes
- **API REST** : Intégration facile dans d'autres applications

### 📊 Visualisation en temps réel
- **Progression d'entraînement** : Statistiques live des parties
- **Visualisation du réseau** : Représentation graphique des activations
- **Historique des coups** : Notation algébrique avec évaluations
- **Graphiques de performance** : Évolution du taux de victoire

## 🚀 Installation rapide

### Prérequis
- Python 3.8+
- 4GB RAM (8GB recommandé)
- GPU optionnel (recommandé pour l'entraînement)

### Installation
```bash
# Cloner le repository
git clone https://github.com/your-username/chess-ai-enhanced.git
cd chess-ai-enhanced

# Installer les dépendances
pip install -r requirements.txt

# Pour le support GPU (optionnel)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🎯 Utilisation

### Interface graphique (par défaut)
```bash
python main.py
```

### Interface web
```bash
python main.py --web
# Ouvrir http://localhost:5000 dans votre navigateur
```

### Mode entraînement
```bash
python main.py --train --games 1000 --speed 0.1
```

### Interface ligne de commande
```bash
python main.py --cli
```

### Options complètes
```bash
python main.py --help
```

## 📁 Architecture du projet

```
chess-ai-enhanced/
├── chess_engine.py      # Moteur d'échecs principal
├── neural_network.py    # Réseau de neurones PyTorch
├── chess_trainer.py     # Système d'auto-entraînement
├── chess_gui.py         # Interface graphique Tkinter
├── web_api.py          # API REST et WebSocket
├── main.py             # Point d'entrée principal
├── app.js              # Frontend JavaScript
├── styles.css          # Styles CSS modernes
├── index.html          # Interface web
├── requirements.txt    # Dépendances Python
└── models/             # Modèles sauvegardés
```

## 🎮 Guide d'utilisation

### Interface graphique
1. **Démarrage** : `python main.py`
2. **Jouer** : Cliquez sur une pièce puis sur la destination
3. **Coup IA** : Bouton "AI Move" pour laisser l'IA jouer
4. **Entraînement** : Onglet "Training" pour démarrer l'auto-entraînement
5. **Visualisation** : Onglet "Network" pour voir le réseau de neurones

### Interface web
1. **Démarrage** : `python main.py --web`
2. **Navigation** : Onglets Game/Training/Network/Stats
3. **Temps réel** : Mises à jour automatiques via WebSocket
4. **Responsive** : Fonctionne sur mobile et tablette

### Mode entraînement
```bash
# Entraînement rapide (100 parties)
python main.py --train --games 100 --speed 0.5

# Entraînement long (1000 parties)
python main.py --train --games 1000 --speed 0.1

# Entraînement avec debug
python main.py --train --debug --log-level DEBUG
```

## 🧠 Architecture du réseau de neurones

### Structure du modèle
```
Input (12x8x8) → Conv2D(32) → Conv2D(64) → Conv2D(128) → Dense(512) → Dense(256) → Output(1)
```

### Encodage de l'échiquier
- **12 canaux** : 6 types de pièces × 2 couleurs
- **8×8 grille** : Position de chaque pièce
- **Normalisation** : Valeurs entre -1 et 1

### Fonction d'évaluation
- **Score matériel** : Valeur des pièces (P=1, T=5, etc.)
- **Bonus positionnel** : Contrôle du centre, avancement des pions
- **Pénalités** : Roi en échec, parties nulles
- **Apprentissage** : Récompenses/punitions selon le résultat

## 🔧 Configuration avancée

### Paramètres d'entraînement
```python
config = TrainingConfig(
    max_games=1000,          # Nombre de parties
    save_interval=10,        # Sauvegarde tous les N jeux
    merge_interval=50,       # Fusion des modèles tous les N jeux
    max_moves_per_game=200,  # Limite de coups par partie
    training_speed=0.1,      # Délai entre les coups (s)
    alpha_beta_depth=2       # Profondeur de recherche
)
```

### Paramètres du réseau
```python
model = ChessNet(
    input_channels=12,       # Canaux d'entrée
    hidden_dim=512          # Dimension des couches cachées
)
```

## 📊 Métriques et statistiques

### Pendant l'entraînement
- **Parties jouées** : Nombre total de parties
- **Taux de victoire** : Pourcentage de victoires du modèle 1
- **Parties nulles** : Nombre de matchs nuls
- **Temps d'entraînement** : Durée totale
- **Coups moyens** : Longueur moyenne des parties

### Évaluation des performances
- **Précision** : Capacité à prédire le résultat
- **Vitesse** : Temps de calcul par coup
- **Convergence** : Stabilité du taux de victoire
- **Diversité** : Variété des ouvertures jouées

## 🌐 API REST

### Endpoints principaux
```
GET  /api/health          # Statut du serveur
GET  /api/board           # État de l'échiquier
POST /api/move            # Jouer un coup
POST /api/ai-move         # Coup de l'IA
POST /api/reset           # Nouvelle partie
POST /api/training/start  # Démarrer l'entraînement
POST /api/training/stop   # Arrêter l'entraînement
GET  /api/model/info      # Informations du modèle
```

### WebSocket events
```javascript
socket.on('board_update', data => {
    // Mise à jour de l'échiquier
});

socket.on('training_update', data => {
    // Progression de l'entraînement
});

socket.on('ai_move', data => {
    // Coup joué par l'IA
});
```

## 🧪 Tests et développement

### Lancer les tests
```bash
pytest tests/ -v --cov=.
```

### Formatage du code
```bash
black *.py
flake8 *.py
mypy *.py
```

### Mode développement
```bash
python main.py --web --debug --log-level DEBUG
```

## 🚀 Optimisations de performance

### GPU
- **Détection automatique** : Utilise CUDA si disponible
- **Mixed precision** : Entraînement plus rapide
- **Batch processing** : Traitement par lots optimisé

### CPU
- **Multi-threading** : Entraînement en arrière-plan
- **Optimisations PyTorch** : Compilation JIT si disponible
- **Cache des évaluations** : Réduction des calculs redondants

### Mémoire
- **Modèles légers** : Architecture optimisée
- **Garbage collection** : Nettoyage automatique
- **Streaming des données** : Pas de stockage complet en mémoire

## 🤝 Contribution

### Améliorations possibles
1. **Algorithmes** : Implémentation MCTS, réseaux residuels
2. **Interface** : Amélioration du design, animations
3. **Performance** : Optimisations supplémentaires
4. **Fonctionnalités** : Analyse de parties, puzzles
5. **Tests** : Couverture de test plus complète

### Structure de contribution
1. Fork le repository
2. Créer une branche feature
3. Développer et tester
4. Soumettre une pull request

## 📜 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## 🙏 Remerciements

- **PyTorch** : Framework de deep learning
- **Flask** : Framework web Python
- **Socket.IO** : Communication temps réel
- **Communauté chess.com** : Inspiration et règles

## 📞 Support

- **Issues** : Utilisez GitHub Issues pour les bugs
- **Discussions** : GitHub Discussions pour les questions

---

**Version** : 2.0.0  
**Auteur** : DrapNard 

> "Un échiquier est le monde, les pièces sont les phénomènes de l'univers, les règles du jeu sont ce que nous appelons les lois de la Nature." - T.H. Huxley
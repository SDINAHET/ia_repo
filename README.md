# 🤖 IA Locale — FastAPI & Ollama

🧠 **API d’intelligence artificielle locale**
🔒 Données privées • ⚡ Rapide • 🖥️ 100 % local

Projet IA basé sur **Ollama** et **FastAPI**, permettant d’exécuter un **LLM local (Llama 3.1)** et des **embeddings sémantiques**, sans dépendance au cloud.

---

## 🎯 Objectif du projet

Créer une API capable de :

✔️ Interroger un modèle de langage local
✔️ Générer des réponses textuelles
✔️ Exploiter des embeddings sémantiques
✔️ Fonctionner sans API externe

👉 Projet adapté pour :
- Intranet
- Portfolio
- Projets sensibles
- Recherche & Développement IA

---

## 🛠️ Stack technique

🧠 **Ollama** — moteur IA local
🤖 **Llama 3.1 (8B)** — modèle de langage
📐 **Nomic Embed Text** — embeddings sémantiques
⚙️ **FastAPI** — framework API Python
🚀 **Uvicorn** — serveur ASGI
🐍 **Python ≥ 3.10**
🖥️ **Linux / WSL**

---

## 📦 Installation d’Ollama

### Installation

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Vérification
```bash
ollama --version
```

Lancement du service
```bash
ollama serve
```

⚠️ Le service Ollama doit rester actif pendant l’utilisation de l’API.

🧠 Installation des modèles IA
Modèle de langage
```bash
ollama pull llama3.1:8b
```

Modèle d’embeddings
```bash
ollama pull nomic-embed-text
```

Vérification
```bash
ollama list
```


➡️ Tous les modèles sont stockés localement
➡️ Aucune donnée n’est envoyée vers le cloud

🐍 Configuration de l’environnement Python
Création de l’environnement virtuel
```python
python3 -m venv venv
source venv/bin/activate
```

Mise à jour de pip
```bash
pip install --upgrade pip
```
Installation des dépendances
```bash
pip install -r requirements.txt
```

Si le fichier requirements.txt n’existe pas encore :

```bash
pip install fastapi uvicorn requests
```

🚀 Lancement de l’API
Démarrage du serveur FastAPI
```bash
uvicorn app:app --reload --port 8001
```
Accès à l’application

🌐 API :
http://127.0.0.1:8001

📚 Documentation Swagger :
http://127.0.0.1:8001/docs

🧩 Architecture du projet
```bash
ia/
├── app.py
├── requirements.txt
├── README.md
├── venv/
```

📦 Architecture simple et lisible
🔧 Facilement extensible (Docker, UI, sécurité)

🔐 Pourquoi une IA locale ?

✅ Confidentialité totale des données
✅ Aucun coût d’API externe
✅ Fonctionnement hors ligne
✅ Contrôle complet de l’infrastructure
✅ Performances constantes

🚀 Évolutions possibles

💬 Interface web (Chat UI)
🔐 Authentification JWT
🧠 Mémoire conversationnelle
📊 Recherche sémantique avancée
🐳 Docker / Docker Compose
🌍 Reverse proxy Apache + HTTPS

```bash
.curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
ollama pull nomic-embed-text

uvicorn app:app --reload --port 8001
```
ouvrir
http://127.0.0.1:8001/docs

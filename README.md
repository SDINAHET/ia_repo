curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
ollama pull nomic-embed-text

root@UID7E:/mnt/d/Users/steph/Documents/6ème trimestre/ia# uvicorn app:app --reload --port 8001

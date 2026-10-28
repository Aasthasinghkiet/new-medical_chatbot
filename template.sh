
#!/usr/bin/env bash

# Stop if a command fails
set -e

echo "Creating Medical RAG Chatbot project structure..."

# -------------------- Directories --------------------

mkdir -p src
mkdir -p research
mkdir -p data
mkdir -p knowledge_base
mkdir -p templates
mkdir -p static

# -------------------- Python files --------------------

touch src/__init__.py
touch src/helper.py
touch src/prompt.py

touch app.py
touch store_index.py
touch setup.py
touch requirements.txt

# -------------------- Frontend files --------------------

touch templates/chat.html
touch static/style.css

# -------------------- Environment and research --------------------

touch .env
touch research/trials.ipynb

# -------------------- Git configuration --------------------

if [ ! -f .gitignore ]; then
    touch .gitignore
fi

# Add common exclusions without overwriting existing entries
for entry in ".env" "__pycache__/" "*.py[cod]" ".venv/" "venv/" \
             ".pytest_cache/" ".ipynb_checkpoints/" "vector_store/"; do
    grep -qxF "$entry" .gitignore 2>/dev/null || echo "$entry" >> .gitignore
done

echo ""
echo "Project structure created successfully!"
echo ""
echo "Medical-RAG-Chatbot/"
echo "├── app.py"
echo "├── store_index.py"
echo "├── setup.py"
echo "├── requirements.txt"
echo "├── .env"
echo "├── .gitignore"
echo "├── data/"
echo "├── knowledge_base/"
echo "├── research/"
echo "│   └── trials.ipynb"
echo "├── src/"
echo "│   ├── __init__.py"
echo "│   ├── helper.py"
echo "│   └── prompt.py"
echo "├── templates/"
echo "│   └── chat.html"
echo "└── static/"
echo "    └── style.css"
echo ""
echo "Next: populate the files, configure .env, and install dependencies."

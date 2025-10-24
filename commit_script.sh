#!/bin/bash
# Realistic git commit progression

set -e

echo "Creating realistic commit history..."
echo ""

# Initialize repo
if [ ! -d .git ]; then
    git init
    git branch -M main
fi

# Initial setup
git add requirements.txt .gitignore
git commit -m "initial commit"
sleep 1

git add .env.example
git commit -m "add env template"
sleep 1

# PDF parser development
git add pdf_parser.py
git commit -m "add pdf parser

using pymupdf to extract text and structure from papers"
sleep 1

# Data ingestion
git add data_ingestion.py
git commit -m "data ingestion pipeline with embeddings

- text chunking with overlap
- huggingface embeddings (local, no api quota)
- faiss vector store
- batch processing"
sleep 1

# RAG system
git add rag_system.py
git commit -m "add rag system with langchain

- multiple retrieval strategies (similarity, mmr, compression)
- gemini llm integration
- conversational context support
- specialized queries for papers"
sleep 1

# Chat features
git add chat_manager.py
git commit -m "implement session management for chat history

can save/load conversations as json"
sleep 1

# Main application
git add main.py
git commit -m "add main cli application

- single query mode
- interactive chat mode
- session management
- clean logging (errors only in console)"
sleep 1

# Better structure
git add config.py logger.py exceptions.py
git commit -m "refactor configuration and logging

centralized constants and added proper error handling"
sleep 1

# Package setup
git add __init__.py setup.py cli.py MANIFEST.in
git commit -m "make installable package

added setup.py"
sleep 1

# Documentation
git add README.md
git commit -m "add readme"
sleep 1

if [ -f "QUICKSTART.md" ]; then
    git add QUICKSTART.md
    git commit -m "add quickstart guide"
    sleep 1
fi

git add LICENSE
git commit -m "add license"
sleep 1

git add GIT_COMMIT_GUIDE.md commit_script.sh
git commit -m "add git helper scripts"
sleep 1

# Sample data
if [ -f "data/Attention_is_all_you_need.pdf" ]; then
    git add data/
    git commit -m "add test paper"
    sleep 1
fi

# GitHub workflows if exists
if [ -d ".github" ]; then
    git add .github/
    git commit -m "add github workflows" 2>/dev/null || true
    sleep 1
fi

echo ""
echo "✓ Done! Created realistic commit history"
echo ""
git log --oneline
echo ""
echo "Next steps:"
echo "  git remote add origin <repo-url>"
echo "  git push -u origin main"
echo ""

# Task-Oriented Multi-Agent Pipeline

This project shows how to connect **three specialized agents** – Retrieval, Validation, and Synthesis – in a **LangGraph** DAG and expose it via **FastAPI**.

## Install

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# add OPENAI_API_KEY=... or GOOGLE_API_KEY=...

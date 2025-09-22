## AI Persona Simulator

Create an AI persona from public signals (name, LinkedIn, X/Twitter, extra notes). The app synthesizes a concise, redacted profile (no explicit full name in output), then lets you chat with an assistant that mimics the person’s communication style, background, and likely personality type.

### What it can do
- Persona creation via sidebar (name, LinkedIn, X, additional info)
- Web-informed research to compile a structured profile (Background, Experience, Interests, Communication Style, Personality & Behavior, Constraints)
- Redaction: avoids including the explicit full name in the generated profile
- Chat UI grounded in the generated persona using OpenAI Responses API
- Safety guardrails: no doxxing or sensitive personal data

### Prerequisites
- Python 3.9+
- An OpenAI API key

### 1) Download the repo
```bash
git clone <your-repo-url> AIPersonaSimulation
cd AIPersonaSimulation
```

### 2) Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3) Add environment variables (.env)
Create a file named `.env` in the project root:
```bash
cat > .env << 'EOF'
OPENAI_API_KEY=sk-...
# Optional overrides
# OPENAI_MODEL_CHAT=gpt-5
# OPENAI_REASONING_EFFORT=low
EOF
```
Optionally load into your shell session:
```bash
set -a; source ./.env; set +a
```

### 4) Install dependencies
```bash
pip install -r requirements.txt
```

### 5) Run the app
```bash
streamlit run app.py
```

### Using the app
1. In the sidebar, enter a full name, LinkedIn/X URLs, and optional notes.
2. Click "Create persona" to synthesize a profile (the explicit full name is redacted in the profile text).
3. Chat with the persona in the main panel. Responses are guided by the profile and developer instructions to align with background, expertise, and inferred personality type.

### Troubleshooting
- Missing key: ensure `.env` contains `OPENAI_API_KEY` and is loaded (the app auto-loads `.env`).
- Install issues with Arrow/Pandas: we pin `pyarrow==14.0.2`. Make sure your `pip` is up to date:
  ```bash
  python -m pip install --upgrade pip setuptools wheel
  ```
- Streamlit not found: verify the virtual environment is activated before running.

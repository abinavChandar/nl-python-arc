# nl-arc-nlsearch (Ollama-only)

A minimal ARC v2 “natural language search” solver that runs **only** on a local Ollama model
(e.g., `qwen2.5-coder:7b`) via the OpenAI-compatible API.

## Quickstart

1) Install deps
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Set up Ollama
```bash
brew install ollama
ollama serve
ollama pull qwen2.5-coder:7b
```

3) Env & data
```bash
cp .env.example .env
# edit OLLAMA_MODEL if needed; defaults are fine

# Put ARC JSON here (name must match):
# data/arc-prize-2025/arc-agi_evaluation_challenges.json
# data/arc-prize-2025/arc-agi_evaluation_solutions.json  (optional)
```

4) Run
```bash
python -m src.run
```

## Notes

- The system prompts the model to write **instructions**, picks the best via simple
  leave-one-out scoring on the training set, then **follows** those instructions to
  produce **two** diverse test guesses per task.

- Everything is plain JSON I/O. If the model returns extra text, the runner extracts
  the first top-level JSON object automatically.

- Tweak `.env` to adjust temperature, candidates, samples, tokens, and task limit.

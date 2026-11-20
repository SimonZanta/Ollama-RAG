# Ollama Docs RAG

Ask questions about the Ollama REST API, answered from `ollama-api.md`.

## Stack

- Python, numpy, requests
- Ollama running locally on `127.0.0.1:11434`
- `nomic-embed-text` for embeddings
- `qwen3:4b-instruct` for query rewriting and answers
- Vectors stored in a `.npy` file, no vector DB

## Run

```
ollama pull nomic-embed-text
ollama pull qwen3:4b-instruct

python -m venv .venv
source .venv/bin/activate
pip install numpy requests

python ollamaEmbedder.py
python ollamaQuestionEmbedder.py
```

## Files

| File | |
|---|---|
| `ollama-api.md` | source docs |
| `ollamaParser.py` | splits the docs by headings into `ollamaChunks.json` |
| `ollamaEmbedder.py` | embeds the chunks into `ollamaVectors.npy` |
| `ollamaQuestionEmbedder.py` | asks for a question, rewrites it, finds the top 5 chunks, streams the answer |
| `testQuestions.json` | 40 test questions |

After changing the docs or the parser, run `ollamaParser.py` and `ollamaEmbedder.py` again.

## Tests

`testQuestions.json`: 30 questions the docs answer (with the expected chunk and answer keywords) and 10 they don't.

| | correct chunk at #1 | correct chunk in top 5 |
|---|---|---|
| raw question | 22 / 30 | 26 / 30 |
| rewritten question | 23 / 30 | 28 / 30 |

- Out-of-scope questions: 10 / 10 answered "I don't know".
- Prompt injection: 11 / 15 attacks blocked. Attacks that add an extra request to a real Ollama question still get through.

# Ollama Docs RAG

A small, local RAG (retrieval-augmented generation) system that answers questions about the [Ollama REST API](https://github.com/ollama/ollama/blob/main/docs/api.md) from its own documentation. Built from scratch to learn how RAG works, with no frameworks and no vector database, just Python, numpy and a local Ollama.

```
$ python ollamaQuestionEmbedder.py
What do you need to know about Ollama?
which models are eating my RAM
Use GET /api/ps. It lists the models currently loaded into memory, including size_vram [1].
```

## How it works

```
ollama-api.md ──► chunk ──► embed ──► vectors (.npy)
                                          │
question ──► rewrite ──► embed ──► cosine search ──► top 5 chunks ──► answer (streamed)
```

1. **Chunking** (`ollamaParser.py`): splits the markdown by headings. Each chunk starts with its heading path (`API > List Running Models > Examples`), and Request/Response sub-sections stay with their example. Output: `ollamaChunks.json` (77 chunks).
2. **Embedding** (`ollamaEmbedder.py`): embeds every chunk with `nomic-embed-text` (768 dimensions, `search_document:` prefix), normalizes the vectors and saves them to `ollamaVectors.npy`.
3. **Asking** (`ollamaQuestionEmbedder.py`):
   - **Query rewriting**: `qwen3:4b-instruct` turns casual questions into documentation wording ("eating my RAM" → "list models by memory usage").
   - **Search**: dot product of the question vector against all chunk vectors, top 5.
   - **Answer**: `qwen3:4b-instruct` answers only from those 5 chunks, cites them like `[2]`, and says "I don't know" when the docs don't cover the question. The answer streams to the terminal, with a spinner while waiting.

## Setup

Requires [Ollama](https://ollama.com) running locally.

```
ollama pull nomic-embed-text
ollama pull qwen3:4b-instruct

python -m venv .venv
source .venv/bin/activate
pip install numpy requests

python ollamaEmbedder.py          # builds ollamaVectors.npy
python ollamaQuestionEmbedder.py  # ask a question
```

`ollamaChunks.json` is committed, so you don't need to run the parser first.

## Testing

`testQuestions.json` holds 40 questions: 30 answered by the docs (with the expected chunk heading and answer keywords) and 10 that the docs don't cover.

| | correct chunk at #1 | correct chunk in top 5 |
|---|---|---|
| raw question | 22 / 30 | 26 / 30 |
| with query rewriting | 23 / 30 | 28 / 30 |

- 10 / 10 out-of-scope questions answered "I don't know".
- Prompt injection: the question and documents are wrapped in tags, and a reminder follows the question. 11 of 15 test attacks were blocked; attacks that hide an extra request inside a real Ollama question still get through.

## Notes

- `127.0.0.1` is used instead of `localhost`, which can resolve to IPv6 and fail.
- `qwen3:4b` is a thinking-only model (thinking can't be turned off), so this project uses `qwen3:4b-instruct`.
- `manual-ollama-chunk.md` contains the first hand-written chunks, used to decide the chunk format.

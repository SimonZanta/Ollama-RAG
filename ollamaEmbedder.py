import numpy as np
import requests
import json

with open('ollamaChunks.json') as f:
    chunks = json.load(f)
    chunksParsed = ['search_document: ' + c['text'] for c in chunks]

r = requests.post('http://127.0.0.1:11434/api/embed',
            json={"model": "nomic-embed-text", "input": chunksParsed})

vectors = r.json()["embeddings"]

v = np.array(vectors)
v = v / np.linalg.norm(v, axis=1, keepdims=True)

np.save("ollamaVectors.npy", v)   # save

print(v)

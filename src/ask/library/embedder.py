"""Local CPU embeddings via fastembed.

The model name and cache dir come from module constants + env. fastembed is
imported lazily inside get_embedding_model so merely importing this module (e.g.
to read EMBED_DIM in store.py) does not pull in the heavy dependency or trigger a
model download.

Model and dimension default to bge-small-en-v1.5 / 384. The dimension is baked
into the sqlite-vec schema, so changing the model means re-indexing. Embeddings
are always computed locally, even when a corpus answers with a cloud model.
"""

import os

EMBED_MODEL = os.getenv("ASK_EMBED_MODEL") or "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
CACHE_DIR = os.getenv("ASK_FASTEMBED_CACHE") or ".fastembed_cache"
# Default to a single inference thread: lighter and more stable on small/low-RAM
# boxes (raise with ASK_EMBED_THREADS if you have cores + memory to spare).
EMBED_THREADS = int(os.getenv("ASK_EMBED_THREADS", "1"))

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(
            model_name=EMBED_MODEL, cache_dir=CACHE_DIR, threads=EMBED_THREADS
        )
    return _model


def embed_texts(texts):
    """Embed a list of strings -> list of 384-dim float lists."""
    texts = list(texts)
    if not texts:
        return []
    model = get_embedding_model()
    return [embedding.tolist() for embedding in model.embed(texts)]

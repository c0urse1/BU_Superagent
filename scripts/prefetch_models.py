from __future__ import annotations

import os


def main() -> None:
    # Avoid torchvision import for text-only models
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

    # Prefetch E5 embeddings model
    try:
        from sentence_transformers import SentenceTransformer as ST

        ST("intfloat/multilingual-e5-large-instruct")
        print("[OK] Cached: intfloat/multilingual-e5-large-instruct")
    except Exception as e:
        print(f"[WARN] E5 prefetch failed: {e}")

    # Prefetch BGE reranker model + tokenizer
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
        AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
        print("[OK] Cached: BAAI/bge-reranker-v2-m3")
    except Exception as e:
        print(f"[WARN] BGE prefetch failed: {e}")


if __name__ == "__main__":
    main()

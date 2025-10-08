# Apple Zero Memory System

Apple Zero mirrors Agent Zero’s memory stack:

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face) loads locally. No OpenAI key is required.
- **Storage**: Each session maps to `memory/<subdir>`, backed by the SQLite+FAISS hybrid store (`tmp/memory.sqlite`).
- **Consolidation**: `python/helpers/memory_consolidation.py` decides whether to merge, replace, or keep memories separate.
- **Knowledge preload**: Markdown files in `knowledge/<profile>/<area>` are ingested on first access and tagged with `area` metadata.

### Developer Tips

1. Update the active memory subdirectory via Settings → Memory.
2. Use `memory_save`, `memory_load`, `memory_delete`, and `memory_forget` tools for manual control.
3. Run `pytest tests/runtime/test_memory_recall.py` after dependency install to validate retrieval quality.

Keep content in this directory factual and vendor-neutral; it loads for every runtime profile that includes the `default` knowledge pack.

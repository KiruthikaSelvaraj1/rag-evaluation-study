install:
	pip install -e ".[dev,models]"
test:
	pytest -q
smoke:  # offline end-to-end on synthetic data, no downloads
	python -m rag_study run configs/smoke_bm25.yaml
	python -m rag_study run configs/smoke_hybrid.yaml
	python -m rag_study compare --baseline smoke_bm25 --names smoke_bm25,smoke_hybrid
all:    # real study (downloads SciFact + models)
	for c in bm25 vector hybrid hybrid_rerank; do python -m rag_study run configs/$$c.yaml; done
	python -m rag_study compare --baseline bm25 --names bm25,vector,hybrid,hybrid_rerank

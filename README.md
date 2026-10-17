



# HierRAG: Single-Document Multimodal Medical Question Answering

Dataset: https://huggingface.co/datasets/yubo2333/MMLongBench-Doc.

HierRAG is a hierarchical retrieval-augmented question-answering system for long medical PDF/OCR Markdown documents. The system simultaneously uses PDF page-level visual information, local text nodes, and chapter-level topic summaries to retrieve evidence **within a single document specified by the user**. The retrieved evidence is fused and reranked by page before answers are generated.

## 1. Core Constraints

- The index-building interface accepts either a directory or a single Markdown path. When a directory is provided, all `.md` files are processed recursively.
- Each Markdown file has an independent index directory. Merging multiple documents into a global retrieval domain is prohibited.
- Queries must provide both `question` and `document_name`.
- The query service first resolves the unique document, then loads only the three indexes belonging to that document.
- If Markdown files with the same name appear in different directories—for example, two `001.md` files—index construction is rejected to prevent cross-document retrieval contamination.

## 2. System Workflow

```text
PDF ──page rendering──> ColPali visual page multi-vector index ─┐
                                                               │
OCR Markdown ─┬─image/table VLM description─> visual semantic index ┤
              ├─heading tree/text chunking────> text node index    ├─page-level fusion and deduplication─> Jina reranking─> LLM answer
              └─bottom-up chapter summarization─> topic cluster index ┘
```

Responsibilities of the three retrieval channels:

- `visual`: ColPali/ColQwen2 page-level visual retrieval, preserving page layout, charts, and image-text relationships.
- `text`: Jina Embeddings + FAISS node-level retrieval, locating local evidence such as paragraphs, sentences, and table descriptions.
- `cluster`: Two-stage retrieval based on KMeans clustering of chapter summaries, supplementing cross-page topic evidence.

Candidate evidence is first mapped to PDF page numbers and deduplicated at the page level. RRF fusion is then applied, followed by a second-stage reranking step when a reranking model is available. Answer generation submits both textual evidence and the retrieved page images, requiring the model to perform step-by-step reasoning internally and return only the final answer.

## 3. Project Structure

```text
app/
├── main.py                         # FastAPI entry point
├── router/
│   ├── index.py                    # Index construction API
│   └── retrieval.py                # Single-document retrieval API
├── models/
│   ├── api_models.py               # HTTP request/response models
│   └── hier_rag.py                 # Evidence and document domain models
├── database/
│   ├── dense_index.py              # Text/visual-description FAISS index
│   └── topic_index.py              # Chapter-summary clustering index
└── service/
    ├── document_registry.py        # Document discovery, PDF pairing, name resolution
    ├── multimodal_preprocessor.py  # Image/table natural-language descriptions
    ├── doc2tree.py                 # Markdown hierarchy tree and text nodes
    ├── page_mapping.py             # Node-to-PDF-page mapping
    ├── summary_service.py          # Bottom-up chapter summarization
    ├── visual_index_service.py     # ColPali page indexing
    ├── index_service.py            # Index construction orchestration
    ├── hier_retrieval_service.py   # Single-document three-channel retrieval
    ├── evidence_fusion.py          # Page-level fusion, deduplication, reranking
    └── answer_service.py           # Multimodal answer generation
```

The `baseline/`, `new_exp/`, `ablation_study/`, and `evaluation/` directories in the project root contain historical baselines, experiments, and evaluation artifacts. They are not loaded by the FastAPI main entry point.

## 4. Input Data Conventions

OCR is not part of this service’s API. The indexing interface requires PDFs that have already undergone OCR, with corresponding Markdown files available. The recommended directory structure is:

```text
medical_documents/
├── 001.md
├── 001.pdf
├── 002.md
├── 002.pdf
└── images/
    ├── 001_page_12_chart.jpg
    └── 002_page_7_table.jpg
```

Pairing rules:

- `001.md` is automatically matched with `001.pdf` in the same input root directory.
- The `_processed`, `_ocr`, and `_markdown` suffixes are ignored during pairing.
- If the PDF is missing, the text and topic indexes can still be built; the visual channel falls back to retrieval based on Markdown image/table descriptions.
- Markdown image paths should be relative to the Markdown file, for example: `![](images/001_page_12_chart.jpg)`.
- Page numbers are parsed preferentially from `<!-- page: 12 -->`, `[[PAGE: 12]]`, or resource names such as `page_12`. If unavailable, the system attempts to match directory headings with page numbers, and finally falls back to page 1.

## 5. Environment Installation

The current environment baseline is Python 3.10. Page rendering for the visual pipeline also requires Poppler to be installed on the system.

```bash
conda activate search
cd /home/asus/chen/knowledge_system

# In a new environment, first install torch/torchvision compatible with the machine's CUDA driver.
pip install -r requirements.txt
```

The project also provides a user-requested `requirement.txt`, which redirects to the same `requirements.txt`.

Copy the configuration template:

```bash
cp .env.example .env
```

At minimum, check the following configuration:

```dotenv
MODEL_NAME=/path/to/jina-embeddings-v3
RERANK_MODEL_NAME=/path/to/jina-reranker-m0
VISUAL_MODEL_NAME=/path/to/colqwen2-v1.0

EMBEDDING_DEVICE=auto
RERANK_DEVICE=auto
VISUAL_DEVICE=auto

OPENAI_API_KEY=your-key
```

`auto` prioritizes `cuda:0` and falls back to the CPU when CUDA is unavailable. The models are relatively large, so using a GPU in production is recommended. Download the models locally in advance.

## 6. Starting the Service

```bash
conda activate search
cd /home/asus/chen/knowledge_system
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- Health check: `GET http://localhost:8000/health`
- OpenAPI: `http://localhost:8000/docs`

## 7. Building Indexes

Endpoint: `POST /api/v1/index/build`

```bash
curl -X POST 'http://localhost:8000/api/v1/index/build' \
  -H 'Content-Type: application/json' \
  -d '{
    "source_path": "/data/medical_documents",
    "rebuild": true,
    "describe_multimodal": true,
    "generate_summaries": true,
    "enable_visual_index": true
  }'
```

Main parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `source_path` | Required | A single Markdown file or a directory to scan recursively |
| `rebuild` | `true` | Overwrite the existing index for the document |
| `describe_multimodal` | `true` | Use a VLM to read actual images and describe images/tables |
| `generate_summaries` | `true` | Use an LLM to generate chapter summaries; when disabled, deterministic compressed summaries are generated |
| `enable_visual_index` | `true` | Build a ColPali page index when a matching PDF is found |

Indexes are written to the following location by default:

```text
db/hier_rag/
├── registry.json
└── documents/<document_id>/
    ├── manifest.json
    ├── artifacts/
    │   ├── document.md
    │   ├── tree.json
    │   └── page_map.json
    ├── text/
    ├── visual/
    ├── visual_semantic/
    └── cluster/
```

## 8. Single-Document Retrieval and Question Answering

Endpoint: `POST /api/v1/retrieval/query`

```bash
curl -X POST 'http://localhost:8000/api/v1/retrieval/query' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "What multidisciplinary care model is recommended for hip fracture patients?",
    "document_name": "001.pdf",
    "top_k": 6,
    "channel_top_k": 10,
    "generate_answer": true
  }'
```

Only `question` and `document_name` are required. `document_name` may use the `.md` filename recorded during indexing, the matching `.pdf` filename, or the name without an extension.

The response includes:

- `answer`: An answer generated from the fused evidence; `null` when no API key is configured or answer generation is disabled.
- `evidence`: The final evidence, page numbers, primary channel, contributing channels, scores, and node information.
- `channel_counts`: The number of raw results retrieved by each of the three channels.

The system does not perform fallback retrieval from other documents. If the document does not exist, the service returns HTTP 404. If the name is ambiguous or the indexes are inconsistent, it returns HTTP 400.

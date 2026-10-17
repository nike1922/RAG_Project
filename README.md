# HierRAG：单文档多模态医学问答


Dataset: https://huggingface.co/datasets/yubo2333/MMLongBench-Doc.

HierRAG 是一个面向长篇医学 PDF/OCR Markdown 的分层检索增强问答系统。系统同时利用 PDF 页面视觉信息、局部文本节点和章节主题摘要，在**用户指定的单份文档内**检索证据，经按页融合与重排序后生成答案。

## 1. 核心约束

- 索引构建接口接收一个目录或单个 Markdown 路径；传入目录时递归处理全部 `.md` 文件。
- 每份 Markdown 建立独立索引目录，禁止合并成全局检索域。
- 查询必须同时提供 `question` 和 `document_name`。
- 查询服务先解析唯一文档，再只加载该文档的三路索引。
- 不同目录下出现同名 Markdown（例如两个 `001.md`）时拒绝构建，避免检索串库。

## 2. 系统流程

```text
PDF ──页面渲染──> ColPali 视觉页面多向量索引 ─┐
                                                   │
OCR Markdown ─┬─图像/表格 VLM 描述─> 视觉语义索引 ┤
              ├─标题树/文本切块────> 文本节点索引 ├─按页融合去重─> Jina 重排─> LLM 答案
              └─章节自底向上摘要───> 主题聚类索引 ┘
```

三路检索职责：

- `visual`：ColPali/ColQwen2 页面级视觉检索，保留页面布局、图表和图文关系。
- `text`：Jina Embeddings + FAISS 节点级检索，定位段落、句子、表格描述等局部证据。
- `cluster`：章节摘要 KMeans 聚类后两阶段检索，补充跨页主题证据。

候选证据先映射到 PDF 页码，以页面为单位去重；随后使用 RRF 融合，并在重排序模型可用时执行二次排序。答案生成会同时提交文本证据和已召回页面图像，要求模型在内部进行逐步推理，仅返回最终答案。

## 3. 项目结构

```text
app/
├── main.py                         # FastAPI 入口
├── router/
│   ├── index.py                    # 索引构建接口
│   └── retrieval.py                # 单文档检索接口
├── models/
│   ├── api_models.py               # HTTP 请求/响应模型
│   └── hier_rag.py                 # 证据与文档领域模型
├── database/
│   ├── dense_index.py              # 文本/视觉描述 FAISS
│   └── topic_index.py              # 章节摘要聚类索引
└── service/
    ├── document_registry.py        # 文档发现、PDF 配对、名称解析
    ├── multimodal_preprocessor.py  # 图像/表格自然语言描述
    ├── doc2tree.py                 # Markdown 层次树和文本节点
    ├── page_mapping.py             # 节点到 PDF 页码映射
    ├── summary_service.py          # 自底向上章节摘要
    ├── visual_index_service.py     # ColPali 页面索引
    ├── index_service.py            # 索引构建编排
    ├── hier_retrieval_service.py   # 单文档三通道检索
    ├── evidence_fusion.py          # 按页融合、去重、重排
    └── answer_service.py           # 多模态答案生成
```

根目录中的 `baseline/`、`new_exp/`、`ablation_study/` 和 `evaluation/` 是历史基线、实验与评测产物，不会被 FastAPI 主入口加载。

## 4. 输入数据约定

OCR 不是本服务接口的一部分。索引接口要求 PDF 已经完成 OCR，并存在对应 Markdown。推荐目录如下：

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

配对规则：

- `001.md` 自动匹配同一输入根目录下的 `001.pdf`。
- `_processed`、`_ocr`、`_markdown` 后缀在配对时会被忽略。
- PDF 缺失时仍可构建文本和主题索引；视觉通道降级为 Markdown 图像/表格描述检索。
- Markdown 图片路径应相对于 Markdown 文件，例如 `![](images/001_page_12_chart.jpg)`。
- 页码优先从 `<!-- page: 12 -->`、`[[PAGE: 12]]` 或资源名 `page_12` 解析，其次使用目录标题与页码匹配，最后回退第 1 页。

## 5. 环境安装

当前环境基线是 Python 3.10。视觉页面转换还需要系统安装 Poppler。

```bash
conda activate search
cd /home/asus/chen/knowledge_system

# 新环境应先安装与机器 CUDA 驱动匹配的 torch/torchvision。
pip install -r requirements.txt
```

项目同时提供用户指定名称的 `requirement.txt`，它会转向同一份 `requirements.txt`。

复制配置模板：

```bash
cp .env.example .env
```

至少检查以下配置：

```dotenv
MODEL_NAME=/path/to/jina-embeddings-v3
RERANK_MODEL_NAME=/path/to/jina-reranker-m0
VISUAL_MODEL_NAME=/path/to/colqwen2-v1.0

EMBEDDING_DEVICE=auto
RERANK_DEVICE=auto
VISUAL_DEVICE=auto

OPENAI_API_KEY=your-key
```

`auto` 会优先选择 `cuda:0`，CUDA 不可用时回退到 CPU。模型较大，生产运行建议使用 GPU，并将模型提前下载到本地。

## 6. 启动服务

```bash
conda activate search
cd /home/asus/chen/knowledge_system
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- 健康检查：`GET http://localhost:8000/health`
- OpenAPI：`http://localhost:8000/docs`

## 7. 构建索引

接口：`POST /api/v1/index/build`

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

主要参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `source_path` | 必填 | 单个 Markdown 或递归扫描目录 |
| `rebuild` | `true` | 覆盖该文档已有索引 |
| `describe_multimodal` | `true` | 用 VLM 读取实际图片，并描述图像/表格 |
| `generate_summaries` | `true` | 用 LLM 生成章节摘要；关闭时生成确定性压缩摘要 |
| `enable_visual_index` | `true` | 匹配到 PDF 时构建 ColPali 页面索引 |

索引默认写入：

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

## 8. 单文档检索与问答

接口：`POST /api/v1/retrieval/query`

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

只有 `question` 和 `document_name` 必填。`document_name` 可以使用构建时记录的 `.md`、匹配的 `.pdf` 或不带扩展名的名称。

响应包括：

- `answer`：基于融合证据生成的答案；未配置 API Key 或关闭生成时为 `null`。
- `evidence`：最终证据、页码、主通道、融合通道、分数和节点信息。
- `channel_counts`：三路各自的原始召回数量。

系统不会在其他文档中补召回。文档不存在返回 HTTP 404，名称有歧义或索引不一致返回 HTTP 400。


# Baseline系统架构说明

## 概述

这是一个基于ColPali的多模态RAG（检索增强生成）系统，用于处理PDF文档并进行多模态问答。

## 系统架构

### 核心模块

#### 1. `embedding.py` - 统一的Embedding引擎
- **功能**: 提供统一的ColPali模型接口
- **类**: `ColPaliEngine`
- **主要方法**:
  - `encode_image()`: 将图片转换为patch嵌入向量
  - `encode_query()`: 将查询文本转换为token嵌入向量
- **模型**: 使用 `google/paligemma-3b-mix-224` (通过modelscope)
- **特点**: 所有其他模块都统一调用这个embedding引擎

#### 2. `index.py` - 索引构建模块
- **功能**: 构建和管理FAISS向量索引
- **类**: `ColPaliIndexer`
- **主要方法**:
  - `add_image()`: 将单张图片添加到索引
  - `add_images()`: 批量添加图片到索引
  - `save()`: 保存索引和元数据
  - `load()`: 加载索引和元数据
- **依赖**: 使用 `embedding.py` 中的 `ColPaliEngine`

#### 3. `retriever.py` - 检索模块
- **功能**: 执行多模态检索
- **类**: `ColPaliRetriever`
- **主要方法**:
  - `search()`: 根据查询检索相关图片
  - `retrieve_and_answer()`: 检索并调用GPT-4o回答问题
- **依赖**: 使用 `embedding.py` 中的 `ColPaliEngine`

#### 4. `process_documents.py` - 文档处理模块
- **功能**: 批量处理PDF文档并构建索引
- **主要功能**:
  - 将PDF按页转换为图片
  - 调用embedding引擎提取向量
  - 构建FAISS索引
- **依赖**: 使用 `embedding.py` 和 `index.py`

#### 5. `batch_query_processor.py` - 批量查询处理模块
- **功能**: 批量处理MMLongBench-Doc数据集的查询
- **类**: `MMLongBenchQueryProcessor`
- **主要功能**:
  - 从Parquet文件加载查询
  - 批量执行检索和问答
  - 保存结果
- **依赖**: 使用 `retriever.py`

### 辅助模块

#### 6. `openai.py` - GPT-4o集成模块
- **功能**: 封装OpenAI GPT-4o API调用
- **类**: `GPT4oHandler`
- **主要方法**: `ask_with_image()`: 使用图片和文本进行问答

#### 7. `test_*.py` - 测试模块
- `test_imports.py`: 测试所有模块导入
- `test_index_creation.py`: 测试索引创建流程
- `embedding.py`: 包含embedding测试功能

#### 8. `run_*.sh` - 运行脚本
- `run_index_creation.sh`: 索引创建脚本

## 数据流

### 1. 索引构建流程
```
PDF文档 → process_documents.py → 图片提取 → embedding.py → 向量化 → index.py → FAISS索引
```

### 2. 查询处理流程
```
用户查询 → batch_query_processor.py → retriever.py → embedding.py → 向量化 → FAISS检索 → GPT-4o问答 → 结果输出
```

## 文件结构

```
baseline/
├── embedding.py              # 统一的embedding引擎
├── index.py                  # 索引构建模块
├── retriever.py              # 检索模块
├── process_documents.py      # 文档处理模块
├── batch_query_processor.py  # 批量查询处理模块
├── openai.py                 # GPT-4o集成模块
├── test_imports.py           # 导入测试
├── test_index_creation.py    # 索引创建测试
├── run_index_creation.sh     # 索引创建脚本
└── README_system_architecture.md  # 本文件
```

## 使用方法

### 1. 创建索引
```bash
# 测试索引创建
python test_index_creation.py

# 完整索引创建
bash run_index_creation.sh
```

### 2. 批量处理查询
```bash
python batch_query_processor.py \
    --parquet_file /path/to/queries.parquet \
    --output_dir ./results \
    --index_path /path/to/index
```

### 3. 测试embedding
```bash
python embedding.py
```

## 关键特性

1. **统一Embedding**: 所有模块都使用 `embedding.py` 中的 `ColPaliEngine`
2. **模块化设计**: 每个模块职责清晰，易于维护
3. **错误处理**: 完善的错误处理和日志记录
4. **批量处理**: 支持批量处理PDF文档和查询
5. **多模态**: 支持图片和文本的多模态检索和问答

## 依赖关系

```
embedding.py (核心)
    ↓
index.py, retriever.py
    ↓
process_documents.py, batch_query_processor.py
```

所有模块都依赖于 `embedding.py` 中的 `ColPaliEngine`，确保了系统的一致性和可维护性。

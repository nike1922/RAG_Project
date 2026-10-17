#!/bin/bash

# 使用Byaldi创建ColPali多模态索引
# 使用方法: bash run_byaldi_index.sh

set -e

echo "=== 使用Byaldi创建ColPali多模态索引 ==="

PDF_DIR="/media/asus/35C588B63481173F/dataset/chen/MMLongBench-Doc/documents"
INDEX_NAME="colpali_index"
MODEL_PATH="/media/asus/35C588B63481173F/model/colpali-v1.3"
INDEX_ROOT="/home/asus/chen/knowledge_system/db/byaldi_db"
OUTPUT_CSV="/home/asus/chen/knowledge_system/db/byaldi_db/search_results.csv"

# 检查
if [ ! -d "$PDF_DIR" ]; then echo "错误: PDF目录不存在: $PDF_DIR"; exit 1; fi
mkdir -p "$INDEX_ROOT"

echo "PDF目录: $PDF_DIR"
echo "索引名称: $INDEX_NAME"
echo "模型路径: $MODEL_PATH"
echo "索引根目录: $INDEX_ROOT"
echo "开始创建索引..."

# 运行Byaldi索引创建脚本
python byaldi_index.py \
  --pdf_dir "$PDF_DIR" \
  --index_name "$INDEX_NAME" \
  --model_path "$MODEL_PATH" \
  --index_root "$INDEX_ROOT" \
  --overwrite

echo "✓ 索引创建完成"

# 如果需要测试查询，取消下面的注释
# echo "执行测试查询..."
# python byaldi_index.py \
#   --pdf_dir "$PDF_DIR" \
#   --index_name "$INDEX_NAME" \
#   --model_path "$MODEL_PATH" \
#   --index_root "$INDEX_ROOT" \
#   --query "这是一个测试查询" \
#   --top_k 5 \
#   --output_csv "$OUTPUT_CSV"
# 
# echo "✓ 查询完成，结果保存在: $OUTPUT_CSV"

#!/usr/bin/env python3
"""
使用Byaldi库创建ColPali多模态索引
自动处理PDF文档，创建索引并支持查询
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from byaldi import RAGMultiModalModel
    logger.info("成功导入Byaldi库")
except ImportError as e:
    logger.error(f"导入Byaldi库失败: {e}")
    logger.error("请确保已安装Byaldi: pip install Byaldi")
    raise

def create_index(
    pdf_dir: str,
    index_name: str,
    model_path: str,
    index_root: str = ".byaldi",
    store_collection: bool = False,
    overwrite: bool = False
) -> RAGMultiModalModel:
    """
    创建多模态文档索引
    
    Args:
        pdf_dir: PDF文档目录路径
        index_name: 索引名称
        model_path: 模型路径，如 "vidore/colpali-v1.3"
        index_root: 索引保存根目录
        store_collection: 是否存储文档集合（会增加内存和存储需求）
        overwrite: 是否覆盖已存在的索引
        
    Returns:
        RAGMultiModalModel 实例
    """
    logger.info(f"开始创建索引: {index_name}")
    logger.info(f"PDF目录: {pdf_dir}")
    logger.info(f"模型路径: {model_path}")
    
    # 检查目录是否存在
    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"PDF目录不存在: {pdf_dir}")
    
    # 创建索引根目录
    os.makedirs(index_root, exist_ok=True)
    
    # 加载模型
    logger.info(f"加载模型: {model_path}")
    rag_model = RAGMultiModalModel.from_pretrained(
        model_path,
        index_root=index_root
    )
    
    # 收集PDF文件元数据
    logger.info("收集文件元数据...")
    pdf_files = []
    metadata_list = []
    pdf_dir_path = Path(pdf_dir)
    
    # 遍历目录收集所有PDF文件
    for file_path in pdf_dir_path.glob("**/*.pdf"):
        relative_path = file_path.relative_to(pdf_dir_path)
        pdf_files.append(str(file_path))
        
        # 为每个文件创建元数据字典
        metadata_list.append({
            "filename": file_path.name,
            "filepath": str(relative_path),
            "full_path": str(file_path)
        })
    
    logger.info(f"找到 {len(pdf_files)} 个PDF文件")
    
    # 创建索引，传递文件元数据
    logger.info(f"开始索引文档: {pdf_dir}")
    rag_model.index(
        input_path=pdf_dir,
        index_name=index_name,
        store_collection_with_index=store_collection,
        overwrite=overwrite,
        metadata=metadata_list  # 传递元数据列表
    )
    
    logger.info(f"索引创建完成: {index_name}")
    return rag_model

def search_index(
    query: str,
    index_name: str,
    index_root: str = ".byaldi",
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    搜索索引
    
    Args:
        query: 查询文本
        index_name: 索引名称
        index_root: 索引根目录
        top_k: 返回结果数量
        
    Returns:
        搜索结果列表
    """
    logger.info(f"加载索引: {index_name}")
    rag_model = RAGMultiModalModel.from_index(
        index_name,
        index_root=index_root
    )
    
    logger.info(f"执行查询: {query}")
    results = rag_model.search(query, k=top_k)
    
    # 转换结果为字典列表
    results_list = []
    for result in results:
        result_dict = {
            "doc_id": result["doc_id"],
            "page_num": result["page_num"],
            "score": result["score"],
            "metadata": result["metadata"]
        }
        results_list.append(result_dict)
    
    return results_list

def main():
    parser = argparse.ArgumentParser(description='使用Byaldi创建和搜索多模态文档索引')
    
    parser.add_argument('--pdf_dir', type=str, required=True,
                        help='PDF文档目录路径')
    parser.add_argument('--index_name', type=str, default='colpali_index',
                        help='索引名称')
    parser.add_argument('--model_path', type=str, default='vidore/colpali-v1.3',
                        help='模型路径或名称')
    parser.add_argument('--index_root', type=str, default='.byaldi',
                        help='索引根目录')
    parser.add_argument('--store_collection', action='store_true',
                        help='是否存储文档集合（会增加内存和存储需求）')
    parser.add_argument('--overwrite', action='store_true',
                        help='是否覆盖已存在的索引')
    parser.add_argument('--query', type=str,
                        help='查询文本，如果提供则执行搜索')
    parser.add_argument('--top_k', type=int, default=5,
                        help='返回结果数量')
    
    args = parser.parse_args()
    
    # 创建索引
    rag_model = create_index(
        pdf_dir=args.pdf_dir,
        index_name=args.index_name,
        model_path=args.model_path,
        index_root=args.index_root,
        store_collection=args.store_collection,
        overwrite=args.overwrite
    )
    
    # 如果提供了查询，则执行搜索
    if args.query:
        results = search_index(
            query=args.query,
            index_name=args.index_name,
            index_root=args.index_root,
            top_k=args.top_k
        )
        
        # 打印结果
        logger.info(f"搜索结果 (top {args.top_k}):")
        for i, result in enumerate(results):
            logger.info(f"{i+1}. 文档ID: {result['doc_id']}, 页码: {result['page_num']}, 得分: {result['score']}")
            if result['metadata']:
                logger.info(f"   元数据: {result['metadata']}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Byaldi搜索命令行工具 - 使用官方推荐的最简用法
支持文本查询和自定义embedding查询
"""

import os
import argparse
import logging
import time
import torch

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

def search(query, index_name="colpali_index", index_root=".byaldi", top_k=5, device="cuda"):
    """
    使用Byaldi进行搜索 - 完全按照官方文档的用法
    
    Args:
        query: 查询文本
        index_name: 索引名称
        index_root: 索引根目录
        top_k: 返回结果数量
        device: 设备，默认为cuda
        
    Returns:
        搜索结果列表
    """
    logger.info(f"从索引加载RAG模型: {index_name}")
    start_time = time.time()
    
    try:
        # 从索引加载，完全按照官方文档的用法
        RAG = RAGMultiModalModel.from_index(
            index_name,
            index_root=index_root,
            device=device
        )
        logger.info(f"RAG模型加载完成，耗时: {time.time() - start_time:.2f}秒")
        
        # 执行搜索
        start_time = time.time()
        logger.info(f"执行查询: {query}")
        
        # 使用官方API进行搜索
        results = RAG.search(query=query, k=top_k)
        
        logger.info(f"搜索完成，找到 {len(results)} 个结果，耗时: {time.time() - start_time:.2f}秒")
        return results
        
    except Exception as e:
        logger.error(f"搜索失败: {e}")
        raise

def main():
    """
    主函数，使用官方推荐的最简用法进行搜索
    """
    parser = argparse.ArgumentParser(description='使用Byaldi搜索多模态文档')
    
    parser.add_argument('--query', type=str, required=True,
                        help='查询文本')
    parser.add_argument('--index_name', type=str, default='colpali_index',
                        help='索引名称')
    parser.add_argument('--index_root', type=str, default='/home/asus/chen/knowledge_system/db/byaldi_db',
                        help='索引根目录')
    parser.add_argument('--top_k', type=int, default=5,
                        help='返回结果数量')
    
    args = parser.parse_args()
    
    # 执行搜索
    results = search(
        query=args.query,
        index_name=args.index_name,
        index_root=args.index_root,
        top_k=args.top_k
    )
    
    # 打印结果
    print("\n搜索结果:")
    for i, result in enumerate(results):
        print(f"{i+1}. 文档ID: {result['doc_id']}, 页码: {result['page_num']}, 得分: {result['score']:.4f}")
        if result['metadata']:
            print(f"   元数据: {result['metadata']}")


if __name__ == "__main__":
    main()

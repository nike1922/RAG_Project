#!/usr/bin/env python3
"""
批量查询处理工具 - 从parquet文件读取查询，执行搜索，保存结果
"""

import os
import pandas as pd
import logging
import time
import csv
from tqdm import tqdm
import argparse

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

class BatchSearchProcessor:
    """批量查询处理器类"""
    
    def __init__(self, index_name="colpali_index", index_root="/home/asus/chen/knowledge_system/db/byaldi_db", device="cuda"):
        """
        初始化批量查询处理器
        
        Args:
            index_name: 索引名称
            index_root: 索引根目录
            device: 设备，默认为cuda
        """
        self.index_name = index_name
        self.index_root = index_root
        self.device = device
        self.rag_model = None
        
    def load_model(self):
        """加载RAG模型（只需加载一次）"""
        logger.info(f"从索引加载RAG模型: {self.index_name}")
        start_time = time.time()
        
        try:
            # 从索引加载
            self.rag_model = RAGMultiModalModel.from_index(
                self.index_name,
                index_root=self.index_root,
                device=self.device
            )
            logger.info(f"RAG模型加载完成，耗时: {time.time() - start_time:.2f}秒")
            return True
        except Exception as e:
            logger.error(f"RAG模型加载失败: {e}")
            return False
    
    def search(self, query, top_k=5):
        """
        执行单个查询
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        if self.rag_model is None:
            logger.error("RAG模型未加载，请先调用load_model()")
            return []
        
        try:
            # 执行搜索
            start_time = time.time()
            results = self.rag_model.search(query=query, k=top_k)
            # logger.info(f"查询完成，找到 {len(results)} 个结果，耗时: {time.time() - start_time:.2f}秒")
            return results
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return []
    
    def process_batch(self, queries, top_k=5):
        """
        批量处理查询
        
        Args:
            queries: 查询列表
            top_k: 每个查询返回的结果数量
            
        Returns:
            处理结果列表，每个元素为 (query, results) 元组
        """
        if self.rag_model is None:
            if not self.load_model():
                logger.error("无法加载RAG模型，批处理终止")
                return []
        
        total_queries = len(queries)
        logger.info(f"开始处理 {total_queries} 个查询")
        
        results = []
        pbar = tqdm(total=total_queries, desc="处理查询", unit="查询")
        
        for i, query in enumerate(queries):
            # 显示当前进度和估计剩余时间
            pbar.set_description(f"查询 {i+1}/{total_queries} [{(i+1)/total_queries*100:.1f}%]")
            
            # 执行查询
            search_results = self.search(query, top_k)
            results.append((query, search_results))
            
            # 更新进度条
            pbar.update(1)
            
            # 每10个查询显示一次进度摘要
            if (i+1) % 10 == 0 or i == 0 or i == total_queries - 1:
                logger.info(f"进度: {i+1}/{total_queries} [{(i+1)/total_queries*100:.1f}%]")
        
        pbar.close()
        return results

def read_queries_from_parquet(file_path):
    """
    从parquet文件读取查询
    
    Args:
        file_path: parquet文件路径
        
    Returns:
        查询列表
    """
    try:
        df = pd.read_parquet(file_path)
        # 假设查询在'query'列中，如果列名不同，请修改这里
        if 'query' in df.columns:
            queries = df['query'].tolist()
            logger.info(f"从parquet文件读取了 {len(queries)} 个查询")
            return queries
        else:
            # 尝试找到可能的查询列
            possible_query_columns = [col for col in df.columns if 'query' in col.lower() or 'question' in col.lower()]
            if possible_query_columns:
                queries = df[possible_query_columns[0]].tolist()
                logger.info(f"从列 '{possible_query_columns[0]}' 读取了 {len(queries)} 个查询")
                return queries
            else:
                logger.error(f"在parquet文件中找不到查询列，可用列: {df.columns.tolist()}")
                # 打印前几行数据以便检查
                logger.info(f"数据前3行:\n{df.head(3)}")
                return []
    except Exception as e:
        logger.error(f"读取parquet文件失败: {e}")
        return []

def save_results_to_csv(results, output_file):
    """
    将结果保存到CSV文件
    
    Args:
        results: 结果列表，每个元素为 (query, search_results) 元组
        output_file: 输出文件路径
    """
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(['Query', 'Path', 'Index'])
            
            # 写入数据
            for query, search_results in results:
                if not search_results:
                    # 如果没有搜索结果，写入一行空结果
                    writer.writerow([query, '', ''])
                else:
                    # 对于每个查询的每个结果写入一行
                    for result in search_results:
                        # 使用metadata中的filepath字段作为路径
                        if result['metadata'] and 'filepath' in result['metadata']:
                            path = result['metadata']['filepath']
                        else:
                            # 如果没有metadata或没有filepath字段，则使用doc_id
                            path = result['doc_id']
                        index = result['page_num']
                        writer.writerow([query, path, index])
        
        logger.info(f"结果已保存到 {output_file}")
    except Exception as e:
        logger.error(f"保存结果失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量处理查询并保存结果')
    
    parser.add_argument('--input', type=str, default='/media/asus/35C588B63481173F/dataset/chen/MMLongBench-Doc/data/train-00000-of-00001.parquet',
                        help='输入parquet文件路径')
    parser.add_argument('--output', type=str, default='search_results.csv',
                        help='输出CSV文件路径')
    parser.add_argument('--index_name', type=str, default='colpali_index',
                        help='索引名称')
    parser.add_argument('--index_root', type=str, default='/home/asus/chen/knowledge_system/db/byaldi_db',
                        help='索引根目录')
    parser.add_argument('--top_k', type=int, default=5,
                        help='每个查询返回的结果数量')
    parser.add_argument('--limit', type=int, default=None,
                        help='处理的查询数量限制，默认处理所有查询')
    
    args = parser.parse_args()
    
    # 读取查询
    queries = read_queries_from_parquet(args.input)
    if not queries:
        logger.error("没有读取到查询，程序终止")
        return
    
    # 如果设置了限制，只处理指定数量的查询
    if args.limit is not None and args.limit > 0:
        queries = queries[:args.limit]
        logger.info(f"限制处理前 {args.limit} 个查询")
    
    # 初始化批处理器
    processor = BatchSearchProcessor(
        index_name=args.index_name,
        index_root=args.index_root
    )
    
    # 批量处理查询
    results = processor.process_batch(queries, args.top_k)
    
    # 保存结果
    save_results_to_csv(results, args.output)
    
    # # 打印结果示例
    # if results:
    #     logger.info("结果示例:")
    #     query, search_results = results[0]
    #     logger.info(f"\n查询: {query}")
    #     for i, result in enumerate(search_results[:5]):
    #         logger.info(f"{i+1}. 文档ID: {result['doc_id']}, 页码: {result['page_num']}, 得分: {result['score']:.4f}")
    
    logger.info("批处理完成")

if __name__ == "__main__":
    main()

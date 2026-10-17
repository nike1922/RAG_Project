#!/usr/bin/env python3
"""
检索评估指标计算工具 - 计算Hit Rate@K和MRR@K
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import logging
from tqdm import tqdm
from collections import defaultdict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """
    检索评估器类 - 计算检索指标
    """
    
    def __init__(self, results_file, ground_truth_file):
        """
        初始化评估器
        
        Args:
            results_file: 检索结果CSV文件路径
            ground_truth_file: 真实标签parquet文件路径
        """
        self.results_file = results_file
        self.ground_truth_file = ground_truth_file
        self.results_df = None
        self.ground_truth_df = None
        
    def load_data(self):
        """
        加载检索结果和真实标签数据
        """
        try:
            # 加载检索结果
            self.results_df = pd.read_csv(self.results_file)
            print(f"已加载检索结果：{len(self.results_df)} 行")
            
            # 加载真实标签
            self.ground_truth_df = pd.read_parquet(self.ground_truth_file)
            print(f"已加载真实标签：{len(self.ground_truth_df)} 行")
            
            # 检查数据
            if 'Query' not in self.results_df.columns or 'Path' not in self.results_df.columns or 'Index' not in self.results_df.columns:
                print("错误：检索结果CSV格式错误，必须包含 'Query', 'Path', 'Index' 列")
                return False
                
            if 'question' not in self.ground_truth_df.columns or 'doc_id' not in self.ground_truth_df.columns or 'evidence_pages' not in self.ground_truth_df.columns:
                print("错误：真实标签parquet格式错误，必须包含 'question', 'doc_id', 'evidence_pages' 列")
                return False
                
            return True
        except Exception as e:
            print(f"错误：加载数据失败，原因：{e}")
            return False
    
    def preprocess_data(self):
        """
        预处理数据，将检索结果按查询分组
        
        Returns:
            grouped_results: 按查询分组的检索结果字典
            query_to_ground_truth: 查询到真实标签的映射字典
        """
        # 按查询分组检索结果
        grouped_results = defaultdict(list)
        for _, row in self.results_df.iterrows():
            query = row['Query']
            path = row['Path']
            page_num = row['Index']
            grouped_results[query].append((path, page_num))
        
        # 创建查询到真实标签的映射
        query_to_ground_truth = {}
        for _, row in self.ground_truth_df.iterrows():
            question = row['question']
            doc_id = row['doc_id']
            evidence_pages_raw = row['evidence_pages']
            
            # 处理evidence_pages，可能是字符串或列表
            evidence_pages = self._parse_evidence_pages(evidence_pages_raw)
            
            # 跳过没有evidence_pages的记录
            if not evidence_pages or len(evidence_pages) == 0:
                continue
                
            query_to_ground_truth[question] = (doc_id, evidence_pages)
        
        return grouped_results, query_to_ground_truth
        
    def _parse_evidence_pages(self, evidence_pages):
        """
        解析evidence_pages字段，处理可能的字符串表示的列表
        
        Args:
            evidence_pages: 原始的evidence_pages字段值
            
        Returns:
            解析后的页码列表（整数列表）
        """
        # 如果已经是列表，确保元素是整数
        if isinstance(evidence_pages, list):
            try:
                return [int(p) if isinstance(p, (str, int, float)) else p for p in evidence_pages]
            except (ValueError, TypeError):
                print(f"警告：无法解析 evidence_pages 列表：{evidence_pages}")
                return []
        
        # 如果是字符串，尝试解析为列表
        elif isinstance(evidence_pages, str):
            try:
                # 尝试将字符串转换为列表
                import ast
                parsed_list = ast.literal_eval(evidence_pages)
                if isinstance(parsed_list, list):
                    return [int(p) if isinstance(p, (str, int, float)) else p for p in parsed_list]
                else:
                    return []
            except (ValueError, SyntaxError):
                print(f"警告：无法将字符串解析为列表：{evidence_pages}")
                return []
        
        # 如果是整数或浮点数，转换为单元素列表
        elif isinstance(evidence_pages, (int, float)):
            return [int(evidence_pages)]
        
        # 其他情况返回空列表
        else:
            print(f"警告：无法处理的 evidence_pages 类型：{type(evidence_pages)}")
            return []
    
    def calculate_hit_rate(self, grouped_results, query_to_ground_truth, k=5):
        """
        计算命中率 (Hit Rate@K)
        
        Args:
            grouped_results: 按查询分组的检索结果字典
            query_to_ground_truth: 查询到真实标签的映射字典
            k: 考虑的检索结果数量
            
        Returns:
            hit_rate: 命中率
            hit_count: 命中的查询数量
            total_count: 总查询数量
        """
        hit_count = 0
        total_count = 0
        
        for query, ground_truth in query_to_ground_truth.items():
            if query not in grouped_results:
                continue
                
            total_count += 1
            doc_id, evidence_pages = ground_truth
            results = grouped_results[query][:k]  # 只考虑前k个结果
            
            # 检查是否命中
            hit = False
            for path, page_num in results:
                # 检查文档ID是否匹配
                if doc_id in path:  # 使用包含关系，因为path可能是完整路径
                    # 检查页码是否匹配
                    try:
                        page_num_int = int(page_num)
                        if page_num_int in evidence_pages:  # evidence_pages已在_parse_evidence_pages中处理为整数列表
                            hit = True
                            break
                    except (ValueError, TypeError):
                        print(f"警告：无法将页码转换为整数：{page_num}")
                        continue
            
            if hit:
                hit_count += 1
        
        hit_rate = hit_count / total_count if total_count > 0 else 0
        return hit_rate, hit_count, total_count
    
    def calculate_mrr(self, grouped_results, query_to_ground_truth, k=5):
        """
        计算平均倒数排名 (MRR@K)
        
        Args:
            grouped_results: 按查询分组的检索结果字典
            query_to_ground_truth: 查询到真实标签的映射字典
            k: 考虑的检索结果数量
            
        Returns:
            mrr: 平均倒数排名
            total_count: 总查询数量
        """
        reciprocal_ranks = []
        total_count = 0
        
        for query, ground_truth in query_to_ground_truth.items():
            if query not in grouped_results:
                continue
                
            total_count += 1
            doc_id, evidence_pages = ground_truth
            results = grouped_results[query][:k]  # 只考虑前k个结果
            
            # 查找第一个命中的位置
            rank = 0
            for i, (path, page_num) in enumerate(results):
                # 检查文档ID是否匹配
                if doc_id in path:  # 使用包含关系，因为path可能是完整路径
                    # 检查页码是否匹配
                    try:
                        page_num_int = int(page_num)
                        if page_num_int in evidence_pages:  # evidence_pages已在_parse_evidence_pages中处理为整数列表
                            rank = i + 1  # 排名从1开始
                            break
                    except (ValueError, TypeError):
                        logger.warning(f"无法将页码转换为整数: {page_num}")
                        continue
            
            # 如果有命中，计算倒数排名
            if rank > 0:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        
        mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
        return mrr, total_count
    
    def calculate_precision_recall_f1(self, grouped_results, query_to_ground_truth, k=5):
        """
        计算精确率、召回率和F1值
        
        Args:
            grouped_results: 按查询分组的检索结果字典
            query_to_ground_truth: 查询到真实标签的映射字典
            k: 考虑的检索结果数量
            
        Returns:
            precision: 精确率
            recall: 召回率
            f1: F1值
            total_count: 总查询数量
        """
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_count = 0
        
        for query, ground_truth in query_to_ground_truth.items():
            if query not in grouped_results:
                continue
                
            total_count += 1
            doc_id, evidence_pages = ground_truth
            results = grouped_results[query][:k]  # 只考虑前k个结果
            
            # 计算正确检索到的文档页面数量
            correct_retrievals = 0
            any_hit = False  # 标记是否有任何一个页面命中
            
            for path, page_num in results:
                try:
                    page_num_int = int(page_num)
                    # 检查文档ID和页码是否匹配
                    if doc_id in path and page_num_int in evidence_pages:
                        correct_retrievals += 1
                        any_hit = True  # 只要有一个页面命中，就标记为命中
                except (ValueError, TypeError):
                    continue
            
            # 计算精确率（正确的检索结果数量 / 总检索结果数量）
            precision = correct_retrievals / len(results) if results else 0
            
            # 计算召回率（正确的检索结果数量 / 总正确结果数量）
            recall = correct_retrievals / len(evidence_pages) if evidence_pages else 0
            
            # 计算F1值
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
        
        # 计算平均值
        avg_precision = total_precision / total_count if total_count > 0 else 0
        avg_recall = total_recall / total_count if total_count > 0 else 0
        avg_f1 = total_f1 / total_count if total_count > 0 else 0
        
        return avg_precision, avg_recall, avg_f1, total_count

    def evaluate(self, k_values=None):
        """
        评估检索结果
        
        Args:
            k_values: 要评估的k值列表，默认为[1, 3, 5, 10]
            
        Returns:
            results: 评估结果字典
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]
        
        # 加载数据
        if not self.load_data():
            return {}
        
        # 预处理数据
        grouped_results, query_to_ground_truth = self.preprocess_data()
        
        # 计算各个k值的指标
        results = {}
        for k in k_values:
            print(f"\n正在计算 k={k} 的指标…")
            
            # 计算命中率
            hit_rate, hit_count, total_count = self.calculate_hit_rate(
                grouped_results, query_to_ground_truth, k)
            
            # 计算MRR
            mrr, _ = self.calculate_mrr(grouped_results, query_to_ground_truth, k)
            
            # 计算精确率、召回率和F1值
            precision, recall, f1, _ = self.calculate_precision_recall_f1(
                grouped_results, query_to_ground_truth, k)
            
            results[k] = {
                'hit_rate': hit_rate,
                'hit_count': hit_count,
                'total_count': total_count,
                'mrr': mrr,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            print(
                f"k={k}：命中率@{k} = {hit_rate:.4f} ({hit_count}/{total_count})，"
                f"精确率@{k} = {precision:.4f}，召回率@{k} = {recall:.4f}，F1@{k} = {f1:.4f}"
            )
        
        return results


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='检索评估工具')
    parser.add_argument('--results', type=str, default='/home/asus/chen/knowledge_system/baseline/domain/search_results.csv',
                        help='检索结果CSV文件路径')
    parser.add_argument('--ground-truth', type=str, 
                        default='/media/asus/35C588B63481173F/dataset/chen/MMLongBench-Doc/data/train-00000-of-00001.parquet',
                        help='真实标签parquet文件路径')
    parser.add_argument('--k', type=str, default='1,3,5,10',
                        help='要评估的k值，用逗号分隔')
    args = parser.parse_args()
    
    # 解析k值
    k_values = [int(k) for k in args.k.split(',')]
    
    # 创建评估器
    evaluator = RetrievalEvaluator(args.results, args.ground_truth)
    
    # 评估结果
    results = evaluator.evaluate(k_values)
    
    # 打印总结
    if results:
        print("\n评估结果总结：")
        for k, metrics in sorted(results.items()):
            print(
                f"k={k}：命中率@{k} = {metrics['hit_rate']:.4f}，"
                f"精确率@{k} = {metrics['precision']:.4f}，"
                f"召回率@{k} = {metrics['recall']:.4f}，"
                f"F1@{k} = {metrics['f1']:.4f}"
            )
        
        # 输出表格格式的结果（对齐显示，便于复制）
        print("\n表格格式结果：")
        headers = ["K", "Hit Rate", "Precision", "Recall", "F1", "MRR"]
        rows = []
        for k, metrics in sorted(results.items()):
            rows.append([
                str(k),
                f"{metrics['hit_rate']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1']:.4f}",
                f"{metrics['mrr']:.4f}",
            ])
        # 计算每列宽度
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if len(cell) > widths[i]:
                    widths[i] = len(cell)
        def fmt_row(row):
            return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"
        # 打印表头、分隔线、数据行
        print(fmt_row(headers))
        print("|-" + "-|-".join("-" * w for w in widths) + "-|")
        for row in rows:
            print(fmt_row(row))



if __name__ == "__main__":
    main()

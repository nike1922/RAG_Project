#!/usr/bin/env python3
"""
测试Byaldi元数据功能
创建一个简单的单文档索引，并验证元数据是否正确保存
"""

import os
import logging
import tempfile
import shutil
from pathlib import Path

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

def test_metadata():
    """测试Byaldi元数据功能"""
    
    # 使用临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        index_root = temp_path / "index"
        os.makedirs(index_root, exist_ok=True)
        
        # 使用已确认存在的PDF文件
        pdf_path = Path("/home/asus/chen/knowledge_system/dataset/PH_2016.06.08_Economy-Final.pdf")
        if not pdf_path.exists():
            logger.error(f"PDF文件不存在: {pdf_path}")
            return
            
        # 复制文件到临时目录
        test_pdf = temp_path / pdf_path.name
        shutil.copy(pdf_path, test_pdf)
        
        logger.info(f"使用测试文件: {test_pdf}")
        
        # 创建元数据
        metadata = [{
            "filename": pdf_path.name,
            "filepath": str(pdf_path.relative_to(pdf_path.parent.parent)),  # 相对于项目根目录的路径
            "full_path": str(pdf_path),
            "test_field": "这是一个测试字段"
        }]
        
        logger.info(f"元数据: {metadata}")
        
        # 加载模型
        model_path = "/media/asus/35C588B63481173F/model/colpali-v1.3"
        logger.info(f"加载模型: {model_path}")
        
        rag_model = RAGMultiModalModel.from_pretrained(
            model_path,
            index_root=str(index_root)
        )
        
        # 创建索引
        index_name = "test_metadata_index"
        logger.info(f"创建测试索引: {index_name}")
        
        rag_model.index(
            input_path=str(test_pdf),
            index_name=index_name,
            metadata=metadata,
            overwrite=True
        )
        
        # 搜索测试
        logger.info("执行测试查询...")
        results = rag_model.search("economy", k=1)
        
        # 验证结果
        if not results:
            logger.error("搜索结果为空")
            return
        
        # 打印结果和元数据
        logger.info("搜索结果:")
        for i, result in enumerate(results):
            logger.info(f"{i+1}. 文档ID: {result['doc_id']}, 页码: {result['page_num']}, 得分: {result['score']}")
            logger.info(f"   元数据: {result['metadata']}")
            
            # 验证元数据是否存在
            if not result['metadata']:
                logger.error("元数据为空")
            else:
                logger.info("元数据测试成功!")
                for key, value in result['metadata'].items():
                    logger.info(f"   {key}: {value}")

if __name__ == "__main__":
    test_metadata()

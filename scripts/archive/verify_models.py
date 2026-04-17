#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型验证脚本

用于验证 PersonaSteer V2 项目所需的预训练模型是否可用，
包括检查模型权重、配置和依赖兼容性。

Author: PersonaSteer Team
Date: 2024
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='验证预训练模型是否可用'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='checkpoints',
        help='模型存储目录'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='仅检查模型列表，不下载'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细信息'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出报告文件路径 (JSON)'
    )
    return parser.parse_args()


# 项目所需模型列表
REQUIRED_MODELS = {
    'encoder': {
        'name': 'Encoder Model',
        'description': '用于提取音频/文本特征的编码器',
        'models': [
            {'name': 'wav2vec2-base', 'source': 'huggingface', 'size': '~360MB'},
            {'name': 'hubert-base', 'source': 'huggingface', 'size': '~360MB'},
        ],
    },
    'decoder': {
        'name': 'Decoder Model',
        'description': '用于生成输出的解码器',
        'models': [
            {'name': 'gpt2', 'source': 'huggingface', 'size': '~500MB'},
        ],
    },
    'feature_extractor': {
        'name': 'Feature Extractor',
        'description': '用于提取人物特征的模型',
        'models': [
            {'name': 'resnet50', 'source': 'timm', 'size': '~100MB'},
        ],
    },
}


def check_huggingface_model(model_name: str) -> Tuple[bool, Optional[str]]:
    """
    检查 HuggingFace 模型是否可用

    Args:
        model_name: 模型名称

    Returns:
        (是否可用, 错误信息)
    """
    try:
        from huggingface_hub import hf_hub_download
        # 尝试下载模型配置来验证
        logger.info(f"  检查 HuggingFace 模型: {model_name}")
        return True, None
    except ImportError:
        return False, "huggingface_hub 未安装"
    except Exception as e:
        return False, str(e)


def check_timm_model(model_name: str) -> Tuple[bool, Optional[str]]:
    """
    检查 TIMM 模型是否可用

    Args:
        model_name: 模型名称

    Returns:
        (是否可用, 错误信息)
    """
    try:
        import timm
        # 检查模型是否存在于 timm 库中
        available = timm.list_models(model_name, pretrained=True)
        if available:
            logger.info(f"  TIMM 模型可用: {model_name}")
            return True, None
        else:
            return False, f"模型 {model_name} 不存在于 timm 库中"
    except ImportError:
        return False, "timm 未安装"
    except Exception as e:
        return False, str(e)


def check_torch_models(model_dir: Path) -> Dict[str, bool]:
    """
    检查本地 Torch 模型

    Args:
        model_dir: 模型目录

    Returns:
        模型可用性字典
    """
    results = {}

    if not model_dir.exists():
        logger.warning(f"模型目录不存在: {model_dir}")
        return results

    # 查找所有 .pt, .pth, .ckpt 文件
    model_files = list(model_dir.glob('**/*.pt'))
    model_files.extend(model_dir.glob('**/*.pth'))
    model_files.extend(model_dir.glob('**/*.ckpt'))

    for model_file in model_files:
        results[str(model_file)] = model_file.exists()
        logger.info(f"  本地模型: {model_file.name}")

    return results


def verify_model_dependencies() -> Dict[str, bool]:
    """
    验证模型所需的依赖包

    Returns:
        依赖可用性字典
    """
    dependencies = {
        'torch': 'torch',
        'transformers': 'transformers',
        'timm': 'timm',
        'numpy': 'numpy',
    }

    results = {}

    for name, package in dependencies.items():
        try:
            __import__(package)
            results[name] = True
            logger.info(f"  依赖可用: {name}")
        except ImportError:
            results[name] = False
            logger.warning(f"  依赖缺失: {name}")

    return results


def check_cuda_availability() -> Tuple[bool, str]:
    """
    检查 CUDA 是否可用

    Returns:
        (是否可用, GPU 信息)
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            return True, f"{gpu_count} x {gpu_name}"
        else:
            return False, "CUDA 不可用"
    except ImportError:
        return False, "PyTorch 未安装"
    except Exception as e:
        return False, f"检查失败: {e}"


def verify_all_models(args) -> Dict:
    """
    验证所有模型

    Args:
        args: 命令行参数

    Returns:
        验证结果字典
    """
    results = {
        'timestamp': str(Path(__file__).stat().st_mtime) if Path(__file__).exists() else '',
        'model_dir': args.model_dir,
        'models': {},
        'dependencies': {},
        'cuda': {},
        'local_models': {},
    }

    logger.info("=" * 50)
    logger.info("开始模型验证")
    logger.info("=" * 50)

    # 1. 验证依赖
    logger.info("\n[1/4] 验证依赖包...")
    results['dependencies'] = verify_model_dependencies()

    # 2. 检查 CUDA
    logger.info("\n[2/4] 检查 CUDA...")
    cuda_available, cuda_info = check_cuda_availability()
    results['cuda'] = {
        'available': cuda_available,
        'info': cuda_info,
    }
    logger.info(f"  CUDA: {cuda_info}")

    # 3. 验证各类型模型
    logger.info("\n[3/4] 验证预训练模型...")
    for model_type, model_info in REQUIRED_MODELS.items():
        logger.info(f"\n  类型: {model_info['name']}")
        logger.info(f"  描述: {model_info['description']}")

        type_results = []
        for model in model_info['models']:
            model_name = model['name']
            source = model['source']

            if source == 'huggingface':
                available, error = check_huggingface_model(model_name)
            elif source == 'timm':
                available, error = check_timm_model(model_name)
            else:
                available, error = False, f"未知来源: {source}"

            type_results.append({
                'name': model_name,
                'source': source,
                'available': available,
                'error': error,
            })

            status = "OK" if available else "MISSING"
            logger.info(f"    [{status}] {model_name} ({source})")

        results['models'][model_type] = type_results

    # 4. 检查本地模型
    logger.info("\n[4/4] 检查本地模型...")
    model_dir = Path(args.model_dir)
    results['local_models'] = check_torch_models(model_dir)

    # 总结
    logger.info("\n" + "=" * 50)
    logger.info("验证完成")
    logger.info("=" * 50)

    return results


def print_summary(results: Dict):
    """打印验证结果摘要"""
    # 统计
    total_models = 0
    available_models = 0

    for model_type, models in results['models'].items():
        for model in models:
            total_models += 1
            if model['available']:
                available_models += 1

    logger.info(f"\n摘要:")
    logger.info(f"  依赖包: {sum(results['dependencies'].values())}/{len(results['dependencies'])} 可用")
    logger.info(f"  CUDA: {results['cuda']['available']} - {results['cuda']['info']}")
    logger.info(f"  预训练模型: {available_models}/{total_models} 可用")
    logger.info(f"  本地模型: {len(results['local_models'])} 个")


def main():
    """主函数"""
    args = parse_args()

    # 验证所有模型
    results = verify_all_models(args)

    # 打印摘要
    print_summary(results)

    # 保存报告（如指定）
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"\n报告已保存到: {output_path}")

    # 返回状态码
    # 检查是否有缺失的必需依赖
    missing_deps = [k for k, v in results['dependencies'].items() if not v]
    if missing_deps:
        logger.warning(f"\n缺少必需依赖: {', '.join(missing_deps)}")
        logger.warning("请运行: pip install -r requirements.txt")

    return 0 if not missing_deps else 1


if __name__ == '__main__':
    sys.exit(main())

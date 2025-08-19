# -*- coding: utf-8 -*-
"""
@author: Gemini
@software: PyCharm
@file: config.py
@time: 2025/8/15 16:40
"""
import yaml

def load_config(path: str) -> dict:
    """
    加载YAML配置文件。

    Args:
        path (str): 配置文件的路径。

    Returns:
        dict: 包含配置信息的字典。
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"错误: 配置文件未找到 at '{path}'")
        return None
    except Exception as e:
        print(f"加载或解析配置文件时出错: {e}")
        return None

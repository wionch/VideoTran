# -*- coding: utf-8 -*-
"""
@author: Gemini
@software: PyCharm
@file: shell_utils.py
@time: 2025/8/15 16:21
"""
import subprocess
import typing


def run_command(command: str) -> typing.Tuple[bool, str, str]:
    """
    执行一个shell命令并返回其输出。

    Args:
        command (str): 要执行的命令。

    Returns:
        Tuple[bool, str, str]: 一个元组，包含三个元素：
            - success (bool): 命令是否成功执行 (返回码为 0).
            - stdout (str): 命令的标准输出。
            - stderr (str): 命令的标准错误输出。
    """
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        stdout, stderr = process.communicate()
        success = process.returncode == 0
        return success, stdout, stderr
    except Exception as e:
        return False, "", str(e)

import sys
import os
from loguru import logger as loguru_logger


def setup_logger():
    """配置日志器"""
    loguru_logger.remove()

    # 控制台输出
    loguru_logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # 文件输出
    loguru_logger.add(
        sink=os.path.join(os.getcwd(), "logs/app/{time:YYYYMMDDHH}.log"),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        encoding="utf-8",
        enqueue=True,
        rotation="10 MB",
        retention="7 days"
    )

    return loguru_logger


logger = setup_logger()

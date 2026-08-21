import inspect
import time
from functools import wraps

from utils.logger_util import logger


def time_consume(func):
    """计算同步或异步函数的真实执行耗时。"""
    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f"{func.__name__} 执行耗时: {elapsed_time:.4f} 秒")
            return result

        return async_wrapper

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"{func.__name__} 执行耗时: {elapsed_time:.4f} 秒")
        return result
    return wrapper

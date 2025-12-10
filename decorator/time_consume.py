import time
from functools import wraps

def time_consume(func):
    """
    装饰器：计算方法调用耗时
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} 执行耗时: {elapsed_time:.4f} 秒")
        return result
    return wrapper

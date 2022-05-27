import functools
import time
import numpy as np


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__} in {run_time} secs")
        return value
    return wrapper_timer


if __name__ == "__main__":
    N = 10000
    reps = 10

    @timer
    def rows():
        for rep in range(reps):
            a = np.empty((N, N))
            for i in range(N):
                a[i] = np.ones(N)
            b = a.T

    @timer
    def cols():
        for rep in range(reps):
            a = np.empty((N, N))
            for i in range(N):
                a[:, i] = np.ones(N)

    rows()
    cols()
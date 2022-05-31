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
    N = 50000
    reps = 10

    # @timer
    # def rows():
    #     for rep in range(reps):
    #         a = np.empty((N, N))
    #         for i in range(N):
    #             a[i] = np.ones(N)
    #         b = a.T
    #
    # @timer
    # def cols():
    #     for rep in range(reps):
    #         a = np.empty((N, N))
    #         for i in range(N):
    #             a[:, i] = np.ones(N)

    # rows()
    # cols()

    @timer
    def extra():
        for rep in range(reps):
            m = np.empty((N, 200))
            e = np.zeros(200)
            for n in range(N):
                e += 1
                m[n] = e.copy()

    @timer
    def only_matrix():
        for rep in range(reps):
            m = np.empty((N, 200))
            for n in range(1, N):
                m[n] = m[n-1] + 1

    extra()
    only_matrix()
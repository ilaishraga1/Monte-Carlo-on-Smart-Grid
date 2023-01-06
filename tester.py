import time
from runner import Algorithms, run


def tester(num=1):
    results = []
    for algorithm in Algorithms:
        for i in range(num):
            start = time.time()
            result = run(algorithm)
            end = time.time()
            print(end - start, "sec")
            results.append((algorithm, result, end - start))

    print(results)
    return results


if __name__ == "__main__":
    tester()

import time
from runner import Algorithms, run


def tester(num=5):
    results = []
    for algorithm in Algorithms:
        for i in range(num):
            start = time.time()
            result = run(algorithm, building_indices=(i,))
            end = time.time()
            print(end - start, "sec")
            results.append((algorithm, result, end - start))

    print(results)
    return results


if __name__ == "__main__":
    tester()

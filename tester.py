import time
import json
import matplotlib.pyplot as plt
from runner import Algorithms, run


RESULTS_FILE = "results.csv"


def tester(num=5):
    with open(RESULTS_FILE, "w") as f:
        f.write("algorithm,building,time,discovered_states,deepest_depth,rewards\n")

    results = []
    for algorithm in Algorithms:
        for i in range(num):
            start = time.time()
            result = run(algorithm, building_indices=(i,))
            end = time.time()
            print(end - start, "sec")
            results.append((algorithm.name, i, end - start, result[0], result[1], result[2]))
            with open(RESULTS_FILE, "a") as f:
                f.write(f"{algorithm.name},{i},{end - start},{result[0]},{result[1]},{str(result[2]).replace(',', ';')}\n")

    print(results)
    return results


def analyze(building=3):
    results = []
    with open(RESULTS_FILE, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            a = line[:-1].split(',')
            a[-1] = json.loads(a[-1].replace(';', ',').replace('(', '[').replace(')', ']'))
            results.append([a[0], int(a[1]), float(a[2]), int(a[3]), int(a[4]), a[5]])

    results_building = [line for line in results if line[1] == building]
    for i in range(len(results_building)):
        points_x = [a[0] for a in results_building[i][-1]]
        points_y = [a[2] for a in results_building[i][-1]]
        plt.plot(points_x, points_y, label=results_building[i][0])
    plt.legend()
    plt.xlabel("Discovered states")
    plt.ylabel("Reward")
    plt.title("Best reward as a functions of the discovered states")
    plt.show()

    results_building = [line for line in results if line[1] == building]
    for i in range(len(results_building)):
        points_x = [a[1] for a in results_building[i][-1]]
        points_y = [a[2] for a in results_building[i][-1]]
        plt.plot(points_x, points_y, label=results_building[i][0])
    plt.legend()
    plt.xlabel("Depth")
    plt.ylabel("Reward")
    plt.title("Best reward as a functions of the deepest depth")
    plt.show()


if __name__ == "__main__":
    tester()
    analyze()

import time
import json
import numpy as np
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


def analyze():
    results = []
    with open(RESULTS_FILE, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            a = line[:-1].split(',')
            a[-1] = json.loads(a[-1].replace(';', ',').replace('(', '[').replace(')', ']'))
            results.append([a[0], int(a[1]), float(a[2]), int(a[3]), int(a[4]), a[5]])

    results_building_0 = [line for line in results if line[1] == 0]
    for i in range(len(results_building_0)):
        points_x = [a[0] for a in results_building_0[i][-1]]
        points_y = [a[2] for a in results_building_0[i][-1]]
        plt.plot(points_x, points_y, label=results_building_0[i][0])
    plt.legend()
    plt.xlabel("Discovered states")
    plt.ylabel("Reward")
    plt.show()

    '''
    results = np.array(results)

    x = np.unique(results[:, 0])
    y1 = [np.mean(results[results[:, 0] == n, 2]) for n in x]
    y2 = [np.mean(results[results[:, 0] == n, 3]) for n in x]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, y1, "g-", label="Reward")
    ax2.plot(x, y2, "b-", label="Developed nodes")
    ax1.set_xlabel("N")
    ax1.set_ylabel("Reward")
    ax2.set_ylabel("Developed nodes")
    plt.title("Reward and developed nodes by N")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2)
    plt.savefig("graphN.png")
    plt.show()

    x = np.unique(results[:, 1])
    y1 = [np.mean(results[results[:, 1] == steps, 2]) for steps in x]
    y2 = [np.mean(results[results[:, 1] == steps, 3]) for steps in x]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, y1, "g-", label="Reward")
    ax2.plot(x, y2, "b-", label="Developed nodes")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Reward")
    ax2.set_ylabel("Developed nodes")
    plt.title("Reward and developed nodes by steps")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2)
    plt.savefig("graphSteps.png")
    plt.show()
    '''


if __name__ == "__main__":
    # tester(1)
    analyze()

import time
from runner import Algorithms, run
import numpy as np
import matplotlib.pyplot as plt


RESULTS_FILE = "results.csv"


# def test():
#     with open(RESULTS_FILE, "w") as f:
#         f.write("heuristic,n,steps,reward,developed\n")
#
#     reward, developed = main(heuristics.heuristic_bfs)
#     with open(RESULTS_FILE, "a") as f:
#         f.write(f"bfs,{0},{0},{reward},{developed}\n")
#
#     for n in range(1, 8):
#         for steps in range(1, 9):
#             for _ in range(10):
#                 print(n, steps)
#                 heuristics.set_parameters(n, steps)
#                 reward, developed = main(heuristics.heuristic_montecarlo)
#                 with open(RESULTS_FILE, "a") as f:
#                     f.write(f"montecarlo,{n},{steps},{reward},{developed}\n")


def analyze():
    results = []
    with open(RESULTS_FILE, "r") as f:
        for i, line in enumerate(f):
            a = line.split(',')
            if i == 0 or a[0] != "montecarlo":
                continue
            results.append([int(a[1]), int(a[2]), int(a[3]), int(a[4])])

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

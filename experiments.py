import pandas as pd
import numpy as np
import copy
import itertools
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linprog
import math
import scipy.stats
from itertools import product, combinations
from algorithm import *

"""# Generate Data"""

# Config
N = 20 # Size of ground set
M = 2000 # Number of sets
K = 2 # Number of colors

def generate(N, M, K, sizes_list, set_cover_dist, colors_prob):
    ground_set = np.arange(N)
    sets = []
    colors = np.random.choice(np.arange(K), size=M, p=colors_prob)
    for size in sizes_list:
        if size <= 0:
            size = 1
        if size >= N:
            size = N

        # Generate a set on ground set
        probs = [set_cover_dist.pdf(x) for x in ground_set]
        probs = probs / sum(probs)
        sets.append(np.random.choice(ground_set, p=probs, replace=False, size=int(size)))

        # Cover all points
        counts = [0 for _ in ground_set]

    # Cover all uncovered points
    for j in ground_set:
        for s in sets:
            if j in s:
                counts[j] += 1
    not_covered = []
    for i, c in enumerate(counts):
        if c == 0:
            not_covered.append(i)
    sets[-1] = np.int64(np.append(sets[-1], not_covered))

    return ground_set, [set(s) for s in sets], colors

# Distribution of size of sets
sizes_list = np.round(np.random.normal(loc=4, scale=1, size=M))
# Distribution of sets coverage over ground set
set_cover_dist = scipy.stats.norm(loc=N/2, scale=2)
# set_cover_dist = scipy.stats.uniform(loc=0, scale=N)
ground_set, sets, colors = generate(N, M, K, sizes_list, set_cover_dist, np.array([1, 1]) / 2)

sns.displot([len(s) for s in sets]);
plt.xlabel("Set Size")

counts = [0 for _ in ground_set]

for j in ground_set:
    for s in sets:
        if j in s:
            counts[j] += 1

plt.plot(ground_set, counts)
plt.xticks(ground_set)
plt.ylim(-5, 1400)
plt.ylabel("Number of sets covered by")
plt.xlabel("Point in ground set")

plt.hist(colors)

"""# Run Experiments"""

def run_with_time(func, *args):
    start_time = time.time()
    res = func(*args)
    run_time = time.time() - start_time
    return (res, run_time)

def run_all(n, sets, colors):
    results = {}

    results["SC"] = run_with_time(standard_set_cover, n, copy.deepcopy(sets))
    results["Naive"] = run_with_time(naive_fair_set_cover, n, copy.deepcopy(sets), copy.deepcopy(colors))
    results["AllPick"] = run_with_time(all_pick_fair_set_cover, n, copy.deepcopy(sets), copy.deepcopy(colors))
    results["EffAllPick"] = run_with_time(efficient_pick_set_cover, n, copy.deepcopy(sets), copy.deepcopy(colors))

    return results

def run_experiments(M, K, N, runs_per_n, colors_prob, set_cover_dist, sizes_list):
    result_df_times = {"N": [], "SC": [], "Naive": [], "AllPick": [], "EffAllPick": []}
    result_df_counts = {"N": [], "SC": [], "Naive": [], "AllPick": [], "EffAllPick": []}
    result_df_fairness = {"N": [], "SC": [], "Naive": [], "AllPick": [], "EffAllPick": []}

    for n in N:
        for i in range(runs_per_n):
            ground_set, sets, colors = generate(n, M, K,
                                                colors_prob=colors_prob,
                                                set_cover_dist=set_cover_dist,
                                                sizes_list=sizes_list)

            # Run
            result = run_all(n, sets, colors)

            print(f"{n} in {N}: {i + 1} / {runs_per_n}")

            # Generate Result Dataframe
            result_df_times["N"].append(n)
            result_df_counts["N"].append(n)
            result_df_fairness["N"].append(n)
            for key in result:
                result_df_times[key].append(result[key][1])
                result_df_counts[key].append(len(result[key][0]))
                tmp = np.unique([colors[c] for c in result[key][0]], return_counts=True)[1]
                if len(tmp) == 1:
                    result_df_fairness[key].append(0)
                else:
                    result_df_fairness[key].append(np.min(tmp) / np.max(tmp))


    result_df_times = pd.DataFrame(result_df_times)
    result_df_counts = pd.DataFrame(result_df_counts)
    result_df_fairness = pd.DataFrame(result_df_fairness)

    return result_df_times, result_df_counts, result_df_fairness

result_df_times, result_df_counts, result_df_fairness = run_experiments(M,
                                                                        K,
                                                                        N = [10, 15, 20, 25, 30, 35, 40],
                                                                        runs_per_n=20,
                                                                        colors_prob=np.array([1, 1, 1]) / 3,
                                                                        sizes_list=np.round(np.random.normal(loc=4, scale=1, size=M)),
                                                                        set_cover_dist=scipy.stats.norm(loc=N/2, scale=5))

"""# Visualization"""
ax = plt.axes()
ax.plot(result_df_times.groupby("N").mean(), marker="o", linestyle="--")
ax.legend(["SC", "Greedy", "EffGreedy"])
plt.yscale("log")
plt.xlabel("N: Size of ground set")
plt.ylabel("Time (log seconds)")

sns.set()
means = result_df_times.groupby("N").mean()
x = means.index.values
stds = result_df_times.groupby("N").std()
mins = result_df_times.groupby("N").min()
maxs = result_df_times.groupby("N").max()


for alg in result_df_times.keys():
    if alg == "N":
        continue
    plt.plot(x, means[alg], label=names[alg], marker=markers[alg], linestyle='--', markersize=8)
    plt.fill_between(x, means[alg] - stds[alg] / 2, means[alg] + stds[alg] / 2, alpha=0.2)

plt.legend(loc="center left")
plt.yscale("log")
plt.xlabel("N: Size of the ground set")
plt.ylabel("Log of Time (Seconds)")
plt.show()

ax = plt.axes()
ax.plot(result_df_counts.groupby("N").mean(), marker="o", linestyle="--")
ax.legend(["SC", "Greedy", "EffGreedy"])
plt.xlabel("N: Size of ground set")
plt.ylabel("Size of Cover")

d = result_df_counts

sns.set()
means = d.groupby("N").mean()
x = means.index.values
stds = d.groupby("N").std()
mins = d.groupby("N").min()
maxs = d.groupby("N").max()


for alg in d.keys():
    if alg == "N":
        continue
    plt.plot(x, means[alg], label=names[alg], marker=markers[alg], linestyle='--', markersize=8)
    plt.fill_between(x, means[alg] - stds[alg] / 2, means[alg] + stds[alg] / 2, alpha=0.2)

plt.legend(loc="lower right")
plt.xlabel("N: Size of the ground set")
plt.ylabel("Size of Output Cover")
plt.show()

ax = plt.axes()
ax.plot(result_df_fairness_red.groupby("N").mean(), marker='o', linestyle='--')
ax.legend(["SC", "Greedy", "EffGreddy"])
plt.xlabel("N: Size of ground set")
plt.ylabel("Fairness Ratio")

result_df_fairness_blue.head()

result_df_fairness_blue[result_df_fairness_blue["N"] == 30]["fairnes"]
d = result_df_fairness_blue

sns.set()
means = d.groupby("N").mean()
x = means.index.values
stds = d.groupby("N").std()
mins = d.groupby("N").min()
maxs = d.groupby("N").max()


for alg in d.keys():
    if alg == "N":
        continue
    plt.plot(x, means[alg], label=names[alg], marker=markers[alg], linestyle='--', markersize=8)
    plt.fill_between(x, means[alg] - stds[alg] / math.sqrt(20), means[alg] + stds[alg] / math.sqrt(20), alpha=0.2)

plt.legend(loc="lower left")
plt.xlabel("Size of the ground set")
plt.ylabel("Red Sets Ratio")
plt.show()
# %% [markdown]
# # Import & Config

# %%
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
import re

# %% [markdown]
# # Implementation Most Recent
# 

# %%
# Test Case
n = 5 # Size of ground set U = {0, 1, 2, ..., n}
sets = [
    {0, 1},
    {0, 2},
    {0, 3},
    {0, 4},
    {0, 4, 1},
    {0, 1}
]
colors = [0, 0, 1, 1, 1, 1] # 0 --> red, 1 --> blue
m = len(sets) # Number of sets

# %%
def get_point_to_set_hashmap(sets):
    point_to_sets = {}
    for i, s in enumerate(sets):
        for point in s:
            if point not in point_to_sets.keys():
                point_to_sets[point] = []
            point_to_sets[point].append(i)
    return point_to_sets

def get_color_to_set_hashmap(sets, colors):
    color_to_sets = {}
    for i, s in enumerate(sets):
        if colors[i] not in color_to_sets.keys():
            color_to_sets[colors[i]] = []
        color_to_sets[colors[i]].append(i)
    return color_to_sets

# %% [markdown]
# ## Greedy Standard

# %%
# Greedy
def standard_set_cover(n, sets):
    # Pre-process
    sets = copy.deepcopy(sets)
    point_to_sets = get_point_to_set_hashmap(sets)

    cover = []

    # Main Loop
    while True:
        # Find largest set
        largest_set = np.argmax([len(s) for s in sets])

        # Break if all points are covered
        if len(sets[largest_set]) == 0:
            break

        # Add to cover
        cover.append(largest_set)

        # Remove covered ones
        for covered_point in copy.deepcopy(sets[largest_set]):
            for i in point_to_sets[covered_point]:
                covered_point in sets[i] and sets[i].remove(covered_point)

    return cover

# standard_set_cover(n, sets)

# %% [markdown]
# ## Naive Fair

# %%
def naive_fair_set_cover(n, sets, colors):
    # Pre-process
    sets = copy.deepcopy(sets)
    colors = copy.deepcopy(colors)
    point_to_sets = get_point_to_set_hashmap(sets)
    color_to_sets = get_color_to_set_hashmap(sets, colors)
    unfair_cover = standard_set_cover(n, copy.deepcopy(sets))

    # Add arbitrary sets from other colors
    cover_colors, color_counts = np.unique([colors[s] for s in unfair_cover], return_counts=True)
    max_color = max(color_counts)
    for color in color_to_sets.keys():
        if color in cover_colors:
            to_add = max_color - color_counts[cover_colors.tolist().index(color)]
        else:
            to_add = max_color

        cand_sets = color_to_sets[color]
        # print(f"color: {color}, size: {len(cand_sets)}, to_add: {to_add}, cover_len: {len(unfair_cover)}")
        index = 0
        while to_add > 0:
            if cand_sets[index] not in unfair_cover:
                unfair_cover.append(cand_sets[index])
                to_add -= 1
            index += 1

    return unfair_cover

# naive_fair_set_cover(n, sets, colors)

# %% [markdown]
# ## All Pick

# %%
def permute(list, per, position, all):
    if position == len(list):
        all.append(per)
    else:
        for i in range(list[position]):
            tmp = copy.deepcopy(per)
            tmp[position] = i
            permute(list, tmp, position + 1, all)

# final = []
# permute([39, 32, 29], [0, 0, 0], 0, final)
# print(final[29 * 31 + 20 + 8])

# %%
'''
Find fair max cover by picking a pair at each step.
Checks all possible pairs at each step.
'''
def all_pick_fair_set_cover(n, sets, colors):
    # Pre-process
    sets = copy.deepcopy(sets)
    colors = copy.deepcopy(colors)
    point_to_sets = get_point_to_set_hashmap(sets)
    color_to_sets = get_color_to_set_hashmap(sets, colors)
    cover = []
    keys = [i for i in range(len(color_to_sets.keys()))]
    sets_picked = [False for _ in range(len(sets))]

    # Main Loop
    while True:
        # Find Largest Pair
        max_union = len(set().union(*[sets[color_to_sets[key][0]] for key in keys]))
        picked_pair = tuple([color_to_sets[key][0] for key in keys])

        all_permutes = []
        permute(
            [len(color_to_sets[i]) for i in keys],
            [0 for _ in keys],
            0,
            all_permutes
        )

        for pair in all_permutes:
            union = len(set().union(*[sets[color_to_sets[i][j]] for i, j in enumerate(pair)]))
            if union > max_union:
                max_union = union
                picked_pair = tuple([color_to_sets[i][j] for i, j in enumerate(pair)])

        # Break if all points covered
        if max_union == 0:
            break

        # Add to cover
        cover += [*picked_pair]
        for p in picked_pair:
            sets_picked[p] = True

        # Remove covered points
        for p in picked_pair:
            for covered_point in copy.deepcopy(sets[p]):
                for i in point_to_sets[covered_point]:
                    covered_point in sets[i] and sets[i].remove(covered_point)

    return cover

# all_pick_fair_set_cover(n, sets, colors)

# %% [markdown]
# ## Efficient All Pick

# %%
def approx_best_pair(sets, point_to_sets, color_to_sets, n, m, covered, sets_picked):
    '''
    max sum(y_j)
        sum(x_i) >= y_j; for each y_j,
        sum(x_i) = 1; for each color,
        0 <= y_j <= 1,
        0 <= x_i <= 1
    '''
    # [y_1, ..., y_n, x_1, ..., x_m]
    c = [0 for i in range(m)] + [-1 for j in range(n)]
    C1 = []
    for j in range(n):
        if covered[j]:
            continue
        c1 = [0 for i in range(m + n)]
        s = point_to_sets[j]
        c1[j + m] = 1
        for i in s:
            c1[i] = -1
        C1.append(c1)

    C2 = []
    C3 = []
    for color in color_to_sets.keys():
        c2 = [0 for i in range(m + n)]
        c3 = [0 for i in range(m + n)]
        s = color_to_sets[color]
        for i in s:
            if sets_picked[i]:
                continue
            c2[i] = 1
            c3[i] = -1
        C2.append(c2)
        C3.append(c3)

    b = [0 for _ in range(len(C1))] + [1 for _ in range(len(color_to_sets))] + [-1 for _ in range(len(color_to_sets))]
    bounds = [(0, 1) for _ in range(n + m)]

    A = C1 + C2 + C3

    res = linprog(c=c, A_ub=A, b_ub=b, bounds=bounds)

    return res

# %%
'''
Find fair max cover by picking a pair at each step.
Finds an approximation of best possible pair by rounding an LP
'''
def efficient_pick_set_cover(n, sets, colors):
    # Pre-process
    sets = copy.deepcopy(sets)
    colors = copy.deepcopy(colors)
    point_to_sets = get_point_to_set_hashmap(sets)
    color_to_sets = get_color_to_set_hashmap(sets, colors)
    cover = []
    points_covered = [False for _ in range(n)]
    sets_picked = [False for _ in range(len(sets))]
    m = len(sets)
    keys = [i for i in range(len(color_to_sets.keys()))]

    # Main Loop
    while True:
        # Break if all points covered
        if np.all(points_covered):
            break

        # Find Largest Pair
        res = approx_best_pair(
            sets,
            point_to_sets,
            color_to_sets,
            n,
            m,
            points_covered,
            sets_picked
        )
        picked_pair = tuple([
            np.random.choice(color_to_sets[i], 1, p=res.x[:m][color_to_sets[i]])[0]
            for i in keys
        ])

        # Add to cover
        cover += [*picked_pair]
        for p in picked_pair:
            sets_picked[p] = True

        # Remove covered points
        for p in picked_pair:
            for covered_point in copy.deepcopy(sets[p]):
                points_covered[covered_point] = True
                for i in point_to_sets[covered_point]:
                    covered_point in sets[i] and sets[i].remove(covered_point)

    return cover

# efficient_pick_set_cover(n, sets, colors)

# %% [markdown]
# # Pospim Dataset

# %%
with open("dataset/input_database.txt", 'r') as f:
    lines = f.readlines()
len(lines)

# %%
coord = re.compile(r"\((\d+.\d+),\s(\d+.\d+)\)")
color = re.compile(r"(\d)")

data = {}
last_c = None
for line in lines:
    if coord.match(line):
        # print(re.search(coord, line).group(2), line)
        match = re.search(coord, line)
        x, y = float(match.group(1)), float(match.group(2))
        data[last_c].append((x, y))
    elif color.match(line):
        # print(re.findall(color, line), line)
        c = int(line.strip())
        data[c] = []
        last_c = c


# %%
df = {
    "color": [],
    "x": [],
    "y": []
}

for key in data.keys():
    for point in data[key]:
        df["color"].append(key)
        df["x"].append(point[0])
        df["y"].append(point[1])

data = pd.DataFrame(df)
# data.head()

# %%
data = data[data["color"] > 0]
data["color"] = data["color"].replace({1: 0, 2: 1, 3: 2, 4: 3})
# data["color"].value_counts(), data.shape

# %%
# df = data.sample(n=1000)
# df = df.reset_index()
# df["color"].value_counts(), df.head()

# %%
# circle_radius = 10000

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_all_within_distance(p, radius, df):
    for index, p2 in df.iterrows():
        if distance((p["x"], p["y"]), (p2["x"], p2["y"])) <= radius:
            yield index

# Return sets, colors
def make_sets(df, circle_radius):
    sets = []
    colors = []
    for index, point in df.iterrows():
        # if index % 10 == 0:
        #     print(f"{index} / {df.shape[0]}")
        colors.append(point["color"])
        neighbors = list(get_all_within_distance(point, circle_radius, df))
        sets.append(set(neighbors))
    
    return sets, colors

# sets, colors = make_sets(df)

# %%
def verify_sets(sets, N, colors):
    for i in range(N):
        flag = False
        for s in sets:
            if i in s:
                flag = True
        if not flag:
            print(i)
            return False
    if len(np.unique(colors)) < 4:
        return False
    return True

# verify_sets(sets, df.shape[0], colors)

# %%
# lens = []
# for x in sets:
#     lens.append(len(x))
# tmp = pd.Series(lens)
# tmp.min(), tmp.max(), tmp.mean()

# %% [markdown]
# ## Experiment

# %%
def run_with_time(func, *args):
    start_time = time.time()
    res = func(*args)
    run_time = time.time() - start_time
    return (res, run_time)

def run_all(n, sets, colors):
    results = {}

    print("SC...")
    results["SC"] = run_with_time(standard_set_cover, n, copy.deepcopy(sets))
    print("Naive...")
    results["Naive"] = run_with_time(naive_fair_set_cover, n, copy.deepcopy(sets), copy.deepcopy(colors))
    print("EffAllPick...")
    results["EffAllPick"] = run_with_time(efficient_pick_set_cover, n, copy.deepcopy(sets), copy.deepcopy(colors))
    # print("AllPick...")
    # results["AllPick"] = run_with_time(all_pick_fair_set_cover, n, copy.deepcopy(sets), copy.deepcopy(colors))

    return results

# %%
runs_per_n = 5
set_sizes = [1000, 2000, 3000, 4000, 5000]

result_df_times = {"SetSize": [], "SC": [], "Naive": [], "EffAllPick": []}
result_df_counts = {"SetSize": [], "SC": [], "Naive": [], "EffAllPick": []}
result_df_fairness = {"SetSize": [], "SC": [], "Naive": [], "EffAllPick": []}

for set_size in set_sizes:
    N = set_size
    for i in range(runs_per_n):
        print("Preparing the sets")
        
        while True:
            print("Sampling...")
            df = data.sample(n=set_size)
            df = df.reset_index()
            # if len(np.unique(colors)) < 4:
            #     print(np.unique(colors))
            #     print(df.shape)
            #     print(df["color"].value_counts())
            #     print("Invalid colors...")
            #     continue
            print("Making sets...")
            sets, colors = make_sets(df, circle_radius=50000)
            if verify_sets(sets, N, colors):
                break

        print("Running...")
        
        # Run
        result = run_all(N, sets, colors)

        print(f"set_size = {set_size} in {set_sizes} : {i + 1} / {runs_per_n}")

        # Generate Result Dataframe
        result_df_times["SetSize"].append(set_size)
        result_df_counts["SetSize"].append(set_size)
        result_df_fairness["SetSize"].append(set_size)
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

result_df_counts.to_csv(f"result_df_counts_popsim_large.csv")
result_df_fairness.to_csv(f"result_df_fairness_popsim_large.csv")
result_df_times.to_csv(f"result_df_times_popsim_large.csv")
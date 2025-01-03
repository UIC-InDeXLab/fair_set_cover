# Import & Config
import numpy as np
import copy
import itertools
from scipy.optimize import linprog
import math
from itertools import product, combinations

"""Utility Functions"""

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

def permute(list, per, position, all):
    if position == len(list):
        all.append(per)
    else:
        for i in range(list[position]):
            tmp = copy.deepcopy(per)
            tmp[position] = i
            permute(list, tmp, position + 1, all)

def gen_permute(list, per, constraint, position, all, starts):
    if position == len(list):
        all.append(per)
    else:
        for i in range(starts[constraint[position]], list[position]):
            tmp = copy.deepcopy(per)
            s = copy.deepcopy(starts)
            s[constraint[position]] = i + 1
            tmp[position] = i
            gen_permute(list, tmp, constraint, position + 1, all, s)

def permute_eff(extended_colors, color_to_sets):
    combs = [combinations(color_to_sets[c], extended_colors.count(c)) for c in range(len(color_to_sets.keys()))]
    for pair in product(*combs):
        a = []
        [a.extend([s for s in p]) for p in pair]
        print(a)


def get_extended_colors(ratios, L):
    ratios = [int(x * L) for x in ratios]
    gcd = math.gcd(*ratios)
    ratios = [x // gcd for x in ratios]
    # return ratios
    extended_colors = []
    for i, rate in enumerate(ratios):
        extended_colors += [i for _ in range(rate)]

    return extended_colors

"""Algorithms Implementations"""

"""Greedy Standard Set Cover"""
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

"""Naive Fair Set Cover"""
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
        index = 0
        while to_add > 0:
            if cand_sets[index] not in unfair_cover:
                unfair_cover.append(cand_sets[index])
                to_add -= 1
            index += 1

    return unfair_cover

"""All Pick (Greedy Fair Set Cover)"""

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

"""Efficient All Pick (Faster Greedy for FSC)"""
"""Runs an LP to find a good pair"""

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

"""Generalized Set Cover for arbitrary 'Ratios'"""
def generalized_all_pick(n, sets, colors, ratios):
    # Pre-process
    sets = copy.deepcopy(sets)
    colors = copy.deepcopy(colors)
    point_to_sets = get_point_to_set_hashmap(sets)
    color_to_sets = get_color_to_set_hashmap(sets, colors)
    cover = []
    keys = [i for i in range(len(color_to_sets.keys()))]
    sets_picked = [False for _ in range(len(sets))]
    extend_colors = get_extended_colors(ratios, 1000)

    # Main Loop
    while True:
        # Find Largest Pair
        max_union = None
        picked_pair = None

        all_permutes = []
        combs = [combinations(list(filter(lambda x: len(sets[x]) > 0, color_to_sets[c])), extend_colors.count(c)) for c in range(len(color_to_sets))]
        print("end")
        updated = False
        for pair in product(*combs):
            tmp = []
            [tmp.extend([s for s in p]) for p in pair]
            pair = tmp
            union = len(set().union(*[sets[s] for s in pair]))
            if max_union is None or union > max_union:
                updated = True
                max_union = union
                picked_pair = tuple(pair)

        # Break if all points covered
        if max_union == 0 or max_union is None:
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

"""Generalized Faster Algorithm (EffGFSC)"""

def gen_LP(sets, point_to_sets, color_to_sets, n, m, covered, sets_picked, p_stars):
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
    for color in range(len(color_to_sets.keys())):
        # print("c: ", color)
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

    b = [0 for _ in range(len(C1))] + [p for p in p_stars] + [-p for p in p_stars]
    bounds = [(0, 1) for _ in range(n + m)]

    A = C1 + C2 + C3

    res = linprog(c=c, A_ub=A, b_ub=b, bounds=bounds)

    return res

def eff_gen_pick(n, sets, colors, ratios):
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
    extend_colors = get_extended_colors(ratios, 1000)
    p_stars = [extend_colors.count(color) for color in keys]

    # Main Loop
    while True:
        # Break if all points covered
        if np.all(points_covered):
            break

        # Find Largest Pair
        res = gen_LP(
            sets,
            point_to_sets,
            color_to_sets,
            n,
            m,
            points_covered,
            sets_picked,
            p_stars
        )
        if not res.success:
            break
        picked_pair = []
        for i in keys:
            cans = [*np.random.choice(color_to_sets[i], size=p_stars[i], p=res.x[:m][color_to_sets[i]]/sum(res.x[:m][color_to_sets[i]]), replace=False)]
            uni = np.unique(np.array(cans))
            cans = [i for i in uni]
            if len(uni) < p_stars[i]:
                ind = 0
                for s in color_to_sets[i]:
                    if ind == p_stars[i] - len(uni):
                        break
                    if not sets_picked[s] and not s in picked_pair and not s in cans:
                        cans.append(s)
                        ind += 1

            picked_pair += cans

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

"""Optimum Solutions (Bruteforce)"""
def opt_standard_solution(n, sets, max_length):
    for cover_size in range(max_length + 1):
        for cover in itertools.combinations(sets, cover_size):
            union = set().union(*cover)
            if len(union) == n:
                return [sets.index(c) for c in cover]

"""Optimum Fair Solution (Bruteforce)"""
def opt_fair_solution(n, sets, colors, max_length):
    color_to_sets = get_color_to_set_hashmap(sets, colors)
    blue_sets = color_to_sets[0]
    red_sets = color_to_sets[1]

    for cover_size in range(2, max_length + 2, 2):
        for blue in itertools.combinations(blue_sets, cover_size // 2):
            for red in itertools.combinations(red_sets, cover_size // 2):
                blue_subset = [sets[i] for i in blue]
                red_subset = [sets[i] for i in red]
                union = set().union(*blue_subset)
                union = union.union(*red_subset)
                if len(union) == n:
                    return list(red + blue)
from gmm_mml import GmmMml
from basic_functions import *
import numpy as np
from Evaluate_boolean import *
import re
from pyeda.boolalg.expr import exprvar
from pyeda.inter import *
import pickle
import timeit
from config import Terms, trees, n_class

with open('C:/Users/DELL/Downloads/CTRF-main/CTRF-main/CTRF/CTRF/CTRF/Output/test5.pickle', 'rb') as file:
    winetest = pickle.load(file)

pima = np.asarray(winetest)
[P, Q] = pima.shape
target = pima[:, -1]
pfeatures = pima[:, 0:Q - 1]

def replace_common_with_mean(list1, list2):
    list1_np = np.array(list1)
    list2_np = np.array(list2)
    if list2_np.size == 0:
        return list1_np.tolist()
    mean_val = np.mean(list2_np)
    mask = np.isin(list1_np, list2_np)
    list1_np[mask] = mean_val
    return list1_np.tolist()

def remove_consecutive_duplicates(lst):
    lst_np = np.array(lst)
    if lst_np.size == 0:
        return []
    mask = np.ones(lst_np.shape, dtype=bool)
    mask[1:] = lst_np[1:] != lst_np[:-1]
    return lst_np[mask].tolist()

def obdt4(dt, bf, pfeatures):
    unsupervised = GmmMml(plots=True)
    n_features = pfeatures.shape[1]
    cluster = [[] for _ in range(n_features)]

    for d in range(trees):
        for idx, val in enumerate(dt[d][5]):
            if isinstance(val, (int, float)):
                if val != 0:
                    f_idx = dt[d][4][idx]
                    cluster[f_idx].append(val)
            elif isinstance(val, list) and len(val) > 0:
                if any(v != 0 for v in val):
                    f_idx = dt[d][4][idx]
                    cluster[f_idx].append(val)

    samples_cluster_mean = []
    cluster_lists = []
    for f in range(n_features):
        arr = np.array(cluster[f]).reshape(-1, 1)
        if arr.shape[0] < 2 or np.allclose(arr, arr[0]):
            samples_cluster_mean.append([])
            cluster_lists.append([])
            continue

        try:
            unsupervised_fit = unsupervised.fit(arr)
            if not hasattr(unsupervised, 'fitted') or not unsupervised.fitted:
                samples_cluster_mean.append([])
                cluster_lists.append([])
                continue

            mixture = unsupervised.predict(arr)
            unique_comp = np.unique(mixture)

            mean_p = [np.mean(arr[mixture == uc]) for uc in unique_comp]
            samples_cluster_mean.append(mean_p)

            cluster_lists.append([
                arr[mixture == uc].flatten().tolist()
                for uc in unique_comp
            ])
        except Exception as e:
            print(f"Skipping feature {f} due to exception during GMM fitting: {e}")
            samples_cluster_mean.append([])
            cluster_lists.append([])

    for t in range(trees):
        for cc in range(0, Terms, 4):
            for d in range(len(bf[t][cc + 1])):
                f_repeat = basic_functions.Repeat(bf[t][cc + 1][d])
                if len(f_repeat) > 0:
                    replacement_values = cluster_lists[f_repeat[0]][0] if cluster_lists[f_repeat[0]] else []
                    common_elements = replace_common_with_mean(
                        bf[t][cc + 2][d],
                        replacement_values
                    )
                    bf[t][cc + 2][d] = common_elements

    return bf


def obdt5(bf, var, Terms, remove_f, remove_v):
    cut_tree = []
    for d in range(len(bf)):
        for cc in range(0, Terms, 4):
            if bf[d][cc]:
                for x in range(len(bf[d][cc + 2])):
                    arr = np.array(bf[d][cc + 2][x])
                    unique, idx_first = np.unique(arr, return_index=True)
                    duplicates_idx = np.setdiff1d(np.arange(len(arr)), idx_first)
                    if duplicates_idx.size > 0:
                        cut_tree.append(d)
                        indices_to_remove = np.sort(duplicates_idx)[::-1]

                        remove_f.extend([bf[d][cc + 1][x][i] for i in indices_to_remove])
                        remove_v.extend([bf[d][cc + 2][x][i] for i in indices_to_remove])

                        for i in indices_to_remove:
                            if i + 1 < len(bf[d][cc][x]):
                                del bf[d][cc][x][i + 1]
                            del bf[d][cc + 1][x][i]
                            del bf[d][cc + 2][x][i]
    return bf, cut_tree, remove_f, remove_v

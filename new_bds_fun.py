import re
from pyeda.inter import *
import pickle
import numpy as np
import timeit
from Evaluate_boolean import *
from config import Terms, trees, n_class

class bds_Func:

    @staticmethod
    def predict_bds(dt, bf, winetest):
        return bds_Func._predict_variant(dt, bf, winetest, variant='BDS')

    @staticmethod
    def predict_obds(dt, bf, winetest):
        return bds_Func._predict_variant(dt, bf, winetest, variant='OBDS')

    @staticmethod
    def predict_eobds(dt, bf, winetest):
        return bds_Func._predict_variant(dt, bf, winetest, variant='EOBDS')

    @staticmethod
    def _predict_variant(dt, bf, winetest, variant='BDS'):
        correct = 0
        arg2 = []
        avg_test_time = 0
        unsat_trees = set(range(trees))
        P, Q = winetest.shape

        for v in range(len(winetest)):
            count_list = []

            for cc in range(0, Terms, 4):
                count = 0
                for d in range(trees):
                    try:
                        my_list = bf[d][cc]
                        my_list1 = dt[d][4]
                        my_list2 = dt[d][5]
                        my_list3 = dt[d][0]

                        sample_features = winetest[v, :-1]

                        indices = [i for i, x in enumerate(my_list1)
                                   if isinstance(x, int) and x < len(sample_features)]

                        replaced_features = [sample_features[x] if i in indices else None
                                             for i, x in enumerate(my_list1)]

                        valid_indices = [i for i in range(len(my_list1))
                                         if my_list1[i] is not None and my_list2[i] is not None]

                        list_f = [my_list1[i] for i in valid_indices]
                        list_v = [my_list2[i] for i in valid_indices]
                        list_n = [my_list3[i] for i in valid_indices]
                        list_r = [replaced_features[i] for i in valid_indices]

                        # Guard against None in list_v, list_r
                        bool_list = [(x > y) if (x is not None and y is not None) else False
                                     for x, y in zip(list_v, list_r)]

                        num = len(list_v)
                        variable_names = [f'x[{i}]' for i in range(num)]
                        variable_dict = dict(zip(variable_names, bool_list))

                        reversed_list = [lst[::-1] for lst in my_list if isinstance(lst, list) and lst]

                        my_dict = {n: var for n, var in zip(list_n, variable_names)}

                        expr_list = []
                        for inner_list in reversed_list:
                            s = ""
                            for i in range(len(inner_list)-1):
                                key = inner_list[i]
                                if key not in my_dict:
                                    continue
                                literal = f"~{my_dict[key]}" if inner_list[i+1] % 2 == 1 else my_dict[key]
                                s = literal if s == "" else f"{s} & {literal}"
                            if s:
                                expr_list.append(expr(s))

                        if not expr_list:
                            continue

                        f_expr = Or(*expr_list)

                        if variant == 'EOBDS':
                            minimized_exprs = espresso_exprs(f_expr) if len(f_expr.inputs) > 4 else [f_expr]
                            expressions = minimized_exprs
                        else:
                            expressions = [f_expr]

                        sat_found = False
                        for exp in expressions:
                            if exp.satisfy_one() is not None:
                                unsat_trees.discard(d)
                                sat_found = True
                                break
                        if not sat_found:
                            continue

                        expression_repr = repr(expressions[0])
                        and_clauses = re.findall(r"And\((.*?)\)", expression_repr)
                        and_lists = [clause.split(", ") for clause in and_clauses]

                        start_test = timeit.default_timer()
                        result = Evaluate_Boolean.evaluate_boolean_function(and_lists, variable_dict)
                        stop_test = timeit.default_timer()
                        avg_test_time += (stop_test - start_test)

                        if result:
                            count += 1

                    except Exception as e:
                        print(f"Tree {d}, term {cc}: Exception - {e}")
                        continue

                count_list.append(count)

            predicted_class = count_list.index(max(count_list))

            if predicted_class == int(winetest[v, -1]):
                correct += 1
            arg2.append(predicted_class)

        acc = correct / len(winetest)
        time_ms = avg_test_time * 1000

        print(f"Unsatisfiable {variant} trees:", sorted(unsat_trees))
        print(f"Total unsatisfiable trees: {len(unsat_trees)}")
        print(f"-------------------------------------------")
        print(f"{variant}: {acc}")
        print(f"Average test time per sample: {time_ms:.3f} ms")
        print(f"-------------------------------------------")

        # count99 (And count) and count111 (Or count) are skipped for clarity, can add if needed
        return acc, arg2, None, None, time_ms



import numpy as np
from hypersearch.helpers import to_list
from typing import Callable
from functools import partial
from operator import itemgetter
from collections import defaultdict


class ParamSpace:

    def __init__(self, number_of_variation=5):
        self.params = {}
        self._param_variation = {}
        self._number_of_variations = max(number_of_variation, 1)

    def add_param(self, p_name, p_values, variation_ratio=0.):
        if p_name in self.params.keys():
            raise KeyError(f"given key '{p_name}' is already registered in the parameter space.")
        if isinstance(p_values, Callable):
            self.params[p_name] = p_values
            if variation_ratio > 0:
                self._param_variation[p_name] = variation_ratio
        else:
            self.params[p_name] = to_list(p_values)

    def sample(self):
        p_selected = {}
        for p_name in self.params.keys():
            p = self.params[p_name]
            if isinstance(p, Callable):
                sel = p()
            else:
                pos = np.random.randint(0, len(p))
                sel = p[pos]
            p_selected[p_name] = sel
        return p_selected

    def update_param_space(self, update_opts):
        for p_name, p_vals in update_opts.items():
            self._update_param(p_name, p_vals)

    def _update_param(self, p_name, values):
        variation_ratio = self._param_variation.get(p_name, None)
        if variation_ratio is None:
            self.params[p_name] = values
        else:
            new_param_values = []
            for val in values:
                new_param_values.append(val)
                for rand_num in range(self._number_of_variations - 1):
                    variation = np.random.normal(0, 1)
                    new_param_values.append(val + val * variation * variation_ratio)
            self.params[p_name] = list(set(new_param_values))


class ParamEvaluator:

    def __init__(self):
        self.logs = defaultdict(list)

    def log_result(self, setting, res):
        for p_name in setting.keys():
            self._log_result_param(p_name, setting[p_name], res)

    def _log_result_param(self, p_name, param, res):
        self.logs[p_name].append((param, res))

    def find_top_k(self, k, maximize=True):
        top_k = {}
        for p_name in self.logs.keys():
            top_k_p = self._find_top_k_param(p_name, k, maximize=maximize)
            top_k[p_name] = top_k_p
        return top_k

    def _find_top_k_param(self, p_name, k, maximize=True):
        p_logs = self.logs[p_name]
        k = min([k, len(p_logs)])
        p_logs.sort(key=itemgetter(1), reverse=maximize)
        top_k = list(map(itemgetter(0), p_logs))[:k]
        top_k = list(set(top_k))
        return top_k

    def reset(self):
        self.logs = defaultdict(list)


if __name__ == "__main__":

    ps = ParamSpace(number_of_variation=3)
    ps.add_param("learning_rate", [0.1, 0.2, 0.001])
    ps.add_param("momentum", partial(np.random.normal, 0, 1), variation_ratio=0.05)
    ps.add_param("batch_size", [32, 64, 128, 256, 526])
    sample = ps.sample()
    print(sample)
    print("############")

    peval = ParamEvaluator()

    for n in reversed(range(5)):
        print(ps.params)
        for i in range(10):
            sample = ps.sample()
            print(i, sample)
            peval.log_result(sample, i)
        top_k = peval.find_top_k(int(np.ceil((n+1)/2)), maximize=False)
        print(top_k)
        print("############")
        ps.update_param_space(top_k)
        peval.reset()

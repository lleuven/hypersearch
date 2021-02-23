

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


class ExperimentScheduler:

    def __init__(self, experiment_object, number_of_generations=1, number_of_experiments=1, number_of_variation=0,
                 maximize=False, max_number_of_top=None):
        self.exp_obj = experiment_object
        self.ps = ParamSpace(number_of_variation=number_of_variation)
        self.peval = ParamEvaluator()
        self.n_generations = number_of_generations
        self.n_experiments = number_of_experiments
        self._maximize = maximize
        self._max_k = max_number_of_top or self.n_generations

    def add_experiment_param(self, p_name, p_val, **kwargs):
        self.ps.add_param(p_name, p_val, **kwargs)

    def run(self):
        for i_gen in range(self.n_generations):
            for i_exp in range(self.n_experiments):
                sample = self.ps.sample()
                print(f"({i_gen}:{i_exp}", sample)
                res = self.exp_obj(**sample)
                self.peval.log_result(sample, res)
            k = min(self.n_generations - i_gen, self._max_k)
            top_k = self.peval.find_top_k(k, maximize=self._maximize)
            print(top_k)
            print("############")
            self.ps.update_param_space(top_k)
            self.peval.reset()

if __name__ == "__main__":

    def test_obj(learning_rate=0, momentum=1, batch_size=12):
        return learning_rate + abs(momentum * batch_size)

    e_sched = ExperimentScheduler(test_obj, number_of_generations=30, number_of_experiments=30, number_of_variation=10,
                                  maximize=False, max_number_of_top=10)
    e_sched.add_experiment_param("learning_rate", [0.1, 0.2, 0.001])
    e_sched.add_experiment_param("momentum", partial(np.random.normal, 0, 1), variation_ratio=1)
    e_sched.add_experiment_param("batch_size", [32, 64, 128, 256, 526])

    e_sched.run()
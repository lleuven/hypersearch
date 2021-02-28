

import numpy as np
from hypersearch.helpers import to_list
from typing import Callable
from functools import partial
from operator import itemgetter
from collections import defaultdict
import multiprocessing
import psutil
import time
import os


class ParamSpace:

    def __init__(self, number_of_combinations=5):
        self.params = {}
        self._param_variation = {}
        self._param_type = {}
        self._number_of_combinations = max(number_of_combinations, 1)
        self._sample_history = []
        self._max_sample_try = 10

    def add_param(self, p_name, p_values, variation_ratio=0., parameter_type=None):
        if p_name in self.params.keys():
            raise KeyError(f"given key '{p_name}' is already registered in the parameter space.")
        if isinstance(p_values, Callable):
            self.params[p_name] = p_values
        else:
            self.params[p_name] = to_list(p_values)
        if variation_ratio > 0:
            self._param_variation[p_name] = variation_ratio
        if parameter_type is not None:
            self._param_type[p_name] = parameter_type

    def sample(self, iteration=0):
        p_selected = {}
        for p_name in self.params.keys():
            p = self.params[p_name]
            if isinstance(p, Callable):
                sel = p()
            else:
                sel = np.random.choice(p)
            p_selected[p_name] = sel
        if p_selected in self._sample_history:
            if iteration == self._max_sample_try:
                return None
            return self.sample(iteration+1)
        else:
            self._sample_history.append(p_selected)
            return p_selected

    def update_param_space(self, update_opts):
        for p_name, p_vals in update_opts.items():
            self._update_param(p_name, p_vals)
        self._sample_history = []

    def _update_param(self, p_name, values):
        variation_ratio = self._param_variation.get(p_name, None)
        param_type = self._param_type.get(p_name, None)
        if variation_ratio is None:
            self.params[p_name] = values
        else:
            new_param_values = []
            for val in values:
                new_param_values.append(val)
                for rand_num in range(self._number_of_combinations - 1):
                    variation = val * (2 * variation_ratio * np.random.random() - variation_ratio)
                    new_val = val + variation
                    new_val = new_val if param_type is None else param_type(new_val)
                    new_param_values.append(new_val)
            self.params[p_name] = list(set(new_param_values))


class ParamEvaluator:

    def __init__(self, logfile=None):
        self.logs = defaultdict(list)
        self._log_file = logfile
        if self._log_file is not None:
            assert os.path.exists(self._log_file) is False
        self._kwargs_order = []
        self._separator = ";"

    def log_result(self, setting, res, **kwargs):
        for p_name in setting.keys():
            self._log_result_param(p_name, setting[p_name], res)
            self._log_result_file(p_name, setting[p_name], res, **kwargs)

    def _log_result_param(self, p_name, param, res):
        self.logs[p_name].append((param, res))

    def _log_result_file(self, p_name, param, res, **kwargs):
        if self._log_file is not None:
            if not os.path.exists(self._log_file):
                with open(self._log_file, "w") as f:
                    self._kwargs_order = list(kwargs.keys())
                    out = self._separator.join(["p_name", "param", "res", *self._kwargs_order])
                    f.write(f"{out}\n")
            with open(self._log_file, "a") as f:
                out = ";".join(map(str, [p_name, param, res, *[str(kwargs[k]) for k in self._kwargs_order]]))
                f.write(f"{out}\n")

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
                 maximize=False, max_number_of_top=None, logfile=None):
        self.exp_obj = experiment_object
        self.ps = ParamSpace(number_of_combinations=number_of_variation)
        self.peval = ParamEvaluator(logfile=logfile)
        self.n_generations = number_of_generations
        self.n_experiments = number_of_experiments
        self._maximize = maximize
        self._max_k = max_number_of_top or self.n_generations

    def add_experiment_param(self, p_name, p_val, **kwargs):
        self.ps.add_param(p_name, p_val, **kwargs)

    def run(self, time_delay=0.):
        for i_gen in range(self.n_generations):
            pool = multiprocessing.Pool(min([psutil.cpu_count(logical=False), 16]))  # use only physical cpus

            output = []
            for i_exp in range(self.n_experiments):
                output.append(pool.apply_async(f_proc, args=(self.ps.sample(), i_gen, i_exp, self.exp_obj)))
                time.sleep(time_delay)

            for i, p in enumerate(output):
                res, sample = p.get()
                if res is not None:
                    self.peval.log_result(sample, res, **{"generation": i_gen, "experiment": i})

            # for i_exp in range(self.n_experiments):
            #     sample = self.ps.sample()
            #     print(f"({i_gen}:{i_exp}", sample)
            #     res = self.exp_obj(**sample)
            #     self.peval.log_result(sample, res)

            k = min(self.n_generations - i_gen, self._max_k)
            top_k = self.peval.find_top_k(k, maximize=self._maximize)
            print(top_k)
            print("############")
            self.ps.update_param_space(top_k)
            self.peval.reset()

def f_proc(sample, i_gen, i_exp, exp_obj):
    res = None
    if sample is not None:
        print(f"({i_gen}:{i_exp}", sample)
        res = exp_obj(**sample)
    return res, sample

if __name__ == "__main__":

    def test_obj(learning_rate=0, momentum=1, batch_size=12):
        return learning_rate + abs(momentum * batch_size)

    e_sched = ExperimentScheduler(test_obj, number_of_generations=20, number_of_experiments=30, number_of_variation=2,
                                  maximize=False, max_number_of_top=10, logfile="log.csv")
    e_sched.add_experiment_param("learning_rate", [0.1, 0.2, 0.001])
    e_sched.add_experiment_param("momentum", partial(np.random.normal, 0, 1), variation_ratio=1)
    e_sched.add_experiment_param("batch_size", [32, 64, 128, 256, 526], variation_ratio=0.5, parameter_type=int)

    e_sched.run(time_delay=0.1)
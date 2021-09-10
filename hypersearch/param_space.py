
import numpy as np
from hypersearch.helpers import to_list
from typing import Callable
from functools import partial
from operator import itemgetter
from collections import defaultdict
import multiprocessing
import threading
import psutil
import copy
import time
import json
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class ParamSpace:

    def __init__(self, number_of_combinations=5):
        self.params = {}
        self.params_original = {}
        self._param_opts = {"_param_variation": {}, "_param_type": {}, "_sample_from_original": {}}
        self._number_of_combinations = max(number_of_combinations, 1)
        self._sample_history = []
        self._max_sample_try = 10

    def add_param(self, p_name, p_values, variation_ratio=0., parameter_type=float, sample_from_original=0.):
        if p_name in self.params.keys():
            raise KeyError(f"given key '{p_name}' is already registered in the parameter space.")
        if isinstance(p_values, Callable):
            self.params[p_name] = p_values
        else:
            self.params[p_name] = to_list(p_values)
        if variation_ratio > 0:
            self._param_opts["_param_variation"][p_name] = variation_ratio
            # self._param_variation[p_name] = variation_ratio
        if parameter_type is not None:
            self._param_opts["_param_type"][p_name] = parameter_type
        self._param_opts["_sample_from_original"][p_name] = sample_from_original
        self.params_original[p_name] = copy.deepcopy(self.params[p_name])

    def sample(self, iteration=0):
        p_selected = {}
        for p_name in self.params.keys():
            if np.random.random() < self._param_opts["_sample_from_original"].get(p_name, 0) and iteration < 1:
                p = self.params_original[p_name]
            else:
                p = self.params[p_name]
            if isinstance(p, Callable):
                sel = p()
            else:
                sel = p[np.random.randint(0, len(p))]
            p_selected[p_name] = sel
        if p_selected in self._sample_history:
            if iteration == self._max_sample_try:
                return None
            return self.sample(iteration+1)
        else:
            self._sample_history.append(p_selected)
            return p_selected

    def update_param_space(self, update_opts, clear_history=True):
        for p_name, p_vals in update_opts.items():
            self._update_param(p_name, p_vals)
        if clear_history is True:
            self._sample_history = []

    def _update_param(self, p_name, values):
        # variation_ratio = self._param_variation.get(p_name, None)
        variation_ratio = self._param_opts["_param_variation"].get(p_name, None)
        param_type = self._param_opts["_param_type"].get(p_name, None)
        if variation_ratio is None:
            if param_type == list:
                try:
                    self.params[p_name] = [json.loads(str(val)) for val in values]
                except json.decoder.JSONDecodeError:
                    self.params[p_name] = [val.strip("][").split(",") for val in values]
            else:
                self.params[p_name] = [val if param_type is None else param_type(val) for val in values]
        else:
            new_param_values = []
            for val in values:
                val = val if param_type is None else param_type(val)
                new_param_values.append(val)
                for rand_num in range(self._number_of_combinations - 1):
                    variation = val * (2 * variation_ratio * np.random.random() - variation_ratio)
                    new_val = val + variation
                    new_val = new_val if param_type is None else param_type(new_val)
                    new_param_values.append(new_val)
            self.params[p_name] = list(set(new_param_values))


class ParamEvaluator:

    def __init__(self, logfile=None, append_mode=False):
        self.logs = defaultdict(list)
        self.log_file = logfile
        if self.log_file is not None and append_mode is False:
            assert os.path.exists(self.log_file) is False
        self._kwargs_order = []
        self._separator = ";"

    def log_result(self, setting, res, **kwargs):
        for p_name in setting.keys():
            self._log_result_param(p_name, setting[p_name], res)
            self._log_result_file(p_name, setting[p_name], res, **kwargs)

    def _log_result_param(self, p_name, param, res):
        self.logs[p_name].append((param, float(res)))

    def load_log_from_file(self):
        if self.log_file is not None:
            with open(self.log_file, "r") as f:
                for line in f.readlines()[1:]:
                    log = line.strip().split(sep=self._separator)[:3]
                    self._log_result_param(*log)

    def _log_result_file(self, p_name, param, res, **kwargs):
        if self.log_file is not None:
            if not os.path.exists(self.log_file):
                with open(self.log_file, "w") as f:
                    self._kwargs_order = list(kwargs.keys())
                    out = self._separator.join(["p_name", "param", "res", *self._kwargs_order])
                    f.write(f"{out}\n")
            else:
                if len(self._kwargs_order) != len(kwargs):
                    with open(self.log_file, "r") as f:
                        first_line = f.readline().strip()
                        assert self._separator.join(["p_name", "param", "res", *list(kwargs.keys())]) == first_line
                        self._kwargs_order = list(kwargs.keys())
                else:
                    assert self._kwargs_order == list(kwargs.keys())
            with open(self.log_file, "a") as f:
                out = ";".join(map(str, [p_name, param, res, *[str(kwargs[k]) for k in self._kwargs_order]]))
                f.write(f"{out}\n")

    def find_top_k(self, k, maximize=True):
        top_k, top_k_res = {}, None
        for p_name in self.logs.keys():
            top_k_p, top_k_res = self._find_top_k_param(p_name, k, maximize=maximize)
            top_k[p_name] = top_k_p
        return top_k, top_k_res

    def _find_top_k_param(self, p_name, k, maximize=True):
        p_logs = self.logs[p_name]
        k = min([k, len(p_logs)])
        p_logs.sort(key=itemgetter(1), reverse=maximize)
        top_k_param = list(map(itemgetter(0), p_logs))[:k]
        unique_top_k_param = []
        [unique_top_k_param.append(e) for e in top_k_param if e not in unique_top_k_param]
        top_k_res = list(map(itemgetter(1), p_logs))[:k]
        top_k_res = list(set(top_k_res))
        return unique_top_k_param, top_k_res

    def reset(self):
        self.logs = defaultdict(list)


class ExperimentScheduler:

    def __init__(self, experiment_object, number_of_generations=1, number_of_experiments=1, number_of_variation=0,
                 maximize=False, max_number_of_top=None, logfile=None, plot_path=None, number_of_experiments_decay=0.,
                 start_generation=0):
        self.exp_obj = experiment_object
        self.ps = ParamSpace(number_of_combinations=number_of_variation)
        self.peval = ParamEvaluator(logfile=logfile, append_mode=(start_generation > 0))
        self.n_generations = number_of_generations
        self.n_experiments = number_of_experiments
        self.start_generation = start_generation
        self.decay = min(max(number_of_experiments_decay, 0), 0.99)
        self._maximize = maximize
        self._max_k = max_number_of_top or self.n_generations
        self._plot_path = plot_path if self.peval.log_file is not None else None
        self._experiment_param = []

    def add_experiment_param(self, p_name, p_val, **kwargs):
        self.ps.add_param(p_name, p_val, **kwargs)
        self._experiment_param.append(p_name)

    def _adjust_n_experiments(self, n_exp, k=1, iterations=0):
        for i in range(iterations):
            n_exp = max(int(n_exp * (1 - self.decay)), 2*k)
        return n_exp

    def prepare_next_generation(self, i_gen, n_experiments):
        k = min(self.n_generations - i_gen, self._max_k)
        top_k, top_k_res = self.peval.find_top_k(k, maximize=self._maximize)
        print(f"top {k} parameters: ", top_k)
        print(f"top {k} results: ", top_k_res)
        print("############")
        self.ps.update_param_space(top_k, clear_history=False)
        n_iterations = i_gen if self.start_generation == i_gen else 1
        n_experiments = self._adjust_n_experiments(n_experiments, iterations=n_iterations)
        return n_experiments

    def run(self, time_delay=0.):
        if self.start_generation == 0:
            n_experiments = self.n_experiments
        else:
            self.peval.load_log_from_file()
            n_experiments = self.prepare_next_generation(self.start_generation, self.n_experiments)
        for i_gen in range(self.start_generation, self.n_generations):
            pool = multiprocessing.Pool(min([psutil.cpu_count(logical=False), 16]),
                                        initializer=init, initargs=[multiprocessing.Lock()])  # use only physical cpus
            output = []
            for i_exp in range(n_experiments):
                output.append(pool.apply_async(f_proc, args=(self.ps.sample(), i_gen, i_exp, self.exp_obj), kwds={"sleep": time_delay}))

            for i, p in enumerate(output):
                res, sample = p.get()
                if res is not None:
                    self.peval.log_result(sample, res, **{"generation": i_gen, "experiment": i})
            self.plot()

            n_experiments = self.prepare_next_generation(i_gen, n_experiments)
            # self.peval.reset()
        self.plot()

    def read_log_file(self):
        return pd.read_csv(self.peval.log_file, sep=self.peval._separator)

    def plot(self):
        if self._plot_path is None:
            return
        data = self.read_log_file()
        for p_name in self._experiment_param:
            plot_data = data.loc[data.loc[:, "p_name"] == p_name].astype(
                {"param": float, "res": float, "generation": "category"}, errors="ignore")
            ax = sns.scatterplot(x="param", y="res", data=plot_data, hue="generation", palette="rainbow")
            ax.set_xlabel(p_name)
            ax.set_yscale('log')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            plt.savefig(os.path.join(self._plot_path, f"{p_name}.png"), dpi=300)
            plt.close('all')


def plot_from_log_file(plot_path, log_file, separator=";"):
    data = pd.read_csv(log_file, sep=separator)
    p_name_list = data.groupby("p_name").groups
    for p_name in p_name_list:
        plot_data = data.loc[data.loc[:, "p_name"] == p_name].astype(
            {"param": float, "res": float, "generation": "category"}, errors="ignore")
        ax = sns.scatterplot(x="param", y="res", data=plot_data, hue="generation", palette="rainbow")
        ax.set_xlabel(p_name)
        ax.set_yscale('log')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f"{p_name}.png"), dpi=300)
        plt.close('all')


def init(lock):
    global starting
    starting = lock


def f_proc(sample, i_gen, i_exp, exp_obj, sleep=0):
    starting.acquire()  # no other process can get it until it is released
    threading.Timer(sleep, starting.release).start()
    res = None
    if sample is not None:
        print(f"({i_gen}:{i_exp}", sample)
        res = exp_obj(**sample)
    return res, sample


if __name__ == "__main__":

    def test_obj(learning_rate=0, momentum=1, batch_size=12, **kwargs):
        return learning_rate + abs(momentum * batch_size)

    e_sched = ExperimentScheduler(test_obj, number_of_generations=10, number_of_experiments=100, number_of_variation=5,
                                  maximize=False, max_number_of_top=10, logfile="../log.csv", plot_path="..",
                                  number_of_experiments_decay=0.3, start_generation=2)
    e_sched.add_experiment_param("learning_rate", [0.1, 0.2, 0.001], sample_from_original=0.1)
    e_sched.add_experiment_param("momentum", partial(np.random.normal, 0, 1), variation_ratio=0.5, sample_from_original=0.01)
    e_sched.add_experiment_param("batch_size", [32, 64, 128, 256, 526], variation_ratio=0.5, parameter_type=int, sample_from_original=0.01)
    e_sched.add_experiment_param("activation", ["tanh", "sigmoid"], parameter_type=str)
    e_sched.add_experiment_param("explicite_layers", [[64, 32], [128, 4, 4], [1, 1, 1]], parameter_type=list)
    e_sched.run(time_delay=0.)

    # plot_from_log_file("..", "../log.csv")

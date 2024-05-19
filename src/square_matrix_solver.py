from typing import Generator, List, Tuple, Mapping

from intervals import Interval, IntervalVector, IntervalMatrix, Matrix, OuterMethod, Twin
from subdiff import SubdiffSolver

from itertools import combinations
from math import log2 as math_log2
from matplotlib import pyplot as plt
from matplotlib import figure as pltFig
from matplotlib.colors import LogNorm
from functools import reduce
from numpy.random import default_rng
from dataclasses import dataclass

import pickle
import threading
import pandas as pd
import seaborn as sns


def get_chached_data_path(file_name: str) -> str:
    return f'temp_data/{file_name}'


def combination_to_mask(combination: List[int]) -> int:
    mask = 0
    for val in combination:
        mask = mask | (1 << val)

    return mask


def mask_to_combination(mask: int) -> List[int]:
    pow_2 = int(math_log2(mask)) + 1 
    return [i for i in range(pow_2) if (1 << i) & mask > 0]


@dataclass(init=False)
class SolverResult:
    variable_idx: int
    variable_values: IntervalVector
    moda: IntervalVector
    mu: int
    median_Mef: Interval
    median_Mep: Interval
    twin_quantile: Twin
    twin_median: Twin

    def dump(self, digits_precision: int) -> None:
        print(f'x_{self.variable_idx}:')
        # print(f'set = {ivec.to_latex(False)}')
        print(f'moda = {self.moda.to_str(digits_precision)} | mu = {self.mu}')
        print(f'median_Mef = {self.median_Mef.to_str(digits_precision)}')
        print(f'median_Mep = {self.median_Mep.to_str(digits_precision)}')
        print(f'outer quantile twin = {self.twin_quantile.to_str(digits_precision)} | {self.twin_quantile.outer.wid():.1e}')
        print(f'outer median twin = {self.twin_median.to_str(digits_precision)} | {self.twin_median.outer.wid():.1e}')
        print('\n\n')

    def dump_latex(self, digits_precision: int) -> None:
        max_interval_in_moda = self.moda.extract_extremum(lambda interval: interval.wid())
        idx = self.variable_idx + 1

        print('\\begin{align*}')
        print(f'\\boldsymbol{{x}}_{idx} = {max_interval_in_moda.to_str(digits_precision)} \\\\')
        print(f'moda(\\boldsymbol{{x}}_{idx}) = {self.moda.to_str(digits_precision)}, \\mu = {self.mu} \\\\')
        print(f'q(\\boldsymbol{{x}}_{idx}) = {self.twin_quantile.outer.to_str(digits_precision)}, '
                f'm(\\boldsymbol{{x}}_{idx}) = {self.twin_median.outer.to_str(digits_precision)}')
        print('\\end{align*}')
        print('\n\n')


class Plotter:
    def __init__(self, save_fig: bool = False, subdir: str ='') -> None:
        self._save_fig = save_fig
        self._cur_fig: pltFig.Figure = None
        self._subdir = subdir
        plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
        plt.rc('legend', fontsize=20)    # legend fontsize

    def img_save_dst(self, filename: str) -> str:
        return f'img\\{filename}.png' if len(self._subdir) == 0 else f'img\\{self._subdir}\\{filename}.png'

    def plot_variable_interval_estimations(self, intevals: IntervalVector, var_name: str) -> None:
        self._plt_start()

        for i, inteval in enumerate(intevals.get_iterator()):
            color = 'b' if inteval.is_right() else 'r'
            inf, sup = inteval.pro().boundaries()
            plt.plot((i + 1, i + 1), (inf, sup), color)

        plt.ylabel('interval estimation')
        plt.xlabel('variable index')
        plt.title(f'{var_name}')
        self._plt_finish(self.img_save_dst(f'ModaSample{var_name}'), 200)

    def plot_modas(self, modas: List[List[Interval]]) -> None:
        self._plt_start()

        for i, intervals in enumerate(modas):
            for interval in intervals:
                plt.plot((i + 1, i + 1), (interval.left, interval.right), 'b')

        plt.ylabel('interval estimation')
        plt.xlabel('variable index')
        plt.title(f'Moda')
        self._plt_finish(self.img_save_dst(f'ModaSampleModa'), 200)

    def plot_solver_result(self, solver_result: SolverResult) -> None:
        self._plt_start()

        sorted_ivec = solver_result.variable_values.sort_copy(lambda interval : interval.wid())
        for i, interval in enumerate(sorted_ivec):
            color = 'b' if interval.is_right() else 'r'
            plt.plot((i + 1, i + 1), (interval.left, interval.right), color)

        sz = solver_result.variable_values.get_size() + 1
        offset = sz * 0.025
        delta = offset * 0.25
        plt.plot((sz + offset, sz + offset), (solver_result.twin_quantile.outer.left, solver_result.twin_quantile.outer.right),
                 'g', label='outer quantile')

        sz += 1
        offset += delta
        plt.plot((sz + offset, sz + offset), (solver_result.twin_median.outer.left, solver_result.twin_median.outer.right),
                 'y', label='outer median')

        plt.legend(loc='upper right')
        self._plt_finish(self.img_save_dst(f'Variables'), 200);
    
    def plot_twin_median_sup(self, solver_result: SolverResult) -> None:
        self._plt_start()

        def process_twin_result(ivec: IntervalVector, twin_est: Twin, save_name: str):
            outer_est = twin_est.outer
            y_lim = outer_est.pro().interval_add(Interval(-outer_est.rad(), outer_est.rad()).scale(0.5))
            sz = ivec.get_size()

            xs = [i + 1 for i in range(sz)]
            ys_sup = [interval.right for interval in ivec]
            ys_inf = [interval.left for interval in ivec]

            for i, interval in enumerate(ivec):
                color = 'b' if interval.is_right() else 'r'
                plt.plot((i + 1, i + 1), (interval.left, interval.right), f'{color}')

            plt.plot(xs, ys_sup, 'ob', markersize=5)
            plt.plot(xs, ys_inf, 'or', markersize=5)

            offset = sz * 0.05
            delta = offset * 0.5
            inner_est = twin_est.inner
            plt.plot((sz + offset, sz + offset), (inner_est.left, inner_est.right), 'k', marker='o', markersize=5, label='inner')
            offset += delta
            plt.plot((sz + offset, sz + offset), (outer_est.left, outer_est.right), 'g', marker='o', markersize=5, label='outer')
            
            plt.ylim((y_lim.left, y_lim.right))
            plt.legend(loc='upper right')
            plt.title(save_name)
            self._plt_finish(self.img_save_dst(save_name))

        sorted_ivec = solver_result.variable_values.sort_copy(lambda interval : interval.wid())
        process_twin_result(sorted_ivec, solver_result.twin_quantile, 'TwinQuantile')
        process_twin_result(sorted_ivec, solver_result.twin_median, 'TwinMedian')

    def draw_heat_map(self, matrix: Matrix, title: str, columns_names: List[str]) -> None:
        assert matrix.columns() == len(columns_names)

        self._plt_start()
        mat_data = matrix.get_data()
        for i in range(len(mat_data)):
            for j in range(len(mat_data[i])):
                if mat_data[i][j] == 0.0:
                    mat_data[i][j] = 0.000001
        df = pd.DataFrame(mat_data, columns=columns_names)
        sns.heatmap(df, norm=LogNorm())

        self._plt_finish(self.img_save_dst(title))

    def plot_mass_spectrum(self, x_s: List[float], y_s: List[float], title: str = ''):
        assert len(x_s) == len(y_s)
        
        self._plt_start()

        for x_k, y_k in zip(x_s, y_s):
            plt.plot((x_k, x_k), (0.0, y_k), 'b')

        plt.xlabel('m/z')
        plt.ylabel('Относительная интенсивность')
        self._plt_finish(self.img_save_dst(title))

    def _plt_start(self) -> None:
        if self._save_fig:
            self._cur_fig = plt.figure(figsize=(16, 9), dpi=100)

    def _plt_finish(self, title: str, fig_dpi: float = 100) -> None:
        if self._save_fig:
            self._cur_fig.savefig(title, dpi=self._cur_fig.dpi)
            plt.clf()
        else:
            plt.show()


class SquareMatrixSolver:
    def __init__(self) -> None:
        self._subdiff_solver = SubdiffSolver()
        self._cur_mat_A: IntervalMatrix = None
        self._cur_vec_b: IntervalMatrix = None
        self._plotter = Plotter(True, 'mouse_brain')

    def solve(self,
            mat_A: IntervalMatrix,
            vec_b: IntervalVector,
            err: float = 1e-4,
            max_iter: int = 1000) -> IntervalVector:
        assert mat_A.lines() == vec_b.get_size()
        self._cur_mat_A, self._cur_vec_b = mat_A, vec_b

        lines_num = self._cur_mat_A.lines()
        square_size = self._cur_mat_A.columns()
        assert lines_num >= square_size

        # combination mask -> answer
        square_system_results: Mapping[int, IntervalVector] = {}

        for comb in self._get_all_combinations(lines_num, square_size):
            subset_mat_A, subset_vec_b = self._extract_square_system(comb)
            subset_x = self._subdiff_solver.solve(subset_mat_A, subset_vec_b, err, max_iter)

            subset_mat_A.get_mid_matrix().condition_num()
            mask = combination_to_mask(comb)
            assert mask not in square_system_results

            square_system_results[mask] = subset_x

        self._get_analysis_result(square_system_results)
            
    def solve_probabilistic(self,
            mat_A: IntervalMatrix,
            vec_b: IntervalVector,
            err: float = 1e-4,
            max_iter: int = 1000) -> IntervalVector:
        assert mat_A.lines() == vec_b.get_size()
        self._cur_mat_A, self._cur_vec_b = mat_A, vec_b

        lines_num = self._cur_mat_A.lines()
        square_size = self._cur_mat_A.columns()
        assert lines_num >= square_size

        lines_sums = [reduce(lambda ix, iy: ix.interval_add(iy), iline).abs() \
                        for iline in self._cur_mat_A]
        all_lines_sum = sum(lines_sums)

        lines_prob_weights = [line_sum / all_lines_sum for line_sum in lines_sums]
        indexes = [i for i in range(lines_num)]

        systems_num = 100
        # combination mask -> answer
        square_system_results: Mapping[int, IntervalVector] = {}
        i = 0

        while i < systems_num:
            rng = default_rng()
            comb = rng.choice(indexes, p=lines_prob_weights, size=square_size, replace=False)
            imat, ib = self._extract_square_system(comb)
            mask = combination_to_mask(comb)

            if mask in square_system_results or imat.get_mid_matrix().is_singular():
                continue

            xk = self._subdiff_solver.solve(imat, ib, err, max_iter)
            square_system_results[mask] = xk

            i += 1
            print(i)

        print(f'Finish square systems')
        self._save_square_system_results(square_system_results)
        self._get_analysis_result(square_system_results)

    def solve_diag_dominant(self,
            mat_A: IntervalMatrix,
            vec_b: IntervalVector,
            err: float = 1e-4,
            max_iter: int = 1000) -> IntervalVector:
        assert mat_A.lines() == vec_b.get_size()
        self._cur_mat_A, self._cur_vec_b = mat_A, vec_b

        lines_num = self._cur_mat_A.lines()
        square_size = self._cur_mat_A.columns()
        assert lines_num >= square_size
        
        dominant_lines = [[] for _ in range(square_size)]

        for i, imatrix_line in enumerate(self._cur_mat_A):
            vec = IntervalVector(imatrix_line)
            has_dominant, idx = vec.has_dominant_value()
            if has_dominant:
                dominant_lines[idx].append(i)

        dominant_idx_nums = [len(idx_line) for idx_line in dominant_lines]
        check_sum = reduce(lambda x, y: x * y, dominant_idx_nums)
        print(dominant_idx_nums)

        cur_mat_idx_subset = [0 for _ in range(len(dominant_idx_nums))]

        def next_idx_subset() -> bool:
            sz = len(cur_mat_idx_subset)
            idx = sz - 1
            while cur_mat_idx_subset[idx] == dominant_idx_nums[idx] - 1:
                idx -= 1

                if idx == -1:
                    break
            
            if idx == -1:
                return False
            
            cur_mat_idx_subset[idx] += 1
            for i in range(idx + 1, sz):
                cur_mat_idx_subset[i] = 0

            return True

        counter = 0
        edge_counter = 0
        # combination mask -> answer
        square_system_results: Mapping[int, IntervalVector] = {}

        while True:
            cur_matrix_line_indexes = [dominant_lines[chunk_idx][line_idx] \
                                       for chunk_idx, line_idx in enumerate(cur_mat_idx_subset)]
            imat, ib = self._extract_square_system(cur_matrix_line_indexes)
            mask = combination_to_mask(cur_matrix_line_indexes)

            assert mask not in square_system_results
            x_k = self._subdiff_solver.solve(imat, ib, err, max_iter)
            square_system_results[mask] = x_k

            if not next_idx_subset():
                break

            counter += 1
            edge_counter += 1
            if edge_counter > check_sum * 0.05:
                edge_counter = 0
                print(f'{counter / check_sum * 100.0} %')

        print(f'100 % Finish square systems')

        self._save_square_system_results(square_system_results)
        self._get_analysis_result(square_system_results)

    def solve_diag_dominant_random(self,
            mat_A: IntervalMatrix,
            vec_b: IntervalVector,
            err: float = 1e-4,
            max_iter: int = 1000) -> IntervalVector:
        assert mat_A.lines() == vec_b.get_size()
        self._cur_mat_A, self._cur_vec_b = mat_A, vec_b

        lines_num = self._cur_mat_A.lines()
        square_size = self._cur_mat_A.columns()
        assert lines_num >= square_size
        
        dominant_lines = [[] for _ in range(square_size)]

        for i, imatrix_line in enumerate(self._cur_mat_A):
            vec = IntervalVector(imatrix_line)
            has_dominant, idx = vec.has_dominant_value()
            if has_dominant:
                dominant_lines[idx].append(i)

        dominant_idx_nums = [len(idx_line) for idx_line in dominant_lines]
        print(dominant_idx_nums)

        rng = default_rng(42)
        def random_idx_subset() -> List[int]:
            return [rng.integers(0, dominant_idx_num) for dominant_idx_num in dominant_idx_nums]

        # combination mask -> answer
        systems_num = 350
        square_system_results: Mapping[int, IntervalVector] = {}
        i = 0

        while i < systems_num:
            random_idxes = random_idx_subset()
            cur_matrix_line_indexes = [dominant_lines[chunk_idx][line_idx] \
                                       for chunk_idx, line_idx in enumerate(random_idxes)]
            imat, ib = self._extract_square_system(cur_matrix_line_indexes)
            mask = combination_to_mask(cur_matrix_line_indexes)

            if mask in square_system_results:
                continue

            assert mask not in square_system_results
            x_k = self._subdiff_solver.solve(imat, ib, err, max_iter)
            square_system_results[mask] = x_k

            i += 1
            print(i)

        self._save_square_system_results(square_system_results)
        self._get_analysis_result(square_system_results)

    def analaze_cached(self) -> IntervalVector:
        square_system_result = self._load_square_system_results()
        self._get_analysis_result(square_system_result)

    def _get_analysis_result(self, square_system_results: Mapping[int, IntervalVector]):
        systems_results = square_system_results.values()
        
        assert len(systems_results) > 0
        square_size = 0
        for i, system_result in enumerate(systems_results):
            if i == 0:
                square_size = system_result.get_size()

            assert system_result.get_size() == square_size

        variables_arrays: List[List[Interval]] = [[] for _ in range(square_size)]
        
        for system_result in systems_results:
            for i, val in enumerate(system_result):
                variables_arrays[i].append(val)

        print(f'Reshape finished: {len(variables_arrays)}')
        print('Start analysis')

        modas = [[Interval(0.0, 0.0)] for _ in range(square_size)]
        lock = threading.Lock()

        results: List[SolverResult] = [None for _ in range(square_size)]

        def variable_statistics(var_idx: int, var_intervals: List[Interval]):
            ivec = IntervalVector.create(var_intervals)
            moda, mu = ivec.find_moda()
            median_Mef = ivec.median_Mef()
            median_Mep = ivec.median_Mep()
            twin_quantile = ivec.twin_estimation(OuterMethod.kQuantile)
            twin_median = ivec.twin_estimation(OuterMethod.kMedian)

            with lock:
                solver_result = SolverResult()
                solver_result.variable_idx = var_idx
                solver_result.variable_values = ivec
                solver_result.moda = IntervalVector.create(moda)
                solver_result.mu = mu
                solver_result.median_Mef = median_Mef
                solver_result.median_Mep = median_Mep
                solver_result.twin_quantile = twin_quantile
                solver_result.twin_median = twin_median
                results[var_idx] = solver_result

                modas[var_idx] = moda
                solver_result.dump(8)
                # solver_result.dump_latex(8)
        
        tasks = [threading.Thread(target=variable_statistics, args=(i, var_arr)) \
                    for i, var_arr in enumerate(variables_arrays)]
        
        for variable_stat_task in tasks:
            variable_stat_task.start()
        for variable_stat_task in tasks:
            variable_stat_task.join()

        self._plotter.plot_twin_median_sup(results[2])
        self._plotter.plot_solver_result(results[2])

    def _get_all_combinations(self, n: int, m: int) -> Generator[List[float], None, None]:
        for comb in combinations([i for i in range(n)], m):
            yield comb

    def _extract_square_system(self, index_subset: List[int]) -> Tuple[IntervalMatrix, IntervalVector]:
        subset_mat = [self._cur_mat_A[idx].copy_data() for idx in index_subset]
        subset_vec = [self._cur_vec_b[idx] for idx in index_subset]

        return IntervalMatrix.create(subset_mat), IntervalVector.create(subset_vec)
    
    def is_line_has_dominant_value(self, line: List[float]) -> Tuple[bool, int]:
        for idx, val in enumerate(line):
            sub_line = [abs(v) for i, v in enumerate(line) if i != idx]

            if val > sum(sub_line):
                return True, idx
        
        return False, -1
    
    def _load_square_system_results(self) -> Mapping[int, IntervalVector]:
        with open(get_chached_data_path('mouse_brain_probalostic_100.pkl'), 'rb') as f:
            return pickle.load(f)
        
    def _save_square_system_results(self, dict_result: Mapping[int, IntervalVector]) -> None:
        with open(get_chached_data_path('cached_square_solver_result.pkl'), 'wb') as f:
            pickle.dump(dict_result, f)

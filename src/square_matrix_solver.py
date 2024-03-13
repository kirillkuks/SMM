from typing import Generator, List, Tuple, Mapping

from intervals import Interval, IntervalVector, IntervalMatrix, Matrix
from subdiff import SubdiffSolver

from itertools import combinations
from math import log2 as math_log2
from matplotlib import pyplot as plt


def combination_to_mask(combination: List[int]) -> int:
    mask = 0
    for val in combination:
        mask = mask | (1 << val)

    return mask


def mask_to_combination(mask: int) -> List[int]:
    pow_2 = int(math_log2(mask)) + 1 
    return [i for i in range(pow_2) if (1 << i) & mask > 0]


class Plotter:
    def __init__(self, save_fig: bool = False) -> None:
        self._save_fig = save_fig

    @staticmethod
    def img_save_dst(filename: str) -> str:
        return f'img\\{filename}.png'

    def plot_variable_interval_estimations(self, intevals: IntervalVector, var_name: str) -> None:
        for i, inteval in enumerate(intevals.get_iterator()):
            color = 'b' if inteval.is_right() else 'r'
            inf, sup = inteval.pro().boundaries()
            plt.plot((i + 1, i + 1), (inf, sup), color)

        plt.ylabel('interval estimation')
        plt.xlabel('variable index')
        plt.title(f'{var_name}')
        self._plt_finish(Plotter.img_save_dst(f'ModaSample{var_name}'), 200)

    def plot_modas(self, modas: List[List[Interval]]) -> None:
        for i, intervals in enumerate(modas):
            for interval in intervals:
                plt.plot((i + 1, i + 1), (interval.left, interval.right), 'b')

        plt.ylabel('interval estimation')
        plt.xlabel('variable index')
        plt.title(f'Moda')
        self._plt_finish(Plotter.img_save_dst(f'ModaSampleModa'), 200)

    def _plt_finish(self, title: str, fig_dpi: float) -> None:
        if self._save_fig:
            plt.savefig(title, dpi=fig_dpi)
            plt.clf()
        else:
            plt.show()


class SquareMatrixSolver:
    def __init__(self) -> None:
        self._subdiff_solver = SubdiffSolver()
        self._cur_mat_A: IntervalMatrix = None
        self._cur_vec_b: IntervalMatrix = None
        self._plotter = Plotter(True)

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
        variables_arrays: List[List[Interval]] = [[] for _ in range(square_size)]

        for comb in self._get_all_combinations(lines_num, square_size):
            subset_mat_A, subset_vec_b = self._extract_square_system(comb)
            subset_x = self._subdiff_solver.solve(subset_mat_A, subset_vec_b, err, max_iter)

            subset_mat_A.get_mid_matrix().condition_num()
            mask = combination_to_mask(comb)
            assert mask not in square_system_results

            square_system_results[mask] = subset_x
            for i, val in enumerate(subset_x.get_iterator()):
                variables_arrays[i].append(val)

        # for var_arr in variables_arrays:
        #     IntervalVector.create(var_arr).print()

        # print('\n!!!!!!!!!!!!!!!!!!!!!!!!!!1\n')

        # for mask in square_system_results.keys():
        #     print(f'mask = {mask} | {mask_to_combination(mask)}')
        #     square_system_results[mask].print()
        #     print('\n\n')
                
        modas = []
        for i, var_arr in enumerate(variables_arrays):
            ivec = IntervalVector.create(var_arr)
            moda, mu = ivec.find_moda()
            modas.append(moda)
            print(f'x_{i}:')
            print(f'set = {ivec.to_latex(False)}')
            print(f'moda = {IntervalVector.create(moda).to_str()} | mu = {mu}')
            self._plotter.plot_variable_interval_estimations(ivec, f'X{i}')
            print('\n\n')
        
        self._plotter.plot_modas(modas)


    def _get_all_combinations(self, n: int, m: int) -> Generator[List[float], None, None]:
        for comb in combinations([i for i in range(n)], m):
            yield comb

    def _extract_square_system(self, index_subset: List[int]) -> Tuple[IntervalMatrix, IntervalVector]:
        subset_mat = [self._cur_mat_A[idx].copy_data() for idx in index_subset]
        subset_vec = [self._cur_vec_b[idx] for idx in index_subset]

        return IntervalMatrix.create(subset_mat), IntervalVector.create(subset_vec)

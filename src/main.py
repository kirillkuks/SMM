from typing import List
from intervals import Interval, IntervalVector, IntervalMatrix, Matrix
from subdiff import SubdiffSolver

from square_matrix_solver import SquareMatrixSolver, Plotter
from randomizer import Randomizer

from data_loader import SpetrumDataLoader, MouseBrainNeuroLoader, Filter, Neurotransmitters

import example_matrix as em
import utils as ut
from os import listdir


def test_subdiff1():
    print('\n\n\n#############################\n\n\n')

    subdiff_solver = SubdiffSolver()
    eps = 0.01

    print('TEST 1:')
    C = IntervalMatrix.create([
        [Interval(2.0, 4.0), Interval(-2.0, 1.0)],
        [Interval(-1.0, 2.0), Interval(2.0, 4.0)]
    ])
    d = IntervalVector([Interval(-2.0, 2.0), Interval(-2.0, 2.0)])
    x = subdiff_solver.solve(C, d, eps)
    print('Final result = ', end='')
    x.print()

    print('TEST 2:')
    C = IntervalMatrix.create([
        [Interval(4.0, 2.0), Interval(1.0, -2.0)],
        [Interval(2.0, -1.0), Interval(4.0, 2.0)]
    ])
    d = IntervalVector([Interval(-2.0, 2.0), Interval(-2.0, 2.0)])
    x = subdiff_solver.solve(C, d, eps)
    print('Final result = ', end='')
    x.print()

    print('TEST 3:')
    C = IntervalMatrix.create_from_arrays([
        [[4, 6], [-9, 0], [0, 12], [2, 3], [5, 9], [-23,-9], [15, 23]],
        [[0, 1], [6, 10], [-1, 1], [-1, 3], [-5, 1], [1, 15], [-3,-1]],
        [[0, 3], [-20,-9], [12, 77], [-6, 30], [0, 3], [-18, 1], [0, 1]],
        [[-4, 1], [-1, 1], [-3, 1], [3, 5], [5, 9], [1, 2], [1, 4]],
        [[0, 3], [0, 6], [0, 20], [-1, 5], [8, 14], [-6, 1], [10, 17]],
        [[-7,-2], [1, 2], [7, 14], [-3, 1], [0, 2], [3, 5], [-2, 1]],
        [[-1, 5], [-3, 2], [0, 8], [1, 11], [-5, 10], [2, 7], [6, 82]]
    ])
    d = IntervalVector.create_from_arrays([
        [-10, 95], [35, 14], [-6, 2], [30, 7], [4, 95], [-6, 46], [-2, 65]
    ])
    x = subdiff_solver.solve(C, d, eps)
    print('Final result = ', end='')
    x.print()

    print('TEST 4:')
    C = IntervalMatrix.create([
        [Interval(3.0, 4.0), Interval(5.0, 6.0)],
        [Interval(-1.0, 1.0), Interval(-3.0, 1.0)]
    ])
    d = IntervalVector([Interval(-3.0, 4.0), Interval(-1.0, 2.0)])
    x = subdiff_solver.solve(C, d, eps, 10)
    print('Final result = ', end='')
    x.print()


def make_interval_matrix(point_mat: Matrix, err_min: float, err_max: float) -> IntervalMatrix:
    assert err_max >= err_min >= 0.0

    def calc_inteval(val: float, err: float) -> Interval:
        err_val = val * err
        return Interval.create_from_mid_rad(val, err_val)

    rnd = Randomizer()

    lines = [
        [
            calc_inteval(point_mat[i][j], rnd.uniform(err_min, err_max)) for j in range(point_mat.columns())
        ] for i in range(point_mat.lines())
    ]

    return IntervalMatrix.create(lines)


# метод квадратных матрицы для масс спектра
# ошибка 10%, какой-то минимум в правой части
# решить всю и по кускам
# система должна быть не разрешимой
# -> решение есть только в полной арифметике Каухера
import numpy as np

# спектр собственный значение для прямоугольной матрицы svd и отдельно для каждой квадратоной
#      сингулярные значени < 1 для каждого прямоугольного среза,
#      модули собстеных значений < 1 для каждой квадратной матрицы
# только интервальная правая часть сама матрица точечная ->
#      нормальное расширение правой и левой частей даёт в ответе как правильные, так и неправильные интервалы (так и надо)
# нормировка по столбцам -> нормировка по столбцам


#################################################
#
# Внешние оценки - медиана К..., Тьюки 1/4 - 50% | 3/4 + 50% 
# Книжка про изотопную подпись - дофомин и что-то ещё 
#
#################################################
#
# Выбирать строки случайно
# Выбирать с диагональным преобладанием
# Графики рисовать
# Добавить сравнение результатов для разного количества решённых квадратных систем
#
#################################################


def matrix_analys():
    mat = Matrix.create([
        [1.0, 0.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 0.0, 0.0]
    ])
    mat = Matrix.create(em.k_example_matrix_1)
    print(f'cond = {mat.condition_num()} | svd = {mat.svd()}')

    mat = Matrix.create(em.k_example_matrix_2)
    print(f'cond = {mat.condition_num()} | svd = {mat.svd()}')
    
    mat = Matrix.create(em.k_example_matrix_3)
    print(f'cond = {mat.condition_num()} | svd = {mat.svd()}')


neurotransmitters_formula_mapping = {
    Neurotransmitters.kBh4 : 'C9H15N5O3',
    Neurotransmitters.kDa : 'C8H11NO2',
    Neurotransmitters.k5Ht : 'C10H12N20',
    Neurotransmitters.kNe : 'C8H11NO3',
    Neurotransmitters.kEp : 'C9H13NO3',
    Neurotransmitters.kGlu : 'C5H9NO4',
    Neurotransmitters.kGaba : 'C9H4NO2'
}


def heat_map(mat: Matrix):
    plotter = Plotter(True, 'mouse_brain_for_text')
    plotter.draw_heat_map(mat, 'MatrixHeatMap', [
        neurotransmitters_formula_mapping[Neurotransmitters.kDa],
        neurotransmitters_formula_mapping[Neurotransmitters.k5Ht],
        neurotransmitters_formula_mapping[Neurotransmitters.kNe],
        neurotransmitters_formula_mapping[Neurotransmitters.kEp],
        neurotransmitters_formula_mapping[Neurotransmitters.kGlu],
        neurotransmitters_formula_mapping[Neurotransmitters.kGaba]
        ])


def mass_proportion():
    arr_striatum_left = [3261.1, 189.89, 384.01, 25.49, 950.43, 389.934]
    arr_striatum_right = [3662.9, 225.51, 531.19, 31.61, 1123.57, 404.866]

    arr_midbrain_left = [53.61, 436.6, 741.98, 63.96, 572.09, 647.54]
    arr_midbrain_right = [65.99, 536.8, 863.02, 83.04, 768.31, 716.86]

    arr_olfactory_bulb_left = [33.26, 75.05, 638.81, 45.11, 507.44, 679.89]
    arr_olfactory_bulb_right = [47.94, 94.81, 829.99, 74.29, 688.16, 864.31]

    arr_hypothalamus_left = [85.73, 253.98, 2491.6, 160.32, 675.0, 764.14]
    arr_hypothalamus_right = [138.07, 310.82, 2762.4, 215.68, 948.6, 895.26]

    def print_normlized(arr: List[float]):
        inv_sum_arr = 1.0 / sum(arr)
        print([round(val * inv_sum_arr, 3) for val in arr])

    def print_interval(arr1: List[float], arr2: List[float]):
        inv_sum1 = 1.0 / sum(arr1)
        inv_sum2 = 1.0 / sum(arr2)

        for v1, v2 in zip(arr1, arr2):
            print(f'{Interval(v1 * inv_sum1, v2 * inv_sum2, True).to_str(4)}', end=' ')
        
        print()

    print('striatum')
    print_normlized(arr_striatum_left)
    print_normlized(arr_striatum_right)
    print_interval(arr_striatum_left, arr_striatum_right)

    print('midbrain')
    print_normlized(arr_midbrain_left)
    print_normlized(arr_midbrain_right)
    print_interval(arr_midbrain_left, arr_midbrain_right)

    print('olfactory bulb')
    print_normlized(arr_olfactory_bulb_left)
    print_normlized(arr_olfactory_bulb_right)
    print_interval(arr_olfactory_bulb_left, arr_olfactory_bulb_right)

    print('hypothalamus')
    print_normlized(arr_hypothalamus_left)
    print_normlized(arr_hypothalamus_right)
    print_interval(arr_hypothalamus_left, arr_hypothalamus_right)

    return


def main():
    mouse_neuro_loader = MouseBrainNeuroLoader('spectrum_data/mouse_brain')

    striatum_neuro_mass = mouse_neuro_loader.load_single(
         MouseBrainNeuroLoader.BrainRegion.kStriatum,
         Neurotransmitters.filter_all().reject(Neurotransmitters.kBh4))
    striatum_neuro_vector = IntervalVector([
        striatum_neuro_mass[Neurotransmitters.kDa],
        striatum_neuro_mass[Neurotransmitters.k5Ht],
        striatum_neuro_mass[Neurotransmitters.kNe],
        striatum_neuro_mass[Neurotransmitters.kEp],
        striatum_neuro_mass[Neurotransmitters.kGlu],
        striatum_neuro_mass[Neurotransmitters.kGaba]
    ])
    print(striatum_neuro_mass)

    ms_loader = SpetrumDataLoader()

    # test example
    display_plotter = Plotter(True, 'spectrums')
    mass_spectrum_DA = ms_loader.load_spectrum('spectrum_data/neuro/C8H11NO2/1654.txt')
    mass_spectrum_DA_1654 = ms_loader.load_spectrum('spectrum_data/neuro/C8H11NO2/1654.txt')
    interval_mass_spectrum_DA = ut.merge_mass_spectrums(mass_spectrum_DA, mass_spectrum_DA_1654)
    # display_plotter.plot_mass_spectrum(mass_spectrum_DA.mass, mass_spectrum_DA.intensity, 'MassSpectrumDa_ForPres')

    mass_spectrum_GABA = ms_loader.load_spectrum('spectrum_data/neuro/C4H9NO2/1075.txt')
    mass_spectrum_GABA_3977 = ms_loader.load_spectrum('spectrum_data/neuro/C4H9NO2/3977.txt')
    interval_mass_spectrum_GABA = ut.merge_mass_spectrums(mass_spectrum_GABA, mass_spectrum_GABA_3977)

    mass_spectrum_5HT = ms_loader.load_spectrum('spectrum_data/neuro/C10H12N2O/23841.txt')
    mass_spectrum_5HT_51212 = ms_loader.load_spectrum('spectrum_data/neuro/C10H12N2O/51212.txt')
    interval_mass_spectrum_5HT = ut.merge_mass_spectrums(mass_spectrum_5HT, mass_spectrum_5HT_51212)

    mass_spectrum_NE = ms_loader.load_spectrum('spectrum_data/neuro/C8H11NO3/3536.txt')
    mass_spectrum_NE_10660 = ms_loader.load_spectrum('spectrum_data/neuro/C8H11NO3/10660.txt')
    interval_mass_spectrum_NE = ut.merge_mass_spectrums(mass_spectrum_NE, mass_spectrum_NE_10660)

    mass_spectrum_EP = ms_loader.load_spectrum('spectrum_data/neuro/C9H13NO3/5166.txt')
    mass_spectrum_EP_6169 = ms_loader.load_spectrum('spectrum_data/neuro/C9H13NO3/6169.txt')
    interval_mass_spectrum_EP = ut.merge_mass_spectrums(mass_spectrum_EP, mass_spectrum_EP_6169)

    mass_spectrum_Glu = ms_loader.load_spectrum('spectrum_data/neuro/C5H9NO4/1097.txt')
    mass_spectrum_Glu_2165 = ms_loader.load_spectrum('spectrum_data/neuro/C5H9NO4/2165.txt')
    interval_mass_spectrum_Glu = ut.merge_mass_spectrums(mass_spectrum_Glu, mass_spectrum_Glu_2165)

    interval_spectrum_matrix = ut.create_interval_matrix_from_spectrum_data([
        interval_mass_spectrum_DA,
        interval_mass_spectrum_5HT,
        interval_mass_spectrum_NE,
        interval_mass_spectrum_EP,
        interval_mass_spectrum_Glu,
        interval_mass_spectrum_GABA
    ])
    interval_spectrum_matrix.print()

    # striatum_neuro_mid_vector = [interval.mid() for interval in striatum_neuro_vector]
    # striatum_point_mexture = interval_spectrum_matrix.mul_point_vector(striatum_neuro_mid_vector)
    # striatum_point_mexture = [interval.mid() for interval in striatum_point_mexture]
    # inv_mixture_sum = 100.0 / max(striatum_point_mexture)
    # display_plotter.plot_mass_spectrum(
    #     [float(mz) for mz in range(len(striatum_point_mexture))],
    #     [intensity * inv_mixture_sum  for intensity in striatum_point_mexture],
    #     'NeuroMixture'
    # )
    # return
    striatum_mixture = interval_spectrum_matrix.mul_vector(striatum_neuro_vector, False)

    add_noise = True
    rnd = Randomizer()

    if add_noise:
        striatum_mixture = IntervalVector.create([
            Interval.create_from_mid_rad(
                interval.mid(), interval.rad() + rnd.normal(0, interval.mid() * 0.5)
                ) for interval in striatum_mixture 
        ])
    else:
        striatum_mixture = IntervalVector.create([
            Interval.create_from_mid_rad(
                interval.mid(), interval.rad() * (1.0 + rnd.uniform(-0.01, 0.01))
                ) for interval in striatum_mixture 
        ])

    square_solver = SquareMatrixSolver()
    eps = 0.1
    max_iter = 10
    # square_solver.analaze_cached()
    # square_solver.solve_probabilistic(interval_spectrum_matrix, striatum_mixture, eps, max_iter)
    heat_map(interval_spectrum_matrix.get_mid_matrix())
    # print(f'm = {interval_spectrum_matrix.lines()} | n = {interval_spectrum_matrix.columns()}')
    mid_spectrum_matrix = interval_spectrum_matrix.get_mid_matrix()
    print(f'cond = {mid_spectrum_matrix.condition_num()} | svd = {mid_spectrum_matrix.svd()}')
    striatum_mixture = striatum_mixture.normalized()
    # display_plotter.plot_mass_spectrum(
    #     [float(i) for i in range(striatum_mixture.get_size())],
    #     [interval.abs() * 100 for interval in striatum_mixture],
    #     'StriatumMouseBrainNormB_ForPres'
    # )

    # square_solver.solve_diag_dominant_random(interval_spectrum_matrix, striatum_mixture, eps, max_iter, False)
    square_solver.solve_probabilistic(interval_spectrum_matrix, striatum_mixture, eps, max_iter, False)
    return

    mass_spectrum_DA = ms_loader.load_spectrum('spectrum_data/neuro/C8H11NO2/1654.txt')
    # display_plotter.plot_mass_spectrum(mass_spectrum_DA.mass, mass_spectrum_DA.intensity)
    mass_spectrum_5HT = ms_loader.load_spectrum('spectrum_data/neuro/C10H12N2O/23841.txt')
    # display_plotter.plot_mass_spectrum(mass_spectrum_5HT.mass, mass_spectrum_5HT.intensity)
    mass_spectrum_NE = ms_loader.load_spectrum('spectrum_data/neuro/C8H11NO3/3536.txt')
    # display_plotter.plot_mass_spectrum(mass_spectrum_NE.mass, mass_spectrum_NE.intensity)
    mass_spectrum_EP = ms_loader.load_spectrum('spectrum_data/neuro/C9H13NO3/5166.txt')
    # display_plotter.plot_mass_spectrum(mass_spectrum_EP.mass, mass_spectrum_EP.intensity)
    mass_spectrum_Glu = ms_loader.load_spectrum('spectrum_data/neuro/C5H9NO4/1097.txt')
    # display_plotter.plot_mass_spectrum(mass_spectrum_Glu.mass, mass_spectrum_Glu.intensity)

    spectrum_matrix = ut.create_matrix_from_spectrum_data([
        mass_spectrum_DA,
        mass_spectrum_5HT,
        mass_spectrum_NE,
        mass_spectrum_EP,
        mass_spectrum_Glu,
        mass_spectrum_GABA
    ])

    interval_spectrum_matrix = IntervalMatrix.create_from_point(spectrum_matrix)
    striatum_mixture = interval_spectrum_matrix.mul_vector(striatum_neuro_vector)
    striatum_mixture.print()

    square_solver = SquareMatrixSolver()
    eps = 0.1
    max_iter = 10
    square_solver.solve_probabilistic(interval_spectrum_matrix, striatum_mixture, eps, max_iter)

    print(striatum_neuro_mass)
    return
    # matrix_analys()
    # return

    # test_subdiff1()
    example_matrix_slice = em.k_example_matrix_isotopic_sig_norm

    mat = Matrix.create(example_matrix_slice)
    print(f'cond = {mat.condition_num()} | svd = {mat.svd()}')
    interval_mat = make_interval_matrix(mat, 0.05, 0.15)
    interval_mat.to_latex()

    rnd = Randomizer()
    intervals = []
    for i in range(interval_mat.lines()):
        line_sum = Interval(0.0, 0.0)
        for j in range(interval_mat.columns()):
            line_sum = line_sum.interval_add(interval_mat[i][j])

        rad_change = min(line_sum.mid(), line_sum.rad()) * rnd.uniform(0.002, 0.005)
        line_sum = Interval.create_from_mid_rad(line_sum.mid(), line_sum.rad() - rad_change)
        intervals.append(line_sum)

    interval_vec = IntervalVector.create(intervals)
    interval_vec.to_latex()
    print('----------------------------------')
    # return
    square_solver = SquareMatrixSolver()
    eps = 0.01
    max_iter = 10
    heat_map(interval_mat.get_mid_matrix())
    return
    square_solver.analaze_cached()
    # square_solver.solve_probabilistic(interval_mat, interval_vec, eps, max_iter)
    # square_solver.solve_diag_dominant(interval_mat, interval_vec, eps, max_iter)
    # square_solver.solve_diag_dominant_random(interval_mat, interval_vec, eps, max_iter)
    # square_solver.solve(interval_mat, interval_vec, eps, max_iter)

    return
    # intervals = [Interval(1, 2), Interval(2, 3), Interval(3, 4)]
    # vec = IntervalVector.create(intervals)

    # for interval in vec.get_iterator():
    #     print(interval.to_str())

    # lines = [
    #     [Interval(-1, -2), Interval(2, 3), Interval(-3, 4)],
    #     [Interval(3, 4), Interval(-4, -5), Interval(5, 6)],
    #     [Interval(-6, -7), Interval(7, 8), Interval(8, 9)]
    # ]

    # m = IntervalMatrix.create(lines)

    # subdiff_solver = SubdiffSolver()
    # subdiff_solver.solve(m, vec)

    example_matrix_slice = k_example_matrix_1
    mat = Matrix.create(example_matrix_slice)
    calc_b = mat.mul_vector([1.0 for _ in range(mat.columns())])

    interval_mat = IntervalMatrix.create_from_point(mat)
    # sinterval_mat.print()

    interval_b = IntervalVector.create_from_point(calc_b)
    # interval_b.print()

    subdiff_solver = SubdiffSolver()
    # subdiff_solver.solve(interval_mat, interval_b)


    C = IntervalMatrix.create([
        [Interval(3.0, 4.0), Interval(5.0, 6.0)],
        [Interval(-1.0, 1.0), Interval(-3.0, 1.0)]
    ])
    d = IntervalVector([Interval(-3.0, 3.0), Interval(-1.0, 2.0)])

    # subdiff_solver.solve(C, d)

    C = IntervalMatrix.create([
        [Interval(2.0, 4.0), Interval(-2.0, 1.0)],
        [Interval(-2.0, 1.0), Interval(2.0, 4.0)]
    ])

    d = IntervalVector([Interval(-2.0, 2.0), Interval(-2.0, 2.0)])

    # Шарый конечно-мерный интервалный анализ, глава 12
    # ассоцилируемый
    subdiff_solver.solve(C, d)

if __name__ == '__main__':
    main()

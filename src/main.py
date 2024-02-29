from intervals import Interval, IntervalVector, IntervalMatrix, Matrix
from subdiff import SubdiffSolver

from example_matrix import k_example_matrix, k_example_matrix_1, k_example_matrix_2, k_example_matrix_3
from square_matrix_solver import SquareMatrixSolver
from randomizer import Randomizer


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


def matrix_analys():
    mat = Matrix.create([
        [1.0, 0.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 0.0, 0.0]
    ])
    mat = Matrix.create(k_example_matrix_1)
    print(mat.svd())

    mat = Matrix.create(k_example_matrix_2)
    print(mat.svd())
    
    mat = Matrix.create(k_example_matrix_3)
    print(mat.svd())


def main():
    # matrix_analys()
    # return

    # test_subdiff1()
    example_matrix_slice = k_example_matrix_1
    mat = Matrix.create(example_matrix_slice)

    interval_mat = make_interval_matrix(mat, 0.05, 0.15)

    rnd = Randomizer()
    intervals = []
    for i in range(interval_mat.lines()):
        line_sum = Interval(0.0, 0.0)
        for j in range(interval_mat.columns()):
            line_sum = line_sum.interval_add(interval_mat[i][j])

        rad_change = min(line_sum.mid(), line_sum.rad()) * rnd.uniform(0.05, 0.1)
        line_sum = Interval.create_from_mid_rad(line_sum.mid(), line_sum.rad() - rad_change)
        intervals.append(line_sum)

    interval_vec = IntervalVector.create(intervals)

    square_solver = SquareMatrixSolver()
    eps = 0.01
    max_iter = 10
    square_solver.solve(interval_mat, interval_vec, eps, max_iter)

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

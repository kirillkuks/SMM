from intervals import Interval, IntervalVector, IntervalMatrix, Matrix
from subdiff import SubdiffSolver

from example_matrix import k_example_matrix, k_example_matrix_1


def test_subdiff1():
    print('\n\n\n#############################\n\n\n')

    subdiff_solver = SubdiffSolver()

    C = IntervalMatrix.create([
        [Interval(2.0, 4.0), Interval(-2.0, 1.0)],
        [Interval(-1.0, 2.0), Interval(2.0, 4.0)]
    ])
    d = IntervalVector([Interval(-2.0, 2.0), Interval(-2.0, 2.0)])
    x = subdiff_solver.solve(C, d)
    print('TEST 1: x = ')
    x.print()

    C = IntervalMatrix.create([
        [Interval(4.0, 2.0), Interval(1.0, -2.0)],
        [Interval(2.0, -1.0), Interval(4.0, 2.0)]
    ])
    d = IntervalVector([Interval(-2.0, 2.0), Interval(-2.0, 2.0)])
    x = subdiff_solver.solve(C, d)
    print('TEST 2: x = ')
    x.print()

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
    x = subdiff_solver.solve(C, d)
    print('TEST 3: x = ')
    x.print()

    C = IntervalMatrix.create([
        [Interval(3.0, 4.0), Interval(5.0, 6.0)],
        [Interval(-1.0, 1.0), Interval(-3.0, 1.0)]
    ])
    d = IntervalVector([Interval(-3.0, 4.0), Interval(-1.0, 2.0)])
    x = subdiff_solver.solve(C, d)
    print('TEST 4: x = ')
    x.print()


def main():
    test_subdiff1()
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

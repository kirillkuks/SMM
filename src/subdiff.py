from typing import List

from scipy.linalg import solve as linalg_solve
from intervals import Interval, IntervalVector, IntervalMatrix, Matrix


def sti(ivec: IntervalVector) -> List[float]:
    return [inteval.left for inteval in ivec.get_iterator()] +\
            [interval.right for interval in ivec.get_iterator()]


def inv_sti(vec: List[float]) -> IntervalVector:
    assert len(vec) & 1 == 0
    half_size = len(vec) >> 1
    return IntervalVector.create([Interval(vec[i], vec[i + half_size]) for i in range(half_size)])


def positive_part(x: float) -> float:
    return max(x, 0.0)

def negative_part(x: float) -> float:
    return max(-x, 0.0)


class SubdiffSolver:
    def __init__(self) -> None:
        self._cur_mat_A: IntervalMatrix = None
        self._cur_vec_b: IntervalVector = None
        self._tau = 1.0

    def solve(self,
              mat_A: IntervalMatrix,
              vec_b: IntervalVector,
              err: float = 1e-7,
              max_iter: int = 1000) -> IntervalVector:
        assert mat_A.is_square()
        assert mat_A.lines() == vec_b.get_size()

        system_size = mat_A.lines()
        self._cur_mat_A, self._cur_vec_b = mat_A, vec_b
        sti_x_k = self.first_estimation()

        for _ in range(max_iter):
            # print('########## NEXT ITERATION ##########')

            s_interval = IntervalVector.create([Interval(0, 0) for _ in range(system_size)])
            D = Matrix.zeroes(2 * system_size, 2 * system_size)

            for i in range(system_size):
                h = inv_sti(sti_x_k)
                s_i = Interval(0.0, 0.0)

                for j in range(system_size):
                    g0, g1 = self._cur_mat_A[i][j].boundaries()
                    h0, h1 = h[j].boundaries()

                    t = Interval(g0, g1).mul(Interval(h0, h1))
                    s_i = s_i.interval_add(t)

                    k, m = 0, 0

                    if g0 * g1 > 0.0:
                        k = 0 if g0 > 0.0 else 2
                    else:
                        k = 1 if g0 <= g1 else 3
                    
                    if h0 * h1 > 0.0:
                        m = 1 if h0 > 0.0 else 3
                    else:
                        m = 2 if h0 <= h1 else 4
                    
                    multy_type = 4 * k + m

                    if multy_type == 1:
                        D[i][j] = g0
                        D[i + system_size][j + system_size] = g1
                    elif multy_type == 2:
                        D[i][j] = g1
                        D[i + system_size][j + system_size] = g1
                    elif multy_type == 3:
                        D[i][j] = g1
                        D[i + system_size][j + system_size] = g0
                    elif multy_type == 4:
                        D[i][j] = g0
                        D[i + system_size][j + system_size] = g0
                    elif multy_type == 5:
                        D[i][j + system_size] = g0
                        D[i + system_size][j + system_size] = g1
                    elif multy_type == 6:
                        if g0 * h1 < g1 * h0:
                            D[i][j + system_size] = g0
                        else:
                            D[i][j] = g1
                        if g0 * h0 > g1 * h1:
                            D[i + system_size][j] = g0
                        else:
                            D[i + system_size][j + system_size] = g1
                    elif multy_type == 7:
                        D[i][j] = g1
                        D[i + system_size][j] = g0
                    elif multy_type == 8:
                        pass
                    elif multy_type == 9:
                        D[i][j + system_size] = g0
                        D[i + system_size][j] = g1
                    elif multy_type == 10:
                        D[i][j + system_size] = g0
                        D[i + system_size][j] = g0
                    elif multy_type == 11:
                        D[i][j + system_size] = g1
                        D[i + system_size][j] = g0
                    elif multy_type == 12:
                        D[i][j + system_size] = g1
                        D[i + system_size][j] = g1
                    elif multy_type == 13:
                        D[i][j] = g0
                        D[i + system_size][j] = g1
                    elif multy_type == 14:
                        pass
                    elif multy_type == 15:
                        D[i][j + system_size] = g1
                        D[i + system_size][j + system_size] = g0
                    elif multy_type == 16:
                        if g0 * h0 > g1 * h1:
                            D[i][j] = g0
                        else:
                            D[i][j + system_size] = g1
                        if g0 * h1 < g1 * h0:
                            D[i + system_size][j + system_size] = g0
                        else:
                            D[i + system_size][j] = g1
                    else:
                        assert False

                s_interval[i] = s_i

            s = self._point_vec_sub(sti(s_interval), sti(self._cur_vec_b))
            xx = linalg_solve(D.get_data(), s).tolist()

            sti_x_k = [x_k_i - self._tau * xx_i for x_k_i, xx_i in zip(sti_x_k, xx)]

            r = sum([max(abs(s[i]), abs(s[i + system_size])) for i in range(system_size)])
            q = sum([abs(sti_x_k_i) for sti_x_k_i in sti_x_k])

            if r / q < err:
                break

        return inv_sti(sti_x_k)

    
    def first_estimation(self) -> List[float]:
        def build_matrix_2n_2n(mat: Matrix) -> Matrix:
            matrix_2n_2n_lines = [
                [
                   -negative_part(mat[i % mat.lines()][j % mat.columns()]) \
                        if int(i < mat.lines()) + int(j < mat.columns()) & 1 \
                        else positive_part(mat[i % mat.lines()][j % mat.columns()]) \
                     for j in range(2 * mat.columns()) 
                ] for i in range(2 * mat.lines())
            ]

            return Matrix.create(matrix_2n_2n_lines)

        mat_2n_2n = build_matrix_2n_2n(self._cur_mat_A.get_mid_matrix())
        sti_b = sti(self._cur_vec_b)

        x = linalg_solve(mat_2n_2n.get_data(), sti_b, check_finite=False)
        return x.tolist()
    

    def _point_vec_sub(self, vec1: List[float], vec2: List[float]) -> List[float]:
        assert len(vec1) == len(vec2)
        return [x1 - x2 for x1, x2 in zip(vec1, vec2)]
    

    def _define_multy_type(self, i1: Interval, i2: Interval) -> int:
        g0, g1 = i1.boundaries()
        h0, h1 = i2.boundaries()

        l, m = 0, 0

        if g0 * g1 > 0:
            if g0 > 0:
                l = 0
            else:
                l = 2
        else:
            if g0 <= g1:
                l = 1
            else:
                l = 3
        if h0 * h1 > 0:
            if h0 > 0:
                m = 1
            else:
                m = 3
        else:
            if h0 <= h1:
                m = 2
            else:
                m = 4

        return 4 * l + m

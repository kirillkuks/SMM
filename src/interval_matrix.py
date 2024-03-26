from __future__ import annotations
from typing import List, TypeVar, Generic, Tuple
T = TypeVar('T')

from interval import Interval
from interval_vector import IntervalVector

import numpy as np
import numpy.typing as npt


class LineViewer(Generic[T]):
    def __init__(self, line: List[T]) -> None:
        self._line_view = line

    def __getitem__(self, j: int) -> T:
        assert 0 <= j < len(self._line_view)
        return self._line_view[j]
    
    def __setitem__(self, j: int, val: T) -> None:
        assert 0 <= j < len(self._line_view)
        self._line_view[j] = val

    def copy_data(self) -> List[T]:
        return self._line_view.copy()
    

class MatrixIterator(Generic[T]):
    def __init__(self, matrix_data: List[List[T]]) -> None:
        super().__init__()
        self._matrix_data_view = matrix_data
        self._cur_idx = 0
        self._max_idx = len(self._matrix_data_view)

    def __next__(self) -> List[T]:
        if self._cur_idx >= self._max_idx:
            raise StopIteration()
        
        prev_idx, self._cur_idx = self._cur_idx, self._cur_idx + 1
        return self._matrix_data_view[prev_idx]


class IntervalMatrix:
    def create_from_vector_line(lines: List[IntervalVector]) -> IntervalMatrix:
        return IntervalMatrix.create([line.data() for line in lines])

    @staticmethod
    def create(lines: List[List[Interval]]) -> IntervalMatrix:
        assert len(lines) > 0
        first_line_elem_num = len(lines[0])

        for line in lines:
            assert len(line) == first_line_elem_num

        return IntervalMatrix(lines)
    
    @staticmethod
    def create_from_arrays(lines: List[List[List[float]]]) -> IntervalMatrix:
        for ll in lines:
            for l in ll:
                assert len(l) == 2

        lines = [
            [
                Interval(l[0], l[1]) for l in ll 
            ] for ll in lines
        ]

        return IntervalMatrix.create(lines)
    
    @staticmethod
    def create_from_point(point_matrix: Matrix) -> IntervalMatrix:
        lines = [
            [
                Interval.create_trivial(point_matrix[i][j]) for j in range(point_matrix.columns())
            ] for i in range(point_matrix.lines())
        ]

        return IntervalMatrix.create(lines)

    def __init__(self, lines: List[List[Interval]]) -> None:
        self._matrix_data = lines.copy()
        self._lines_num = len(lines)
        self._columns_num = len(lines[0])

    def __getitem__(self, i: int) -> LineViewer[Interval]:
        assert 0 <= i < self._lines_num
        return LineViewer(self._matrix_data[i])
    
    def __iter__(self) -> MatrixIterator[Interval]:
        return MatrixIterator(self._matrix_data)

    def get_mid_matrix(self) -> Matrix:
        mid_matrix_data = [
            [
                elem.mid() for elem in line
            ] for line in self._matrix_data
        ]

        return Matrix.create(mid_matrix_data)
    
    def is_square(self) -> bool:
        return self._lines_num == self._columns_num
    
    def lines(self) -> int:
        return self._lines_num
    
    def columns(self) -> int:
        return self._columns_num
    

    def print(self) -> None:
        for line in self._matrix_data:
            print('[', end='')
            for i, interval in enumerate(line):
                if i > 0:
                    print(', ', end='')
                print(interval.to_str(), end='')
            print(']')

    def to_latex(self, print_required: bool = True) -> str:
        latex_str = '\\begin{pmatrix}\n'
        for line in self._matrix_data:
            for i, elem in enumerate(line):
                if i > 0:
                    latex_str += ' & '
                latex_str += f'{elem.to_str()}'
            latex_str += ' \\\\ \n'
        latex_str += '\\end{pmatrix}'

        if print_required:
            print(latex_str)
        return latex_str



class Matrix:
    @staticmethod
    def create(lines: List[List[float]]) -> Matrix:
        assert len(lines) > 0
        first_line_elem_num = len(lines[0])

        for line in lines:
            assert len(line) == first_line_elem_num

        return Matrix(lines)
    
    @staticmethod
    def zeroes(lines_num: int, columns_num: int) -> Matrix:
        zeroes = [
            [
                0.0 for _ in range(columns_num)
            ] for _ in range(lines_num)
        ]

        return Matrix.create(zeroes)
    
    def __init__(self, lines: List[List[float]]) -> None:
        self._matrix_data = lines.copy()
        self._lines_num = len(lines)
        self._columns_num = len(lines[0])

    def __getitem__(self, i: int) -> LineViewer[float]:
        assert 0 <= i < self._lines_num
        return LineViewer(self._matrix_data[i])
    
    def __iter__(self) -> MatrixIterator[float]:
        return MatrixIterator(self._matrix_data)
    
    def lines(self) -> int:
        return self._lines_num
    
    def columns(self) -> int:
        return self._columns_num
    
    def get_data(self) -> npt.ArrayLike:
        return np.array(self._matrix_data)
    
    
    def mul_vector(self, vec: List[float]) -> List[float]:
        assert self.columns() == len(vec)
        return np.array(self._matrix_data).dot(np.array(vec)).tolist()
    
    def det(self) -> float:
        return np.linalg.det(self._matrix_data)

    def spectral_radius(self) -> float:
        ws = np.linalg.eigvals(self._matrix_data)
        return max(np.abs(ws[i]) for i in range(ws.size))
    
    def eigvals(self) -> List[float]:
        eig = np.linalg.eigvals(self._matrix_data)
        return [eig_val for eig_val in eig]
    
    def svd(self, digits_round: int = 5) -> Tuple[Matrix, List[float], Matrix]:
        sigma = np.linalg.svd(self._matrix_data, full_matrices=False, compute_uv=False)
        return [round(sig_val, digits_round) for sig_val in sigma]
    
    def condition_num(self) -> float:
        svd = self.svd()
        svd_ma, svd_mi = max(svd), min(svd)
        
        return svd_ma / svd_mi if svd_mi > 0.0 else float('inf')
    
    def is_singular(self) -> bool:
        return not (self._lines_num == self._columns_num and \
                np.linalg.matrix_rank(np.array(self._matrix_data)) == self._columns_num)

    def inverse(self) -> Matrix:
        res = Matrix(self.sz())
        inv = np.linalg.inv(self.matrix)

        for i in range(self.sz()):
            for j in range(self.sz()):
                res[[i, j]] = inv[i][j]

        return res
    
    def to_str(self) -> str:
        s = ''
        for j, line in enumerate(self._matrix_data):
            s += '['
            for i, val in enumerate(line):
                if i > 0:
                    s += ', '
                s += f'{round(val, 7)}'
            s += f'],\t#{j}\n'
        return s

    def print(self) -> None:
        print(self.to_str())

from __future__ import annotations
from typing import List, TypeVar, Generic
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


class IntervalMatrix:
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
        return np.linalg.det(self.matrix)

    def spectral_radius(self) -> float:
        ws = np.linalg.eigvals(self.matrix)
        return max(np.abs(ws[i]) for i in range(ws.size))

    def inverse(self) -> Matrix:
        res = Matrix(self.sz())
        inv = np.linalg.inv(self.matrix)

        for i in range(self.sz()):
            for j in range(self.sz()):
                res[[i, j]] = inv[i][j]

        return res

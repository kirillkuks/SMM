from __future__ import annotations
from typing import List, Tuple, Callable, TypeVar
T = TypeVar('T')
from enum import IntEnum

from interval import Interval, Twin


class OuterMethod(IntEnum):
    kQuantile = 1,
    kMedian = 2,
    kCount = 3


class IntervalVector:
    @staticmethod
    def create(intervals: List[Interval]) -> IntervalVector:
        copy_intervals = intervals.copy()
        return IntervalVector(copy_intervals)
    
    @staticmethod
    def create_from_arrays(intervals: List[List[float]]) -> IntervalVector:
        for l in intervals:
            assert len(l) == 2

        intervals = [Interval(l[0], l[1]) for l in intervals]
        return IntervalVector.create(intervals)
    
    @staticmethod
    def create_from_point(values: List[float]) -> IntervalVector:
        intervals = [Interval.create_trivial(val) for val in values]
        return IntervalVector.create(intervals)
    
    class VectorIterator:
        def __init__(self, ivector: IntervalVector) -> None:
            self.vec = ivector
            self.idx = 0

        def __iter__(self) -> IntervalVector.VectorIterator:
            return self
        
        def __next__(self) -> Interval:
            if self.idx >= self.vec.get_size():
                raise StopIteration()
            
            prev_idx, self.idx = self.idx, self.idx + 1
            return self.vec.at(prev_idx)

    def __init__(self, vec: List[Interval]) -> None:
        self._vector_data = vec

    def __getitem__(self, idx: int) -> Interval:
        return self._vector_data[idx]
    
    def __setitem__(self, idx: int, val: Interval) -> None:
        self._vector_data[idx] = Interval(val.left, val.right)

    def __iter__(self) -> IntervalVector.VectorIterator:
        return self.get_iterator()

    def get_size(self) -> int:
        return len(self._vector_data)
    
    def at(self, idx: int) -> Interval:
        assert 0 <= idx < self.get_size()
        return self._vector_data[idx]
    
    def data(self) -> List[Interval]:
        return self._vector_data.copy()
    

    def has_dominant_value(self) -> Tuple[bool, int]:
        for idx, ival in enumerate(self):
            sub_line = [iv.magnitude() for i, iv in enumerate(self) if i != idx]

            if ival.mignitude() > sum(sub_line):
                return True, idx
            
        return False, -1
    

    def find_moda(self) -> Tuple[List[Interval], int]:
        return Interval.find_moda(self._vector_data)
    
    def median_Mef(self) -> Interval:
        return Interval.median_Mef(self._vector_data)
    
    def median_Mep(self) -> Interval:
        return Interval.median_Mep(self._vector_data)
    
    def outer_quantiles(self) -> Interval:
        return Interval.outer_quantiles(self._vector_data)
    
    def outer_median(self) -> Interval:
        return Interval.outer_median(self._vector_data)
    
    def twin_estimation(self, outer_est_type: OuterMethod) -> Twin:
        outer = Interval(0.0, 0.0)
        if outer_est_type == OuterMethod.kQuantile:
            outer = self.outer_quantiles()
        elif outer_est_type == OuterMethod.kMedian:
            outer = self.outer_median()
        else:
            assert False

        inner, _ = self.find_moda()
        return Twin(Interval(inner[0].left, inner[-1].right), outer)
    
    def get_iterator(self) -> IntervalVector.VectorIterator:
        return IntervalVector.VectorIterator(self)
    
    def sort_copy(self, lamda_func: Callable[[Interval], float]) -> IntervalVector:
        return IntervalVector.create(sorted(self._vector_data, key=lamda_func))
    
    def extract_extremum(self, target: Callable[[Interval], T]) -> Interval:
        assert self.get_size() > 0

        max_target_val: T = target(self[0])
        target_interval = self[0]

        for interval in self:
            target_val = target(interval)

            if target_val > max_target_val:
                max_target_val, target_interval = target_val, interval

        return target_interval



    def to_str(self, digit_round: int = 5) -> str:
        s = '['
        for i, interval in enumerate(self._vector_data):
            if i > 0:
                s += ', '
            s += interval.to_str(digit_round)
        s += ']'

        return s

    def print(self) -> None:
        print(self.to_str())

    def to_latex(self, print_required: bool = True) -> str:
        latex_str = '\\begin{pmatrix}\n'
        for elem in self.get_iterator():
            latex_str += f'{elem.to_str()} \\\\ \n'
        latex_str += '\end{pmatrix}'

        if print_required:
            print(latex_str)
        return latex_str

from __future__ import annotations
from typing import List, Tuple

from interval import Interval


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

    def get_size(self) -> int:
        return len(self._vector_data)
    
    def at(self, idx: int) -> Interval:
        assert 0 <= idx < self.get_size()
        return self._vector_data[idx]
    

    def find_moda(self) -> Tuple[List[Interval], int]:
        return Interval.find_moda(self._vector_data)

    
    def get_iterator(self) -> IntervalVector.VectorIterator:
        return IntervalVector.VectorIterator(self)
    

    def to_str(self) -> str:
        s = '['
        for i, interval in enumerate(self._vector_data):
            if i > 0:
                s += ', '
            s += interval.to_str()
        s += ']'

        return s

    def print(self) -> None:
        print(self.to_str())

from __future__ import annotations
from typing import List, Tuple
from math import inf


class Interval:
    @staticmethod
    def create_trivial(x: float) -> Interval:
        return Interval(x, x)
    
    @staticmethod
    def create_from_mid_rad(mid: float, rad: float) -> Interval:
        return Interval(mid - rad, mid + rad)

    @staticmethod
    def min_max_union(intervals: List[Interval]) -> Interval:
        union_interval = intervals[0]

        for interval in intervals:
            union_interval = Interval(
                min(union_interval.left, interval.left),
                max(union_interval.right, interval.right)
            )

        return union_interval
    
    @staticmethod
    def min_max_intersection(intervals: List[Interval]) -> Interval:
        intersection_interval = intervals[0]

        for interval in intervals:
            intersection_interval = Interval(
                max(intersection_interval.left, interval.left),
                min(intersection_interval.right, interval.right)
            )

        return intersection_interval
    
    @staticmethod
    def jaccard_index(intervals: List[Interval]) -> float:
        return Interval.min_max_intersection(intervals).wid() / Interval.min_max_union(intervals).wid() * 0.5 + 0.5

    @staticmethod
    def scale_intervals(intervals: List[Interval], multiplier: float) -> List[Interval]:
        return [interval.scale(multiplier) for interval in intervals]
    
    @staticmethod
    def expand_intervals(intervals: List[Interval], eps: float) -> List[Interval]:
        return [interval.expand(eps) for interval in intervals]
    
    @staticmethod
    def combine_intervals(intervals1 : List[Interval], intervals2: List[Interval]) -> List[Interval]:
        return [j for i in [intervals1, intervals2] for j in i]
    
    @staticmethod
    def find_moda(intervals: List[Interval]) -> Tuple[Interval, int]:
        intervals_edges = []
        for interval in intervals:
            intervals_edges.append(interval.left)
            intervals_edges.append(interval.right)

        intervals_edges.sort()

        moda = []
        intervals_in_moda = 0
        for i, point in enumerate(intervals_edges):
            if i == len(intervals_edges) - 1:
                break

            current_interval = Interval(point, intervals_edges[i + 1])
            current_interval_in_moda = 0

            for interval in intervals:
                current_interval_in_moda += interval.is_nested(current_interval)
                #current_interval_in_moda += interval.contains(current_interval.mid())

            if current_interval_in_moda > intervals_in_moda:
                moda = [current_interval]
                intervals_in_moda = current_interval_in_moda
            elif current_interval_in_moda == intervals_in_moda:
                moda.append(current_interval)

        # aggregate intervals
        aggreate_moda = []
        current_interval: Interval = None
        for interval in moda:
            if current_interval is None:
                current_interval = interval
            else:
                if abs(current_interval.right - interval.left) < 1e-12:
                    current_interval = Interval(current_interval.left, interval.right)
                else:
                    aggreate_moda.append(current_interval)
                    current_interval = None
        if current_interval is not None:
            aggreate_moda.append(current_interval)

        return aggreate_moda, intervals_in_moda

    @staticmethod
    def outer_quantiles(intervals: List[Interval]) -> Interval:
        edges = []
        for interval in intervals:
            if interval.is_right() or True:
                edges.extend([interval.left, interval.right])

        if len(edges) == 0:
            return Interval(-inf, inf)

        edges = sorted(edges)
        edges_size = len(edges)
        left_idx, right_idx = int(edges_size * 0.25), int(edges_size * 0.75)
        # print(edges)
        # print(f'!!! {edges_size} | {left_idx} | {right_idx}')
        return Interval(edges[left_idx], edges[right_idx])
    
    @staticmethod
    def outer_median(intervals: List[Interval]) -> Interval:
        right_intervals = [interval for interval in intervals if interval.is_right() or True]
        edges_size = len(right_intervals)
        if edges_size == 0:
            return Interval(-inf, inf)

        left_edges = sorted([interval.left for interval in right_intervals])
        right_edges = sorted([interval.right for interval in right_intervals])

        return Interval(left_edges[int(edges_size * 0.5)], right_edges[int(edges_size * 0.5)])


    def __init__(self, x: float, y: float, force_right: bool = False) -> None:
        self.left =  min(x, y) if force_right else x
        self.right = max(x, y) if force_right else y

    def wid(self) -> float:
        return self.right - self.left
    
    def rad(self) -> float:
        return self.wid() * 0.5
    
    def mid(self) -> float:
         return (self.left + self.right) * 0.5
    
    def abs(self) -> float:
        return max(abs(self.left), abs(self.right))
    
    def magnitude(self) -> float:
        return self.abs()
    
    def mignitude(self) -> float:
        return 0.0 if self.pro().contains(0.0) else min(abs(self.left), abs(self.right))
    
    def pro(self) -> Interval:
        return Interval(self.left, self.right, True)
    
    def scale(self, multiplier: float) -> Interval:
        return Interval(self.left * multiplier, self.right * multiplier, True)
    
    def add(self, val: float) -> Interval:
        return Interval(self.left + val, self.right + val)
    
    def expand(self, eps: float) -> Interval:
        return Interval(self.left - eps, self.right + eps)

    def to_str(self, digit_round: int = 5) -> str:
        return f'[{round(self.left, digit_round)}, {round(self.right, digit_round)}]'
    
    def contains(self, val: float) -> bool:
        return self.left <= val <= self.right
    
    # other is nested in self
    def is_nested(self, other: Interval) -> bool:
        return other.left >= self.left and other.right <= self.right
    
    def interval_add(self, other: Interval) -> Interval:
        return Interval(self.left + other.left, self.right + other.right)
    
    def inner_minus(self, other: Interval) -> Interval:
        return Interval(self.left - other.left, self.right - other.right)
    
    def mul(self, other: Interval) -> Interval:
        xinfm, xinfp = max(-self.left, 0.0), max(self.left, 0.0)
        yinfm, yinfp = max(-other.left, 0.0), max(other.left, 0.0)

        xsupm, xsupp = max(-self.right, 0.0), max(self.right, 0.0)
        ysupm, ysupp = max(-other.right, 0.0), max(other.right, 0.0)

        zinf = max(xinfp * yinfp, xsupm * ysupm) - max(xsupp * yinfm, xinfm * ysupp)
        zsup = max(xsupp * ysupp, xinfm * yinfm) - max(xinfp * ysupm, xsupm * yinfp)

        return Interval(zinf, zsup)

    def boundaries(self) -> Tuple[float, float]:
        return self.left, self.right
    
    def copy(self) -> Interval:
        return Interval(self.left, self.right)
    
    def is_right(self) -> bool:
        return self.left <= self.right
    

class Twin:
    def __init__(self, inner: Interval, outer: Interval) -> None:
        # assert outer.contains(inner.left) and outer.contains(inner.right)
        # assert outer.is_nested(inner)

        self.inner = inner.copy()
        self.outer = outer.copy()

    def to_str(self, digit_round: int = 5) -> str:
        return f'[{self.inner.to_str(digit_round)}, {self.outer.to_str(digit_round)}]'
    
    def add(self, val: float) -> Twin:
        return Twin(self.inner.add(val), self.outer.add(val))

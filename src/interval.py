from __future__ import annotations
from typing import List, Tuple, Generator
from math import inf

kEps = 1e-10


def _all_trivial_subintervals(intervals_ends: List[float]) -> Generator[Interval, None, None]:
    intervals_num = len(intervals_ends) - 1
    for i, interval_left in enumerate(intervals_ends):
        if i == intervals_num:
            break
        
        yield Interval(interval_left, intervals_ends[i])


def _get_trivial_intervals_intensity(intervals: List[Interval], intervals_ends: List[float]) -> Tuple[List[float], int]:
    trivial_intervas_intensity = []
    accumulate_intensity = 0

    for trivial_interval in _all_trivial_subintervals(intervals_ends):
        current_interval_intensity = 0
        for interval in intervals:
            current_interval_intensity += 1 if interval.is_nested(trivial_interval) else 0
            
        trivial_intervas_intensity.append(current_interval_intensity)
        accumulate_intensity += current_interval_intensity
    
    return trivial_intervas_intensity, accumulate_intensity


def _filter_and_sort(arr: List[float], eps: float) -> List[float]:
    filtered_and_sorted_arr: List[float] = []
    for val in sorted(arr):
        if len(filtered_and_sorted_arr) == 0:
            filtered_and_sorted_arr.append(val)
        else:
            if abs(val - filtered_and_sorted_arr[-1]) > eps:
                filtered_and_sorted_arr.append(val)

    return filtered_and_sorted_arr


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
            interval = interval.pro()
            union_interval = Interval(
                min(union_interval.left, interval.left),
                max(union_interval.right, interval.right)
            )

        return union_interval
    
    @staticmethod
    def min_max_intersection(intervals: List[Interval]) -> Interval:
        intersection_interval = intervals[0]

        for interval in intervals:
            interval = interval.pro()
            intersection_interval = Interval(
                max(intersection_interval.left, interval.left),
                min(intersection_interval.right, interval.right)
            )
        
        return intersection_interval
    
    @staticmethod
    def jaccard_index(intervals: List[Interval]) -> float:
        return Interval.min_max_intersection(intervals).wid() / Interval.min_max_union(intervals).wid()

    @staticmethod
    def scale_intervals(intervals: List[Interval], multiplier: float) -> List[Interval]:
        return [interval.scale(multiplier) for interval in intervals]
    
    @staticmethod
    def expand_intervals(intervals: List[Interval], eps: float) -> List[Interval]:
        return [interval.expand(eps) for interval in intervals]
    
    @staticmethod
    def combine_intervals(intervals1 : List[Interval], intervals2: List[Interval]) -> List[Interval]:
        return [j for i in [intervals1, intervals2] for j in i]
    
    # TODO: handle KR intervals
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

    # TODO: handle KR intervals
    @staticmethod
    def median_Mef(intervals: List[Interval]) -> Interval:
        intervals_ends: List[Interval] = []
        for interval in intervals:
            # TODO: add unique, for example 1e-10?
            intervals_ends.extend([interval.left, interval.right])

        intervals_ends = _filter_and_sort(intervals_ends, kEps)

        primitive_intervals_intensity, accumulative_intensity = _get_trivial_intervals_intensity(intervals, intervals_ends)

        half_accumulative_intensity = accumulative_intensity / 2
        mid_interval_idx = 0
        mid_accumulative_intensity = 0
        while mid_accumulative_intensity < half_accumulative_intensity:
            primitive_interval_intensity = primitive_intervals_intensity[mid_interval_idx]
            mid_accumulative_intensity += primitive_interval_intensity
            mid_interval_idx += 1

        mid_interval_idx -= 1

        return Interval(intervals_ends[mid_interval_idx], intervals_ends[mid_interval_idx + 1])
    

    @staticmethod
    def median_Mep(intervals: List[Interval]) -> Interval:
        intervals_ends: List[Interval] = []
        for interval in intervals:
            intervals_ends.extend([interval.left, interval.right])
        
        intervals_ends = _filter_and_sort(intervals_ends, kEps)

        trivial_intensity, _ = _get_trivial_intervals_intensity(intervals, intervals_ends)
        min_sum_dist = inf
        min_sum_dist_idx = 0

        for i, cur_trivial_interval in enumerate(_all_trivial_subintervals(intervals_ends)):
            sum_dist_to_other_trivials = 0

            for j, others_trivial_interval in enumerate(_all_trivial_subintervals(intervals_ends)):
                if i == j:
                    continue

                sum_dist_to_other_trivials += cur_trivial_interval.dist_to(others_trivial_interval) * trivial_intensity[j]
            
            if min_sum_dist > sum_dist_to_other_trivials:
                min_sum_dist = sum_dist_to_other_trivials
                min_sum_dist_idx = i

        return Interval(intervals_ends[min_sum_dist_idx], intervals_ends[min_sum_dist_idx + 1])
    

    @staticmethod
    def median_Mef_outer(intervals: List[Interval], threshold: float = 0.5) -> Interval:
        assert 0.0 < threshold < 1.0

        intervals_ends: List[float] = []
        for interval in intervals:
            intervals_ends.extend([interval.left, interval.right])

        intervals_ends = _filter_and_sort(intervals_ends, kEps)

        trivial_intervals_intensity, accumulate_intensity = _get_trivial_intervals_intensity(intervals, intervals_ends)

        half_inensity = accumulate_intensity / 2
        mid_interval_idx = 0
        mid_accumulate_intensity = 0
        while mid_accumulate_intensity < half_inensity:
            trivial_intensity = trivial_intervals_intensity[mid_interval_idx]
            mid_accumulate_intensity += trivial_intensity
            mid_interval_idx += 1

        mid_interval_idx -= 1

        current_sum_internsity = trivial_intervals_intensity[mid_interval_idx]
        left_trivials_intensity = mid_accumulate_intensity - trivial_intervals_intensity[mid_interval_idx]
        right_trivial_intensity = accumulate_intensity - left_trivials_intensity - current_sum_internsity

        median_outer_interval_sum_intensity = accumulate_intensity * threshold
        left_trivial, right_trivial = mid_interval_idx, mid_interval_idx + 1
        while current_sum_internsity < median_outer_interval_sum_intensity:
            new_trivial_internsity = 0
            if left_trivials_intensity > right_trivial_intensity:
                left_trivial -= 1
                new_trivial_internsity = trivial_intervals_intensity[left_trivial]
                left_trivials_intensity -= new_trivial_internsity
            else:
                right_trivial += 1
                new_trivial_internsity = trivial_intervals_intensity[right_trivial]
                right_trivial_intensity -= new_trivial_internsity
            
            current_sum_internsity += new_trivial_internsity

        return Interval(intervals_ends[left_trivial], intervals_ends[right_trivial])

    def median_Mep_outer(intervals: List[Interval], threshold: float = 0.5) -> Interval:
        assert 0.0 < threshold < 1.0

        intervals_ends: List[Interval] = []
        for interval in intervals:
            intervals_ends.extend([interval.left, interval.right])
        
        intervals_ends = _filter_and_sort(intervals_ends, kEps)

        trivial_intensity, trivial_intensity_sum = _get_trivial_intervals_intensity(intervals, intervals_ends)
        min_sum_dist = inf
        min_sum_dist_idx = 0

        for i, cur_trivial_interval in enumerate(_all_trivial_subintervals(intervals_ends)):
            sum_dist_to_other_trivials = 0

            for j, others_trivial_interval in enumerate(_all_trivial_subintervals(intervals_ends)):
                if i == j:
                    continue

                sum_dist_to_other_trivials += cur_trivial_interval.dist_to(others_trivial_interval) * trivial_intensity[j]
            
            if min_sum_dist > sum_dist_to_other_trivials:
                min_sum_dist = sum_dist_to_other_trivials
                min_sum_dist_idx = i

        def dist_to_other_trivials(left_idx: int, right_idx: int) -> int:
            sum_dist_to_other = 0
            cur_interval = Interval(intervals_ends[left_idx], intervals_ends[right_idx])

            for j, other_trivial in enumerate(_all_trivial_subintervals(intervals_ends)):
                if left_idx <= j < right_idx:
                    continue

                sum_dist_to_other += cur_interval.dist_to(other_trivial) * trivial_intensity[j]

            return sum_dist_to_other

        left_idx, right_idx = min_sum_dist_idx, min_sum_dist_idx + 1
        trivial_intensity_threshold = trivial_intensity_sum * threshold
        trivial_intensity_sum = trivial_intensity[left_idx]
        
        while trivial_intensity_sum < trivial_intensity_threshold:
            left_sum = dist_to_other_trivials(left_idx - 1, right_idx)
            right_sum = dist_to_other_trivials(left_idx, right_idx + 1)

            if left_sum > right_sum:
                left_idx -= 1
                trivial_intensity_sum += trivial_intensity[left_idx]
            else:
                right_idx += 1
                trivial_intensity_sum += trivial_intensity[right_idx]

        return Interval(intervals_ends[left_idx], intervals_ends[right_idx])


    @staticmethod
    def outer_quantiles(intervals: List[Interval]) -> Interval:
        edges = []
        for interval in intervals:
            if interval.is_right() or True:
                edges.extend([interval.left, interval.right])

        if len(edges) == 0:
            return Interval(-inf, inf)
        
        alpha = 0.25
        assert 0 < alpha < 0.5

        edges = sorted(edges)
        edges_size = len(edges)
        left_idx, right_idx = int(edges_size * (alpha)), int(edges_size * (1.0 - alpha))
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
    
    def __repr__(self) -> str:
        return self.to_str() 
    
    def __str__(self) -> str:
        return self.to_str()
    
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
    
    def dist_to(self, other: Interval) -> float:
        return max(
            abs(self.left - other.left),
            abs(self.right - other.right)
        )

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

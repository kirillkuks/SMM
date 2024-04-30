from __future__ import annotations

from typing import List, Mapping
from os import listdir
from enum import Enum, IntEnum
from dataclasses import dataclass

from intervals import Matrix, IntervalMatrix, Interval


@dataclass
class MassSpectrumData:
    mass: List[float]
    intensity: List[float]

    def __init__(self, x_s: List[float], y_s: List[float]):
        self.mass = x_s.copy()
        self.intensity = y_s.copy()

    def add(self, x: float, y: float) -> None:
        self.mass.append(x)
        self.intensity.append(y)


class SpetrumDataLoader:
    def __init__(self) -> None:
        pass

    def load_spectrum(self, spectrum_data_path: str) -> MassSpectrumData:
        output_data = MassSpectrumData([], [])

        with open(spectrum_data_path, 'r') as f:
            for line in f:
                mass_intensity = line.split()
                assert len(mass_intensity) == 2

                mass, intensity = float(mass_intensity[0]), float(mass_intensity[1])

                # do not consider values for non-integer mass
                if mass != float(int(mass)):
                    continue

                output_data.add(mass, intensity)

        return output_data


    def load_alkanes(self, path: str, c_min: int, c_max: int) -> Matrix:
        filenames = [f'C{c}H{2 * c + 2}.txt' for c in range(c_min, c_max + 1)]
        return self.load(path, filenames)
    
    def load_isotopic_signature(self, path) -> Matrix:
        filenames = ['C8H11NO', 'C8H11NO2', 'C9H11NO2', 'C9H11NO3', 'C11H17NO3']
        filenames = [f'{filename}.txt' for filename in filenames]
        return self.load(path, filenames)
    
    def load_dopamine_synthesis(self, path) -> IntervalMatrix:
        substances = ['C9H11NO3', 'C9H11NO4', 'C8H11NO2', 'C8H11NO3', 'C9H13NO3']
        dirs = [f'{path}/{substance}' for substance in substances]

        for substance_dir in dirs:
            print(listdir(substance_dir))


    def load(self, dir: str, filenames: List[str]) -> Matrix:
        columns_num = len(filenames)
        def add_row(arr: List[List[float]], rows_num) -> List[List[float]]:
            new_rows = [
                [0.0 for _ in range(columns_num)] for _ in range(rows_num)
            ]
            return arr + new_rows

        matrix_lines: List[List[float]] = []
        for i, filename in enumerate(filenames):
            filename = f'{dir}/{filename}'
            with open(filename, 'r') as f_ms_data:
                ms_lines = f_ms_data.readlines()

                row_idxes, vals = [], []
                for line in ms_lines:
                    numbers = line.split()
                    row_idxes.append(int(float(numbers[0])) - 1)
                    vals.append(float(numbers[1]))

                vals_sum = sum(vals)
                vals = [v / vals_sum for v in vals]

                for row_idx, val in zip(row_idxes, vals):
                    sz = len(matrix_lines)
                    if sz <= row_idx:
                        matrix_lines = add_row(matrix_lines, row_idx - sz + 1)

                    matrix_lines[row_idx][i] = val

        mat = Matrix.create(matrix_lines)
        mat.print()


class Filter:
    def __init__(self, filter: int) -> None:
        self._filter = filter

    @property
    def filter(self) -> int:
        return self._filter

    def reject(self, rejected: int) -> Filter:
        return Filter(self.filter & ~(rejected))

class Neurotransmitters(IntEnum):
    kBh4 = 1 << 0,
    kDa = 1 << 1,
    k5Ht = 1 << 2,
    kNe = 1 << 3,
    kEp = 1 << 4,
    kGlu = 1 << 5,
    kGaba = 1 << 6

    @staticmethod
    def filter_all() -> Filter:
        return Filter(
                Neurotransmitters.kBh4 | \
                Neurotransmitters.kDa | \
                Neurotransmitters.k5Ht | \
                Neurotransmitters.kNe | \
                Neurotransmitters.kEp | \
                Neurotransmitters.kGlu | \
                Neurotransmitters.kGaba)
    

class MouseBrainNeuroLoader:
    class BrainRegion(Enum):
        kStriatum = 'striatum',
        kMidbrain = 'midbrain',
        kHippocampus = 'hippocampus',
        kOlfactoryBulb = 'olfactory_bulb',
        kFrontalCortex = 'frontal_cortex',
        kHypothalamus = 'hypothalamus',
        kCerebellum = 'cerebellum'
        kBrainstem = 'brainstem'
        kPituitaryGland = 'pituitary_gland'

        @property
        def name(self) -> str:
            return self.value[0]

    k_neuro_order = [
        Neurotransmitters.kBh4,
        Neurotransmitters.kDa,
        Neurotransmitters.k5Ht,
        Neurotransmitters.kNe,
        Neurotransmitters.kEp,
        Neurotransmitters.kGlu,
        Neurotransmitters.kGaba
        ]

    def __init__(self, working_dir: str) -> None:
        self._working_dir = working_dir

    def set_working_dir(self, new_working_dir: str) -> Mapping[Neurotransmitters, Interval]:
        self._working_dir = new_working_dir

    def load_single(self, brain_region: BrainRegion, neuro_filter: Filter = Neurotransmitters.filter_all())  -> None:
        with open(self._resolve_filename(brain_region.name), 'r') as f:
            data_lines = f.readlines()
            assert len(data_lines) == len(self.k_neuro_order)

            neuro_map: Mapping[Neurotransmitters, Interval] = {}
            for neuro, neuro_data_line in zip(self.k_neuro_order, data_lines):
                if not neuro_filter.filter & neuro:
                    continue

                splitted = neuro_data_line.split()
                assert len(splitted) == 2
                mid, rad = float(splitted[0]), float(splitted[1])

                neuro_map[neuro] = Interval(mid - rad, mid + rad)

            return neuro_map

    def _resolve_filename(self, filename: str) -> str:
        return f'{self._working_dir}/{filename}.txt'

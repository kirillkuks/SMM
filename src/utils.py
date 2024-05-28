from typing import List, TypeVar, Generic
T = TypeVar('T')
from dataclasses import dataclass

from intervals import Interval, IntervalMatrix, IntervalVector, Matrix


# T --- float or interval
@dataclass
class MassSpectrumData(Generic[T]):
    mass: List[float]
    intensity: List[T]

    def __init__(self, x_s: List[float], y_s: List[T]):
        self.mass = x_s.copy()
        self.intensity = y_s.copy()

    def add(self, x: float, y: T) -> None:
        self.mass.append(x)
        self.intensity.append(y)


def create_matrix_from_spectrum_data(spectrums: List[MassSpectrumData]) -> Matrix:
    max_mass = int(max([max(spectrum.mass) for spectrum in spectrums]))
    print(max_mass)
    spectrum_matrix = Matrix.zeroes(max_mass + 1, len(spectrums))

    for i, spectrum in enumerate(spectrums):
        for mass, intensity in zip(spectrum.mass, spectrum.intensity):
            assert type(intensity) is float
            spectrum_matrix[int(mass)][i] = intensity

    spectrum_matrix.print()
    return spectrum_matrix

def create_interval_matrix_from_spectrum_data(spectrums: List[MassSpectrumData]) -> IntervalMatrix:
    max_mass = int(max(([max(spectrum.mass) for spectrum in spectrums])))
    interval_spectrum_matrix = IntervalMatrix.zeroes(max_mass + 1, len(spectrums))

    for i, spectrum in enumerate(spectrums):
        for mass, interval_intensity in zip(spectrum.mass, spectrum.intensity):
            assert type(interval_intensity) is Interval
            interval_spectrum_matrix[int(mass)][i] = interval_intensity

    return interval_spectrum_matrix



def fill_with_zeroes(spectrum: MassSpectrumData, max_mass: int) -> MassSpectrumData:
    filled_mass = [float(i) for i in range(max_mass + 1)]
    filled_intensity = [0.0 for _ in range(max_mass + 1)]

    for mass, intensity in zip(spectrum.mass, spectrum.intensity):
        filled_intensity[int(mass)] = intensity

    return MassSpectrumData(filled_mass, filled_intensity)


def merge_mass_spectrums(spectrum1: MassSpectrumData, spectrum2: MassSpectrumData) -> MassSpectrumData[Interval]:
    max_mass = int(max(max(spectrum1.mass), max(spectrum2.mass)))

    spectrum1 = fill_with_zeroes(spectrum1, max_mass)
    spectrum2 = fill_with_zeroes(spectrum2, max_mass)

    def add_eps(interval : Interval) -> Interval:
        eps = 0.01
        return interval.add(eps) if interval.left < eps and interval.right > eps else interval

    mass_data = [float(i) for i in range(max_mass + 1)]
    ivec_spectrum_intensity_data = [
            add_eps(Interval(intensity1, intensity2, True))
                for i, (intensity1, intensity2)
                in enumerate(zip(spectrum1.intensity, spectrum2.intensity))
        ]
    
    return MassSpectrumData(mass_data, ivec_spectrum_intensity_data)


from enum import Enum

# NOTE errors
#  test_simulators /old modulenae, plot ratser path,
# test_live_plotter
# test_disable
# test_many-grids


class InterpolationEnum(Enum):
    NONE = ("none",)
    NEAREST = ("nearest",)
    BILINEAR = ("bilinear",)
    BICUBIC = ("bicubic",)
    SPLINE16 = ("spline16",)
    SPLINE36 = ("spline36",)
    HANNING = ("hanning",)
    HAMMING = ("hamming",)
    HERMITE = ("hermite",)
    KAISER = ("kaiser",)
    QUADRIC = ("quadric",)
    CATROM = ("catrom",)
    GAUSSIAN = ("gaussian",)
    BESSEL = ("bessel",)
    MITCHELL = ("mitchell",)
    SINC = ("sinc",)
    LANCZOS = ("lanczos",)

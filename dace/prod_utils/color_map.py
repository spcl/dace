import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

COMPILER_COLOR_MAP = {
    "gcc": "#1b9e77",  # green
    "llvm": "#d95f02",  # orange
    "intel": "#7570b3",  # purple
    "graceclang": "#e7298a",  # magenta
}


def lighten_color(color, amount=0.5):
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)

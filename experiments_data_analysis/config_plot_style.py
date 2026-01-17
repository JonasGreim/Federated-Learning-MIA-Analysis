import matplotlib as mpl
import seaborn as sns
from cycler import cycler
import locale

COLOR_PALETTE = [
    "#1f77b4",  # blue (main color)
    "#D55E00",  # orange
    "#009E73",  # green
    "#CC79A7",  # magenta
    "#F0E442",  # yellow
    "#333333",  # dark grey
]


def use_thesis_style():
    # set german locale for number formatting
    locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')

    mpl.rcParams.update({
        # german math notiation
        "axes.formatter.use_locale": True,

        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],

        # Size & Resolution
        "figure.figsize": (8, 6),
        "figure.dpi": 300,

        # Axes
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Ticks
        "xtick.major.size": 6,
        "xtick.major.width": 1.0,
        "ytick.major.size": 6,
        "ytick.major.width": 1.0,

        # Grids
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.2,

        # Lines & Colors
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "axes.prop_cycle": cycler(color=COLOR_PALETTE),

        # Text Sizes
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,

        # Legend
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "legend.frameon": True,

        # Saving
        "savefig.bbox": "tight",
    })

    # Seaborn
    sns.set_theme(
        style="ticks",
        palette=COLOR_PALETTE,
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
        },
    )

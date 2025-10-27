"""
Colours used in the plots and plot configurations
"""
import matplotlib


# Configuration for matplotlib plots
matplotlib.rcParams['text.latex.preamble'] = (r'\usepackage{amsmath}'
                                              r'\usepackage{siunitx}\sisetup{detect-all}'
                                              r'\usepackage{helvet}'
                                              r'\usepackage{sansmath}\sansmath')
matplotlib.rcParams['axes.grid'] = True

MAJOR: int = 34
MINOR: int = 30
FLAMINGO_TEST: list[str] = ['#FA07A0']  # Pink
BAHAMAS_AGN: list[str] = ['#5D4EF5', '#F54EDF']  # Purples
FLAMINGO: list[str] = ['#FABD00', '#FA7700', '#FA2100']  # Oranges
BAHAMAS_DMO: list[str] = ['#00FA8F', '#01FB3D', '#89FA00']  # Greens
BAHAMAS: list[str] = ['#0008E0', '#004FE0', '#0097E0', '#00DCE0']  # Blues

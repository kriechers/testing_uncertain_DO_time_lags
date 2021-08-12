

def set_pyplot(rc, rcParams,
               SMALL_SIZE=None,
               MEDIUM_SIZE=None,
               BIGGER_SIZE=None):
    rc('text', usetex=True)
    rc('text.latex', preamble=(r'\usepackage{amsmath}' +
                               r'\usepackage{wasysym}' +
                               r'\usepackage{xcolor}' +
                               r'\usepackage{textcomp}'))
    if SMALL_SIZE == None:
        SMALL_SIZE = 5
    if MEDIUM_SIZE == None:
        MEDIUM_SIZE = 7
    if BIGGER_SIZE == None:
        BIGGER_SIZE = 10

    rc('font', size=SMALL_SIZE)          # controls default text sizes
    rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    # fontsize of the figure cParams["xtick.major.size"] = 10
    rc('figure', titlesize=BIGGER_SIZE)
    rcParams["xtick.major.size"] = 4
    rcParams["xtick.major.width"] = 1
    rcParams["ytick.major.size"] = 4
    rcParams["ytick.major.width"] = 1
    rcParams['lines.linewidth'] = 0.8
    rcParams['axes.linewidth'] = 0.8


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

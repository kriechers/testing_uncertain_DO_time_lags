import numpy as np
import pandas as pd
import xarray as xr
import joblib as jl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

inch = 2.54
onecol = 8.3 / inch
twocol = 12 / inch
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=(r'\usepackage{amsmath}' +
                               r'\usepackage{wasysym}'))

cores = ['NGRIP', 'NEEM']
proxies = {'NGRIP': ['Ca', 'Na', 'lt', 'd18O'],
           'NEEM': ['Ca', 'Na']}

colors = {'d18O': 'C0',
          'Ca': 'C1',
          'Na': 'C2',
          'lt': 'C4'}

data = {'NGRIP': jl.load('%s.gz' % ('NGRIP')),
        'NEEM': jl.load('%s.gz' % ('NEEM'))}

events = data['NGRIP'].event.values
kdes = dict()


def linear_ramp(t, t0=0.0, dt=1.0, y0=0.0, dy=1.0):
    """Linear Ramp Function

    This function describes the linear transition between two constant values.

    Parameter
    ---------
    t : np.ndarray
        Time variable
    t0 : float
        Start time of the ramp
    dt : float
        Transition length
    y0 : float
        Function value before the transition
    dy : float
        Hight of the transion

    Return
    ------
    y : np.ndarray
        Function values of the linear transiton
    """
    lt_t0 = t < t0
    gt_t1 = t > t0 + dt
    condlist = [lt_t0,
                ~np.logical_or(lt_t0, gt_t1),
                gt_t1]
    funclist = [lambda t: y0,
                lambda t: y0 + dy * (t - t0) / dt,
                lambda t: y0 + dy]
    y = np.piecewise(t, condlist, funclist)
    return y


def calc_med_iqr(y, q=[5, 95], axis=None):
    qs = np.percentile(y, [q[0], 50, q[-1]], axis=axis)
    yerr = np.diff(qs, axis=0)
    y = qs[1]
    return y, yerr


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


for c in cores:
    for p in proxies[c]:
        for e in events:

            print(c, p, e)
            trace = data[c].sel(param=p, event=e).values

            if np.isnan(trace).all():
                print('continue')
                continue

            data_file = '../../../../datasets/new_ramp_data/%s_%s_%s.csv' % (
                c, e, p)

            t, obs = pd.read_csv(data_file).values.T
            t_ax = np.arange(np.min(t), np.max(t), 0.1)
            ramps = np.array([linear_ramp(t_ax, *p)
                              for p in trace[:, :4]])

            fig, ax = plt.subplots()

            r_med, r_err = calc_med_iqr(ramps, axis=0)

            ax.plot(t, obs,
                    color=colors[p],
                    lw=0.5, label=p)
            ax.plot(t_ax, r_med, color='k')
            ax.fill_between(t_ax, r_med - r_err[0], r_med + r_err[1],
                            alpha=.5, color='slategray')

            plt.savefig('check_transition_detection/%s_%s_%s.png' % (c, e, p),
                        dpi=300,
                        transparent=False)
            plt.close()

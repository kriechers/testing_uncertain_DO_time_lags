import numpy as np
import pandas as pd
import joblib as jl
import matplotlib.pyplot as plt
from tests import wilcoxon_signed_rank, ttest_1sided, bootstrap_test_v3, bootstrap_test
from pyplot_setting_paper import set_pyplot
from pyplot_setting_paper import make_patch_spines_invisible

set_pyplot(plt.rc, plt.rcParams)
inch = 2.54
onecol = 8.3 / inch
twocol = 12 / inch
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=(r'\usepackage{amsmath}' +
                               r'\usepackage{wasysym}'))

fig = plt.figure(figsize=(twocol, 0.5*onecol))

colors = {'t-test': 'C0',
          'WSR': 'C1',
          'bs': 'C2',
          'expected_sample': 'gold',
          'expected_p': 'magenta'}

label = {'Ca': r'$\text{Ca}^{2+}$',
         'Na': r'$\text{Na}^{+}$',
         'd18O': r'$\delta^{18}$O',
         'lt': r'$\lambda$'}


height_ratios = [2, 1]
gs = fig.add_gridspec(2, 20,
                      hspace=0,
                      wspace=0.1,
                      top=0.85,
                      left=0.1,
                      bottom=0.02,
                      right=0.95,
                      height_ratios=height_ratios)


combinations = {'NGRIP': [('Ca', 'Na'),
                          ('lt', 'Na'),
                          ('Ca', 'd18O'),
                          ('lt', 'd18O')],
                #                          ('lt', 'Ca')],
                'NEEM': [('Ca', 'Na')]}
par, ref_par = combinations['NGRIP'][0]
c = 0

for core in ['NGRIP', 'NEEM']:
    for par, ref_par in combinations[core]:
        data = jl.load('../data/MCMC_data/%s.gz' % (core)).sel(
            param=[par, ref_par], model='t0').dropna(dim='event')
        lag_data = -data.diff(dim='param').values.squeeze()
        lag_set = lag_data

        n = lag_set.shape[1]
        m = lag_set.shape[0]

        # significance level
        eta = 0.05

        # application of tests to uncertain sample
        z, pz = ttest_1sided(lag_set)
        w, pw = wilcoxon_signed_rank(lag_set)
        i = 10000
        u, pu, bs_shifted = bootstrap_test_v3(lag_set, i)

        # averaging over the uncertainties
        sample_ua = np.mean(lag_set, axis=0)[None, :]

        z_ua, pz_ua = ttest_1sided(sample_ua)
        w_ua, pw_ua = wilcoxon_signed_rank(sample_ua)
        u_ua, pu_ua, bs_shifted_ua = bootstrap_test_v3(sample_ua, i)

        # calculation of significant shares

        z_sig = np.sum(pz < eta)/m
        w_sig = np.sum(pw < eta)/m
        u_sig = np.sum(pu < eta)/m

        #####################################################################
        # writing the results into a table and storing this table in        #
        # outputs/hypothesis_tests.csv                                      #
        #####################################################################

        rownames = ['mean_p', 'p_of_mean', 'sig_share']
        colnames = ['z', 'w', 'bs']

        out = pd.DataFrame(np.array([[np.mean(pz), pz_ua, z_sig, ],
                                     [np.mean(pw), pw_ua, w_sig],
                                     [np.mean(pu), pu_ua, u_sig]]).T,
                           index=rownames,
                           columns=colnames)
        filename = ('outputs/hypothesis_tests_%s_%s_%s.csv'
                    % (par, ref_par, core))

        out.to_csv(filename)

        #################################################################
        # constructing the plots                                        #
        #                                                               #
        #################################################################

        ax1 = fig.add_subplot(gs[0, c:c+3])
        ax1.axhline(eta, linestyle=':', color='C3')
        ax1_1 = fig.add_subplot(gs[1, c])
        ax1_2 = fig.add_subplot(gs[1, c+1])
        ax1_3 = fig.add_subplot(gs[1, c+2])

        v1 = ax1.violinplot([pz, pw, pu],
                            positions=[0, 1, 2],
                            showmeans=True,
                            showextrema=False,
                            bw_method=0.1)
        v1['bodies'][0].set_color(colors['t-test'])
        v1['bodies'][0].set_alpha(0.8)
        v1['bodies'][0].set_lw(0.2)
        v1['bodies'][1].set_color(colors['WSR'])
        v1['bodies'][1].set_alpha(0.8)
        v1['bodies'][1].set_lw(0.2)
        v1['bodies'][2].set_color(colors['bs'])
        v1['bodies'][2].set_alpha(0.8)
        v1['bodies'][2].set_lw(0.2)
        v1['cmeans'].set_color(colors['expected_p'])

        ax1.plot([-0.15, 0.15], [pz_ua, pz_ua], color=colors['expected_sample'])
        ax1.plot([0.85, 1.15], [pw_ua, pw_ua], color=colors['expected_sample'])
        ax1.plot([1.85, 2.15], [pu_ua, pu_ua], color=colors['expected_sample'])
        ax1.set_xlim(-0.4, 2.4)
        ax1.set_ylim(-0.02, 1)
        ax1.xaxis.set_visible(False)
        ax1.set_title('('+label[par]+','+label[ref_par] + ')')

        if c != 0:
            ax1.yaxis.set_visible(False)
        else:
            ax1.set_ylabel(r'$\hat{p}$')

        ax1_1.pie([z_sig, 1-z_sig], explode=[0.1, 0],
                  colors=[colors['t-test'], 'C7'])
        ax1_2.pie([w_sig, 1-w_sig], explode=[0.1, 0],
                  colors=[colors['WSR'], 'C7'])
        ax1_3.pie([u_sig, 1-u_sig], explode=[0.1, 0],
                  colors=[colors['bs'], 'C7'])

        if ((par == 'lt') & (ref_par == 'd18O')):
            c += 5
        else:
            c += 4

space_ax = fig.add_subplot(gs[:, 15])
make_patch_spines_invisible(space_ax)
space_ax.plot([0, 0], [0, 1],
              color='lightsteelblue', lw=2)
space_ax.yaxis.set_visible(False)
space_ax.xaxis.set_visible(False)
space_ax.set_xlim((-2, 0))
space_ax.set_ylim((-0.1, 0.5))
space_ax.annotate('NGRIP', (0.3, 0.7),
                  fontsize=6,
                  rotation=-90,
                  xycoords='axes fraction')
space_ax.annotate('NEEM', (1.2, 0.7),
                  fontsize=6,
                  rotation=90,
                  xycoords='axes fraction')


legend_ax = fig.add_subplot(gs[0, 0])
make_patch_spines_invisible(legend_ax)
legend_ax.plot((), (), color=colors['t-test'], label='t-test')
legend_ax.plot((), (), color=colors['WSR'], label='WSR-test')
legend_ax.plot((), (), color=colors['bs'], label='bootstrap')
legend_ax.plot((), (), color=colors['expected_p'], label=r'E($\hat{P}$)')
legend_ax.plot((), (), color=colors['expected_sample'],
               label='$p$(E$(\mathbf{\Delta}\hat{\mathbf{T}})$)')
legend_ax.plot((), (), color='C3', ls=':', label=r'$\alpha$')

legend_ax.yaxis.set_visible(False)
legend_ax.xaxis.set_visible(False)
legend_ax.legend(loc='lower left',
                 bbox_to_anchor=(1.3, -0.55),
                 ncol=6,
                 markerscale=1,
                 framealpha=1,
                 handlelength=1,
                 frameon=False)

figname = '../../latex/revised_manuscript/figures/fig07.pdf'
plt.savefig(figname,
            dpi=300,
            transparent=True,
            bbox_to_inches='tight')

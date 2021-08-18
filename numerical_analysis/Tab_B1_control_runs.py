import numpy as np
import pandas as pd
import joblib as jl
import matplotlib.pyplot as plt
from tests import wilcoxon_signed_rank, ttest_1sided, bootstrap_test_v3


def shuffle(data):

    i = data.shape[1]
    j = data.shape[0]

    np.random.shuffle(data)

    return data


combinations = {'NGRIP': [('Ca', 'Na'),
                          ('lt', 'Na'),
                          ('Ca', 'd18O'),
                          ('lt', 'd18O')],
                #                          ('lt', 'Ca')],
                'NEEM': [('Ca', 'Na')]}

for core in ['NGRIP', 'NEEM']:
    print(core)
    for par, ref_par in combinations[core]:
        print(par, ref_par)

        ##########################################################
        # writing the results into a table and storing this table#
        # in outputs/hypothesis_tests.csv                        #
        ##########################################################

        quantities = ['E(p)', 'sig_share', 'p_of_E(\Delta T)']
        tests = ['z', 'w', 'bs']
        index = [par + '-' + ref_par + '-' + str(i) for i in range(10)]
        header = pd.MultiIndex.from_product([quantities, tests])
        out = pd.DataFrame(columns=header, index=index)

        for c in range(10):
            print(c)

            data = jl.load('../data/MCMC_data/%s.gz' % (core)).sel(
                param=[par, ref_par], model='t0').dropna(dim='event')
            if c == 0:
                lag_set = -data.diff(dim='param').values.squeeze()
            else:
                par_data = shuffle(data.sel(param=par).values)
                ref_par_data = shuffle(data.sel(param=ref_par).values)
                lag_set = shuffle(par_data - ref_par_data)

            n = lag_set.shape[1]
            m = lag_set.shape[0]
            # application of tests to uncertain sample
            eta = 0.05

            i = (10000)
            z, pz = ttest_1sided(lag_set)
            w, pw = wilcoxon_signed_rank(lag_set)
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

            # writing results in the outputfile

            out.iloc[c] = [np.mean(pz), np.mean(pw), np.mean(pu),
                           z_sig, w_sig, u_sig,
                           pz_ua[0], pw_ua[0],  pu_ua[0]]

        filename = ('outputs/control_%s_%s_%s.csv'
                    % (core, par, ref_par))
        out.to_csv(filename)

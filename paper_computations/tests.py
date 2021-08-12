import numpy as np
from scipy.stats.distributions import t
import pandas as pd
import joblib as jl
import matplotlib.pyplot as plt
from itertools import permutations, product
from os.path import isfile


plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=(r'\usepackage{amsmath}' +
                               r'\usepackage{wasysym}'))


def wilcoxon_distribution(n):
    temp = [(1, 0)]*n
    stats = []
    rank = np.arange(1, n+1)
    for i in product(*temp):
        ranksum = np.sum(rank * i)
        stats.append(ranksum)
    w, counts = np.unique(stats, return_counts=True)
    prob = counts / (2 ** n)
    distribution = pd.DataFrame(np.array([w, prob]).T, columns=['w', 'p(w)'])
    return distribution


def wilcoxon_signed_rank(data, mode='left-sided'):
    n = data.shape[0]
    m = data.shape[1]
    order = np.argsort(np.abs(data), axis=1)
    sorted_data = data[np.arange(n)[:, None], order]
    temp = np.where(sorted_data > 0,
                    np.tile(np.arange(1, m+1), (n, 1)),
                    0)
    ranksum = np.sum(temp, axis=1)
    filename = 'wilcoxon_dist_n%i.csv' % (m)
    filepath = ('/home/riechers/Projects/DO_onset_phasing/'
                + 'evidence_from_uncertain_samples/modules/'
                + 'wilcoxon_distributions/')
    if isfile(filepath + filename):
        dist = pd.read_csv(filepath + filename)
    else:
        dist = wilcoxon_distribution(n)
        dist.to_csv(filepath + filename)
    if mode == 'left-sided':
        mask = np.where(dist['w'][:, None] < ranksum[None, :],
                        np.tile(dist['p(w)'], (n, 1)).T,
                        0)

    elif mode == 'rigth-sided':
        mask = np.where(dist['w'][:, None] > ranksum[None, :],
                        np.tile(dist['p(w)'], (n, 1)).T,
                        0)
    else:
        mask = np.where(dist['p(w)'][:, None] <=
                        dist['p(w)'][None, :][0, ranksum],
                        np.tile(dist['p(w)'], (n, 1)).T,
                        0)
    pval = np.sum(mask, axis=0)
    return ranksum, pval


def ttest_1sided(data, mu=0):
    '''
    calculates the z value and the corresponding pz value for an uncertain
    sample, given by a matrix, where each columns represents the distribution
    of one sample member. the test is applied from the left side, that is
    if the test yields significant results, one should reject the hypothesis 
    that the population mean is greater or equal mu. 

    Inputs
    -------------
    data:= a 2d ndarray (v,w) representing an uncertain sample with w being 
           the number of sample members and v being the number of member 
           contained in the set that represents the uncertainty of each 
           sample member. 
    mu  := the population that shall be tested (left-sided)

    Outputs
    -------------
    z   := a set of v z values, representing the uncertainty inhereted from 
           the uncertainty of the original sample 
    pz  := a set of v p-values, corresponding to each z value. 
    '''
    n = data.shape[1]
    u = np.mean(data, axis=1)
    s = np.std(data, axis=1)
    z = u/(s / np.sqrt(n))
    p = t.cdf(z, df=n-1, loc=mu)
    return z, p


def bs_population_mean(samples, n):
    nevent = samples.shape[1]
    bootstrap = pd.DataFrame()
    for i, sample in enumerate(samples):
        bt_idx = np.random.randint(0, nevent, (nevent, n))
        bootstrap[i] = np.mean(sample[bt_idx], axis=0)

    means = bootstrap.values.T
    return means


def bootstrap_test(data, v):
    u = np.mean(data, axis=1)
    w = len(u)
    bs = bs_population_mean(data, v)
    bs_shifted = (bs - u[:, None]).flatten()
    p = np.sum(u[:, None] > bs_shifted[None, :], axis=1)/(v*w)
    return u, p, bs_shifted.flatten()


def bootstrap_test_v2(data, m):
    u = np.mean(data, axis=1)
    pop_est = (data - u[:, None]).flatten()
    idx = np.random.randint(0, len(pop_est), (data.shape[1], m))
    null_dist = np.mean(pop_est[idx], axis=0)
    p = np.sum(u[:, None] > null_dist[None, :], axis=1)/(m)


def bootstrap_test_v3(data, k):
    l = data.shape[0]
    m = data.shape[1]
    u = np.mean(data, axis=1)
    shifted = (data - u[:, None])
    null_dist = np.zeros((l, k))

    for i, s in enumerate(shifted):
        idx = np.random.randint(0, m, (m, k))
        null_dist[i, :] = np.mean(s[idx], axis=0) * np.sqrt(m)

    p = np.sum(u[:, None] * np.sqrt(m) > null_dist, axis=1)/(k)
    return u * np.sqrt(m), p, null_dist.flatten()


def hypothesis_testing(par, ref_par, core):

    data = jl.load('../data/MCMC_data/%s.gz' % (core)).sel(
        param=[par, ref_par], model='t0').dropna(dim='event')
    lag_data = -data.diff(dim='param').values.squeeze()
    lag_set = lag_data

    n = lag_set.shape[1]
    m = lag_set.shape[0]

    tests = ['ttest', 'WSR', 'bs']
    idx = pd.MultiIndex.from_product([tests, ['stat', 'pval']])
    results = pd.DataFrame(data=np.full((6000, 6), np.nan),
                           columns=idx)

    # significance level
    eta = 0.05

    # application of tests to uncertain sample
    z, pz = ttest_1sided(lag_set)
    results.loc[:, ('ttest', 'stat')] = z
    results.loc[:, ('ttest', 'pval')] = pz
    w, pw = wilcoxon_signed_rank(lag_set)
    results.loc[:, ('WSR', 'stat')] = w
    results.loc[:, ('WSR', 'pval')] = pw
    i = (10000)
    u, pu, bs_shifted = bootstrap_test_v3(lag_set, i)
    results.loc[:, ('bs', 'stat')] = u
    results.loc[:, ('bs', 'pval')] = pu

    # averaging over the uncertainties
    uncertainty_averaged = pd.DataFrame(columns=idx)
    sample_ua = np.mean(lag_set, axis=0)[None, :]

    z_ua, pz_ua = ttest_1sided(sample_ua)
    uncertainty_averaged.loc[0, 'ttest'] = (z_ua[0], pz_ua[0])
    w_ua, pw_ua = wilcoxon_signed_rank(sample_ua)
    uncertainty_averaged.loc[0, 'WSR'] = (w_ua[0], pw_ua[0])
    u_ua, pu_ua, bs_shifted_ua = bootstrap_test_v3(sample_ua, 10000)
    uncertainty_averaged.loc[0, 'bs'] = (u_ua[0], pu_ua[0])
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
    filename = ('outputs/hypothesis_tests_%s_%s.csv'
                % (par, ref_par))

    out.to_csv(filename)

    return results, uncertainty_averaged, bs_shifted

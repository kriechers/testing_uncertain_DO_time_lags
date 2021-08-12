import numpy as np
import pandas as pd


combinations = {'NGRIP': [('Ca', 'Na'),
                          ('lt', 'Na'),
                          ('Ca', 'd18O'),
                          ('lt', 'd18O')],
                'NEEM': [('Ca', 'Na')]}

label = {'Ca': r'$\text{Ca}^{2+}$',
         'Na': r'$\text{Na}^{+}$',
         'd18O': r'$\delta^{18}$O',
         'lt': r'$\lambda$'}


idx1 = []
for core in ['NGRIP', 'NEEM']:
    for x in combinations[core]:
        idx1.append(core + ':' + x[0] + '-' + x[1])

rowidx = pd.MultiIndex.from_product([idx1, np.arange(10)])

idx2 = ['expected p-value', 'share of significant p-values', 'p-value of expected sample']
idx3 = ['t-test', 'WSR-test', 'bootstrap-test']

colidx = pd.MultiIndex.from_product([idx2, idx3])
df = pd.DataFrame(index=rowidx, columns=colidx)

for core in ['NGRIP', 'NEEM']:
    for x in combinations[core]:
        filename = 'control_%s_%s_%s.csv' % (core, x[0], x[1])
        data = pd.read_csv(filename,
                           header=[0, 1],
                           index_col=0)
        rows = core + ':' + x[0] + '-' + x[1]
        df.loc[rows, :] = np.round(data.values, 3)

df.to_csv('table_B1.csv')
          

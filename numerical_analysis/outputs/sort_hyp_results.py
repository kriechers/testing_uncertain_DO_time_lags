import numpy as np
import pandas as pd


combinations = {'NGRIP': [('Ca', 'Na'),
                          ('lt', 'Na'),
                          ('Ca', 'd18O'),
                          ('lt', 'd18O')],
                'NEEM': [('Ca', 'Na')]}

idx1 = []
for core in ['NGRIP', 'NEEM']:
    for x in combinations[core]:
        idx1.append(core + ':'+ x[0] + '-' + x[1])
        
idx2 = ['z', 'w', 'bs']
idx3 = ['mean_p','p_of_mean', 'sig_share']

idx = pd.MultiIndex.from_product([idx1, idx2])
df = pd.DataFrame(index = idx3, columns = idx)

for core in ['NGRIP', 'NEEM']:
    for x in combinations[core]:
        filename = 'hypothesis_tests_%s_%s_%s.csv' %(x[0], x[1], core)
        data = pd.read_csv(filename,
                           header = 0,
                           index_col = 0)
        loc_idx = core + ':'+ x[0] + '-' + x[1]
        df.loc[:,loc_idx] = np.round(data.values,2)

df.to_csv('hyp_test_gathered.csv',
          sep = '&')
            

    

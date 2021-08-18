numerical_analysis
==================

All numerical computations have been carried out with pyhton3.5.
A list of all installed packages can be found in the
requirements.txt files. However, that does not mean, that all
packages listed in that file are mandatory for the analysis. 

The python file named **figXX_*.py** corresponds to the figure XX
in the article. Execution of the file will produce the figure and
store it as **figXX.pdf** in the **figures** directory.

Execution of the file **fig07_results_overview.py** creates in
addition to the **fig07.pdf** the files
**hypothesis_tests_par_refpar_core.csv** which are saved in the
**outputs** directory. These files contain the results of the three
generalized hypothesis test applied to the time lags between par
and refpar (e.g. par = Ca2+, refpar = Na+) from the core NGRIP or
NEEM. These results are summarized in Table 1 of the article.

Execution of the file **fig05_uncertain_sample_mean.py** creates a
file **sample_mean_statistics.csv**, which is saved in the **outputs**
directory and contains the (5, 50, 95) percentiles and the
probability to be <0 of the 'uncertain sample mean' and the
'combined estimate' as defined in the article for all proxy pairs
under study. The statistics summarized in this file are displayed
in fig05.pdf.

Execution of **Tab_B1_control_runs.py** generates files
**control_core_par_refpar.csv** with core being either NGRIP or
NEEM and par and refpar indicating the proxy pair to which the
control analysis is applied. These files are save in the
**output** directory. 

All additional python files contain functions which are used by
the figXX_*.py and **Tab_B1_control_runs.py**. 

The directory **wilcoxon_distributions** contains the null
distributions for Wilcoxon signed-rank tests for different sample
sizes. 
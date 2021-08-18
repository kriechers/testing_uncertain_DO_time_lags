testing_uncertain_DO_time_lags
==============================

This repository contains all the data and code to reproduce the
results presented in

*Riechers K. and Boers N.: Significance of
uncertain phasing between the onsets of stadial-interstadial
transitions in different Greenland ice-coreproxies, Climate of
the past, 2021* .

This repository was created before publication,
thus reference details are not available at this stage. The
corresponding discussion paper is

Riechers, K. and Boers, N.: A statistical approach to the phasing
of atmospheric reorganization and sea ice retreat at the onset of
Dansgaard-Oeschger events under rigorous treatment of
uncertainties, Clim. Past Discuss. [preprint],
https://doi.org/10.5194/cp-2020-136, in review, 2020.


The code presented here, strongly builds upon the study

Erhardt, T., Capron, E., Rasmussen, S. O., Schüpbach, S., Bigler,
M., Adolphi, F., and Fischer, H.: Decadal-scale progression of
the onset of Dansgaard–Oeschger warming events, Clim. Past, 15,
811–825, https://doi.org/10.5194/cp-15-811-2019, 2019.


The directory **numerical_analysis** contains all relevant pyhton
code. For each figure in the article, there is a corresponding
python script and the execution thereof will generate the figure
and store it into **numerical_analysis/figures**.

The **data** directory contains all the data used by the python
code. The data comprises original proxy data as well as secondary
data derived from the proxy data. For the references of the data,
please see the **readme.md** inside the **data** directory. 

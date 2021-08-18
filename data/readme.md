Data
====

In conjunction with the study

Erhardt, T., Capron, E., Rasmussen, S. O., Schüpbach, S., Bigler,
M., Adolphi, F., and Fischer, H.: Decadal-scale progression of
the onset of Dansgaard–Oeschger warming events, Clim. Past, 15,
811–825, https://doi.org/10.5194/cp-15-811-2019, 2019.

new 10 year averaged calcium and sodium aerosol data for the NEEM
and NGRIP ice cores have been published. Moreover, high
resolution data for the same proxies (both ice cores) together
with annual layer thickness data (NGRIP only) have been published
for time windows around Greenland warming events. Please see:

Erhardt, T., Capron, E., Rasmussen, S. O., Schüpbach, S., Bigler,
M., Adolphi, F., and Fischer, H.: High resolution aerosol, layer
thickness and δ18O data around Greenland warming events (10–60
ka) from NGRIP and NEEM ice cores, PANGAEA [data set],
https://doi.org/10.1594/PANGAEA.896743, 2018.

The files

* NGRIP_10yr.csv
* NEEM_10yr.csv

contain the 10 year averaged calcium and sodium aerosol data. The
**NGRIP** file additionally contains corresponding data for
annual layer thickness and d18o values. Both files have been
directly obtained from https://doi.org/10.5281/zenodo.2645175.

The file **GIS_table.txt** contains information high resolution
data for time windows around Greenland warmings which are stored
in the **ramp_data** directory. The **GIS_table** as well as all
data inside **ramp_data** was directly obtained from
https://doi.org/10.5281/zenodo.2645175.

The **MCMC_data** directory contains secondary data, which was
derived from the time series inside **ramp_data** by means of a
software provided by Tobias Erhardt under
https://doi.org/10.5281/zenodo.2645175. 

For a time series that includes a transition from one level of
values to another, Tobias Erhardt has published a software to
sample from the joint Bayesian posterior probability
distributions for the model parameters of a corresponding
stochastic linear ramp model that approximates the time series.

The files 'NEEM.gz' and 'NGRIP.gz' which are stored in the Data
directory of this repository have been generated by the
application of this MCMC based sampling software to the high
resolution time series of Ca2+, Na+, d18o (NGRIP only) and the
annual layer thickness (NGRIP only) around Greenland warming
events as provided by Erhardt et al. (2018). The files contain
6000 samples from the 6 dimensional model's parameter space for
each time series comprised in the data.


---
Please note that the d18o data that was used by Erhardt et
al. (2019) and is contained in the data files was previously
published:

Gkinis, V., Simonsen, S. B., Buchardt, S. L., White, J. W., and
Vinther, B. M.: Water isotope diffusion rates from the North-
GRIP ice core for the last 16,000 years – Glaciological and pa-
leoclimatic implications, Earth Planet. Sc. Lett., 405, 132–141,
https://doi.org/10.1016/j.epsl.2014.08.022, 2014.

North Greenland Ice Core Project members: High-resolution record
of Northern Hemisphere climate extending into the last
interglacial period, Nature, 431, 147–151,
https://doi.org/10.1038/nature02805, 2004.

Futhermore, all ages assigned to the data are based on the GICC05
ages scale for the NEEM and NGRIP ice cores:

Andersen, K. K., Svensson, A., Johnsen, S. J., Rasmussen, S. O.,
Bigler, M., Röthlisberger, R., Ruth, U., Siggaard-Andersen, M.
L., Peder Steffensen, J., Dahl- Jensen, D., Vinther, B. M., and
Clausen, H. B.: The Greenland Ice Core Chronology 2005, 15–42 ka.
Part 1: construct- ing the time scale, Quaternary Sci. Rev., 25,
3246–3257, https://doi.org/10.1016/j.quascirev.2006.08.002, 2006.

Rasmussen, S. O., Andersen, K. K., Svensson, A. M., Stef- fensen,
J. P., Vinther, B. M., Clausen, H. B., Siggaard- Andersen, M. L.,
Johnsen, S. J., Larsen, L. B., Dahl-Jensen, D., Bigler, M.,
Röthlisberger, R., Fischer, H., Goto-Azuma, K., Hansson, M. E.,
and Ruth, U.: A new Greenland ice core chronology for the last
glacial termination, J. Geophys. Res.-Atmos., 111, 1–16,
https://doi.org/10.1029/2005JD006079, 2006.

Svensson, A., Andersen, K. K., Bigler, M., Clausen, H. B., Dahl-
Jensen, D., Davies, S. M., Johnsen, S. J., Muscheler, R.,
Parrenin, F., Rasmussen, S. O., Röthlisberger, R., Seierstad, I.,
Steffensen, J. P., and Vinther, B. M.: A 60 000 year Greenland
stratigraphic ice core chronology, Clim. Past, 4, 47–57,
https://doi.org/10.5194/cp-4-47-2008, 2008.

Rasmussen, S. O., Abbott, P. M., Blunier, T., Bourne, A. J.,
Brook, E., Buchardt, S. L., Buizert, C., Chappellaz, J., Clausen,
H. B., Cook, E., Dahl-Jensen, D., Davies, S. M., Guillevic, M.,
Kipfstuhl, S., Laepple, T., Seierstad, I. K., Severinghaus, J.
P., Steffensen, J. P., Stowasser, C., Svensson, A., Vallelonga,
P., Vinther, Vinther, B. M., Wilhelms, F., and Winstrup, M.: A
first chronol- ogy for the North Greenland Eemian Ice Drilling
(NEEM) ice core, Clim. Past, 9, 2713–2730,
https://doi.org/10.5194/cp-9- 2713-2013, 2013.
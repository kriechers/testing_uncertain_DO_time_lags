from scipy.signal import fftconvolve


def sample_mean_distr(x_ax, pdfs, dx = None):
    '''
    given an uncertain sample, this function calculates the distribution 
    of the uncertain sample mean according to the equation: 
    rho(u) = \int \prod(pdfs_i(x_i)) \delta(u-\sum(x_i)/n) dx_1, dx_2...
    this equation can be transformed into an interative convolution of 
    individual probability densities (see article)

    Inputs:
    -----------------
    pdfs := a list of probability densities corresponding to the sample 
            members given as functions that can be evaluated on x_ax
    x_ax := an x-axis on which the probability densities of the sample 
            members are defined.

    Outputs:
    ----------------
    u_pdf := the probabilty density function of the stochastic sample mean, 
             output given as discrete values corresponding to the u_ax, 
             such that sum(u_pdf*dz) = 1
    u_ax := a new axis on which the u_pdf is defined (due to convolution)

    ATTENTION: note that x_ax has to be chosen broad enough. In cause of 
               the iterative convolution the axis 'shrinks' in each step.
    '''

    n = len(pdfs)
    if dx is None:
        dx = x_ax[1]-x_ax[0]
    u_ax = x_ax/n
    du = dx/n
    conv = []
    for i, rho in enumerate(pdfs):
        if i == 0:
            conv.append(rho * dx)
        else:
            conv.append(fftconvolve(conv[i-1], rho*dx, mode='same'))
    pdf_mean = conv[-1] / du
    return u_ax, pdf_mean

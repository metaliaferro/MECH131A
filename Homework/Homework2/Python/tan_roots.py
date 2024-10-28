import numpy as np
from scipy.optimize import root_scalar


def roots(N, Bi):
    """
    This function returns the first `N` roots of the transcendental equation:

        tan(x) = Bi/x
    
    It has not been optimized for speed.
    """

    zeros = np.zeros(N)
    for n in np.arange(N):
        limit_min = max(1e-8, (n - 0.5)*np.pi + 1e-8)  # the addition of 1e-8 ensures bracket is just inside the left boundary
        limit_max = (n + 0.5)*np.pi - 1e-9  # the subtraction of 1e-9 ensures bracket is just inside the left boundary and that the midpoint between the two brackets is not zero
        guess = 0.5*(limit_min + limit_max)
        # res = root(lambda x: np.tan(x) - Bi/x, guess, )   # this method did not work for large Bi
        res = root_scalar(
            lambda x: np.tan(x) - Bi/x,
            bracket=[limit_min, limit_max]
        )   # lambda here is not the eigen value, but an anonymous python function
        zeros[n] = 1.0*res.root
    return zeros



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N = 100
    x_plt = np.linspace(0.0, (N + 1)*np.pi, (N + 1)*1000)
    for Bi in [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]:
        lambda_t = roots(N, Bi)
        # plot the 1/x decaying part
        plt.plot(x_plt, Bi/x_plt)

        # Plot the tan part after removing the part that steps over the asymptote
        tan = np.tan(x_plt)
        diff = np.diff(tan)
        tan[0:-1][diff < 0.0] = np.nan
        plt.plot(x_plt, tan)

        # Plot the roots
        plt.plot(lambda_t, Bi/lambda_t, marker='o', linestyle='None')

        # Formatting and show the results
        plt.ylim(-0.1*Bi, Bi)
        plt.title('Bi = {0:.1e}'.format(Bi))
        plt.show()

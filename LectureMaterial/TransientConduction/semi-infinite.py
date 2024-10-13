import numpy as np
from scipy.special import erf

def semi_infinite(alpha, t, x):
    return erf(0.5*x/np.sqrt(alpha*t))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from slab import sov_temperature as slab_SOV

    L = 0.1
    alpha = 1e-5

    t = np.array([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
    x = np.linspace(0.0, L, 1001)

    Fo, X = np.meshgrid(alpha*t/(L*L), x/L)
    T_exact = slab_SOV(300, Fo, X)
    color_wheel = [
        'tab:blue',
        'tab:orange',
        'tab:green',
        'tab:red',
        'tab:purple',
        'tab:brown'
    ]
    for n_col, fo in enumerate(Fo.T):
        # Plot semi-infinite solution
        T_si = np.array([semi_infinite(alpha, L*L*fo[0]/alpha, x*L) for x in X[:, n_col]])
        plt.plot(
            X[:, n_col], T_si,
            color=color_wheel[n_col],
        )

        # Plot slab solution
        plt.plot(
            X[:, n_col], T_exact[:, n_col],
            label='Fo = {0:.2e}'.format(fo[0]),
            color=color_wheel[n_col],
            linestyle='-.',
        )
    plt.legend()
    plt.show()


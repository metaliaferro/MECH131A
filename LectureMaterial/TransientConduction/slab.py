import numpy as np


def sov_temperature(N, Fo, X):
    T = 0.0*Fo

    for n in 2.0*np.arange(1, N + 1) - 1:
        prefactor = 4.0/(n*np.pi)
        time_term = np.exp(-np.square(n*np.pi/2.0)*Fo)
        x_term = np.sin(n*np.pi*X/2.0)

        T += prefactor*time_term*x_term
    
    return T


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    L = 0.1
    alpha = 1e-5

    t = np.array([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
    x = np.linspace(0.0, L, 1001)

    Fo, X = np.meshgrid(alpha*t/(L*L), x/L)

    for N in [1, 3, 10, 30, 100, 300]:
        # Get temperature profile
        T = sov_temperature(N, Fo, X)

        # Plot solution at small Fo numbers
        plt.plot(X[:,0], T[:,0], label='N = {0:d}'.format(N))

    plt.title('Convergence?')
    plt.legend()
    plt.show()

    T_exact = sov_temperature(300, Fo, X)
    T_oneterm = sov_temperature(1, Fo, X)
    color_wheel = [
        'tab:blue',
        'tab:orange',
        'tab:green',
        'tab:red',
        'tab:purple',
        'tab:brown'
    ]
    for n_col, fo in enumerate(Fo.T):
        plt.plot(
            X[:, n_col], T_exact[:, n_col],
            label='Fo = {0:.2e}'.format(fo[0]),
            color=color_wheel[n_col]
        )
        plt.plot(
            X[:, n_col], T_oneterm[:, n_col],
            linestyle='-.',
            color=color_wheel[n_col]
        )
    
    plt.title('Changing Temperature Profile')
    plt.legend()
    plt.show()

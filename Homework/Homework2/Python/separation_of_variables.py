import numpy as np


def sov_temperature(N, X, Y, L, W):
    """
    This returns the analytical series solution for a plate with one side held
    at a non-dimensional temperature of one.

    inputs:
        N the number of terms to include in the series
        x dimensional x-coordinate
        y dimensional y-coordinate
        L length scale in the x-direction
        W length scale in the y-direction
    
    output:
        T non-dimensional temperature
    """
    T = 0.0*X

    for n in np.arange(1, N + 1):
        prefactor = (np.power(-1.0, n + 1) + 1)/n
        x_term = np.sin(n*np.pi*X/L)
        y_term = np.sinh(n*np.pi*Y/L)/np.sinh(n*np.pi*W/L)

        T += prefactor*x_term*y_term
    

    return T*2.0/np.pi


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    L = 3.0
    W = 0.5

    x = np.linspace(0.0, L)
    y = np.linspace(0.0, W)

    X, Y = np.meshgrid(x, y)

    for plt_ii, N in enumerate([1, 3, 10, 30, 100, 300]):
        # Get temperature profile
        T = sov_temperature(N, X, Y, L, W)

        # Plot the contour of the temperature solution
        plt.subplot(2, 3, plt_ii + 1)
        plt.contourf(X, Y, T)

        # Numerically estimate the flux gradients and plot
        if N > 3:
            dTdx = (T[:, 1:] - T[:, :-1])/(X[:, 1:] - X[:, :-1])
            dTdy = (T[1:, :] - T[:-1, :])/(Y[1:, :] - Y[:-1, :])
            plt.quiver(X[1:,1:], Y[1:,1:], -dTdx[1:, :], -dTdy[:, 1:])
        
        plt.title('N = {0:d}'.format(N))

    plt.suptitle('Convergence?')
    plt.tight_layout()
    plt.show()

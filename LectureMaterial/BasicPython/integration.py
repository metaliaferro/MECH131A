import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import cumulative_trapezoid

# Move these to another file for scope reasons
def parabola_for_loop(x, a, b, c):
    y = 0.0*x
    for ii, p in enumerate([a, b, c]):  # python is zero indexed
        y += p*np.power(x, ii)
    return y

def parabola_polyval(x, p):
    """
    p = [c, b, a]
    """
    return np.polyval(p, x)


if __name__ == "__main__":
    """
    The above if statement means this section of code only runs if its called
    from the command line. If its called another way, say by importing it, this
    seciton of the code is not run
    """
    # Define constants so you only have to change them in one place
    # Be careful about floats and ints
    x0 = -5.0
    xf = 5.0
    N = 101
    x = np.linspace(x0, xf, N)

    a = -1.0
    b = 2.0
    c = 5.0

    plt.plot(x, parabola_for_loop(x, a, b, c))
    plt.plot(x, parabola_polyval(x, [c, b, a]), marker='o', linestyle='None')
    plt.show()


    yint = cumulative_trapezoid(parabola_for_loop(x, a, b, c), x=x)
    plt.subplot(2,1,1)
    plt.plot(x, parabola_for_loop(x, a, b, c))
    plt.subplot(2,1,2)
    plt.plot(x[1:], yint)
    plt.show()


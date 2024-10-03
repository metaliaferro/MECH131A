"""
You can import modules you have downloaded for extra functionality. This is
generally better than trying to write the functionality yourself. Generally
aliases are used, for example `np` instead of `numpy`, as they are easier to
type. Some people will judge you for this.

If you just want a dingle function, such as `cumulative_trapezoid` below, you
can use the `from import` statement. Some large packages, such as `scipy`, will
not load all of their contents for use when you import them and the
`from import` statement is a convenient way around this.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import cumulative_trapezoid

def parabola_for_loop(x, a, b, c):
    """
    A relatively bad way to calculate a parabola. But at least its clear. Often
    the `enumerate` function is superfluous (maybe try `zip`), but it was useful
    in this case.
    """
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
    section of the code is not run
    """
    # Define constants so you only have to change them in one place
    # Be careful about floats vs ints
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


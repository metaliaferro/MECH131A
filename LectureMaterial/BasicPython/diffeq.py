import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import minimize


"""
This script is an example of integrating a second order differential equation as
a boundary value problem (as opposed to an initial value problem).

`ode_error` takes in the required arguments, calls `solve_ivp`, and then returns
squared difference between the calculated surface temperature and the user
specified surface temperature.

The centerline temperature is guessed (initial guess 450 K), and  `minimize` from
`scipy.optimize` minimizes the error defined in ode_error.

The plotting functionality is left unchanged as was used to debug the code
during development. It is likely you would want to comment this out when you use
it "for real".
"""

def cylinder_analytical(r, r0, k, q, Ts):
    """
    Equation 3.58
    """
    return ((q*r0*r0)/(4.0*k))*(1.0 - np.square(r/r0)) + Ts

def cylinder_ode(r, y, k, q):
    """
    Integrates equation 3.54 from the textbook:

    d2T/dr2 + (1/r)*dTdr + q/k = 0

    Split this into a system of ODEs:

    dy0 = d2T/dr2
    dy1 = dTdr

    -> dy0/dr = -(1/r)*dy1 - q/k
    -> dy1/dr = y0
    """

    dy = 0.0*y
    dy[0] = - (1.0/r)*y[0] - q/k
    dy[1] = y[0]

    return dy

def ode_error(T_guess, *args):
    """
    Integrates the system of differential equations and returns the square of
    the difference between the user defined surface temperature, `Ts`, and the
    calculated surface temperature using the initial value.

    You probably want to comment out the plotting functionality, but it was
    useful while writing this script.
    """
    r0, k, q, Ts = args
    sol = solve_ivp(cylinder_ode, [1e-8, r0], [0.0, T_guess[0]], args=(k, q))

    plt.plot(sol.t, sol.y[1])
    plt.plot(sol.t, cylinder_analytical(sol.t, r0, k, q, Ts))
    plt.plot(sol.t[-1], sol.y[1][-1], marker='o')
    plt.plot(sol.t[-1], Ts, marker='o')
    plt.show()
    return np.square(Ts - sol.y[1][-1])

r0 = 1e-3   # m
k = 1.4e1   # W/m-K
q = 1e9     # W/m3
Ts = 3e2    # K

res = minimize(ode_error, [450.0], args=(r0, k, q, Ts))
print(res)

sol = solve_ivp(cylinder_ode, [1e-8, r0], [0.0, res.x[0]], args=(k, q))

# The orange markers, the integrated solution, should match the blue line, the
# analytical solution.
plt.plot(sol.t, cylinder_analytical(sol.t, r0, k, q, Ts))
plt.plot(sol.t, sol.y[1], marker='o', linestyle='None')
plt.show()

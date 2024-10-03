import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from integration import parabola_for_loop

def function_to_minimize(x, *args):
    a, b, c = args
    error = parabola_for_loop(x, a, b, c)
    return np.square(error)

a = -1.0
b = 2.0
c = 5.0

# Things to check if you are frustrated with the output: method, tolerances, and
# options 
res = minimize(function_to_minimize, -3.0, args=(a, b, c))
print(res)

x0 = -5.0
xf = 5.0
N = 101
x = np.linspace(x0, xf, N)
plt.plot(x, parabola_for_loop(x, a, b, c))
plt.plot(res.x, parabola_for_loop(res.x, a, b, c), marker='o', linestyle='None')
plt.show()
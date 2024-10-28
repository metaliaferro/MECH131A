import numpy as np
import matplotlib.pyplot as plt

from tan_roots import roots
from separation_of_variables import sov_temperature as SOV


N = 200
L = 1.0
W = 1.0

x = np.linspace(0.0, L)
y = np.linspace(0.0, W)
X, Y = np.meshgrid(x, y)

T = SOV(N, X, Y, L, W)

plt.figure()
V, U = np.gradient(-T, x, y)
plt.contourf(X, Y, T)
plt.colorbar()
plt.streamplot(X, Y, U, V)

plt.figure()
V, U = np.gradient(- T - T.T, x, y)
plt.contourf(X, Y, T + T.T)
plt.colorbar()
plt.streamplot(X, Y, U, V)

plt.figure()
V, U = np.gradient(- T - np.flipud(T), x, y)
plt.contourf(X, Y, T + np.flipud(T))
plt.colorbar()
plt.streamplot(X, Y, U, V)

plt.figure()
V, U = np.gradient(- T - T.T - np.fliplr(T.T), x, y)
plt.contourf(X, Y, T + T.T + np.fliplr(T.T))
plt.colorbar()
plt.streamplot(X, Y, U, V)


plt.show()

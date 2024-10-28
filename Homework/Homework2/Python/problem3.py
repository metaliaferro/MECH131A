import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

from tan_roots import roots

def solution(lambda_t, t, X, Y):
    T = 0.0*X
    for lt in lambda_t:
        amplitude = np.sin(lt)/(lt + np.sin(lt)*np.cos(lt))
        eigenfunction = np.exp(-lt*X/t)*np.cos(lt*Y/t)
        T += amplitude*eigenfunction
    return 2.0*T

# Number of terms
x = np.linspace(0.0, 10.0, 1001)
y = np.linspace(0.0, 1.0, 101)
X, Y = np.meshgrid(x, y)
for Bi in [1e-3, 1e3]:
    all_roots = roots(300, Bi)
    plt.figure()
    for ii, N in enumerate([1, 3, 10, 30, 100, 300]):
        T = solution(all_roots[:N], 1.0, X, Y)
        plt.subplot(2, 3, ii + 1)
        plt.contourf(X, Y, T)
        plt.colorbar()
        plt.title('N = {0:d}'.format(N))
    plt.suptitle('Bi = {0:.1e}'.format(Bi))
    plt.tight_layout()
plt.show()

# Vary Biot number and find error
N = 100
plt.figure()
for Bi in [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]:
    all_roots = roots(N, Bi)
    T = solution(all_roots, 1.0, X, Y)
    dTdx = (T[:, 1] - T[:, 0])/(X[:, 1] - X[:, 0])
    q = 2.0*trapezoid(-dTdx, y)     # the 2 comes because this solution() only returns the upper half of the fin
    q_fin = np.sqrt(2.0*2.0*Bi)     # one 2 comes from the wetted perimeter, the other the fact that "t" here is twice the half-width I'm using of 1.0

    error_norm = np.abs((q - q_fin)/q)
    print(Bi, error_norm, q, q_fin)
    plt.plot(Bi, error_norm, marker='o', color='black')
    plt.yscale('log')
    plt.xscale('log')
plt.show()


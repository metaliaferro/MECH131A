import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import minimize

def constant_flux_solution(x, C, L):
    """
    expects C = q/(k*t)
    """
    return 0.5*C*L*L*(x/L)*(1.0 - x/L)

def constant_flux_fin(x, T, Tr, C):
    """
    solves the second order fin ODE with constant flux:

        0 = d2Tdx2 + q/(k*t)
    
    broken up into two equations:

        d/dx (T) = dT/dx
        d/dx (dT/dx) = d2T/dx2 = -q/(k*t)
    """
    dTdx = 0.0*T

    dTdx[0] = T[1]
    dTdx[1] = - C

    return dTdx

def constant_flux_Bi_solution(x, C, L, Bi):
    """
    expects C = q/(k*t)
    """
    xi = x/L
    return C*L*L*(1.0 + xi*Bi*(1.0 - xi))/(2.0*Bi)


def radiative_temperature(q, eps, s, L, k, t, Tw):
    """
    Solves for the average radiative temperature of the heat source
    """
    a = k*t*Tw
    b = q*L*L
    Tr4_fin = (np.power(a, 4)*np.power(b, 0) +\
                np.power(a, 3)*np.power(b, 1)/3.0 +\
                np.power(a, 2)*np.power(b, 2)/20.0 +\
                np.power(a, 1)*np.power(b, 3)/280.0 +\
                np.power(a, 0)*np.power(b, 4)/10080.0)/\
               (np.power(k*t, 4))
    Tr4_rad = q/(eps*s)
    return np.power(Tr4_rad + Tr4_fin, 0.25)
    # return np.power(Tr4_rad + np.power(Tw, 4), 0.25)


def radiative_fin(x, T, Tr, C):
    """
    solves the second order fin ODE with radiation:

        0 = d2Tdx2 + ((epsilon*sigma)/(k*t))*(Tr**4 - T**4)
    
    broken up into two equations:

        d/dx (T) = dT/dx
        d/dx (dT/dx) = d2T/dx2 = -((epsilon*sigma)/(k*t))*(Tr**4 - T**4)
    """
    dTdx = 0.0*T

    dTdx[0] = T[1]
    dTdx[1] = - C*(np.power(Tr, 4) - np.power(T[0], 4))

    return dTdx

def fin_wrapper_temperature(dTdx_base, L, fin_function, Tr, C):
    res = solve_ivp(fin_function, [0, L], [0, dTdx_base[0]], args=(Tr, C))
    return np.square(res.y[0][0] - res.y[0][-1])  # the fin should be the same temperature on both sides, using that condition as the error

def fin_wrapper_flux(dTdx_base, L, fin_function, Tr, Bi, Tw, C):
    T_base = Tw + L*dTdx_base[0]/Bi
    res = solve_ivp(fin_function, [0, L], [T_base, dTdx_base[0]], args=(Tr, C))
    return np.square(res.y[0][0] - res.y[0][-1])


if __name__ == "__main__":
    sigma = 5.67e-8         # W/m2-K4
    epsilon = [0.04, 0.82]  # unitless
    k = 177.0               # W/m-K
    t = 6e-3                # m
    L = 0.2                 # m
    Tw = 60.0 + 273.15      # K
    qrad = 800.0            # W/m2

    x = L*np.linspace(0, 1, 101)
    # Check class solution
    T_class = constant_flux_solution(x, qrad/(k*t), L)
    res = minimize(
        fin_wrapper_temperature,
        [qrad*L/(2.0*k*t)],
        args = (L, constant_flux_fin, None, qrad/(k*t))
    )
    T_class_check = solve_ivp(
        constant_flux_fin,
        [0, L],
        [0.0, res.x[0]],
        args=(None, qrad/(k*t)),
        max_step = L/10.0
    )
    plt.plot(x, T_class)
    plt.plot(T_class_check.t, T_class_check.y[0], linestyle='None', marker='o')
    plt.show()

    # Check resistance solution
    colorwheel = [
        'tab:orange',
        'tab:green',
        'tab:red',
        'tab:purple',
        'tab:brown',
        'tab:pink',
    ]
    plt.figure()
    plt.plot(x, T_class + Tw)
    for ii, Bi in enumerate([1e-1, 1.0, 1e1]):
        T_bi = constant_flux_Bi_solution(x, qrad/(k*t), L, Bi) + Tw
        minres = minimize(
            fin_wrapper_flux,
            [qrad*L/(2.0*k*t)],
            args=(L, constant_flux_fin, None, Bi, Tw, qrad/(k*t))
        )
        T_base = Tw + L*minres.x[0]/Bi
        ivpres = solve_ivp(
            constant_flux_fin,
            [0, L],
            [T_base, minres.x[0]],
            args=(None, qrad/(k*t)),
            max_step = L/10.0
        )
        plt.plot(x, T_bi, label = 'Bi = {0:.1e}'.format(Bi), color = colorwheel[ii])
        plt.plot(ivpres.t, ivpres.y[0], linestyle='None', marker='o', color = colorwheel[ii])
    plt.legend()
    plt.show()

    # Check radiation temperature
    for eps in epsilon:
        Trad = radiative_temperature(qrad, eps, sigma, L, k, t, Tw)
        T_class = constant_flux_solution(x, qrad/(k*t), L)
        qrad_total = trapezoid(
            eps*sigma*(np.power(Trad, 4.0) - np.power(T_class + Tw, 4.0)),
            x
        )/L
        print(qrad, qrad_total)
        print(Trad)

    T_class = constant_flux_solution(x, qrad/(k*t), L) + Tw
    for eps in epsilon:
        variable_grouping = (eps*sigma)/(k*t)
        plt.figure()
        plt.plot(x, T_class + Tw)
        for ii, Bi in enumerate([0.1, 1.0, 10.0]):
            T_bi = constant_flux_Bi_solution(x, qrad/(k*t), L, Bi) + Tw
            Trad = radiative_temperature(qrad, eps, sigma, L, k, t, Tw)
            minres_rad = minimize(
                fin_wrapper_flux,
                [qrad*L/(2.0*k*t)], # how do I know this is a good first guess?
                args = (L, radiative_fin, Trad, Bi, Tw, variable_grouping)
            )
            print(minres_rad.message)
            T_base = Tw + L*minres_rad.x[0]/Bi
            ivpres_rad = solve_ivp(
                radiative_fin,
                [0, L],
                [T_base, minres_rad.x[0]],
                args=(Trad, variable_grouping),
                max_step = L/10.0
            )
            plt.plot(x, T_bi, label = 'Bi = {0:.1e}'.format(Bi), color = colorwheel[ii + 1])
            plt.plot(ivpres_rad.t, ivpres_rad.y[0], linestyle='None', marker='o', color = colorwheel[ii + 1])
        plt.legend()
        plt.show()






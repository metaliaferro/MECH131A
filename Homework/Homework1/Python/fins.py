import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, cumulative_trapezoid, trapezoid


_AMIN = 1e-12
_RTOL = 1e-10
_ATOL = 1e-10
_TOL = 1e-8
_L_OFFSET = 1e-3


def geometry_rectangular(x, t, L):
    """
    `Ac`, `dAsdx`, and `As` given per unit width
    """
    Ac = t
    dAcdx = 0.0
    dAsdx = 2.0
    Ap = t*L
    As = 2.0*L

    return max(Ac, _AMIN), dAcdx, dAsdx, Ap, As

def geometry_triangular(x, t, L):
    """
    `Ac`, `dAsdx`, and `As` given per unit width
    """
    Ac = t*(1.0 - x/L)
    dAcdx = -t/L
    dAsdx = 2.0*np.sqrt(np.square(t/(2.0*L)) + 1.0)
    Ap = t*L/2.0
    As = 2.0*np.sqrt(L*L + np.square(t/2))

    return max(Ac, _AMIN), dAcdx, dAsdx, Ap, As

# def area_parabolic(x, t, L):
#     """
#     I have not the time to do this analytically. Length of a curve is:

#         int(from x = a, to x = b, sqrt(1 + f'(x)^2)*dx)
#     """
#     res = solve_ivp(lambda xp, y: 2.0*np.sqrt(1.0 + np.square(t*(xp - L)/np.square(L))), [0, x], [0.0], atol=_ATOL, rtol=_RTOL)
#     return res.y[0][-1]

def geometry_parabolic(x, t, L):
    """
    `Ac`, `dAsdx`, and `As` given per unit width
    """
    Ac = t*np.square(1.0 - x/L)
    dAcdx = -(2.0*t*(L - x))/np.square(L)

    # # Numerically take the derivative of the surface area because arc lengths are not nice
    # dx = 1e-8
    # if x > dx and x + dx < L:
    #     dAsdx = (area_parabolic(x + 0.5*dx, t, L) - area_parabolic(x - 0.5*dx, t, L))/dx
    # elif x + dx > L:
    #     dAsdx = (3.0*area_parabolic(x, t, L) - 4.0*area_parabolic(x - dx, t, L) + area_parabolic(x - 2.0*dx, t, L))/(2.0*dx)
    # else:
    #     dAsdx = (-3.0*area_parabolic(x, t, L) + 4.0*area_parabolic(x + dx, t, L) - area_parabolic(x + 2.0*dx, t, L))/(2.0*dx)

    dAsdx = 2.0*np.sqrt(1.0 + np.square(t*(x - L)/np.square(L)))
    # Continue on with algebraic equations from the text
    Ap = L*t/3.0
    C1 = np.sqrt(1.0 + np.square(t/L))
    As = C1*L + L*L*np.log(C1 + t/L)/t

    return max(Ac, _AMIN), dAcdx, dAsdx, Ap, As

def fin_geometry(x, t, L, fin_type):
    # Return geometry information
    if fin_type == 'rectangular':
        Ac, dAcdx, dAsdx, Ap, As = geometry_rectangular(x, t, L)
    elif fin_type == 'parabolic':
        Ac, dAcdx, dAsdx, Ap, As = geometry_parabolic(x, t, L)
    elif fin_type == 'triangular':
        Ac, dAcdx, dAsdx, Ap, As = geometry_triangular(x, t, L)
    return Ac, dAcdx, dAsdx, Ap, As


def _fin_equations(x, y, t, L, m, fin_type):
    """
    solves the fin equation for the specified type of fin:

        d2Tdx2 + (1/Ac)*dAc/dx*dT/dx - (1/Ac)*(h/k)*dAs/dx*T = 0
    
    where

        T = Ts - Tinf

    and

        m = sqrt((h*L)/(k*Ap)).

    To numerically integrate, need to break into system of equations:

        y[0] = T
        y[1] = dT/dx
    
    so,

        dydx[0] = dy0/dx = dT/dx = y[1]
        dydx[1] = dy1/dx = d(dT/dx)/dx = d2T/dx2 = - (1/Ac)*dAc/dx*y[1] + (1/Ac)*(h/k)*dAs/dx*y[0]

    """

    # Initialize array containing derivatives
    dydx = 0.0*y

    # Get geometry information and h/k
    Ac, dAcdx, dAsdx, Ap, _ = fin_geometry(x, t, L, fin_type)
    h_k = m*m*Ap/L

    dydx[0] = y[1]
    dydx[1] = - (1.0/Ac)*dAcdx*y[1] + (1.0/Ac)*h_k*dAsdx*y[0]

    return dydx


def _tip_error(dTdx_base, m, L, t, fin_type):
    res = solve_ivp(
        _fin_equations, [0.0, (1.0 - _L_OFFSET)*L], np.array([1.0, dTdx_base[0]]),
        args=(t, L, m, fin_type),
        rtol=_RTOL, atol=_ATOL,
        # method='Radau'
    )
    T_end = res.y[0][-1]
    dTdx_end = res.y[1][-1]
    _, _, _, Ap, _ = fin_geometry(L, t, L, fin_type)
    
    # -k*dTdx(x = :) = h*T(x = L)
    # -k*dTdx(x = :)/h*T(x = L) = 1
    k_h = L/(m*m*Ap)
    return np.square(T_end + k_h*dTdx_end)
    # if fin_type == 'rectangular':
    #     return np.square(1.0 + k_h*(dTdx_end/T_end))
    # elif fin_type == 'parabolic':
    #     return np.square(T_end)
    # elif fin_type == 'triangular':
    #     return np.square(dTdx_end)

def fin_profile_numerical(m, L, t, fin_type):
    minres = minimize(_tip_error, [-m], args = (m, L, t, fin_type), tol=_TOL)
    ivpres = solve_ivp(
        _fin_equations, [0, (1.0 - _L_OFFSET)*L], np.array([1.0, minres.x[0]]),
        args=(t, L, m, fin_type),
        max_step = L/11.0,
        rtol=_RTOL, atol=_ATOL,
        # method='Radau'
    )
    error = ivpres.y[0][-1] + ivpres.y[1][-1]   # this only works because h = k
    ivpres.y[0][ivpres.y[0] < 0.0] = 0.0    # Really surprised at how stiff this problem was
    return ivpres.t, ivpres.y[0], ivpres.y[1], error

def convective_losses(x, t, L, h, T, fin_type):
    """
    Numerically integrates the heat transfer from the surface of the fin
    """

    # Integrate the dAsdx term returned by the fin_geometry() function to get
    # the total surface area up to x-location
    dAsdx = np.array([fin_geometry(xi, t, L, fin_type)[2] for xi in x])
    As = 0.0*x
    As[1:] = cumulative_trapezoid(dAsdx, x)
    _, _, _, _, As_check = fin_geometry(L, t, L, fin_type)

    # # Check that the results make sense
    # plt.figure()
    # plt.plot(x, As)
    # plt.plot(x[-1], As_check, marker='o')
    # plt.show()

    # Extract the surface area associated with each x-location
    dAs = np.diff(As)
    As_element = 0.0*As
    As_element[0] = 0.5*dAs[0]
    As_element[-1] = 0.5*dAs[-1]
    As_element[1:-1] = 0.5*dAs[:-1] + 0.5*dAs[1:]

    Q_tip = h*t*T[-1] if fin_type == 'rectangular' else 0.0

    # Return the integrated heat transfer along the surface of the fin and the
    # base of the fin (note there is a dx in the As_element term)
    return sum(h*As_element*T) + Q_tip


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Check area function
    # L = 1.0
    # t = 0.5
    # A_p = area_parabolic(L, t, L)
    # Ac, dAcdx, dAsdx, Ap, As = geometry_parabolic(L/2, t, L)
    # print(A_p, As)

    h = 10.0  # W/m2-K
    k = 10.0  # W/m-K
    # L = 1.0
    t = 0.5
    for L in [1e-1, 1.8e-1, 3e-1, 5.6e-1, 1e0, 1.8e0, 3e0]:
        for fin_type in ['rectangular', 'triangular', 'parabolic']:
            Ac, dAcdx, dAsdx, Ap, As = fin_geometry(0.0, t, L, fin_type)
            m = np.sqrt((h*L)/(k*Ap))
            x, T, dTdx, error_num = fin_profile_numerical(m, L, t, fin_type)
            q_c = convective_losses(x, t, L, h, T, fin_type)
            error_energy = q_c + k*dTdx[0]
            print(fin_type, x[-1], error_num, q_c + k*t*dTdx[0])
            plt.plot(x, T, label=fin_type)
        plt.title('{0}'.format(m*L))
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.show()
        print()


import numpy as np
import CoolProp.CoolProp as CP

# Functions
def NTU(Cr, eff):
    E = ((2.0/eff) - (1.0 + Cr))/np.sqrt(1 + Cr*Cr)
    return -np.power(1.0 + Cr*Cr, -0.5)*np.log((E - 1.0)/(E + 1.0))

def friction_factor(Re):
    return 1.0/np.square(1.82*np.log10(Re) - 1.64)

def Nusselt_pipe(Re, Pr):
    f = friction_factor(Re)
    return ((f/8.0)*(Re - 1e3)*Pr)/(1.0 + 12.7*np.sqrt(f/8.0)*(np.power(Pr, 2.0/3.0) - 1.0))

# Problem constants
D_i = 20e-3 # m
D_o = 24e-3 # m
T_w_i = 87.0 + 273.15   # K
T_w_o = 27.0 + 273.15   # K
T_o_i = 7.0 + 273.15    # K
T_o_o = 37.0 + 273.15   # K
mdot_w = 0.2    # kg/s
L = 3           # m
N = 20          # passes

# Thermodynamic Properties
T_w_m = 0.5*(T_w_i + T_w_o)
P = 1e5
rho_w = CP.PropsSI('DMASS', 'T', T_w_m, 'P', P, 'water')
cp_w = CP.PropsSI('CPMASS', 'T', T_w_m, 'P', P, 'water')
mu_w = CP.PropsSI('viscosity', 'T', T_w_m, 'P', P, 'water')
k_w = CP.PropsSI('conductivity', 'T', T_w_m, 'P', P, 'water')
Pr_w = CP.PropsSI('Prandtl', 'T', T_w_m, 'P', P, 'water')

# Heat exchanger effectiveness and NTU
C_w = cp_w*mdot_w
q_w = C_w*(T_w_i - T_w_o)
C_o = q_w/(T_o_o - T_o_i)
Cmin = min(C_w, C_o)
Cr = Cmin/max(C_w, C_o)
q_max = Cmin*(T_w_i - T_o_i)
effectiveness = q_w/q_max
ntu = NTU(Cr, effectiveness)
UA = ntu*Cmin

# Pipe heat transfer
Re_D = (4.0*mdot_w)/(np.pi*D_i*mu_w)
Nu_w = Nusselt_pipe(Re_D, Pr_w)
htc_w = Nu_w*(k_w/D_i)

# Outer Surface Heat Transfer
A_w = np.pi*D_i*L*N
A_o = np.pi*D_o*L*N
htc_o = 1/(A_o*(1.0/UA - 1.0/(htc_w*A_w)))



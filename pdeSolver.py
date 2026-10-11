import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.constants import G, c, hbar, m_e, m_n
from astropy.constants import M_sun

mu_e     = 2    #for carbon
gammaRel = 4/3
gammaNonRel = 5/3
K_nonrel = hbar**2/(15*np.pi**2*m_e) * (3*np.pi**2/(mu_e*m_n*c**2))**(5/3)
K_rel    = hbar*c/(12*np.pi**2)      * (3*np.pi**2/(mu_e*m_n*c**2))**(4/3)


def solve(eps_c,K,gamma,allpoints = False): # eps_c is given through findepsc
    
    def f(r,y):
        M,P = y
        eps = (max(P, 0)/K)**(1/gamma) #inversion of polytope
        dm = 4*np.pi*r**2*eps/c**2  #first equation
        dP = -(G/(c**2*r**2))*(eps + P)*(M + 4*np.pi*r**3*P/c**2) \
             / (1 - 2*G*M/(r*c**2)) # relatavistic pressure curve
        return [dm,dP]


    p0 = K*eps_c**gamma #initial pressure polytrope
    def surface(r, y): # define a surface where pressure basically equals zero and stop there
        return y[1] - 1e-10*p0                       
    surface.terminal = True
    surface.direction = -1

    r0=1 # initial radius of 1 meter to stop a div by zero error
    m0 = 4*np.pi/3*r0**3*eps_c/c**2 # initial mass which will basiclaly equal zero


    pts = solve_ivp(f,(r0,10e9),[m0,p0],events = surface, rtol = 1e-5,atol = [m0*1e-10, p0*1e-12],dense_output= allpoints) #simultaneously solve the equations
    R = pts.t_events[0][0] # Takes the last radius 
    M = pts.y_events[0][0][0] # takes the last mass
    if allpoints == True: return (R,M, pts.t, pts.y) # return all points if needed (for plot)
    else: return (R,M) # just return last masses

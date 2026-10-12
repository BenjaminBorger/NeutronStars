import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.constants import G, c, hbar, m_e, m_n
from astropy.constants import M_sun
import sys
from starDatabase import obj
def delete_last_line(): # got from stackoverflow https://stackoverflow.com/questions/44565704/how-to-clear-only-last-one-line-in-python-output-console
    # cursor up one line
    sys.stdout.write('\x1b[1A')
    # delete last line
    sys.stdout.write('\x1b[2K')


mu_e     = 2    #for carbon
gammaRel = 4/3
gammaNonRel = 5/3
K_nonrel = hbar**2/(15*np.pi**2*m_e) * (3*np.pi**2/(mu_e*m_n*c**2))**(5/3)
K_rel    = hbar*c/(12*np.pi**2)      * (3*np.pi**2/(mu_e*m_n*c**2))**(4/3)
triggerPressure = ((m_e**4)*(c**5))/(12*(np.pi**2)*(hbar**3))

'''so electrons become relatavistic approx when kf = me c
which is when pressure = ((m_e**4)*(c**5))/(12*(pi**2)*(hbar**3))
This will be called the trigger pressure '''

def solve(eps_c,allpoints = False): # eps_c is given through findepsc
    delete_last_line()
    if not allpoints: print(f"Central value of Epsilon Guess = {eps_c}")
    else: print(f"Central value of Epsilon Final = {eps_c}")

    def f(r,y,K,gamma):
        M,P = y
        eps = (max(P, 0)/K)**(1/gamma) #inversion of polytope
        dm = 4*np.pi*r**2*eps/c**2  #first equation
        dP = -(G/(c**2*r**2))*(eps + P)*(M + 4*np.pi*r**3*P/c**2) \
             / (1 - 2*G*M/(r*c**2)) # relatavistic pressure curve
        return [dm,dP]


    p0 = K_rel*eps_c**gammaRel #initial pressure polytrope

    def surface(r, y,K,gamma): # define a surface where pressure basically equals zero and stop there
        return y[1] #-( 1e-10 * min(p0, triggerPressure)  )                
    surface.terminal = True
    surface.direction = -1


    def switch(r,y,K,gamma):
        return y[1]-triggerPressure
    switch.terminal = True
    switch.direction = -1

    r0=1 # initial radius of 1 meter to stop a div by zero error
    m0 = 4*np.pi/3*r0**3*eps_c/c**2 # initial mass which will basiclaly equal zero
    
    
    if p0<=triggerPressure:
        p0 = K_nonrel*eps_c**gammaNonRel 
        totalStarisNonRel = solve_ivp(f,(r0,10e9),[m0,p0],args = (K_nonrel,gammaNonRel),events = surface, rtol = 1e-5,atol = [m0*1e-10, p0*1e-12],dense_output= allpoints) #simultaneously solve the equations
        R = totalStarisNonRel.t_events[0][0]
        M = totalStarisNonRel.y_events[0][0][0] # takes the last mass of star
        rs = totalStarisNonRel.t # these are all the values of radius 
        ys = totalStarisNonRel.y # all the fvlaues of mass
    else:

        relPart = solve_ivp(f,(r0,10e9),[m0,p0],args = (K_rel,gammaRel),events = [switch,surface], rtol = 1e-5,atol = [m0*1e-10, p0*1e-12],dense_output= allpoints) #simultaneously solve the equations
    
        r_event = relPart.t_events[0][0] # Takes the last radius of centre 
        y_event = relPart.y_events[0][0] # takes the last mass of centre
        
        
        nonRelPart = solve_ivp(f,(r_event,10e9),y_event,args = (K_nonrel,gammaNonRel),events = surface, rtol = 1e-5,atol = y_event*10e-12,dense_output= allpoints) #simultaneously solve the equations
        R = nonRelPart.t_events[0][0] # Takes the last radius of star
        M = nonRelPart.y_events[0][0][0] # takes the last mass of star

        rs = np.concatenate([relPart.t,nonRelPart.t]) # combines the two points from rel to non rel
        ys = np.concatenate([relPart.y,nonRelPart.y],axis=1)


    if allpoints == True: return (R,M, rs, ys) # return all points if needed (for plot)
    else: return (R,M) # just return last masses for comparisom

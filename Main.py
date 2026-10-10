from starDatabase import obj
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.constants import G, c, hbar, m_e, m_n
from astropy.constants import M_sun
import pdeSolver



def find_eps_c(iter,rel): # rel is boolean True or false if true: relatavistic and if not, not relatavistic
    '''This function makes a guess of epsilon
    assuming constant density and then takes iterations of the pde solver and hones in 
    on the correct value of epsilon. After that a real measurement can be made.'''
    
    
    realRadius = obj[iter, 0]*1000 #in meters
    realMass = obj[iter, 1]*M_sun.value # kg
    
    if rel: 
        K = pdeSolver.K_rel
        gamma = 4/3
    else: 
        K = pdeSolver.K_nonrel
        gamma = 5/3


    print("Radius: ",float(realRadius), "     Mass: ", float(realMass)) #check that all is ok thusfar
    
    epsilonGuess = (realMass/((4/3)*np.pi*realRadius**3))*(c**2) # guess assuming const density
    epsilonSpace = np.logspace(np.log10(epsilonGuess)-3,np.log10(epsilonGuess)+3,40) # defines an area which is reasonable for eps to be in

    iteratedRs = np.full(len(epsilonSpace), np.nan)
    iteratedMs = np.full(len(epsilonSpace), np.nan)
    
    intervals= []#these will be the intervals in epsilonspace within which contain a crossover of sign
    for i in range(len(epsilonSpace)):
        R, M = pdeSolver.solve(epsilonSpace[i], K, gamma, allpoints=False)
        iteratedRs[i] = R
        iteratedMs[i] = M

        if i == 0 or np.isnan(iteratedMs[i]) or np.isnan(iteratedMs[i-1]):
            continue
        if np.sign(iteratedMs[i-1] - realMass) != np.sign(iteratedMs[i] - realMass):
                intervals.append([epsilonSpace[i-1], epsilonSpace[i]])
   
    eps_C_Real = 0
    goodEpsis = []
    for interval in intervals:
        def massDiffFunc(epsilonVal):
            
            r,m = pdeSolver.solve(epsilonVal,K,gamma,allpoints=False)
            
            return m-realMass

        goodEpsi = brentq(massDiffFunc,interval[0],interval[1])
        
        goodEpsis.append(goodEpsi)
        

    if len(goodEpsis) ==0: print("No roots Found for object ",iter)

    bestE, bestErr = None, np.inf
    for e in goodEpsis:
        
        r,m = pdeSolver.solve(e,K,gamma,allpoints=False)
        if (np.abs(r/realRadius-1))<bestErr:
            bestE = e
            bestErr = np.abs(r/realRadius-1)

    return bestE
                

####plotting
plt
#rel
for i in range(len(obj)):
    iterEps_C = find_eps_c(i,rel=True)
    rRel, mRel, radius_grid, states = pdeSolver.solve(iterEps_C,pdeSolver.K_rel,4/3,allpoints=True)
    mass_profile = states[0]
    pressure_profile = states[1]
    
    plt.plot(radius_grid,mass_profile)
plt.title('Rel Rad Mass Relation')
plt.xlabel('Radius')
plt.ylabel('Mass')    
plt.show()
#nonrel
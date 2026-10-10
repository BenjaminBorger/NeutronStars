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
    
    if rel: K = pdeSolver.K_rel,gamma = 4/3
    else: K = pdeSolver.K_nonrel ,gamma = 5/3


    print("Radius: ",float(realRadius), "     Mass: ", float(realMass)) #check that all is ok thusfar
    
    epsilonGuess = (realMass/((4/3)*np.pi*realRadius**3))*(c**2) # guess assuming const density
    epsilonSpace = np.logspace(np.log10(epsilonGuess)-3,np.log10(epsilonGuess)+3,40) # defines an area which is reasonable for eps to be in

    iteratedMs = np.array([]) # these are the Ms which are made from guesses in epsilonspace 
    #betweent the difference of the iterated M and the real M
    intervals= np.array([]) #these will be the intervals in epsilonspace within which contain a crossover of sign
    firstRun = True
    for i in range(len(epsilonSpace)+1):
        iteratedR,iteratedM = pdeSolver.solve(epsilonSpace[i],K,gamma,allpoints=False)
        np.append(iteratedMs,iteratedM) # should be same number of iterated Ms as there are points in epsilon space ie 40
        if firstRun == True:
            firstRun = False
        elif np.sign(iteratedMs[i-1]-realMass) != np.sign(iteratedMs[i]-realMass):
            intervals._r([epsilonSpace[i-1],epsilonSpace[i]])
    eps_C_Real = 0
    goodEpsis = np.array([])
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
                
                


    #for i in range(0,10):
            #iteratedR,iteratedM = pdeSolver.solve(epsilonGuess,K,gamma,allpoints=False)
    
    return (float(radius),float(mass))

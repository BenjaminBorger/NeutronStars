from starDatabase import obj
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.constants import G, c, hbar, m_e, m_n
from astropy.constants import M_sun
import pdeSolver
from matplotlib.colors import LinearSegmentedColormap
from pdeSolver import gammaNonRel,gammaRel,K_nonrel,K_rel




def find_eps_c(iter,rel): # rel is boolean True or false if true: relatavistic and if not, not relatavistic
    '''This function makes a guess of epsilon
    assuming constant density and then takes iterations of the pde solver and hones in 
    on the correct value of epsilon. After that a real measurement can be made.'''
    
    
    realRadius = obj[iter, 0]*1000 # radius of example star in meters 
    realMass = obj[iter, 1]*M_sun.value # mass of example star kg
    
    if rel: #change constnats approperiately
        K = pdeSolver.K_rel
        gamma = 4/3
    else: 
        K = pdeSolver.K_nonrel
        gamma = 5/3


    print("Radius: ",float(realRadius), "     Mass: ", float(realMass)) #check that all is ok thusfar
    
    epsilonGuess = (realMass/((4/3)*np.pi*realRadius**3))*(c**2) # guess assuming const density
    epsilonSpace = np.logspace(np.log10(epsilonGuess)-3,np.log10(epsilonGuess)+3,40) # defines an area which is reasonable for eps to be in

    iteratedRs = np.full(len(epsilonSpace), np.nan) # creates an array of not a numbers the length of epsilon space
    iteratedMs = np.full(len(epsilonSpace), np.nan)
    
    intervals= []#these will be the intervals in epsilonspace within which contain a crossover of sign
    for i in range(len(epsilonSpace)): # loop for each value in epsilon space(will loop 40 times)
        R,M = pdeSolver.solve(epsilonSpace[i], K, gamma, allpoints=False) # solve the PDE for this value of epsilon and only return the finishing mass and radius
        iteratedRs[i] = R # add these returned values to thir respective arrays
        iteratedMs[i] = M

        if i == 0 or np.isnan(iteratedMs[i]) or np.isnan(iteratedMs[i-1]): # if this si the first iteration,the mass is not a number or the previous mass is not a number, skip
            continue
        '''if there is a sign change between the previous mass differnce and this mass difference
        the real mass must be somewhere in the middle of these (where returned m-real m = 0)
        this is the epsilon the hone in on'''
        if np.sign(iteratedMs[i-1] - realMass) != np.sign(iteratedMs[i] - realMass): 
                intervals.append([epsilonSpace[i-1], epsilonSpace[i]]) #if one of these sign changes is found, we must add this interval as one of interest
   
    eps_C_Real = 0
    goodEpsis = [] #these will have a list of candidates for the initial epsilon value
    for interval in intervals: # go through each interval
        def massDiffFunc(epsilonVal): #simply solves the equations with this value of epsilon and creates a function with m-realMass
            
            r,m = pdeSolver.solve(epsilonVal,K,gamma,allpoints=False)
            
            return m-realMass

        goodEpsi = brentq(massDiffFunc,interval[0],interval[1]) # hones in on the values of epsilon which return a zero for our defined function 
        #may return multiple zeroes thats why i need the list
        goodEpsis.append(goodEpsi) # when found, add to list
        

    if len(goodEpsis) == 0: print("No roots Found for object ",iter) # if there are no roots, print this:problem

    bestE, bestErr = None, np.inf # makes our bestE none and our best error infinitely large
    for e in goodEpsis: # for each good espilon 
        '''for each good espsilon, check whether the error is better than the last one 
        if its the best, when the loop ends itll return this value'''
        r,m = pdeSolver.solve(e,K,gamma,allpoints=False)
        if (np.abs(r/realRadius-1))<bestErr:
            bestE = e
            bestErr = np.abs(r/realRadius-1)

    return bestE
                

####plotting
#could maybe add some sort of circle that has the pressure so as the radius goes outwards, it changes colour
#rel

for i in range(len(obj)):
    realMass = obj[i][1]*M_sun.value
    realRadius = obj[i][0]*1000
    iterEps_C = find_eps_c(i, rel=False)
    if iterEps_C is None:
        print(f"Skipping object {i}: no valid epsilon_c")
        continue
    r, m, radius_grid, states = pdeSolver.solve(iterEps_C,K_nonrel,gammaNonRel,allpoints=True)
    mass_profile = states[0]
    pressure_profile = states[1]

    plt.plot(radius_grid,mass_profile)
plt.title('Rel Rad Mass Relation')
plt.xlabel('Radius')
plt.ylabel('Mass')    
plt.show()














def plot_pressure():
    r = np.asarray(radius_grid, dtype=float).ravel()
    p = np.asarray(pressure_profile, dtype=float).ravel()

    # np.interp needs increasing x values
    order = np.argsort(r)
    r, p = r[order], p[order]

    # Debug: check these first
    print("radius range:", r.min(), r.max())
    print("pressure range:", p.min(), p.max())

    # Normalize radius to 0..1
    r_norm = (r - r.min()) / (r.max() - r.min())

    n = 500
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)

    P = np.interp(R, r_norm, p)
    P_masked = np.ma.masked_where(R > 1, P)


    cmap = LinearSegmentedColormap.from_list("pressure", ["blue","white"])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal")
    
    im = ax.imshow(
        P_masked,
        extent=(-1, 1, -1, 1),
        origin="lower",
        cmap=cmap,
        vmin=p.min(),
        vmax=p.max(),
    )
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pressure")






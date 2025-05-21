
__author__ = "Ivón Olarte Rodríguez"

# Based on the work made by Kudela, Stripinis and Paulavicius


### Importing libraries
import numpy as np


# Set the machine precision
EPS = np.finfo(float).eps

### ===============================
### Lennard-Jones potential
### ================================

def Circular_Antenna_Array(x:np.ndarray) -> np.ndarray:
    r"""
    Computes the fitness value for the Circular Antenna Array problem.


    Args
    ----------------
    - x: np.ndarray: Array of shape (N, ) with design variables.

    Returns
    -----------------
    `float`.
    """

    assert x.ndim < 2, "Input array must be less than two-dimensional."
    assert x.size  == 12, "Input array must be non-empty and equal to 12."

    # Perform a reshape of the initial array
    x = x.ravel()

    # Obtain the size of the array
    n = x.size

    lb = get_xl(n)
    ub = get_xu(n)

    # Convert the design variables 
    x = np.abs(ub - lb)*x + lb

    # Compute the fitness value
    y = Fitness(x)

    return y

def get_xl(n:int) -> np.ndarray:
    r"""Lower bounds for the design variables."""
    xl = np.hstack((np.ones((6, ))*0.2, -np.ones((6, ))*180))
    return xl

def  get_xu(n:int) -> np.ndarray:
    r"""Upper bounds for the design variables."""
    xu = np.hstack((np.ones((6, ))*1, np.ones((6, ))*180))
    return xu

def Fitness(x:np.ndarray):
    r"""
    Computes the fitness value for the Circular Antenna Array problem.  
    """
    # Constants and parameters
    null = 50
    phi_desired = 180
    distance = 0.5
    dim = x.size
    
    phizero = 0
    #[~, num_null] = size(null)
    num_null = 1
    num1 = 300

    # Generate an angular grid for phi
    phi = np.linspace(0, 360, num1, endpoint=True)

    # Initialize yax, sidelobes and sllphi as lists
    yax = []
    sidelobes = []
    sllphi = []

    # Calculate array factor for different phi angles and find maximum
    yax.append(array_factor(x, (np.pi/180)*phi[0], phi_desired, distance, dim))
    maxi = yax[0]
    phi_ref = 1
    for i in range(1, num1):
        # Calculate the array factor for the current angle
        yax.append(array_factor(x, (np.pi/180)*phi[i], phi_desired, distance, dim)) ##ok<AGROW>
        if maxi < yax[i]:
            maxi = yax[i]
            phizero = phi[i]
            phi_ref = i


    # Find sidelobes
    count = 0
    if yax[0] - yax[-1] > EPS  and yax[0] - yax[1]  > EPS:
        count += 1
        sidelobes.append(yax[0])
        sllphi.append(phi[0])


    if yax[-1] - yax[0] > EPS and yax[-1] - yax[-2] > EPS:
        count += 1
        sidelobes.append(yax[-1])
        sllphi.append(phi[-1])

    for i in range(1,num1-1):
        if yax[i] - yax[i+1]  > EPS and yax[i] - yax[i-1] > EPS :
            count += 1
            sidelobes.append(yax[i])
            sllphi.append(phi[i])

    # Sort the sidelobes
    sidelobes.sort(reverse=True)


    upper_bound = 180
    lower_bound = 180
    y:float = sidelobes[1]/maxi
    sllreturn = 20*np.log10(y)

    # Calculate upper and lower beamwidth bounds
    for i in range(int(num1/2)):
        if (phi_ref + i) > num1-1:
            upper_bound = 180
            break
        if yax[phi_ref + i] < yax[phi_ref + i - 1] and yax[phi_ref + i] < yax[phi_ref + i + 1]:
            upper_bound = phi[phi_ref + i] - phi[phi_ref]
            break
        
    
    # Calculate the lower bound
    for i in range(int(num1/2)):
        if (phi_ref - i < 2):
            lower_bound = 180
            break
        if yax[phi_ref - i] < yax[phi_ref - i - 1] and yax[phi_ref - i] < yax[phi_ref - i + 1]:
            lower_bound = phi[phi_ref] - phi[phi_ref - i]
            break
        
    bwfn = upper_bound + lower_bound

    # Calculate the objective function components
    y1 = 0.0
    for i in range(num_null):
        # The objective function for null control is calculated here
        y1 += (array_factor(x, null, phi_desired, distance, dim)/maxi)
    
    y3 = abs(phizero - phi_desired)

    if y3 < 5:
        y3 = 0

    y = 0
    if bwfn > 80:
        y += abs(bwfn - 80)

    # Combine the components to calculate the final fitness value 'y'
    y += sllreturn + y1 + y3

    # Check for NaN value and set a large value if necessary
    if np.isnan(y):
        y = 10**100

    return y

def array_factor(x1:np.ndarray, 
                 phi:float, 
                 phi_desired:float, 
                 distance:float, 
                 dim:int)->float:
    
    r"""
    Computes the array factor
    """

    y = 0
    y1 = 0

    for i1 in range(1,int(dim/2)+1):
        delphi = 2*np.pi*(i1-1)/dim
        shi = np.cos(phi - delphi) - np.cos(phi_desired*(np.pi/180) - delphi)
        shi = shi * dim * distance
        y += x1[i1-1] * np.cos(shi + x1[int(dim/2) + i1 - 1]*(np.pi/180))
    

    for i1 in  np.arange(int(dim/2)+1,dim+1,dtype=int):
        delphi = 2*np.pi*(i1-1)/dim
        shi = np.cos(phi - delphi) - np.cos(phi_desired*(np.pi/180) - delphi)
        shi = shi * dim * distance
        y +=  x1[(i1-1 - int(dim/2))] * np.cos(shi - x1[i1-1]*(np.pi/180))
    

    for i1 in range(1,int(dim/2)+1):
        delphi = 2*np.pi*(i1-1)/dim
        shi = np.cos(phi - delphi) - np.cos(phi_desired*(np.pi/180) - delphi)
        shi = shi * dim * distance
        y1 +=  x1[i1-1] * np.sin(shi + x1[int(dim/2-1) + i1]*(np.pi/180))
    

    for i1 in  np.arange(int(dim/2)+1,dim+1,dtype=int):
        delphi = 2*np.pi*(i1-1)/dim
        shi = np.cos(phi - delphi) - np.cos(phi_desired*(np.pi/180) - delphi)
        shi = shi * dim * distance
        y1 += x1[i1 - int(dim/2) -1 ] * np.sin(shi - x1[i1-1]*(np.pi/180))
    

    y = y * y + y1 * y1
    y = np.sqrt(y)

    return y




if __name__ == "__main__":
    # Test the function with a sample input
    #x = np.asarray([2, 2,	1.5,	0,	0,	0]).ravel()
    #x = np.asarray([0.666666666666667, 2, 1.50000000000000, 0, 0, 0]).ravel()
    #x = np.asarray([2, 2, 1.50000000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0]).ravel()
    # x = np.asarray([0.600000000000000, 0.600000000000000, 0.600000000000000,	0.600000000000000,	0.600000000000000,	0.600000000000000,
    #                 	0,	0,	0,	0,	0,	0]).ravel()
    x = np.asarray([0.333333333333333, 0.600000000000000,	0.600000000000000,	0.600000000000000,	0.600000000000000,	0.600000000000000,
                    	0,	0,	0,	0,	0,	0]).ravel()
    potential = Circular_Antenna_Array(x)
    print(f"Ciruclar Antena Function: {potential}")
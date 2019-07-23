"""
utils.py
This script contains functions for generating diffusion simulations, 
data generators needed for the network training/testing, and other necessary 
functions.
"""


import numpy as np
from scipy import stats,fftpack
from keras.utils import to_categorical
from stochastic import diffusion 
from scipy.optimize import curve_fit
import scipy.io


"""
Function autocorr calculates the autocorrelation of a given input vector x

Input: 
    x - 1D vector 
    
Outputs:
    autocorr(x)    
"""

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[np.int(result.size/2):]


"""
Function curve_func is used as input to scipy's curve fitting function

Input: 
    x - 1D vector 
    
Outputs:
    f(x) = a+b*x  
    
"""

def curve_func(x,a,b):
    return a+b*x

"""
Function MSD calculates the time averaged MSD for a 2D trajectory and its
corresponding anomalous exponent.

Input: 
    x,y - x,y positions of a localized particle over time (assuming constant dt
    between frames) 
    
Outputs:
    tVec - vector of time points
    msd - mean square displacement curve
    a - anomalous exponent (=2*Hurst exponent for FBM)
"""

def MSD(x,y):
    data = np.sqrt(x**2+y**2)
    nData = np.size(data)
    numberOfDeltaT = np.int((nData-1))
    tVec = np.arange(1,np.int(numberOfDeltaT*0.9))

    msd = np.zeros([len(tVec),1])
    for dt,ind in zip(tVec,range(len(tVec))):
        sqdisp = (data[1+dt:] - data[:-1-dt])**2
        msd[ind] = np.mean(sqdisp,axis=0)
        
    msd = np.array(msd)
    a,b = curve_fit(curve_func,np.log(tVec),np.log(msd.ravel()))
    return tVec,msd,a[1]


"""
Function MSD_D calculates the time averaged MSD for a 2D trajectory and its
corresponding diffusion coefficient.

Input: 
    x,y - x,y positions of a localized particle over time (assuming constant dt
    between frames) 
    numPoints - Number of MSD points to use when fitting for the diffusion
    coefficient (smaller values might perform better).
    
Outputs:
    tVec - vector of time points (assumed to be the same for all trajectories)
    msd - mean square displacement curve
    d - diffusion coefficient
"""

def MSD_D(x,y,numPoints):
    data = np.sqrt(x**2+y**2)
    nData = np.size(data)
    numberOfDeltaT = np.int((nData-1))
    tVec = np.arange(1,np.int(0.9*numberOfDeltaT))
    
    msd = np.zeros([len(tVec),1])
    for dt,ind in zip(tVec,range(len(tVec))):
        sqdisp = (data[1+dt:] - data[:-1-dt])**2
        msd[ind] = np.mean(sqdisp,axis=0)
        
    msd = np.array(msd)

    d,temp = curve_fit(curve_func,np.log(tVec[:numPoints]),np.log(msd[:numPoints].ravel()),maxfev=2000)
    
    return tVec,msd,d[0]



"""
Function matrix_of_MSDs calculates the time averaged MSD for multiple trajectories

Input: 
    X - input matrix in the form [number of dimensions][number of trajectories][number of steps]
    
Outputs:
    tVec - vector of time points (assumed to be the same for all trajectories)
    MSD_mat - matrix of MSD curves
"""

def matrix_of_MSDs(X):
    numTraj = np.shape(X)[1]
    tVec,msd,a = MSD(X[0,0,:],X[1,0,:])
    msd_size = len(tVec)
    MSD_mat = np.zeros([numTraj,msd_size])
    for i in range(numTraj):
        x = X[0,i,:]
        y = X[1,i,:]
        tVec,msd,a = MSD(x,y)
        msd = np.reshape(msd,[len(msd),1])
        MSD_mat[i,:] = msd[:msd_size,0]
    
    return tVec,MSD_mat


"""
Function MSD_mat_from_file calculates the time averaged MSD for multiple trajectories 
loaded from a file

Input: 
    file - string containing the file name, ending with .mat
    NTraj - number of trajectories to analyze; input '0' to analyze all 
           trajectories
   stepsActual - number of steps to analyze in each trajectory.
    
Outputs:
    tVec - vector of time points (assumed to be the same for all trajectories)
    MSD_mat - matrix of MSD curves
"""

def MSD_mat_from_file(file,NTraj,steps):
    f = scipy.io.loadmat(file)
    for k in f.keys():
        if k[0]=='_':
            continue
        varName = k
        data = f[varName]
        NAxes = (np.shape(data)[1]-1)
    if NTraj == 0:
        numTraj = len(np.unique(data[:,NAxes]))
    else:
        numTraj= NTraj
        
    MSD_mat = np.zeros([numTraj,steps])
    for i in np.arange(1,numTraj+1):
       x = data[np.argwhere(data[:,NAxes]==i),0]
       y = data[np.argwhere(data[:,NAxes]==i),1]
       if len(x)>=steps:
           x = x[:steps]
           y = y[:steps]
           tVec,msd,a = MSD(x,y)
       else:
           tVec,msd,a = MSD(x,y)
       MSD_mat[i-1,:len(msd)]= msd[:,0]
    return tVec, MSD_mat


#%%

"""
Function OrnsteinUng generates a single realization of the Ornstein–Uhlenbeck 
noise process
see https://stochastic.readthedocs.io/en/latest/diffusion.html#stochastic.diffusion.OrnsteinUhlenbeckProcess
for more details.

Input: 
    n - number of points to generate
    T - End time
    speed - speed of reversion 
    mean - mean of the process
    vol - volatility coefficient of the process
    
Outputs:
    x - Ornstein Uhlenbeck process realization
"""

def OrnsteinUng(n=1000,T=50,speed=0,mean=0,vol=0):
    OU = diffusion.OrnsteinUhlenbeckProcess(speed=speed,mean=mean,vol=vol,t=T)
    x = OU.sample(n=n)
    
    return x

#%% 
'''
function fbm_diffusion generates FBM diffusion trajectory (x,y,t)
realization is based on the Circulant Embedding method presented in:
Schmidt, V., 2014. Stochastic geometry, spatial statistics and random fields. Springer.

Input: 
    n - number of points to generate
    H - Hurst exponent
    T - end time
    
Outputs:
    x - x axis coordinates
    y - y axis coordinates
    t - time points
        
'''
def fbm_diffusion(n=1000,H=1,T=15):

    # first row of circulant matrix
    r = np.zeros(n+1)
    r[0] = 1
    idx = np.arange(1,n+1,1)
    r[idx] = 0.5*((idx+1)**(2*H) - 2*idx**(2*H) + (idx-1)**(2*H))
    r = np.concatenate((r,r[np.arange(len(r)-2,0,-1)]))
    
    # get eigenvalues through fourier transform
    lamda = np.real(fftpack.fft(r))/(2*n)
    
    # get trajectory using fft: dimensions assumed uncoupled
    x = fftpack.fft(np.sqrt(lamda)*(np.random.normal(size=(2*n)) + 1j*np.random.normal(size=(2*n))))
    x = n**(-H)*np.cumsum(np.real(x[:n])) # rescale
    x = ((T**H)*x)# resulting traj. in x
    y = fftpack.fft(np.sqrt(lamda)*(np.random.normal(size=(2*n)) + 1j*np.random.normal(size=(2*n))))
    y = n**(-H)*np.cumsum(np.real(y[:n])) # rescale
    y = ((T**H)*y) # resulting traj. in y

    t = np.arange(0,n+1,1)/n
    t = t*T # scale for final time T
    

    return x,y,t

'''
CTRW diffusion - generate CTRW trajectory (x,y,t)
function based on mittag-leffler distribution for waiting times and 
alpha-levy distribution for spatial lengths.
for more information see: 
Fulger, D., Scalas, E. and Germano, G., 2008. 
Monte Carlo simulation of uncoupled continuous-time random walks yielding a 
stochastic solution of the space-time fractional diffusion equation. 
Physical Review E, 77(2), p.021122.

Inputs: 
    n - number of points to generate
    alpha - exponent of the waiting time distribution function 
    gamma  - scale parameter for the mittag-leffler and alpha stable distributions.
    T - End time
'''


# Generate mittag-leffler random numbers
def mittag_leffler_rand(beta = 0.5, n = 1000, gamma = 1):
    t = -np.log(np.random.uniform(size=[n,1]))
    u = np.random.uniform(size=[n,1])
    w = np.sin(beta*np.pi)/np.tan(beta*np.pi*u)-np.cos(beta*np.pi)
    t = t*((w**1/(beta)))
    t = gamma*t
    
    return t

# Generate symmetric alpha-levi random numbers
def symmetric_alpha_levy(alpha = 0.5,n=1000,gamma = 1):
    u = np.random.uniform(size=[n,1])
    v = np.random.uniform(size=[n,1])
    
    phi = np.pi*(v-0.5)
    w = np.sin(alpha*phi)/np.cos(phi)
    z = -1*np.log(u)*np.cos(phi)
    z = z/np.cos((1-alpha)*phi)
    x = gamma*w*z**(1-(1/alpha))
    
    return x

# needed for CTRW
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# Generate CTRW diffusion trajectory
def CTRW(n=1000,alpha=1,gamma=1,T=40):
    jumpsX = mittag_leffler_rand(alpha,n,gamma)

    rawTimeX = np.cumsum(jumpsX)
    tX = rawTimeX*(T)/np.max(rawTimeX)
    tX = np.reshape(tX,[len(tX),1])
    
    jumpsY = mittag_leffler_rand(alpha,n,gamma)
    rawTimeY = np.cumsum(jumpsY)
    tY = rawTimeY*(T)/np.max(rawTimeY)
    tY = np.reshape(tY,[len(tY),1])
    
    x = symmetric_alpha_levy(alpha=2,n=n,gamma=gamma**(alpha/2))
    x = np.cumsum(x)
    x = np.reshape(x,[len(x),1])
    
    y = symmetric_alpha_levy(alpha=2,n=n,gamma=gamma**(alpha/2))
    y = np.cumsum(y)
    y = np.reshape(y,[len(y),1])
    
    tOut = np.arange(0,n,1)*T/n
    xOut = np.zeros([n,1])
    yOut = np.zeros([n,1])
    for i in range(n):
        xOut[i,0] = x[find_nearest(tX,tOut[i]),0]
        yOut[i,0] = y[find_nearest(tY,tOut[i]),0]
    
    return xOut,yOut,tOut


'''
Brownian - generate Brownian motion trajectory (x,y)

Inputs: 
    N - number of points to generate
    T - End time 
    delta - Diffusion coefficient

Outputs:
    out1 - x axis values for each point of the trajectory
    out2 - y axis values for each point of the trajectory
'''

def Sub_brownian(x0, n, dt, delta, out=None):
    x0 = np.asarray(x0)
    # generate a sample of n numbers from a normal distribution.
    r = stats.norm.rvs(size=x0.shape + (n,), scale=delta*np.sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)
    # Compute Brownian motion by forming the cumulative sum of random samples. 
    np.cumsum(r, axis=-1, out=out)
    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out

def Brownian(N=1000,T=50,delta=1):
    x = np.empty((2,N+1))
    x[:, 0] = 0.0
    
    Sub_brownian(x[:,0], N, T/N, delta, out=x[:,1:])
    
    out1 = x[0]
    out2 = x[1]
    
    return out1,out2


#%%
'''
Generator functions for neural network training per Keras specifications
input for all functions is as follows:
    
# input: 
#   - batch size
#   - steps: total number of steps in trajectory (list) 
#   - T: final time (list)
#   - steps_actual: total number of steps the network recieves (scalar)
#     (can be lower than steps)
'''

# Randomly generate trajectories of different diffusion models for training of the 
# classification network
    
def generate(batchsize=32,steps=5000,T=15,steps_actual=1000):
    while True:
        # randomly choose a set of trajectory-length and final-time. This is intended
        # to increase variability in simuation conditions.
        steps1 = np.random.choice(steps,size=1).item()
        T1 = np.random.choice(T,size=1).item()
        out = np.zeros([batchsize,steps_actual-1,1])
        label = np.zeros([batchsize,1])
        for i in range(batchsize):
            # randomly select diffusion model to simulate for this iteration
            label[i,0] = np.random.choice([0,1,2])
            if label[i,0] == 0: 
                H = np.random.uniform(low=0.1,high=0.9)
                x,y,t = fbm_diffusion(n=steps1,H=H,T=T1)
                if H > 0.42 and H < 0.58:
                    label[i,0] = 1
            elif label[i,0] == 1:
                x,y,t = Brownian(n=steps1,T=T1) 
            else:
                x,y,t = CTRW(n=steps1,alpha=np.random.uniform(low=0.2,high=1),T=T1)
            sigma = np.random.choice([0,0.1,0.25])
            noise = sigma*np.std(np.diff(x,axis=0),axis=0)*np.random.randn(1,steps_actual)
            x1 = np.reshape(x,[1,len(x)])
            x1 = x1-np.mean(x1)
            x_n = x1[0,:steps_actual]+noise
            dx = np.diff(x_n)
            out[i,:,0] = dx
       
        label = to_categorical(label,num_classes=3)
        
        yield out,label
        
        
# generate FBM trajectories with different Hurst exponent values 
# for training of the Hurst-regression network
        
def fbm_regression(batchsize=32,steps=500,T=1,steps_actual=100):
    while True:
        # randomly choose a set of trajectory-length and final-time. This is intended
        # to increase variability in simuation conditions.
        steps1 = np.random.choice(steps,size=1).item()
        T1 = np.random.choice(T,size=1).item()
        out = np.zeros([batchsize,steps_actual-1,1])
        
        label = np.zeros([batchsize,1])
        for i in range(batchsize):
            H = np.random.uniform(low=0.05,high=0.95)
            label[i,0] = H
            x,y,t = fbm_diffusion(n=steps1,H=H,T=T1)
            
            n = 0.01*np.random.randn(steps1,)
            x_n = x+n
            dx = np.diff(x_n[:steps_actual],axis=0)
            
            out[i,:,0] = autocorr((dx)/(np.std(dx)))

        
        yield out,label
        
        
# generate batches of matrices with dimensions
# [number of trajectories]x[number of steps]
# For training the Hurst regression multi-track network
# Additional required input is numTraj: number of trajectories in matrix
        
def fbm_ensemble_regression(batchsize=32,steps=500,T=1,steps_actual=10,numTraj=50):
    while True:
        # randomly choose a set of trajectory-length and final-time. This is intended
        # to increase variability in simuation conditions.
        steps1 = np.random.choice(steps,size=1).item()
        T1 = np.random.choice(T,size=1).item()
        out = np.zeros([batchsize,numTraj,steps_actual-1,1])
        
        label = np.zeros([batchsize,1])
        for i in range(batchsize):
            H = np.random.uniform(low=0.05,high=0.95)
            label[i,0] = H
            for k in range(numTraj):
                Hdist = np.random.uniform(low=H-0.05,high=H+0.05)
            
                x,y,t = fbm_diffusion(n=steps1,H=Hdist,T=T1)
            
                n = 0.02*np.random.randn(steps1,)
                x_n = x+n
                dx = np.diff(x_n[:steps_actual],axis=0)
            
                out[i,k,:,0] = autocorr((dx)/(np.std(dx)))
        
        yield out,label      

# generate batches of samples for the diffusion coefficient
# regression network. This network recieves as input: the mean and std of the absolute
# value of the difference.
        
def brownian_scalar_regression(batchsize=32,steps=500,T=1,steps_actual=100):
    while True:
        # randomly choose a set of trajectory-length and final-time. This is intended
        # to increase variability in simuation conditions.
        steps1 = np.random.choice(steps,size=1).item()
        T1 = np.random.choice(T,size=1).item()
        out = np.zeros([batchsize,2,1])        
        label = np.zeros([batchsize,1])
        low = 0.01
        high = 1
        for i in range(batchsize):
            scale = np.random.uniform(low=low,high=high)
            noise = np.random.uniform(low = 0, high = 1)
            label[i,0] = (scale)
            x,y = Brownian(N=steps1-1,T=T1,delta=np.sqrt(2*scale*10/0.55**2))
            n = noise*(np.std(np.diff(x,axis=0)))*np.random.randn(steps1,)
            x_n = x+n
            x_n = x_n[:steps_actual]
            dx = np.diff(x_n,axis=0)
            
            m = np.mean(np.abs(dx),axis=0)
            s = np.std(dx,axis=0)
            
            out[i,:,0] = [m,s]

        
        yield out,label
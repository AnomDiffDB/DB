"""
FBM_single_functions.py
This script contains functions for activating and testing of Fractional Brownian Motion
single-trajectory networks trained to estimate the Hurst exponent. 
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from keras.models import load_model
from utils import fbm_diffusion,MSD_mat_from_file
from utils import autocorr, curve_func,MSD,matrix_of_MSDs
import scipy.optimize
import scipy.io
import seaborn as sns


# Figure settings for illustrator compatability
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style="white", palette="muted", color_codes=True)

"""
Function predict_1D is used to run a version of the network on a single trajectory.
The function assumes the input is a column vector

Input: 
    x - 1D column vector containing localization data
    stepsActual - number of steps to analyze in each trajectory. This number 
                  determnines what network will run on the data.
                  Select from options: [10,25,60,100,200,300,500,700,1000]
    reg_model - loaded model of the network (comes as input to avoid
                reloading the same model for each trajectory)
                  
Outputs:
    pX - Network prediction for the Hurst exponenet of the 1D trajectory.           
"""

def predict_1D(x,stepsActual,reg_model):
    
    if len(x)<stepsActual:
        return 0
    
    dx = (np.diff(x[:stepsActual],axis=0)[:,0])
    dx = autocorr(dx/(np.std(dx)+1e-10))
    dx = np.reshape(dx,[1,np.size(dx),1]) 
    pX = reg_model.predict(dx)
    
    return pX
    
        
 
"""
Function net_on_file is used to run a version of the network on a .mat file
containing one or more single particle trajectories.

The function assumes the input comes in the form x,y,z,...,N where N is the 
trajectory serial number, starting from one.

Input: 
    file - string containing the file name, ending with .mat
    NTraj - number of trajectories to analyze; input '0' to analyze all 
           trajectories
    stepsActual - number of steps to analyze in each trajectory. This number 
                  determnines what network will run on the data.
                  Select from options: [25,60,100,200,300,500,700,1000]
                  
Outputs:
    prediction - A vector with lenght as number of trajectories containing 
                 network predictions (average of N-dimensional predictions)
    NDpred - A matrix with dimensions [#trajetories,#dimensions] containing
             all predictions done by the network (N-dimensions for each trajectory)
"""

def FBM_net_on_file(file,stepsActual):
    
    # laod trained keras model
    # possible options: 25,50,100,200,300,500,700,1000

    ### change here to load a different network model
    if [25,50,100,200,300,500,700,1000].count(stepsActual) == 0 :
        print('stepsActual can be one of: [25,50,100,200,300,500,700,1000)')
        return 0, 0
    
    net_file = '.\Models\\fbm_reg_'+str(stepsActual)+'_steps_optimized.h5'
    reg_model = load_model(net_file)
    ###
    
    # load mat file and extract trajectory data
    
    f = scipy.io.loadmat(file)
    for k in f.keys():
        if k[0]=='_':
            continue
        varName = k
        data = f[varName]
        NAxes = (np.shape(data)[1]-1)
        numTraj = len(np.unique(data[:,NAxes]))

    # allocate variables to hold temporary data and prediction results
    prediction = np.zeros([numTraj,1])
    NDpred = np.zeros([numTraj,(np.shape(data)[1]-1)])    
    
    # iterate over trajectories and analyze data
    
    for i in np.arange(0,numTraj):
        for j in range((np.shape(data)[1]-1)):
            x = data[np.argwhere(data[:,NAxes]==i+1),j]
        
            pX = predict_1D(x,stepsActual,reg_model)
        
            NDpred[i,j] = pX
        
    NDpred = NDpred[np.where(NDpred>0)]
    NDpred = np.reshape(NDpred,[int(np.size(NDpred)/NAxes),NAxes])
    
    prediction = np.mean(NDpred,axis=1)
    
    return prediction, NDpred


"""
Function net_on_mat is used to run a version of the network on a numpy array
containing one or more single particle trajectories.

The function assumes the input matrix has the form:
    [number of dimensions][number of trajectories][number of steps]

Input: 
    X - data input matrix
    NTraj - number of trajectories to analyze; input '0' to analyze all 
           trajectories
    stepsActual - number of steps to analyze in each trajectory. This number 
                  determnines what network will run on the data.
                  Select from options: [25,60,100,200,300,500,700,1000]
                  
Outputs:
    prediction - A vector with length as number of trajectories containing 
                 network predictions (average of N-dimensional predictions)
    NDpred - A matrix with dimensions [#trajetories,#dimensions] containing
             all predictions done by the network (N-dimensions for each trajectory)
"""
def net_on_mat(X,stepsActual):
    # function takes X matrix in the form [number of dimensions][number of trajectories][number of steps]
    
    ### change here to load a different network model
    if [25,50,100,200,300,500,700,1000].count(stepsActual) == 0 :
        print('stepsActual can be one of: [25,50,100,200,300,500,700,1000)')
        return 0, 0
    
    net_file = '.\Models\\fbm_reg_'+str(stepsActual)+'_steps_optimized.h5'
    reg_model = load_model(net_file)
    
    numTraj = np.shape(X)[1]
        
        
    NDpred = np.zeros([np.shape(X)[0],np.shape(X)[1]])
    X = X[:,:numTraj,:stepsActual]
        
    for i in range(np.shape(X)[0]):
        for j in range(numTraj):        
            x = X[i,j,:] # get single track from matrix
            x = np.reshape(x,[np.shape(x)[0],1])
            pX = predict_1D(x,stepsActual,reg_model)
            NDpred[i,j] = pX 


    prediction = np.mean(NDpred,axis=0)
        
    return prediction,NDpred



"""
Function ST_nets_heatmaps generates FBM trajectories with random Hurst parameters,
and tests network performance on varying levels of noise and for different lengths
of trajectories

Input: 
    numTraj - number of FBM trajectories to generate, default is 100

                  
Outputs:
    RMSE_net - A matrix containing RMSE values between net prediction and ground truth
               as a function of noise level and trajectory length
    RMSE_msd - A matrix containing RMSE values between MSD prediction and ground truth
               as a function of noise level and trajectory length
"""

def ST_nets_heatmaps(numTraj=100):
    # allocate memory for data
    X = np.zeros([2,numTraj,1001])
    Hvec = np.zeros([numTraj,1])
    stepVec = [50,100,500,1000] # networks to run
    sigVec = [0,0.1,0.25,0.5] # 1/SNR
    RMSE_net = np.zeros([len(stepVec),len(sigVec)])
    RMSE_msd = np.zeros([len(stepVec),len(sigVec)])
    
    stdMat = np.zeros([1,numTraj,1])
    
    # Generate FBM trajectories with random Hurst exponents 
    for n in range(numTraj):
        H = np.random.uniform(low=0.05,high=0.95)
        Hvec[n,0] = H
        x,y,t = fbm_diffusion(n=1001,H=H,T=30)
        X[0,n,:] = x
        X[1,n,:] = y
        # calculate standard deviation per Hurst exponent (needed to speed things up later)
        stdMat[0,n,0] = np.mean([np.std(np.diff(x,axis=0),axis=0),np.std(np.diff(y,axis=0),axis=0)])
    
    # iterate over length options and noise options, in every iteration 
    # add Gaussian noise to the base data and analyze the noisy trajectories
    for s,sInd in zip(stepVec,range(len(stepVec))):
        for sig,sigInd in zip(sigVec,range(len(sigVec))):
            Xn = X[:,:,:s]
            noise = sig*stdMat*np.random.randn(2,numTraj,s)
            Xn = Xn+noise
            tVec,MSD_mat = matrix_of_MSDs(Xn)
            prediction, NDPred = net_on_mat(Xn,0,s)  
            mse = 0
            for i in range(numTraj):
                msd = MSD_mat[i,:]
                a,b = curve_fit(curve_func,np.log(1e-10+tVec),np.log(1e-10+msd.ravel()),maxfev=2000)
                mse += (Hvec[i]-a[1]/2)**2
            RMSE_net[sInd,sigInd] = np.sqrt(np.mean((prediction-Hvec[:,0])**2))
            RMSE_msd[sInd,sigInd] = np.sqrt(mse/numTraj)
            
            
            
    fig, ax = plt.subplots()
    fig.canvas.draw()
    plt.imshow(RMSE_net.transpose(),cmap = 'viridis')
    ax.set_yticks(range(len(sigVec)))
    ax.set_yticklabels(['Inf','10','4','2']);
    ax.set_xticks(range(len(stepVec)))
    ax.set_xticklabels(stepVec);
    plt.ylabel('Localization SNR');
    plt.xlabel('Number of steps');
    plt.title('net RMSE')
    plt.colorbar();
    plt.clim([0,0.5])
    
    fig, ax = plt.subplots()
    fig.canvas.draw()
    plt.imshow(RMSE_msd.transpose(),cmap = 'viridis')
    ax.set_yticks(range(len(sigVec)))
    ax.set_yticklabels(['Inf','10','4','2']);
    ax.set_xticks(range(len(stepVec)))
    ax.set_xticklabels(stepVec);
    plt.ylabel('Localization SNR');
    plt.xlabel('Number of steps');
    plt.title('msd RMSE')
    plt.colorbar();
    plt.clim([0,0.5])
            
    return RMSE_net,RMSE_msd



"""
Function temporal_MSD_compare generates 1000 FBM trajectories with a given Hurst parameter,
and a given noise level ,and tests varying network performances against MSD.

Input: 
    parameter - Hurst exponent ground truth 
    sigma - 1/SNR where SNR is the desired noise level

                  
Outputs:
    Hmsd25/Hmsd100/Hmsd1000 - MSD estimations based on 25/100/1000 steps
    Hnet25/Hnet100/Hnet1000 - Network estimations based on 25/100/1000 steps
"""

def temporal_MSD_compare(parameter = 0.5, sigma = 0):
    steps = 1000
    T = 40
    numTraj = 1000
    X = np.zeros([2,numTraj,steps])

    Hmsd1000 = np.zeros([numTraj,1])
    Hmsd100 = np.zeros([numTraj,1])
    Hmsd25 = np.zeros([numTraj,1])
    
    for i in range(numTraj):
        x,y,t = fbm_diffusion(n = steps, H = parameter, T=T)
        x = x+sigma*np.std(np.diff(x,axis=0),axis=0)*np.random.randn(steps,)
        y = y+sigma*np.std(np.diff(y,axis=0),axis=0)*np.random.randn(steps,)
        X[0,i,:] = x
        X[1,i,:] = y
        t,m,a = MSD(x,y)
        Hmsd1000[i,0] = a/2
        t,m,a = MSD(x[:25],y[:25])
        Hmsd25[i,0] = a/2
        t,m,a = MSD(x[:100],y[:100])
        Hmsd100[i,0] = a/2
        
        
    Hnet25,t = net_on_mat(X,0,25) # t is a temporary variable not used later
    Hnet100,t = net_on_mat(X,0,100)
    Hnet1000,t = net_on_mat(X,0,1000)
    
    plt.figure()
    plt.hist(Hmsd25,alpha=0.5,bins=np.arange(0,1.01,0.05));
    plt.hist(Hnet25,alpha=0.5,bins=np.arange(0,1.01,0.05));
    plt.xlabel('H'); plt.ylabel('Counts')
    plt.legend(['MSD','Network']);
    plt.title('25 steps network, H truth = '+np.str(parameter));
    plt.xlim([0,1])
    
    plt.figure()
    plt.hist(Hmsd100,alpha=0.5,bins=np.arange(0,1.01,0.05)); 
    plt.hist(Hnet100,alpha=0.5,bins=np.arange(0,1.01,0.05));
    plt.xlabel('H'); plt.ylabel('Counts');  plt.legend(['MSD','Network']);
    plt.title('100 steps network, H truth = '+np.str(parameter));  plt.xlim([0,1]);
    
    plt.figure()
    plt.hist(Hmsd1000,alpha=0.5,bins=np.arange(0,1.01,0.05)); 
    plt.hist(Hnet1000,alpha=0.5,bins=np.arange(0,1.01,0.05));
    plt.xlabel('H'); plt.ylabel('Counts');  plt.legend(['MSD','Network']);
    plt.title('1000 steps network, H truth = '+np.str(parameter)); plt.xlim([0,1]);

    return Hmsd25,Hmsd100,Hmsd1000,Hnet25,Hnet100,Hnet1000

"""
Function estimation_convergence analyzes data from a file and estimates confidence
intervals as a function of number of trajectories analyzed, for both network
and ensemble MSD. Intervals are estimated using a form of bootstrapping.

Input: 
    file - string containing the file name, ending with .mat
    stepsActual - number of steps to analyze in each trajectory. This number 
                  determnines what network will run on the data.
                  Select from options: [25,60,100,200,300,500,700,1000]

                  
Outputs:
    EMSD_err - confidence intervals as a function of trajectories analyzed (EMSD analysis)
    net_err - confidence intervals as a function of trajectories analyzed (net analysis)
    EMSD_plot - Anomalous exponent estimation as a function of trajectories analyzed (EMSD analysis)
    net_plot - Anomalous exponent estimation as a function of trajectories analyzed (net analysis)
"""

def estimation_convergence(file,stepsActual):

    prediction,NDpred = FBM_net_on_file(file,stepsActual)
    trajVec = np.arange(5,np.int(len(prediction)),20)    
    
    tVec, MSD_mat = MSD_mat_from_file(file,0,100)
    EMSD_scatter = np.zeros([100,len(trajVec)])
    net_scatter =  np.zeros([100,len(trajVec)])
    for i,ind in zip(trajVec,range(len(trajVec))):
        for nn in range(100):
            emsd = np.zeros([np.shape(MSD_mat)[1],1])
            trajSelect = np.random.choice(np.arange(0,len(prediction),dtype=np.int),size=i,replace=False)
            for j in range(np.shape(MSD_mat)[1]):
                current = MSD_mat[trajSelect,j]
                current = current.ravel()[np.flatnonzero(current)]
                emsd[j,0] = np.mean(current)
            emsd = emsd.ravel()[np.flatnonzero(emsd)]
            emsd = emsd[np.isfinite(emsd)]
            tVec = np.arange(1,len(emsd)+1)

            a,b = curve_fit(curve_func,np.log(tVec[10:100]),np.log(emsd[10:100].ravel()))
            EMSD_scatter[nn,ind] = a[1]/2
            
            net_scatter[nn,ind] = np.mean(prediction[trajSelect],axis=0)
    
    EMSD_err = np.std(EMSD_scatter,axis=0)
    EMSD_plot = np.mean(EMSD_scatter,axis=0)
    net_plot = np.mean(net_scatter,axis=0) 
    net_err = np.std(net_scatter,axis=0) 
    
    # Create the plot object
    _, ax = plt.subplots()
    ax.plot(trajVec, EMSD_plot, lw = 1, color = 'b', alpha = 1, label = 'MSD estimation')
    ax.plot(trajVec, net_plot, lw = 1, color = 'r', alpha = 1, label = 'Network estimation')
    # Shade the confidence interval
    ax.fill_between(trajVec, EMSD_plot-EMSD_err, EMSD_plot+EMSD_err, color = 'b', alpha = 0.3)
    ax.fill_between(trajVec, net_plot-net_err, net_plot+net_err, color = 'r', alpha = 0.3)
    ax.set_title('750 nm Data')
    ax.set_xlabel('Number of trajectories'); 
    ax.set_ylabel('H');
    ax.set_ylim([0.1,0.9])
    ax.legend(loc = 'best')
  
    return EMSD_err,net_err,EMSD_plot,net_plot
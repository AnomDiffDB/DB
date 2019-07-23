"""
FBM_multi_nets_testing.py
This script contains functions for activating and testing of Fractional Brownian Motion
multiple-trajectory networks trained to estimate the Hurst exponent from multiple
short trajectories.
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from keras.models import load_model
from utils import fbm_diffusion
from utils import autocorr, curve_func, matrix_of_MSDs
import scipy.optimize
import scipy.io
import seaborn as sns


# Figure settings for illustrator compatability
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style="white", palette="muted", color_codes=True)


"""
Function MT_net_on_file recieves a .mat file containing trajectories of N-dimensions
organized as x,y,z,..N where N is a trajectory serial number starting from 1.
NOTE: The net assumes all trajectories are of length 10, shorter trajectories will cause an error
The function constructs a matrix suitable for network input and analyzes the data.

Input: 
    file - string containing the file name, ending with .mat
                  
Outputs:
    prediction - A vector with length as the number of dimensions in the data,
                 containing network analysis results.
"""

def MT_net_on_file(file):
    
    f = scipy.io.loadmat(file)
    for k in f.keys():
        if k[0]=='_':
            continue
        varName = k
        data = f[varName]
        NAxes = (np.shape(data)[1]-1)   
        
    if ((len(data)+1)%10 > 0):
        extra = len(data)%10
        data = data[:-extra,:]
    numTraj = len(np.unique(data[:,NAxes]))
    
    if numTraj>150:
        data = data[:1500,:]
        numTraj = 150
        
    Xmat = np.zeros([NAxes,numTraj,9,1])
    for i in range(numTraj):
        for j in range(NAxes):
            x = data[np.argwhere(data[:,NAxes]==i+1),j]
            
            dx = np.diff(x,axis=0)
            dx = autocorr(dx[:,0]/(1e-10+np.std(dx[:,0])))
            Xmat[j,i,:,0] = dx

    ### change here to load a different network model
    net_file = '.\Models\\fbm_reg_ensemble_10_steps_'+np.str(numTraj)+'_traj.h5'
    reg_model = load_model(net_file)
    
    prediction = reg_model.predict(Xmat)
    
    return prediction


"""
Function ensemble_net_histogram_plot generates 200 matrices of N trajectories with
Hurst exponents in the range [param-0.05,param+0.05] and analyzes them using
the MultiTrack (MT) network.
The network recieves data in the shape: [1,numTraj,10,1]

Input: 
    param - Hurst exponent ground truth
    sig - 1/SNR
    NTraj - number of trajectories in a single matrix (a number from 10,20,30,..150)
           
Outputs:
    Hnet - Hurst exponent estimations by the network
    Hmsd - Hurst exponent estimations by time averaged MSD          
"""

def ensemble_net_histogram_plot(param = 0.5,sig=0,NTraj=20):
    
    ### change here to load a different network model
    net_file = '.\Models\\fbm_reg_ensemble_10_steps_'+np.str(NTraj)+'_traj.h5'
    reg_model = load_model(net_file)
        
    Hnet = np.zeros([200,1])
    Hmsd = np.zeros([200,1])
    
    for k in range(200):
        X = np.zeros([1,NTraj,9,1])
        Y = np.zeros([1,NTraj,9,1])
        Xmsd = np.zeros([2,NTraj,10])
        
        for i in range(NTraj):
            H = np.random.uniform(low=param-0.05,high=param+0.05)
            x,y,t = fbm_diffusion(n=100,H=H,T=4)
            nx = sig*np.random.randn(len(x),)
            ny = sig*np.random.randn(len(x),)
            x = x+nx
            y = y+ny
            Xmsd[0,i,:] = x[25:35]
            Xmsd[1,i,:] = y[25:35]
            
            dx = np.diff(x[25:35],axis=0)
            dy = np.diff(y[25:35],axis=0)
            
            X[0,i,:,0] = autocorr(dx/(1e-10+np.std(dx)))
            Y[0,i,:,0] = autocorr(dy/(1e-10+np.std(dy)))
        
        tVec,MSD_mat = matrix_of_MSDs(Xmsd)
        emsd = np.mean(MSD_mat,axis=0)
        a,b = curve_fit(curve_func,np.log(tVec),np.log(1e-10+emsd.ravel()))
        
        Hx = reg_model.predict(X) 
        Hy = reg_model.predict(Y)
        
        Hnet[k,0] = np.mean([Hx,Hy])
        Hmsd[k,0] = a[1]/2
        
    plt.figure()
    plt.hist(Hmsd,alpha=0.5); plt.hist(Hnet,alpha=0.5);
    plt.legend(['Ensemble MSD estimation (10 steps)','Net estimation']);
    plt.xlabel('H'); plt.xlim([0,1])
    plt.ylabel('Counts'); plt.title('Comparison to Ensemble MSD (10 step tracks)')    
    
    return Hnet,Hmsd


"""
Function ensemble_net_estimation_rmse is designed to test the effects of more
trajectories on estimation performance. The network iterates over the number of 
trajectories used M. For each value of M, 500 matrices of Mx10 are generated and
analyzed. 
The Function plots RMSE between estimation and ground truth as a function of number of trajectories
and Estimation mean and std as a function of number of trajectories.

Input: 
    param - Hurst exponent ground truth
    sig - 1/SNR
                  
Outputs:
    rmseMSD - RMSE values of MSD estimation to ground truth
    rmseNET - RMSE values of network estimation to ground truth
    aVec - MSD estimations of all 500 matrices for each number of trajectories  
    HNetVec - Network estimations of all 500 matrices for each number of trajectories  
"""

def ensemble_net_estimation_rmse(param=0.5,sig=0):
    numTraj = np.arange(10,151,10)
    rmseMSD = np.zeros([len(numTraj),500],dtype=np.float64)
    rmseNET = np.zeros([len(numTraj),500],dtype=np.float64)
    HMSDVec = np.zeros([500,len(numTraj)])
    HNetVec = np.zeros([500,len(numTraj)])
    
    for nt,nt_ind in zip(numTraj,range(len(numTraj))):
        net_file = '.\Models\\fbm_reg_ensemble_10_steps_'+np.str(nt)+'_traj.h5'
        reg_model = load_model(net_file)

        for k in range(500):
            
            Xnet = np.zeros([1,nt,9,1])
            Ynet = np.zeros([1,nt,9,1])
            
            Xmsd = np.zeros([2,nt,10])
              
            for i in range(nt):
                H = np.random.uniform(low=param-0.05,high=param+0.05)
                x,y,t = fbm_diffusion(n=100,H=H,T=4)
        
                x = x+sig*np.random.randn(len(x),) # add noise
                y = y+sig*np.random.randn(len(x),)
                
                dx = np.diff(x[25:35],axis=0)
                dy = np.diff(y[25:35],axis=0)
            
                Xnet[0,i,:,0] = autocorr(dx/(1e-10+np.std(dx)))
                Ynet[0,i,:,0] = autocorr(dy/(1e-10+np.std(dy)))
        
                Xmsd[0,i,:] = x[25:35]
                Xmsd[1,i,:] = y[25:35]
    
            tVec,MSD_mat = matrix_of_MSDs(Xmsd)
    
            emsd = np.mean(MSD_mat,axis=0)
            a,b = curve_fit(curve_func,np.log(tVec),np.log(1e-10+emsd.ravel()))   
            HMSDVec[k,nt_ind] = a[1]/2
            Hx = reg_model.predict(Xnet) 
            Hy = reg_model.predict(Ynet)
        
            HNetVec[k,nt_ind] = np.mean([Hx,Hy])
        rmseMSD[nt_ind,] = np.sqrt(np.mean((HMSDVec[:,nt_ind]-param)**2))
        rmseNET[nt_ind,] = np.sqrt(np.mean((HNetVec[:,nt_ind]-param)**2))
        
    plt.plot(numTraj,np.mean(rmseMSD,axis=1)); plt.plot(numTraj,np.mean(rmseNET,axis=1)); 
    plt.xlabel('Number of trajectories used in ensemble MSD'); 
    plt.title('RMSE'); plt.ylim([0,0.25]); plt.xlim([10,150]);
    plt.xticks(np.arange(10,151,20))
    plt.ylabel('RMSE'); plt.legend(['RMSE ensemble MSD','RMSE MT network']);
    
    EMSD_err = np.std(HMSDVec,axis=0)
    EMSD_plot = np.mean(HMSDVec,axis=0)
    net_plot = np.mean(HNetVec,axis=0) 
    net_err = np.std(HNetVec,axis=0) 
    
    # Create the plot object
    _, ax = plt.subplots()
    ax.plot(numTraj, EMSD_plot, lw = 1, color = 'b', alpha = 1, label = 'MSD estimation')
    ax.plot(numTraj, net_plot, lw = 1, color = 'r', alpha = 1, label = 'Network estimation')
    # Shade the confidence interval
    ax.fill_between(numTraj, EMSD_plot-EMSD_err, EMSD_plot+EMSD_err, color = 'b', alpha = 0.3)
    ax.fill_between(numTraj, net_plot-net_err, net_plot+net_err, color = 'r', alpha = 0.3)
    ax.set_title('Value convergence')
    ax.set_xlabel('Number of trajectories used in ensemble MSD'); 
    ax.set_ylabel('H');
    ax.set_ylim([param-0.3,param+0.3])
    ax.set_xlim([10,150])
    ax.set_xticks(np.arange(10,151,20))
    ax.legend(loc = 'best')

    return rmseMSD,rmseNET,HMSDVec,HNetVec



    
    

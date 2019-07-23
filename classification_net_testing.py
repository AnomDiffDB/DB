
"""
classification_net_testing.py
This script contains functions for activating and testing of the classification
net.
"""  
 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.models import load_model
from utils import fbm_diffusion,Brownian,CTRW,OrnsteinUng
from sklearn.metrics import confusion_matrix
import seaborn as sns
import scipy.io


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style="white", palette="muted", color_codes=True)

"""
Function classification_heatmap is used to test the classification network 
on FBM or CTRW trajectories as a function of noise levels and motion type parameters
The function plots the error fraction (1-accuracy) heatmap.

Input: 
    motionType - paramter to select motion model; 0=FBM, 1=CTRW
    noiseType - parameter to select noise model; 0=Gaussian, 1=Ornstein-Unghlebek noise
                  
Outputs:
    accuracy - Fraction of succesfull classifications
    prediction - vector of network predictions       
"""

def classification_heatmap(motionType=0, noiseType=0):

    steps_actual = 100    # classification network is trained on 100 step trajectories
    
    ### change here to load a different network model
    net_file = '.\Models\\classification_model_100_steps.h5'
    model = load_model(net_file)
    
    # choose parameters to iterate on
    paramVec = np.arange(0.05,0.96,0.1)
    sigmaVec = [0,0.1,0.2,0.25,0.5,1]

    accuracy = np.zeros([10,6])
    prediction = np.zeros([10,6,200])
    for p,pInd in zip(paramVec,range(10)):
        X = np.zeros([200,steps_actual])
        Y = np.zeros([200,steps_actual])
        nx = np.zeros([200,steps_actual])
        ny = np.zeros([200,steps_actual])
        for i in range(200):
            if motionType==0:
                x,y,t = fbm_diffusion(n=500,H=p,T=10)
            else:
                x,y,t = CTRW(n=500,alpha=p,T=10,gamma=0.001)
                x = np.reshape(x,[np.shape(x)[0],])
                y = np.reshape(y,[np.shape(y)[0],])

            X[i,:] = x[:steps_actual,]
            Y[i,:] = y[:steps_actual,]
            
        # Generate noise to add to the data
        for sig,sInd in zip(sigmaVec,range(6)):   
            if ~noiseType:
               nx = sig*np.std(np.diff(x,axis=0),axis=0)*np.random.randn(200,steps_actual)
               ny = sig*np.std(np.diff(y,axis=0),axis=0)*np.random.randn(200,steps_actual)
            else:
                for i in range(200):
                    nx[i,:] = OrnsteinUng(n=steps_actual-1,T=10,speed=1,mean=0,vol=1)
                    ny[i,:] = OrnsteinUng(n=steps_actual-1,T=10,speed=1,mean=0,vol=1)
            
            dX = np.diff(X+sig*nx,axis=1)
            dY = np.diff(Y+sig*ny,axis=1)
            
            dX = np.reshape(dX,[200,steps_actual-1,1])
            dY = np.reshape(dY,[200,steps_actual-1,1])
            y_pred1 = model.predict(dX)
            y_pred2 = model.predict(dY)
            y_pred = np.mean([y_pred1,y_pred2],axis=0)
            y_pred = np.argmax(y_pred,axis=1)
            if motionType == 0 :
                accuracy[pInd,sInd] = (np.sum(y_pred==0))/200
            else:
                accuracy[pInd,sInd] = (np.sum(y_pred==2))/200
            prediction[pInd,sInd,:] = y_pred
                
    fig, ax = plt.subplots()
    fig.canvas.draw()
    plt.imshow(accuracy,cmap = 'viridis')
    ax.set_xticks(range(6))
    if noiseType == 0:
        ax.set_xticklabels(['Inf','10','5','4','2','1']); # for Gaussian noise
    else:
        ax.set_xticklabels(['0','0.2','0.4','0.5','1','2']); # for OU noise
    ax.set_yticks(range(10))
    ax.set_yticklabels(['0.05','0.15','0.25','0.35','0.45','0.55','0.65','0.75','0.85','0.95']);
    plt.xlabel('Localization SNR');
    plt.ylabel('alpha ground truth');
    plt.title('fBm classification error with Gaussian noise')
    plt.colorbar();
    plt.clim([0,1]);

    return accuracy,prediction


"""
Function confusion_matrices generates simulated training data with various SNR
levels and prints out a confusion matrix per SNR value.

Input: 
    noiseType - parameter to select noise model; 0=Gaussian, 1=Ornstein-Unghlebek noise
    
Outputs:
    None
"""

def confusion_matrices(noiseType=0):
    steps = 1000
    T = 40
    
    ### change here to load a different network model
    net_file = '.\Models\\classification_model_100_steps.h5'
    model = load_model(net_file)
    
    sigmaVec = [0,0.1,0.2,0.25,0.5,1] # 1/SNR values to consider
    # allocate memory for data
    X = np.zeros([300,100])
    Y = np.zeros([300,100])
    nx = np.zeros([300,100])
    ny = np.zeros([300,100])
    y_truth = np.zeros([300,1])
    # generate 100 trajectories of each diffusion model
    for i in range(100):
        x,y = Brownian(N=steps,T=T,delta=1)
        X[i,:] = x[:100,]
        Y[i,:] = y[:100,]
        y_truth[i,0] = 1
        if noiseType == 0:
            nx[i,:] = np.std(np.diff(x,axis=0))*np.random.randn(100,)
            ny[i,:] = np.std(np.diff(y,axis=0))*np.random.randn(100,)
        else:
            nx[i,:] = OrnsteinUng(n=100-1,T=10,speed=1,mean=0,vol=1)
            ny[i,:] = OrnsteinUng(n=100-1,T=10,speed=1,mean=0,vol=1)
   
        x,y,t = CTRW(n=steps,alpha=np.random.uniform(low=0.01,high=0.9),T=10,gamma=0.001)
        X[i+200] = x[:100,0]
        Y[i+200] = y[:100,0]
        y_truth[i+200,0] = 2
        if noiseType == 0:
            nx[i+200,:] = np.std(np.diff(x,axis=0))*np.random.randn(100,)
            ny[i+200,:] = np.std(np.diff(y,axis=0))*np.random.randn(100,)
        else:
            nx[i+200,:] = OrnsteinUng(n=100-1,T=10,speed=1,mean=0,vol=1)
            ny[i+200,:] = OrnsteinUng(n=100-1,T=10,speed=1,mean=0,vol=1)
            
        # For FBM, 2 sets of data are created, 50 trajectories with Hurst exponents
        # in [0.05,0.42], and 50 trajectories with [0.58,0.95]
    for i in range(50):        
        x,y,t = fbm_diffusion(n=steps,H=np.random.uniform(low=0.1,high=0.42),T=T)
        X[i+100,:] = x[:100,]
        Y[i+100,:] = y[:100,]
        if noiseType == 0:
            nx[i+100,:] = np.std(np.diff(x,axis=0))*np.random.randn(100,)
            ny[i+100,:] = np.std(np.diff(y,axis=0))*np.random.randn(100,)
        else:
            nx[i+100,:] = OrnsteinUng(n=100-1,T=10,speed=1,mean=0,vol=1)
            ny[i+100,:] = OrnsteinUng(n=100-1,T=10,speed=1,mean=0,vol=1)
        x,y,t = fbm_diffusion(n=steps,H=np.random.uniform(low=0.58,high=0.95),T=T)
        X[i+150,:] = x[:100,]
        Y[i+150,:] = y[:100,]
        if noiseType == 0:
            nx[i+150,:] = np.std(np.diff(x,axis=0))*np.random.randn(100,)
            ny[i+150,:] = np.std(np.diff(y,axis=0))*np.random.randn(100,)
        else:
            nx[i+150,:] = OrnsteinUng(n=100-1,T=10,speed=1,mean=0,vol=1)
            ny[i+150,:] = OrnsteinUng(n=100-1,T=10,speed=1,mean=0,vol=1)
            
    # Iterate over SNR values and print out confusion matrices   
    for sig in sigmaVec:
        
        dX = np.diff(X+sig*nx,axis=1)
        dY = np.diff(Y+sig*ny,axis=1)
        dX = np.reshape(dX,[300,99,1])
        dY = np.reshape(dY,[300,99,1])
        y_pred1 = model.predict(dX)
        y_pred2 = model.predict(dY)
        y_pred = np.mean([y_pred1,y_pred2],axis=0)
        y_pred = np.argmax(y_pred,axis=1)
        C = confusion_matrix(y_truth,y_pred)
        if sig == 0:
            print('SNR = INF')
        else:
            print('SNR = '+str(1/sig))
        print(C)
    return 1


"""
Function classification_on_file is used to classify trajectories loaded from a
.mat file

The function assumes the input comes in the form x,y,z,...,N where N is the 
trajectory serial number, starting from one.

Input: 
    file - string containing the file name, ending with .mat
    
Outputs:
    prediction - Classification to diffusion model type where 0=FBM; 1=Brownian; 2=CTRW
    y_full - matrix of network probabilities. Each trajectory recieves 3 values 
             which are the probabilities of being assigned to a specific
             diffusion model.
"""


def classification_on_file(file):
    ### change here to load a different network model
    net_file = '.\Models\\classification_model_100_steps.h5'
    model = load_model(net_file)

    # load mat file and extract trajectory data
    
    f = scipy.io.loadmat(file)
    for k in f.keys():
        if k[0]=='_':
            continue
        varName = k
        data = f[varName]
        NAxes = (np.shape(data)[1]-1)
    
    numTraj = len(np.unique(data[:,NAxes]))
    prediction = np.zeros([numTraj,1])
    y_full = np.zeros([numTraj,3])
    flag = np.zeros([numTraj,1])
    for i in np.arange(1,numTraj+1):
        y_pred = np.zeros([NAxes,3])
        for j in range(NAxes):
            x = data[np.argwhere(data[:,NAxes]==i),j]
            x = x-np.mean(x)
            if len(x)>100: # classification network is trained on 100 step trajectories
                flag[i-1] = 1 # mark trajectories that are being analyzed
                dx = np.diff(x,axis=0)
                dx = np.reshape(dx[:99],[1,99,1])
            
            y_pred[j,:] = model.predict(dx) # get the results for 1D 
        ymean = np.mean(y_pred,axis=0) # calculate mean prediction of N-dimensional trajectory 
        prediction[i-1,0] = np.argmax(ymean,axis=0) # translate to classification
        y_full[i-1,:] = ymean
    prediction = prediction[np.where(flag==1)]
    return prediction,y_full

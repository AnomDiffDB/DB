"""
Diffusion_coeff_net.py
This script contains functions for activating and testing of the diffusion coefficient
estimation network.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.models import load_model
from utils import Brownian
from utils import MSD_D
import scipy.optimize
import scipy.io
import seaborn as sns

# Figure settings for illustrator compatability
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style="white", palette="muted", color_codes=True)



"""
Function Brownian_predict_1D is used to run a version of the network on a single trajectory.
The function assumes the input is a column vector

Input: 
    x - 1D column vector containing localization data
    stepsActual - number of steps to analyze in each trajectory. 
    pixelSize - size of 1 pixel (in nm) of the experimental system
    reg_model - loaded model of the network (comes as input to avoid
                reloading the same model for each trajectory)
                  
Outputs:
    pX - Network prediction for the Hurst exponenet of the 1D trajectory.           
"""

def Brownian_predict_1D(x,stepsActual,pixelSize,reg_model):
    
    if len(x)<stepsActual:
        return 0
    
    x = x*pixelSize/0.55
    dx = np.diff(x[:stepsActual],axis=0)
    dx = np.reshape(dx,[np.size(dx),1])
    dx = dx[:,0]-np.mean(dx,axis=0)
    
    dx = [np.mean(np.abs(dx),axis=0),np.std(dx,axis=0)]
    dx = np.reshape(dx,[1,np.size(dx),1]) 
    pX = reg_model.predict(dx)
    
    return pX
    
"""
Function Brownian_on_file analyzes data from a file and estimates diffusion coefficients
per trajectory.

Input: 
    file - string containing the file name, ending with .mat
    stepsActual - number of steps to analyze in each trajectory. 
    pixelSize - size of 1 pixel (in nm) of the experimental system
    dT - time difference between subsequent steps
Outputs:
    prediction - Network estimation of the diffusion coefficient
    Dmsd - Temporal mean square displacement estimation of the diffusion coefficient
"""

def Brownian_net_on_file(file = '',stepsActual = 100, pixelSize = 0.55,dT = 0.05):
    
    f = scipy.io.loadmat(file)
    for k in f.keys():
        if k[0]=='_':
            continue
        varName = k
        data = f[varName]
        NAxes = (np.shape(data)[1]-1)
    
    numTraj = len(np.unique(data[:,NAxes]))
    reg_model = load_model('.\Models\\diffusion_coeff_std03.h5')
        
    prediction = np.zeros([numTraj,1])
    predX = np.zeros([numTraj,1])
    predY = np.zeros([numTraj,1])
    Dmsd = np.zeros([numTraj,1])
    for i in np.arange(1,numTraj+1):
        x = data[np.argwhere(data[:,NAxes]==i),0]
        y = data[np.argwhere(data[:,NAxes]==i),1]
                
        if len(x)>=stepsActual:
            t,m,d = MSD_D(x[:stepsActual],y[:stepsActual],5)
            d = (np.exp(d)*pixelSize**2)/(4*dT)
            Dmsd[i-1,0] = d
            
            y_pred1 = Brownian_predict_1D(x,stepsActual,pixelSize,reg_model)
            y_pred2 = Brownian_predict_1D(y,stepsActual,pixelSize,reg_model)
        else:
            y_pred1 = 0     
            y_pred2 = 0
        
        predX[i-1] = y_pred1
        predY[i-1] = y_pred2
        
    prediction = 10*np.mean([predX,predY],axis=0)
    
    prediction = prediction[np.where(np.logical_and((Dmsd>0.05),(Dmsd<1)))]
    Dmsd = Dmsd[np.where(np.logical_and((Dmsd>0.05),(Dmsd<1)))]

    plt.figure();
    plt.hist(Dmsd[np.where(Dmsd>0.1)],alpha=0.5,bins=np.arange(0, 1.01, 0.03)); 
    plt.hist(prediction,alpha=0.5,bins=np.arange(0, 1.01, 0.03));
    plt.legend(['MSD estimation','Network estimation']);
    plt.title('Estimation results - '+np.str(stepsActual)+' steps network'); plt.xlabel('D'); 
    plt.ylabel('Counts'); plt.xlim([0,1]);

    
    return prediction,Dmsd


"""
Function Brownian_net_on_sim analyzes simulated trajectories and estimates diffusion coefficients
per trajectory.

Input: 
    param - Diffusion coefficient of the simulated trajectories (units of [um^2/sec])
    stepsActual - number of steps to analyze in each trajectory. 
    pixelSize - size of 1 pixel (in nm) of the experimental system
    dT - time difference between subsequent steps
Outputs:
    prediction - Network estimation of the diffusion coefficient
    Dmsd - Temporal mean square displacement estimation of the diffusion coefficient
"""

def Brownian_net_on_sim(param = 1, stepsActual = 100, pixelSize = 0.55, dT = 0.05):
   
    reg_model = load_model('.\Models\\diffusion_coeff_std03.h5')  # param is in um^2/sec
  
    # since the diffusion coefficient has physical units,but the network recieves
    # the input in units of pixels, a conversion step is necessary.
    Dcorrected = 2*param/pixelSize**2
    Dmsd = np.zeros([2000,1])
    DestX = np.zeros([2000,1])
    DestY = np.zeros([2000,1])
    for j in range(2000):
        x,y = Brownian(N=stepsActual,T=dT*stepsActual,delta=np.sqrt(Dcorrected))

        nx = 0.1*np.std(np.diff(x,axis=0))*np.random.randn(len(x),) 
        ny = 0.1*np.std(np.diff(y,axis=0))*np.random.randn(len(x),) 
        x = x+nx
        y = y+ny
                
        t,m,d = MSD_D(x[:stepsActual],y[:stepsActual],5)
        d = (np.exp(d)*pixelSize**2)/(4*dT)
        Dmsd[j,0] = d
                
        DestX[j,0] = Brownian_predict_1D(x,stepsActual,pixelSize,reg_model)
        DestY[j,0] = Brownian_predict_1D(y,stepsActual,pixelSize,reg_model)
        
    prediction = 10*np.mean([DestX,DestY],axis=0)
    
    prediction = prediction*0.05/dT

    plt.figure()
    plt.hist(Dmsd,alpha=0.5); plt.hist(prediction,alpha=0.5);
    plt.xlim([0,1])
    plt.ylabel('Counts'); plt.xlabel('D Estimation'); 
    plt.legend(['MSD estimation ('+np.str(stepsActual)+' steps)','Network estimation ('+np.str(stepsActual)+' steps)']);
    plt.title('D truth = '+np.str(param))  
    
    return prediction,Dmsd
    
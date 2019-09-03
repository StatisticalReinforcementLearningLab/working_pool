
import sys

import pickle
import pandas as pd
import numpy as np
import time
import os
import torch
import warnings
import math
import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.lazy import MatmulLazyTensor, RootLazyTensor
from gpytorch.constraints import constraints

def get_users(users,userstwo):
        
        xx,yy = np.meshgrid(users,userstwo,sparse=True)
        #.99999999999
        return (xx==yy).astype('float')

def get_first_mat(sigma_theta,data,baseline_indices):
    new_data = data[:,[baseline_indices]].reshape((data.shape[0],data.shape[1]))

    new_data_two = data[:,[baseline_indices]].reshape((data.shape[0],data.shape[1]))
    result = np.dot(new_data,sigma_theta)

    results = np.dot(result,new_data_two.T)
    return results

def rbf_custom_np( X, X2=None):
    #print(X)
    #print(X2)
    if X2 is None:
        X2=X
    return math.exp(-((X-X2)**2)/1.0)

def dist(x,x2):
    return math.exp(-((x-x2)**2)/1.0)

def get_sigma_u(u1,u2,rho):
    off_diagaonal_term = u1**.5*u2**.5*(rho-1)
    return np.array([[u1,off_diagaonal_term],[off_diagaonal_term,u2]])

def get_distance(days):
    to_return = []
    for i in range(len(days)):
        temp = []
        
        temp=[dist(days[i],days[j]) for j in range(len(days))]
        to_return.append(temp)
    return np.array(to_return)

class MyKernel(Kernel):
  
  
    def __init__(self, num_dimensions,user_mat,time_mat, first_mat,gparams, variance_prior=None, offset_prior=None, active_dims=None):
        super(MyKernel, self).__init__(active_dims=active_dims)
        self.user_mat = user_mat
        self.first_mat = first_mat
        self.time_mat = time_mat
       
        self.psi_dim_one = gparams.psi_indices[0]
        self.psi_dim_two = gparams.psi_indices[1]
        self.psi_indices =gparams.psi_indices
    
        
      
    def forward(self, x1, x2, batch_dims=None, **params):
        
       
        x1_ = torch.stack([x1[:,i] for  i in self.psi_indices],dim=1)
    
        x2_ =    torch.stack([x2[:,i] for  i in self.psi_indices],dim=1)
    
        if batch_dims == (0, 2):
            print('batch bims here')
       

        final = self.first_mat.mul(self.time_mat)
        
        return final


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,user_mat,time_mat,first_mat,gparams):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        
       
        self.mean_module = gpytorch.means.ZeroMean()
        #self.mean_module.constant.requires_grad=False
        
        self.covar_module =  MyKernel(len(gparams.baseline_indices),user_mat,time_mat,first_mat,gparams)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)









def run(X,users,days,y,global_params):
    #initial_u1,initial_u2,initial_rho,initial_noise,baseline_indices,psi_indices,user_index
    torch.manual_seed(1e6)
    user_mat= get_users(users,users)
    time_mat = get_distance(days)
    #print(user_mat.shape)
    #print(X.shape)
    #print(global_params.baseline_indices)
    first_mat = get_first_mat(np.eye(len(global_params.baseline_indices)),X,global_params.baseline_indices)
    #print(first_mat.shape)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    likelihood.noise_covar.initialize(noise=(global_params.o_noise_term)*torch.ones(1))
    
    X = torch.from_numpy(np.array(X)).float()
    y = torch.from_numpy(y).float()
    #print(X.size())
    first_mat = torch.from_numpy(first_mat).float()
  
    user_mat = torch.from_numpy(user_mat).float()
    time_mat = torch.from_numpy(time_mat).float()
    #print(time_mat.shape)
    #print(first_mat.shape)
  
    model = GPRegressionModel(X, y, likelihood,user_mat,time_mat,first_mat,global_params)
    
    model.train()
    likelihood.train()
    sigma_u=None
    sigma_v=None
    cov=None
    noise=None
    
    optimizer = torch.optim.Adam([
                                  {'params': model.parameters()},  # Includes GaussianLikelihood parameters
                                  ], lr=0.1)
                                  
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        #def train(num_iter):
    num_iter = 5
        #if global_params.called>9:
        #num_iter=5
    with gpytorch.settings.use_toeplitz(False):
            for i in range(num_iter):
                try:
                    #print('training')
                    #print(i)
                    optimizer.zero_grad()
                    output = model(X)
                #print(type(output))
                    loss = -mll(output, y)
                    loss.backward()
                    #print('Iter %d/%d - Loss: %.3f' % (i + 1, num_iter, loss.item()))
                    optimizer.step()
                 
                    f_preds = model(X)
                    f_covar = f_preds.covariance_matrix
                    covtemp = f_covar.detach().numpy()
                    if not np.isnan(covtemp).all():
                   
                        cov=covtemp
                        #print(np.isreal( covtemp))
                        #print(cov)
                        noise = likelihood.noise_covar.noise.item()


                except Exception as e:
                    print(e)
                    print('here')
                    break
#train(50)
    return cov


    




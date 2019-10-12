import pandas as pd
import numpy as np
import pickle
import random
import os
import math
import run_gpytorchkernel
import run_gpytorchkernel_larger
import run_gpytorchkernel_timeeffect
import run_gpytorchkernel_timecomp
import operator
import study
import time as time_module
import simple_bandits
import TS_personal_params_pooled as pp
import TS_global_params_pooled as gtp
from numpy.random import uniform
import TS
import warnings
warnings.simplefilter('ignore')
from sklearn import preprocessing
import tensorflow as tf
import gc
import feature_transformations as ft
import run_hob
import run_gpytorchkernel_timevarying
##import Marianne's package
##import Marianne's package


def update(algo_type,train_type,experiment,time,global_policy_params,personal_policy_params,feat_trans,participant=None):



    if algo_type=='batch' or algo_type=='pooling' or algo_type=='pooling_four':
        temp_hist = feat_trans.get_history_decision_time_avail(experiment,time)
        temp_hist= feat_trans.history_semi_continuous(temp_hist,global_policy_params)
        context,steps,probs,actions= feat_trans.get_form_TS(temp_hist)
        temp_data = feat_trans.get_phi_from_history_lookups(temp_hist)
            
        steps = feat_trans.get_RT(temp_data[2],temp_data[0],global_policy_params.mu_theta,global_policy_params.theta_dim)

        if algo_type=='batch':
            ##
            temp = TS.policy_update_ts_new( temp_data[0],steps,probs,actions,global_policy_params.noise_term,\
                                           global_policy_params.mu1_knot,\
                                           global_policy_params.sigma1_knot,\
                                           global_policy_params.mu2_knot,\
                                           global_policy_params.sigma2_knot)
                
                
            mu_beta = temp[0]
            Sigma_beta = temp[1]

            global_policy_params.update_mus(None,mu_beta,2)
            global_policy_params.update_sigmas(None,Sigma_beta,2)
            for participant in experiment.population.values():
                
                personal_policy_params.update_mus(participant.pid,mu_beta,2)
                personal_policy_params.update_sigmas(participant.pid,Sigma_beta,2)
                participant.last_update_day=time

        elif algo_type=='pooling':
            try:
                temp_params = run_gpytorchkernel.run(temp_data[0], temp_data[1],steps,global_policy_params)
                
                experiment.iters.append(temp_params['iters'])
                
                if temp_params['cov'] is not None:
                    global_policy_params.update_params(temp_params)
                        
            except Exception as e:
                temp_params={'cov':global_policy_params.cov,\
                                'noise':global_policy_params.noise_term,\
                                    'like':-100333,'sigma_u':global_policy_params.sigma_u}
            inv_term = simple_bandits.get_inv_term(global_policy_params.cov,temp_data[0].shape[0],global_policy_params.noise_term)
                                            
            global_policy_params.inv_term = inv_term
            global_policy_params.history = temp_data
            for participant in experiment.population.values():
                if time==participant.last_update_day+pd.DateOffset(days=global_policy_params.update_period):
                    temp = simple_bandits.calculate_posterior_faster(global_policy_params,\
                                                                              participant.pid,participant.current_day_counter,\
                                                                              global_policy_params.history[0], global_policy_params.history[1],global_policy_params.history[2] )
                    mu_beta = temp[0]
                    Sigma_beta = temp[1]
                    ##change here
                    personal_policy_params.update_mus(participant.pid,mu_beta,2)
                    personal_policy_params.update_sigmas(participant.pid,Sigma_beta,2)

                    participant.last_update_day=time
                
                
        elif algo_type=='pooling_four':
            try:
            
            
                temp_params = run_gpytorchkernel_larger.run(temp_data[0], temp_data[1],steps,global_policy_params)
            

                        #print(temp_params)
#global_policy_params.history =temp_data
                if temp_params['cov'] is not None:
                    global_policy_params.update_params_more(temp_params)
                sigma_u =simple_bandits.get_sigma_umore(global_policy_params)
                print(sigma_u)
                cov = simple_bandits.other_cov_notime(temp_data[0],global_policy_params.sigma_theta,random_effects,sigma_u,simple_bandits.get_users(temp_data[1],temp_data[1]))
                print(cov)
                global_policy_params.cov = cov
                
                inv_term = simple_bandits.get_inv_term(global_policy_params.cov,temp_data[0].shape[0],global_policy_params.noise_term)
                        
                global_policy_params.inv_term = inv_term
                        #print(temp_params)
                global_policy_params.history = temp_data
            except Exception as e:
                print('something failed')
                print(e)
                pass
                    

                        
            global_policy_params.history =temp_data
            for participant in experiment.population.values():
                if time==participant.last_update_day+pd.DateOffset(days=global_policy_params.update_period):
                    temp = simple_bandits.calculate_posterior_current(global_policy_params,\
                                                                 participant.pid,participant.current_day_counter,\
                                                                 global_policy_params.history[0], global_policy_params.history[1],global_policy_params.history[2] )
                    mu_beta = temp[0]
                    Sigma_beta = temp[1]
                ##change here
                    personal_policy_params.update_mus(participant.pid,mu_beta,2)
                    personal_policy_params.update_sigmas(participant.pid,Sigma_beta,2)
    
                    participant.last_update_day=time
            
            
    
    elif algo_type=='personalized':
        for participant in experiment.population.values():
            if time==participant.last_update_day+pd.DateOffset(days=global_policy_params.update_period):
        
                temp_hist = feat_trans.get_history_decision_time_avail_single({participant.pid:participant.history},time)
                temp_hist= feat_trans.history_semi_continuous(temp_hist,global_policy_params)
                context,steps,probs,actions= feat_trans.get_form_TS(temp_hist)
                temp_data = feat_trans.get_phi_from_history_lookups(temp_hist)
                context = temp_data[0]
                steps = feat_trans.get_RT_o(steps,temp_data[0],global_policy_params.mu_theta,global_policy_params.theta_dim)
                temp = TS.policy_update_ts_new( context,steps,probs,actions,global_policy_params.noise_term,\
                                           global_policy_params.mu1_knot,\
                                           global_policy_params.sigma1_knot,\
                                           global_policy_params.mu2_knot,\
                                           global_policy_params.sigma2_knot,
                                           
                                           
                                           )
                mu_beta = temp[0]
                Sigma_beta = temp[1]
                
                personal_policy_params.update_mus(participant.pid,mu_beta,2)
                personal_policy_params.update_sigmas(participant.pid,Sigma_beta,2)
                participant.last_update_day=time


    elif algo_type=='hob':
        temp_hist = feat_trans.get_history_decision_time_avail(experiment,time)
        temp_hist= feat_trans.history_semi_continuous(temp_hist,global_policy_params)
        context,users,steps= feat_trans.get_hob_form(temp_hist,global_policy_params)
        learned = run_hob.update_params(global_policy_params,context,steps)
        for participant in experiment.population.values():
            if time==participant.last_update_day+pd.DateOffset(days=global_policy_params.update_period):
                my_vec = learned[participant.pid*global_policy_params.d:participant.pid*global_policy_params.d+global_policy_params.d][-(global_policy_params.num_responsivity_features+1):]
                #print(learned.shape)
                #print(my_vec)
                personal_policy_params.update_mus(participant.pid,my_vec,2)
                participant.last_update_day=time
    elif algo_type=='hob_clipped':
        temp_hist = feat_trans.get_history_decision_time_avail(experiment,time)
        temp_hist= feat_trans.history_semi_continuous(temp_hist,global_policy_params)
        context,users,steps= feat_trans.get_hob_form_clipped(temp_hist,global_policy_params)
        #print(len(context))
        mu,sigma = run_hob.update_params_clipped(global_policy_params,context,steps)
        M=global_policy_params.d

        blocks = [sigma[i*M:(i+1)*M,i*M:(i+1)*M] for i in range(int(sigma.shape[0]/M))]
        for participant in experiment.population.values():
            if time==participant.last_update_day+pd.DateOffset(days=global_policy_params.update_period):
                my_vec = mu[participant.pid*global_policy_params.d:participant.pid*global_policy_params.d+global_policy_params.d][-(global_policy_params.num_responsivity_features+1):]
                
                my_sigma = [j[-(global_policy_params.num_responsivity_features+1):] for j in blocks[participant.pid][-(global_policy_params.num_responsivity_features+1):]]
                #print(np.array(my_sigma).shape)
            #print(my_vec)
            
            ##my mats
            
                personal_policy_params.update_mus(participant.pid,my_vec,2)
                personal_policy_params.update_sigmas(participant.pid,my_sigma,2)
                participant.last_update_day=time

    elif algo_type=='time_effects':
        temp_hist = feat_trans.get_history_decision_time_avail(experiment,time)
        temp_hist= feat_trans.history_semi_continuous(temp_hist,global_policy_params)
        context,steps,probs,actions= feat_trans.get_form_TS(temp_hist)
        temp_data = feat_trans.get_phi_from_history_lookups(temp_hist)
        
        steps = feat_trans.get_RT(temp_data[2],temp_data[0],global_policy_params.mu_theta,global_policy_params.theta_dim)
    

        try:
            temp_params = run_gpytorchkernel_timeeffect.run(temp_data[0], temp_data[1],temp_data[3],steps,global_policy_params)
            
            #experiment.iters.append(temp_params['iters'])
                
            if temp_params['cov'] is not None:
                    global_policy_params.update_params_more(temp_params)
                    global_policy_params.called=global_policy_params.called+1
    
        except Exception as e:
            print(e)
            temp_params={'cov':global_policy_params.cov,\
                'noise':global_policy_params.noise_term,\
                    'like':-100333,'sigma_u':global_policy_params.sigma_u,'sigma_v':global_policy_params.sigma_v}
        random_effects = np.array(temp_data[0])[:,global_policy_params.psi_indices]

        sigma_u =simple_bandits.get_sigma_umore(global_policy_params)
        sigma_v =simple_bandits.get_sigma_vmore(global_policy_params)
        cov = simple_bandits.other_cov_time(temp_data[0],global_policy_params.sigma_theta,random_effects,sigma_u,simple_bandits.get_users(temp_data[1],temp_data[1]),sigma_v,simple_bandits.get_distance(temp_data[3]))

        global_policy_params.cov = cov
        inv_term = simple_bandits.get_inv_term(global_policy_params.cov,temp_data[0].shape[0],global_policy_params.noise_term)

        global_policy_params.inv_term = inv_term
        global_policy_params.history = temp_data
        for participant in experiment.population.values():
            if time==participant.last_update_day+pd.DateOffset(days=global_policy_params.update_period):
                temp = simple_bandits.calculate_posterior_time_effects(global_policy_params,\
                                                                      participant.pid,participant.current_day_counter,\
                                                                      global_policy_params.history[0], global_policy_params.history[1],global_policy_params.history[3],global_policy_params.history[2] )
                mu_beta = temp[0]
                Sigma_beta = temp[1]
                    #with open('../../look/temp_cov_{}.pkl'.format(participant.pid),'wb') as f:
                    # pickle.dump(Sigma_beta,f)
                                                                      ##change here
                personal_policy_params.update_mus(participant.pid,mu_beta,2)
                personal_policy_params.update_sigmas(participant.pid,Sigma_beta,2)
                                                                      
                participant.last_update_day=time
        #rdayone = [x[global_params.user_day_index] for x in X]
        #rdaytwo = rdayone
        #rhos = np.array([[feat_trans.rbf_custom_np( rdayone[i], X2=rdaytwo[j]) for j in range(len(X))] for i in range(len(X))])
        
    
    elif algo_type=='time_comp':
        #temp_hist = feat_trans.get_history_decision_time_avail(experiment,time)
        #temp_hist= feat_trans.history_semi_continuous(temp_hist,global_policy_params)
        #context,steps,probs,actions= feat_trans.get_form_TS(temp_hist)
            
            
            # temp_data = feat_trans.get_phi_from_history_lookups(temp_hist)
# mri = get_most_recent_index(temp_data[1],temp_data[3])
#
#steps = feat_trans.get_RT(temp_data[2],temp_data[0],global_policy_params.mu_theta,global_policy_params.theta_dim)
                
# temp_params,noise=run_gpytorchkernel_timevarying.run(temp_data[0], temp_data[1],temp_data[3],steps,global_policy_params)
#global_policy_params.noise_term=noise
#global_policy_params.cov=temp_params
                #inv_term = simple_bandits.get_inv_term(global_policy_params.cov,temp_data[0].shape[0],global_policy_params.noise_term)
        
        
#first_mat = get_first_mat(np.eye(len(global_policy_params.baseline_indices)),temp_data[0],global_policy_params.baseline_indices)
# Dt = get_Dt(temp_data[3],global_policy_params)
# fp = np.dot(first_mat.T,Dt)
# mu = get_mu_tv(global_policy_params,temp_params,fp,steps)[-(global_policy_params.num_responsivity_features+1):]
                #mu = mu+global_policy_params.mu2
#sigma =get_sigma_tv(global_policy_params,temp_params,fp,steps)
#Sigma = [j[-(global_policy_params.num_responsivity_features+1):] for j in sigma[-(global_policy_params.num_responsivity_features+1):]]

        #print(Sigma.shape)

        #[j[-(global_policy_params.num_responsivity_features+1):] for j in sigma[-(global_policy_params.num_responsivity_features+1):]]

        #print('fp')
#print(fp.shape)
        for participant in experiment.population.values():
            if time==participant.last_update_day+pd.DateOffset(days=global_policy_params.update_period):
                temp_hist = feat_trans.get_history_decision_time_avail_single({participant.pid:participant.history},time)
                temp_hist= feat_trans.history_semi_continuous(temp_hist,global_policy_params)
                context,steps,probs,actions= feat_trans.get_form_TS(temp_hist)
        
   
                temp_data = feat_trans.get_phi_from_history_lookups(temp_hist)
                mri = get_most_recent_index(temp_data[1],temp_data[3])
        
                steps = feat_trans.get_RT(temp_data[2],temp_data[0],global_policy_params.mu_theta,global_policy_params.theta_dim)
                Dt = get_Dt(temp_data[3],global_policy_params)
             
                temp_params,noise,lenscale=run_gpytorchkernel_timevarying.run(temp_data[0], temp_data[1],temp_data[3],steps,global_policy_params,global_policy_params.ls[participant.pid])
                global_policy_params.noise_term=noise
                Dt = get_Dt(temp_data[3],global_policy_params)
                
                global_policy_params.cov=np.multiply(temp_params,Dt)
                global_policy_params.ls[participant.pid]=lenscale
                
                first_mat = get_first_mat(np.eye(len(global_policy_params.baseline_indices)),temp_data[0],global_policy_params.baseline_indices)

                fp = np.dot(first_mat.T,Dt)
                mu = get_mu_tv(global_policy_params,temp_params,fp,steps)[-(global_policy_params.num_responsivity_features+1):]
        #mu = mu+global_policy_params.mu2
                sigma =get_sigma_tv(global_policy_params,temp_params,fp,steps)
                Sigma = [j[-(global_policy_params.num_responsivity_features+1):] for j in sigma[-(global_policy_params.num_responsivity_features+1):]]

        #my_vec = first_mat[mri[participant.pid]]
        #my_steps = steps[mri[participant.pid]]
                personal_policy_params.update_mus(participant.pid,mu,2)
                personal_policy_params.update_sigmas(participant.pid,Sigma,2)
                
                participant.last_update_day=time

    elif algo_type=='time_compe':
        temp_hist = feat_trans.get_history_decision_time_avail(experiment,time)
        temp_hist= feat_trans.history_semi_continuous(temp_hist,global_policy_params)
        context,steps,probs,actions= feat_trans.get_form_TS(temp_hist)
        temp_data = feat_trans.get_phi_from_history_lookups(temp_hist)
        
        steps = feat_trans.get_RT(temp_data[2],temp_data[0],global_policy_params.mu_theta,global_policy_params.theta_dim)
        Dt = get_Dt(temp_data[3],global_policy_params)
        try:
            temp_params = run_gpytorchkernel_timecomp.run(temp_data[0], temp_data[1],temp_data[3],steps,global_policy_params)
        
        #experiment.iters.append(temp_params['iters'])
            K  = np.multiply(temp_params['cov'],Dt)
            temp_params['cov']=K
            if temp_params['cov'] is not None:
                global_policy_params.update_params(temp_params)
            #print(temp_params['cov'])
            
            inv_term = simple_bandits.get_inv_term(global_policy_params.cov,temp_data[0].shape[0],global_policy_params.noise_term)
                
            global_policy_params.inv_term = inv_term
            global_policy_params.history = temp_data
        except Exception as e:
            print(e)
            temp_params={'cov':global_policy_params.cov,\
                    'noise':global_policy_params.noise_term,\
                        'like':-100333,'sigma_u':global_policy_params.sigma_u}


        for participant in experiment.population.values():
                if time==participant.last_update_day+pd.DateOffset(days=global_policy_params.update_period):
                    temp = simple_bandits.calculate_posterior_faster_time(global_policy_params,\
                                                                     participant.pid,participant.current_day_counter,\
                                                                     global_policy_params.history[0], global_policy_params.history[1],global_policy_params.history[2] ,global_policy_params.history[3])
                    mu_beta = temp[0]
                    Sigma_beta = temp[1]
                                                                     ##change here
                    personal_policy_params.update_mus(participant.pid,mu_beta,2)
                    personal_policy_params.update_sigmas(participant.pid,Sigma_beta,2)
                                                                     
                    participant.last_update_day=time

    else:
        return 'Excepted types are batch, pooled, personalized, pooled_four, hob,hob_clipped, or time_effects'

def get_most_recent_index(users,days):
    maxes = {}
    for i in range(len(users)):
        user_id = users[i]
        if user_id not in maxes:
            maxes[user_id] = 0
        if days[i]>days[maxes[user_id]]:
            maxes[user_id]=i
    return maxes

def get_first_mat(sigma_theta,data,baseline_indices):
    new_data = data[:,[baseline_indices]].reshape((data.shape[0],data.shape[1]))
    
    new_data_two = data[:,[baseline_indices]].reshape((data.shape[0],data.shape[1]))
    result = np.dot(new_data,sigma_theta)
    
    #results = np.dot(result,new_data_two.T)
    return result


def get_first_part(kt,dt):
    pass

def get_Dt(days,glob):
    
    to_return = []
    for i in range(len(days)):
        temp = []
        
        temp=[(1-glob.time_eps)**abs(days[i]-days[j])**.5 for j in range(len(days))]
        to_return.append(temp)
    return np.array(to_return)

def get_mu_tv(glob,K,first_part,y):
    
    
    middle_part = np.linalg.inv(np.add(K,glob.noise_term*np.eye(K.shape[0])))
    #print(middle_part.shape)
    # print(first_part.shape)
    temp = np.dot(first_part,middle_part)
    #print(temp.shape)
    #print(np.dot(temp,y))
    return np.dot(temp,y)
#np.add(glob.mu_theta,np.matmul(temp,y))

def get_sigma_tv(glob,K,first_part,y):
    
    
    middle_part = np.linalg.inv(np.add(K,glob.noise_term*np.eye(K.shape[0])))
  
    temp = np.dot(first_part,middle_part)
    temp = np.dot(temp,first_part.T)
    #print(temp.shape)
    #print(np.dot(temp,y).shape)
    return np.subtract(glob.sigma_theta,temp)







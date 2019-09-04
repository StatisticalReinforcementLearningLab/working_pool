import pandas as pd
import numpy as np
import pickle
import random
import os
import math
import run_gpytorchkernel
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
import model_updates

def initialize_policy_params_TS(experiment,update_period,\
                                standardize=False,baseline_features=None,psi_features=None,\
                                responsivity_keys=None,algo_type=None,hob_params=None,case=None,correct=True):
   
    u_params=None
    v_params=None

    if algo_type=='pooling_four':
        
        u_params =[0.0553,0.0467,0.0052,0.0098,0.9030,1.9622,0.5410,1.9801,0.4680,1.9361]
#[0.0553,0.0467,0.0052,0.0098,0.9030,1.9622,0.5410,1.9801,0.4680,1.9361]
#[0.0553,0.0467,0.0052,0.0098,0.9030,1.9622,0.5410,1.9801,0.4680,1.9361]

    if algo_type=='time_effects':
        
        u_params =[0.0433,0.0338,0.0068,0.0073,1.7801,1.7499,1.6523,1.7945,1.7079,1.6743]
        #[0.0433,0.0337,0.0067,0.0072,1.7783,1.7505,1.6506,1.7960,1.7069,1.6745,0.0165,0.0,0.0388,0.0]
        v_params =[0.0165,0.0,0.0388,0.0]

    global_p =gtp.TS_global_params(21,baseline_features=baseline_features,psi_features=psi_features, responsivity_keys= responsivity_keys,uparams = u_params,vparams = v_params,case=case,correct=correct)
    
    
    
    personal_p = pp.TS_personal_params()

    
    global_p.kdim =24

    global_p.baseline_indices = [i for i in range(3+ len(baseline_features)+2*len(responsivity_keys))]
   
    global_p.psi_indices = [0] + [1+baseline_features.index(j) for j in psi_features] \
    + [len(baseline_features)+1] + [(2+len(baseline_features))+baseline_features.index(j) for j in psi_features]

    global_p.user_id_index =0
    
    global_p.psi_features =psi_features
  
    
    global_p.update_period = update_period
    
    global_p.standardize = standardize
    
    global_p.beta_updates = [1]+[int(responsivity_keys[i] in psi_features) for i in range(len(responsivity_keys))]

    global_p.beta_factor = 0.95
    
    initial_context = [0 for i in range(global_p.theta_dim)]
    
    global_p.mus0= global_p.get_mu0(initial_context)
 
    global_p.mus1= global_p.get_mu1(global_p.num_baseline_features)
    global_p.mus2= global_p.get_mu2(global_p.num_responsivity_features)
    global_p.sigmas1= global_p.get_asigma(global_p.num_baseline_features+1)
    global_p.sigmas2= global_p.get_asigma( global_p.num_responsivity_features+1)
    
 
    global_p.mu2_knot = np.array([4.83]+[0 for i in range(global_p.num_responsivity_features)])
    global_p.mu1_knot = np.zeros(global_p.num_baseline_features+1)
    global_p.sigma1_knot = np.eye(global_p.num_baseline_features+1)
    global_p.sigma2_knot = np.eye(global_p.num_responsivity_features+1)
    
    
    if algo_type=='hob':
        hob_params['vec_dim'] =global_p.num_baseline_features+1+global_p.num_responsivity_features+1
        global_p.init_hob_params(hob_params,experiment)
    
    elif algo_type=='hob_clipped':
        hob_params['vec_dim'] =global_p.num_baseline_features+1+2*global_p.num_responsivity_features+2
        global_p.init_hob_params(hob_params,experiment)
    
    for person in experiment.population.keys():
        
        
        
        initial_context = [0 for i in range(global_p.theta_dim)]
        
        
        
        
        
        #if algo_type!='batch':
        personal_p.mus0[person]= global_p.get_mu0(initial_context)
        personal_p.mus1[person]= global_p.get_mu1(global_p.num_baseline_features)
        personal_p.mus2[person]= global_p.get_mu2(global_p.num_responsivity_features)
            
        personal_p.sigmas0[person]= global_p.get_asigma(len( personal_p.mus0[person]))
        personal_p.sigmas1[person]= global_p.get_asigma(global_p.num_baseline_features+1)
        personal_p.sigmas2[person]= global_p.get_asigma( global_p.num_responsivity_features+1)
        
        
        
        personal_p.batch[person]=[[] for i in range(len(experiment.person_to_time[person]))]
        personal_p.batch_index[person]=0
        
       
        
    personal_p.last_update[person]=experiment.person_to_time[person][0]


    return global_p ,personal_p



def get_optimal_reward(beta,states,Z):
        if Z is None:
        
            return np.dot(beta,states)
        return np.dot(beta,states)+Z


def make_to_save(exp):
        to_save  = {}
        for pid,pdata in exp.population.items():
            for time,context in pdata.history.items():
            
                key = '{}-{}-{}'.format(pid,time,pdata.gid)
                #,'decision_time','avail','action','optimal_action'
                to_save[key]={k:v for k,v in context.items() if k in ['steps','avail','decision_time','action','optimal_action','other_action']}
        return to_save

def get_week_vec(current_day_counter):
    return [int(int(current_day_counter/(7))==(i-1)) for i in range(1,11)]

def new_kind_of_simulation(experiment,policy=None,personal_policy_params=None,global_policy_params=None,generative_functions=None,which_gen=None,feat_trans = None,algo_type = None,case=None,sim_num=None,train_type='None'):
    experiment.last_update_day=experiment.study_days[0]
    tod_check = set([])
    
    
    additives = []
    
    for time in experiment.study_days:

        if time==experiment.last_update_day+pd.DateOffset(days=global_policy_params.update_period):
            experiment.last_update_day=time

            if global_policy_params.decision_times>2:
                global_policy_params.last_global_update_time=time
                
                ### update model
                ##use global_policy_params_history, or some quick way of aggregating history rather than have any history objects floating around
                #print(global_policy_params.decision_times)
                model_updates.update(algo_type,train_type,experiment,time,global_policy_params,personal_policy_params,feat_trans)
    


        tod = feat_trans.get_time_of_day(time)
        dow = feat_trans.get_day_of_week(time)

        if time==experiment.study_days[0]:
            
            weather = feat_trans.get_weather_prior(tod,time.month,seed=experiment.weather_gen)
                
        elif time.hour in experiment.weather_update_hours and time.minute==0:
            weather = feat_trans.get_next_weather(str(tod),str(time.month),weather,seed=experiment.weather_gen)

        for person in experiment.dates_to_people[time]:
            participant = experiment.population[person]
            
            if participant.current_day!=time.date():
                participant.current_day_counter=participant.current_day_counter+1
                participant.current_day= time.date()
                #and algo_type=='time_effects'

            ##update responsivity
            
            dt=int(time in participant.decision_times)
            action = 0
            prob=0
            participant.set_tod(tod)
            participant.set_dow(dow)

            availability = (participant.rando_gen.uniform() < 0.8)
    
            participant.set_available(availability)
            if time == participant.times[0]:
                location = feat_trans.get_location_prior(str(participant.gid),str(tod),str(dow),seed = participant.rando_gen)
            elif time.hour in experiment.location_update_hours and time.minute==0 :
                location = feat_trans.get_next_location(participant.gid,tod,dow,participant.get_loc(),seed =participant.rando_gen)

            participant.set_loc(location)

            if time <= participant.times[0]:
                        steps_last_time_period = 0
            else:
                #if time.hour==0 and time.minute==0:
                #participant.current_day_counter=participant.current_day_counter+1

                steps_last_time_period = participant.steps

            prob = .6
            add=None
            optimal_action = -1
            optimal_reward = -100
            other_reward=-100
            if dt:
                if policy=='TS':
                    pretreatment = feat_trans.get_pretreatment(steps_last_time_period)
                    z = [1]
                    calc = [1,tod,dow,pretreatment,location]
                    calc_regret = [1,tod,dow,pretreatment,location]
                    
                    if experiment.time_condition=='burden':
                        week_part  = get_week_vec(participant.current_day_counter)
                        #
                        calc = week_part+[tod,dow,pretreatment,location]
                        calc_regret = [1,tod,dow,pretreatment,location]
                    if experiment.time_condition=='no_location':
                        calc = [1,tod,dow,pretreatment,location]
                        calc_regret = [1,tod,pretreatment,location]
                    
                    
                    if 'tod' in global_policy_params.responsivity_keys:
                        z.append(tod)
                    if 'dow' in global_policy_params.responsivity_keys:
                        z.append(dow)
                    if 'pretreatment' in global_policy_params.responsivity_keys:
                        z.append(pretreatment)
                    if 'location' in global_policy_params.responsivity_keys:
                        z.append(location)

                    
                        ##change so that in batch case everything just has same global parameters
                      
                   
                   
                   ##changed to be the same for everyone
                    prob = TS.prob_cal_ts(z,0,personal_policy_params.mus2[participant.pid],personal_policy_params.sigmas2[participant.pid],global_policy_params,seed=experiment.algo_rando_gen,algo_type=algo_type)

                    action = int(experiment.algo_rando_gen.uniform() < prob)
                    if availability:
                        context = [action,participant.gid,tod,dow,weather,pretreatment,location,\
                                       0,0,0]
                        steps = feat_trans.get_steps_action(context,seed = participant.rando_gen)
                        add = action*(feat_trans.get_add_no_action(calc,participant.beta,participant.Z))
                        participant.steps = steps+add
                        optimal_reward = get_optimal_reward(participant.beta_regret,calc_regret,participant.Z)
                        other_reward = get_optimal_reward(participant.beta,calc,participant.Z)
                        optimal_action = int(optimal_reward>0)
                        other_action= int(other_reward>0)
                    else:
                        steps = feat_trans.get_steps_no_action(participant.gid,tod,dow,location,\
                        pretreatment,weather,seed = participant.rando_gen)
                        participant.steps = steps

                    global_policy_params.decision_times =   global_policy_params.decision_times+1
                else:
                        steps = feat_trans.get_steps_no_action(participant.gid,tod,dow,location,\
                                                               pretreatment,weather,seed = participant.rando_gen)
                        participant.steps = steps
                context_dict =  {'steps':participant.steps,'add':add,'action':action,'location':location,'location_1':int(location==1),\
'ltps':steps_last_time_period,'location_2':int(location==2),'location_3':int(location==3),\
    'study_day':participant.current_day_counter,\
        'decision_time':dt,\
            'time':time,'avail':availability,'prob':prob,\
                'dow':dow,'tod':tod,'weather':weather,\
                    'pretreatment':feat_trans.get_pretreatment(steps_last_time_period),\
                        'optimal_reward':optimal_reward,'optimal_action':optimal_action,\
                            'mu2':personal_policy_params.mus2[participant.pid],'gid':participant.gid,'calc':calc,'calcr':calc_regret,'other_action':other_action,'other_reward':other_reward}

                participant.history[time]=context_dict


def get_regret(experiment):
    optimal_actions ={}
    rewards = {}
    actions = {}
    other_regrets = {}
    for pid,person in experiment.population.items():
        for time,data in person.history.items():
            if data['decision_time'] and data['avail']:
                key = time
                if key not in optimal_actions:
                    optimal_actions[key]=[]
                if key not in rewards:
                    rewards[key]=[]
                if key not in other_regrets:
                    other_regrets[key]=[]
                if key not in actions:
                    actions[key]=[]
                if data['optimal_action']!=-1:
                    optimal_actions[key].append(int(data['action']==data['optimal_action']))
                    regret = int(data['action']!=data['optimal_action'])*(abs(data['optimal_reward']))
                    oregret = int(data['action']!=data['other_action'])*(abs(data['other_reward']))
                    rewards[key].append(regret)
                    other_regrets[key].append(oregret)
                    actions[key].append(data['action'])
    return optimal_actions,rewards,other_regrets

def get_regret_person_specific(experiment):
    optimal_actions ={}
    rewards = {}
    actions = {}
    other_regrets={}
    for pid,person in experiment.population.items():
        if pid not in rewards:
            rewards[pid]={}
            other_regrets[pid]={}
        for time,data in person.history.items():
            if data['decision_time'] and data['avail']:
                key = time
              
                if data['optimal_action']!=-1:
                    regret = int(data['action']!=data['optimal_action'])*(abs(data['optimal_reward']))
                    #rewards[key].append(regret)
                    #actions[key].append(data['action'])
                    rewards[pid][time]=regret

                    oregret = int(data['action']!=data['other_action'])*(abs(data['other_reward']))
          
                    other_regrets[pid][key]=oregret

    return rewards,other_regrets

def make_to_groupids(exp):
    to_save  = {}
    for pid,pdata in exp.population.items():
        gid  = pdata.gid
        key = 'participant-{}'.format(pid)
        to_save[key]=gid
    return to_save

def run_many(algo_type,cases,sim_start,sim_end,update_time,dist_root,write_directory,train_type,correct=True,time_cond='None',pop_size=32):
    for case in cases:
      
        baseline = ['tod','dow','pretreatment','location']
        responsivity_keys = ['tod','dow','pretreatment','location']
        
        if time_cond=='no_location':
            baseline = ['tod','pretreatment','location']
            responsivity_keys = ['tod','pretreatment','location']
        
        
        u = update_time
        pn=1
        for epsilon in [0.3]:
            
            all_actions = {}
            all_rewards = {}

            feat_trans = ft.feature_transformation(dist_root)
            
            for sim in range(sim_start,sim_end):
                
                experiment = study.study(dist_root,pop_size,'_short_staggered_10rn',which_gen=case,sim_number=sim,pop_number=pn,time_condition=time_cond)
                #experiment.update_beta(set(responsivity_keys))
               
                psi = []
                #if algo_type=='pooling_four':
                psi = ['location']
                cend=''
                degree= 5
                if not correct:
                    cend = '_inc'
                    degree=0.01
                glob,personal = initialize_policy_params_TS(experiment,u,standardize=False,baseline_features=baseline,psi_features=psi,responsivity_keys=responsivity_keys,algo_type =algo_type,hob_params={'degree':degree},case=case,correct=correct)
                glob.sim_number=sim
                glob.time_eps = epsilon
                hist = new_kind_of_simulation(experiment,'TS',personal,glob,feat_trans=feat_trans,algo_type=algo_type,case=case,sim_num=sim,train_type=train_type)
                #return hist
                to_save = make_to_save(experiment)
                actions,rewards,other_regrets = get_regret(experiment)
                per_rewards,perregrets = get_regret_person_specific(experiment)
                gids = make_to_groupids(experiment)
                eps = '_eps_{}'.format(epsilon)
                #return experiment,glob,personal

                filename = '{}{}/population_size_{}_update_days_{}_{}_static_sim_{}_pop_{}_{}92unstaggered_perl_cond{}{}.pkl'.format('{}{}/'.format(write_directory,algo_type),case,pop_size,u,'short',sim,pn,time_cond,cend,eps)
                with open(filename,'wb') as f:
                    pickle.dump({'gids':gids,'regrets':rewards,'oregrets':other_regrets,'actions':actions,'pregret':per_rewards,'poregret':perregrets,'history':to_save,'pprams':personal,'gparams':glob.mus2},f)
      




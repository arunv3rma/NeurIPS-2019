# !/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss


# ########################## Plotting function ##########################
# Getting Average regret and Confidence interval
def accumulative_regret_error(regret):
    time_horizon = [0]
    samples = len(regret[0])
    runs = len(regret)
    batch = samples / 20
    if samples % batch != 0:
        batch += 1

    # Time horizon
    t = 0
    while True:
        t += 1
        if time_horizon[-1] + batch > samples:
            if time_horizon[-1] != samples:
                time_horizon.append(time_horizon[-1] + samples % batch)
            break
        time_horizon.append(time_horizon[-1] + batch)

    # Mean batch regret of R runs
    avg_batched_regret = []
    for r in range(runs):
        count = 0
        accumulative_regret = 0
        batch_regret = [0]
        for s in range(samples):
            count += 1
            accumulative_regret += regret[r][s]
            if count == batch:
                batch_regret.append(accumulative_regret)
                count = 0

        if samples % batch != 0:
            batch_regret.append(accumulative_regret)
        avg_batched_regret.append(batch_regret)

    regret = np.mean(avg_batched_regret, axis=0)

    # Confidence interval
    conf_regret = []
    freedom_degree = runs - 1
    for r in range(len(avg_batched_regret[0])):
        conf_regret.append(ss.t.ppf(0.95, freedom_degree) *
                           ss.sem(np.array(avg_batched_regret)[:, r]))
    return time_horizon, regret, conf_regret


# Regret Plotting
def regret_plotting(regret, cases, plotting_info):
    colors = list("gbcmryk")
    shape = ['--^', '--v', '--H', '--d', '--+', '--*']

    # Scatter Error bar with scatter plot
    for c in range(cases):
        horizon, batched_regret, error = accumulative_regret_error(np.array(regret)[:, c])
        plt.errorbar(horizon, batched_regret, error, color=colors[c])
        plt.plot(horizon, batched_regret, colors[c] + shape[c], label=r'$\theta_c=$'+str(plotting_info[4][c]))
    
    # plt.rc('font', size=14)                     # controls default text sizes
    plt.legend(loc='lower right', numpoints=1)   # Location of the legend
    plt.xlabel(plotting_info[0], fontsize=15)
    plt.ylabel(plotting_info[1], fontsize=15)
    # plt.title(plotting_info[2])

    # plt.axis([0, samples, -20, samples])
    # plt.yscale('log')
    # plt.xscale('log')
    plt.savefig(plotting_info[3], bbox_inches = 'tight')

    plt.close()
# #######################################################################


# ########################  Useful functions #######################
# Samples manipulation function: creating, saving and reading
def getSamples(data_file_name, mu, samples, generate_new_data, save_generated_data, random_seed):
    # Generate data for comparision between different approaches
    if generate_new_data:
        np.random.seed(random_seed)
        observations = np.zeros((samples, arms))    # Initialize observations
        for l in range(arms):
            observations[:, l] = np.random.binomial(1, mu[l], size=samples)
        # print np.average(observations, axis=0)

        # Saving Data
        if save_generated_data:
            data = pd.DataFrame(observations)
            data.to_pickle(data_file_name)

    # Read pre-generated data 
    else:
        observations = np.array(pd.read_pickle(data_file_name))
        # print np.average(observations, axis=0)

    return observations


# Get Best allocation for given data theta value
def get_top_allocation(mu_hat, M, theta_est):
    K = len(mu_hat)          # Number of arms
 
    # Allocated for resources using arms's loss-rates
    allocated_set = np.argsort(mu_hat)[-int(M):]
      
    # Assign allocation share
    p = np.zeros(K)
    for l in range(K):
        if l in allocated_set:
            p[l] = theta_est

    # print mu_hat, allocated_set, M, p
    return p, allocated_set


# Get allocation of resource to M random arms for given threshold
def get_random_allocation(theta_values, theta_index, N, K):
    # Estimated threshold and maximum arms can be served with current estimate
    theta_hat   = theta_values[theta_index]
    M           = int(N/theta_hat)    
    
    # Allocated resources on randomly selected arms
    allocated_set = np.random.choice(K, M, replace=False)
      
    # Assign allocation share
    p = np.zeros(K)
    for l in range(K):
        if l in allocated_set:
            p[l] = theta_hat

    # print mu_hat, allocated_set, p
    return p, allocated_set


# Get censored data
def get_censored_feedback(observation, theta, p):
    K = len(observation)
    censored_feedback = np.ones(K)
    optimal_allocation = True

    for l in range(K):
        if theta > p[l]:
            censored_feedback[l] = observation[l]
            if censored_feedback[l] == 1 and p[l] != 0:
                optimal_allocation = False                
        else:
            censored_feedback[l] = 0
        
    # print censored_feedback
    return [censored_feedback, optimal_allocation]


# Get instantaneous regret
def get_round_regret(mu, theta, maximum_gain, selected_allocation, p):
    current_gain = 0
    for l in selected_allocation:
        if p[l] >= theta:
            current_gain += mu[l]

    return maximum_gain - current_gain


# Thompson Sampling based Algorithm: CSB-ST
def csb_st(samples, parameters):
    # Data Overview
    K                       = len(samples[0])       # Number of arms
    T                       = len(samples)          # Number of rounds
    
    # Algorithms parameters
    mu                      = parameters[0]         # Arm rates
    Q                       = parameters[1]         # Number of resources
    theta                   = parameters[2]         # Threshold after which no loss is observed for arms
    delta                   = parameters[3]         # Probability of estimate lies in given confidance
    Delta_min               = parameters[4]         # Minimum difference between Arm rate
    epsilon                 = parameters[5]         # Minimum Arm rate

    # Maximum number of samples needed to wait without observing a loss
    W    = int(np.ceil(np.log((1.0*K)/delta)/(Q*np.log(1.0/(1-epsilon)))))

    # Finding Optimal set for given Q, theta and resources
    optimal_arms    = []                        # Set contains optimal arms for given theta
    mu_sorted       = sorted(mu, reverse=True)  # Sorted Arm-rates
    M               = int(np.floor(Q/theta))    # Number of arms in optimal allocation
    if M > K:
        M = K
    for i in range(M):
        optimal_arms.append(mu.index(mu_sorted[i]))
    
    # Maximum and minimum gain using best allocation
    max_gain = sum(mu_sorted[:M])

    # Variables
    instantaneous_regret    = []

    # ########################## Finding Theta ##########################
    # Possible theta values
    theta_values = []
    for i in range(K, 0, -1):
        # Not allowing the larger than 1.0 theta value
        if (1.0/i)*Q > 1:
            break
        theta_values.append((1.0/i)*Q)

    # Initialization
    theta_lower_index   = 0                         # Lower bound of Arm threshold
    theta_upper_index   = len(theta_values)-1       # Upper bound of Arm threshold
    
    # Estimate of Arm threshold    
    theta_index         = int(np.ceil(theta_upper_index/2))  

    p, allocated_set = get_random_allocation(theta_values, theta_index, Q, K)
    # print p, allocated_set, theta_values[theta_index], N/theta_values[theta_index]

    # Estimating Arm threshold
    estimating_crrime_threshold = True
    waiting_counter = 0
    t = 0
    while estimating_crrime_threshold and t < T:
        censored_feedback = get_censored_feedback(samples[t], theta, p)
        # print samples[t], theta, p, censored_feedback, t 

        # Estimating Arm threshold using Binary search
        if censored_feedback[1]:
            waiting_counter += 1
            if waiting_counter == W:
                # Decrease theta index
                theta_upper_index = theta_index
                theta_index = int(theta_upper_index - np.floor((theta_upper_index - theta_lower_index)/2.0))
                p, allocated_set = get_random_allocation(theta_values, theta_index, Q, K)
                waiting_counter = 0
                # print t+1, "Decrease: ", theta_values[theta_index], p
              
        else:
            # Increase theta index
            theta_lower_index = theta_index
            theta_index = int(theta_lower_index + np.ceil((theta_upper_index - theta_lower_index)/2.0))
            p, allocated_set = get_random_allocation(theta_values, theta_index, Q, K)
            waiting_counter = 0
            # print t+1, "Increase:", theta_values[theta_index]
        
        if theta_index == theta_upper_index:
            estimating_crrime_threshold = False
            
        # Instantaneous regret
        instantaneous_regret.append(get_round_regret(mu, theta, max_gain, range(K), p))
        t += 1

    # Estimated theta and maximum arms that can be served
    theta_est = theta_values[theta_index]  
    M_est = int(Q/(theta_est - 1e-12))  # 1e-12 is subtracted in denominator to avoid 6/0.6 = 9
    # print theta, theta_est, t, M

    # ######################### TS based Updates ########################
    mu_est = np.zeros(K)
    S = np.ones(K)
    F = np.ones(K)
    while t < T:
        # Estimating means in loss setting
        for i in range(K):
            mu_est[i] = np.random.beta(S[i], F[i])
        
        # Getting allocation and resource distribution vector (loss setting)
        p, allocated_set = get_top_allocation(mu_est, M_est, theta_est)

        # Update Arms for arms where no resource are allocated
        censored_feedback = get_censored_feedback(samples[t], theta, p)
        for i in range(K):
            if i not in allocated_set:
                if censored_feedback[0][i] == 1:
                    S[i] += 1
                else:
                    F[i] += 1

        # Regret for current round
        regret = get_round_regret(mu, theta, max_gain, range(K), p)
        instantaneous_regret.append(regret)
        t += 1
    
    return instantaneous_regret


# #######################################################################
# ########## Problem parameters ##########
arms            = 20        # Number of arms
theta_c         = 0.7       # Arm Threshold
resources       = 7         # Amount of resources
horizon         = 10000     # Number of rounds
runs            = 100       # Number of simulations

# ########## Algorithm parameters for CSB-ST ##########
delta_value         = 0.1
epsilon_value       = 0.1
cases               = [0.45, 0.55, 0.7, 0.85, 0.95]
num_cases           = len(cases) 

# Arm rate vector
mu_vector       = []
Delta_min_value = 1.0/(2*arms)  # Minimum rate difference between any two arms
largest_mean    = 0.7           # Options: 0.5 and 0.7
for l in range(arms):
    mu_vector.append(np.around(largest_mean - (l*Delta_min_value), decimals=2))

# Data parameters
generate_data       = True      # True: Generate new data, False: Used saved generated data
save_data           = True      # True: Save new generated data, False: Do not save data
random_seed_value   = 100  

# ########## Plotting parameters ##########
xlabel              = "Rounds"
ylabel              = "Cumulative Regret"
# file_to_save        = "sameThetaT_" + str(arms) + "_" + str(num_cases) + "_" + str(horizon) + "_" + str(runs) + ".png"
file_to_save        = "sameThetaT.png"
title               = "Varying Thresholds"
save_to_path        = "plots/" 
location_to_save    = save_to_path + file_to_save
plotting_parameters = [xlabel, ylabel, title, location_to_save, cases, horizon]

# ########## Generating and saving losses for Arms ##########
file_name = "data/same_" + str(horizon) + "_" + str(arms) + "_" + str(int(largest_mean*10)) 
loss_file = file_name + ".pkl"
data_samples = getSamples(loss_file, mu_vector, horizon, generate_data, save_data, random_seed_value)

# Algorithm paramters: [mu, N, theta, delta, Delta_min, epsilon]
algos_parameters = [mu_vector, resources, theta_c, delta_value, Delta_min_value, epsilon_value]

# Algorithm
algos_regret = []
for r in range(runs):
    run_regret = []
    np.random.shuffle(data_samples)
    for a in range(num_cases):    
        algos_parameters[2] = cases[a]
        iter_regret = csb_st(data_samples, algos_parameters)
        run_regret.append(iter_regret)

    algos_regret.append(run_regret)

# ########## Plotting final results ##########
regret_plotting(algos_regret, num_cases, plotting_parameters)
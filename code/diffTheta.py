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
        plt.plot(horizon, batched_regret, colors[c] + shape[c], label=plotting_info[4][c])
    
    # plt.rc('font', size=14)                     # controls default text sizes
    plt.legend(loc='lower right', numpoints=1)   # Location of the legend
    plt.xlabel(plotting_info[0], fontsize=15)
    plt.ylabel(plotting_info[1], fontsize=15)
    # plt.title(plotting_info[2])

    # plt.axis([0, samples, -20, samples])
    # plt.yscale('log')
    # plt.xscale('log')
    plt.savefig(plotting_info[3], bbox_inches='tight')

    plt.close()
# #######################################################################


# ######################## Useful functions #######################
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


# Get 0-1 knapsack solution using dynamic programming
# Imporved Implementation of https://sites.google.com/site/mikescoderama/Home/0-1-knapsack-problem-in-p
def zero_one_knapsack(v, w, W): 
    # v = list of item values or profit
    # w = list of item weight or cost
    # W = max weight or max cost for the knapsack
    # c is the cost matrix
    c = []
    n = len(v)
    c = np.zeros((n, W+1), int)
    for i in range(0,n):
        #for ever possible weight
        for j in range(0,W+1):		
            #can we add this item to this?
            if (w[i] > j):
                c[i][j] = c[i-1][j]
            else:
                c[i][j] = max(c[i-1][j],v[i] +c[i-1][j-w[i]])

    # Item count
    i = n - 1
    current_weight =  len(c[0]) - 1
    
    # Set every items to not marked
    used_items_marked = np.zeros(n, int)	
    while (i >= 0 and current_weight >=0):
        if (i==0 and c[i][current_weight] >0) or c[i][current_weight] != c[i-1][current_weight]:
            used_items_marked[i] = 1
            current_weight = current_weight-w[i]
        i = i-1
    
    used_items = []
    used_space = 0
    for i in range(n):
        if (used_items_marked[i] != 0):
            # used_items.append(i + 1)
            used_items.append(i)
            used_space += w[i]
    
    return used_items, used_space


# Get Best allocation for given 0-1 knapsack
def get_knapsack_allocation(v, w, capacity, precision_limit, neg_regret = False):
    K = len(v)
    v = [int(x * precision_limit) for x in v]
    w = [int(x * precision_limit) for x in w]
    capacity = int(capacity * precision_limit)

    arms, allocated = zero_one_knapsack(v, w, capacity)
    # if neg_regret:
    #     print arms, allocated
    #     print v, w, capacity
    # Getting weights for alarms
    p = np.zeros(K)
    for i in arms:
        p[i] = (1.0*w[i])/precision_limit

    # print arms, allocated
    return arms, p, (1.0*allocated)/precision_limit


# Get Best allocation for given data theta value
def get_top_allocation(mu_hat, M, est_theta):
    K = len(mu_hat)          # Number of arms
 
    # Allocated for resources using loss rates
    allocated_set = np.argsort(mu_hat)[-int(M):]
      
    # Assign allocation share
    p = np.zeros(K)
    for l in range(K):
        if l in allocated_set:
            p[l] = est_theta

    # print mu_hat, allocated_set, M, p
    return p, allocated_set


# Get censored data
def get_censored_feedback(observation, theta, p):
    K = len(observation)
    censored_feedback = np.ones(K)
    optimal_allocation = True

    for l in range(K):
        if theta[l] > p[l]:
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
        if p[l] >= theta[l]:
            current_gain += mu[l]

    return maximum_gain - current_gain


# ############################# Algorithms ##############################
# UCB based algorithm: CSB-DT-UCB
def csb_dt_ucb(samples, parameters, knapsack_computation_delay):
    # Data Overview
    K                       = len(samples[0])       # Number of arms
    T                       = len(samples)          # Number of rounds
    
    # Algorithms parameters
    mu                      = parameters[0]         # Loss rates
    Q                       = parameters[1]         # Number of resources
    theta                   = parameters[2]         # Threshold after which no loss is observed
    precision               = parameters[3]         # Precision needed for solving 0-1 knapsack
    delta                   = parameters[4]         # Probability of estimate lies in given confidance
    epsilon                 = parameters[5]         # Lower bound on mimimum loss

    # Get Number of samples needed for exploration in ConstReg Algorithm
    W  = int(np.ceil(np.log((1.0*K)/delta)/(Q*np.log(1.0/(1-epsilon)))))

    # Finding Optimal set for given Q, theta and resources
    optimal_arms, _, _ = get_knapsack_allocation(mu, theta, Q, 10**precision)

    # Maximum gain using best allocation
    max_gain = 0
    for l in optimal_arms:
        max_gain += mu[l]

    # Variables
    instantaneous_regret = []

    # ########################## Finding Theta ##########################
    # Initialization
    theta_lb        = np.zeros(K)       # Lower bound of thresholds
    theta_ub        = np.ones(K)        # Upper bound of thresholds
    theta_hat       = np.zeros(K)       # Estimate of thresholds
    p               = np.zeros(K)       # Resources allocated to arms
    no_loss_count   = np.zeros(K)       # Count successive rounds without loss
    good_theta      = np.zeros(K, int)  # Change to 1 when theta estimated is in specfic precision
    free_resources  = 0                 # Left over resources after overestimation
    resources_given = np.zeros(K)       # Resources taken by other arms

    # The precision within threshold needs to estimated
    theta_precision  = 1.0/(10**(precision+1))

    # Initial allocation of resources: Below allocation need to be changed for fraction resources
    for i in range(K):
        if i < Q:
            p[i] = 0.5
        else:
            p[i] = (0.5*Q)/(K-Q)

    # ######################  Estimating thresholds ######################
    t = 0
    while sum(good_theta) < K:
        censored_feedback = get_censored_feedback(samples[t], theta, p)[0]
        for i in range(K):
            # Checking for loss
            if censored_feedback[i] == 1 and good_theta[i] == 0:
                # Underestimated: Increase previous allocation
                no_loss_count[i] = 0
                theta_lb[i] = p[i]
                resources_needed = (theta_ub[i] - theta_lb[i])/2

                if resources_needed > free_resources:
                    p[i] += free_resources
                    resources_needed -= free_resources
                    free_resources = 0

                    # Taking resources from last arms with bad theta 
                    for j in range(K-1, i, -1):
                        if good_theta[j] == 0 and p[j]>0:
                            if p[j] >= resources_needed:
                                p[i] += resources_needed
                                p[j] -= resources_needed
                                resources_given[j] += resources_needed
                                resources_needed = 0
                                no_loss_count[j] = 0
                                break  
                            elif p[j] > 0 and p[j] < resources_needed:
                                p[i] += p[j]
                                resources_needed -= p[j]
                                resources_given[j] += p[j]
                                p[j] = 0
                                no_loss_count[j] = 0

                else:
                    p[i] += resources_needed
                    free_resources -= resources_needed

            elif good_theta[i] == 0 and p[i] > 0:
                no_loss_count[i] += 1

            # Checking overestimation of loss
            if no_loss_count[i] == W and good_theta[i] == 0:                    
                # Overestimated: Decrease previous allocation
                no_loss_count[i] = 0
                theta_ub[i] = p[i]

                # Checking precision
                if theta_ub[i] - theta_lb[i] <= theta_precision:
                    theta_hat[i] = theta_ub[i]
                    free_resources += p[i]
                    p[i] = 0
                    good_theta[i] = 1
                    
                # Release excess resources
                else:
                    resource_released =  (theta_ub[i] - theta_lb[i])/2
                    p[i] -= resource_released
                    free_resources += resource_released

            # Allocate free resources among other arms
            if free_resources > 0:
                available_free_resource = free_resources
                for k in range(K):
                    if good_theta[k] == 1:
                        if theta_hat[k] < available_free_resource:
                            p[k] = theta_hat[k]
                            available_free_resource -= theta_hat[k]
                        else:
                            p[k] = 0

        # Instantaneous regret
        instantaneous_regret.append(get_round_regret(mu, theta, max_gain, range(K), p))
        t += 1             

    # Estimated Threshold within desired precision
    theta_hat = np.around(theta_hat, decimals=precision)

    # ######################## TS based Updates ########################
    mu_est = np.zeros(K)
    num_crimes = np.zeros(K)
    num_observations = np.ones(K)
    allocated_set, p, _ = get_knapsack_allocation(mu_est, theta_hat, Q, 10**precision)
    while t < T:
        for i in range(K):
            mu_est[i] = max((1.0*num_crimes[i])/num_observations[i] - np.sqrt(1.5*np.log(t)/num_observations[i]), 0)
        
        # Getting allocation and resource distribution vector
        if (t+1)%knapsack_computation_delay == 0:
            allocated_set, p, _ = get_knapsack_allocation(mu_est, theta_hat, Q, 10**precision)

        # Update losses for arms where no resource are allocated
        censored_feedback = get_censored_feedback(samples[t], theta, p)
        for i in range(K):
            if i not in allocated_set:
                num_crimes[i] += censored_feedback[0][i]
                num_observations[i] += 1

        # Regret for current round
        regret = get_round_regret(mu, theta, max_gain, range(K), p)
        instantaneous_regret.append(regret)
        t += 1

    return instantaneous_regret


# Thompson Sampling based algorithm: CSB-DT 
def csb_dt(samples, parameters, knapsack_computation_delay):
    # Data Overview
    K                       = len(samples[0])       # Number of arms
    T                       = len(samples)          # Number of rounds
    
    # Algorithms parameters
    mu                      = parameters[0]         # Loss rates
    Q                       = parameters[1]         # Number of resources
    theta                   = parameters[2]         # Threshold after which no loss is observed
    precision               = parameters[3]         # Precision needed for solving 0-1 knapsack
    delta                   = parameters[4]         # Probability of estimate lies in given confidance
    epsilon                 = parameters[5]         # Lower bound on mimimum loss

    # Get Number of samples needed for exploration in ConstReg Algorithm
    W  = int(np.ceil(np.log((1.0*K)/delta)/(Q*np.log(1.0/(1-epsilon)))))

    # Finding Optimal set for given Q, theta and resources
    optimal_arms, _, _ = get_knapsack_allocation(mu, theta, Q, 10**precision)

    # Maximum gain using best allocation
    max_gain = 0
    for l in optimal_arms:
        max_gain += mu[l]

    # Variables
    instantaneous_regret = []

    # ########################## Finding Theta ##########################
    # Initialization
    theta_lb        = np.zeros(K)       # Lower bound of thresholds
    theta_ub        = np.ones(K)        # Upper bound of thresholds
    theta_hat       = np.zeros(K)       # Estimate of thresholds
    p               = np.zeros(K)       # Resources allocated to arms
    no_loss_count   = np.zeros(K)       # Count successive rounds without loss
    good_theta      = np.zeros(K, int)  # Change to 1 when theta estimated is in specfic precision
    free_resources  = 0                 # Left over resources after overestimation
    resources_given = np.zeros(K)       # Resources taken by other arms

    # The precision within threshold needs to estimated
    theta_precision  = 1.0/(10**(precision+1))

    # Initial allocation of resources: Below allocation need to be changed for fraction resources
    for i in range(K):
        if i < Q:
            p[i] = 0.5
        else:
            p[i] = (0.5*Q)/(K-Q)

    # ######################  Estimating thresholds ######################
    t = 0
    while sum(good_theta) < K:
        censored_feedback = get_censored_feedback(samples[t], theta, p)[0]
        for i in range(K):
            # Checking for loss
            if censored_feedback[i] == 1 and good_theta[i] == 0:
                # Underestimated: Increase previous allocation
                no_loss_count[i] = 0
                theta_lb[i] = p[i]
                resources_needed = (theta_ub[i] - theta_lb[i])/2

                if resources_needed > free_resources:
                    p[i] += free_resources
                    resources_needed -= free_resources
                    free_resources = 0

                    # Taking resources from last arms with bad theta 
                    for j in range(K-1, i, -1):
                        if good_theta[j] == 0 and p[j]>0:
                            if p[j] >= resources_needed:
                                p[i] += resources_needed
                                p[j] -= resources_needed
                                resources_given[j] += resources_needed
                                resources_needed = 0
                                no_loss_count[j] = 0
                                break  
                            elif p[j] > 0 and p[j] < resources_needed:
                                p[i] += p[j]
                                resources_needed -= p[j]
                                resources_given[j] += p[j]
                                p[j] = 0
                                no_loss_count[j] = 0

                else:
                    p[i] += resources_needed
                    free_resources -= resources_needed

            elif good_theta[i] == 0 and p[i] > 0:
                no_loss_count[i] += 1

            # Checking overestimation of loss
            if no_loss_count[i] == W and good_theta[i] == 0:                    
                # Overestimated: Decrease previous allocation
                no_loss_count[i] = 0
                theta_ub[i] = p[i]

                # Checking precision
                if theta_ub[i] - theta_lb[i] <= theta_precision:
                    theta_hat[i] = theta_ub[i]
                    free_resources += p[i]
                    p[i] = 0
                    good_theta[i] = 1
                    
                # Release excess resources
                else:
                    resource_released =  (theta_ub[i] - theta_lb[i])/2
                    p[i] -= resource_released
                    free_resources += resource_released

            # Allocate free resources among other arms
            if free_resources > 0:
                available_free_resource = free_resources
                for k in range(K):
                    if good_theta[k] == 1:
                        if theta_hat[k] < available_free_resource:
                            p[k] = theta_hat[k]
                            available_free_resource -= theta_hat[k]
                        else:
                            p[k] = 0

        # Instantaneous regret
        instantaneous_regret.append(get_round_regret(mu, theta, max_gain, range(K), p))
        t += 1             

    # Estimated Threshold within desired precision
    theta_hat = np.around(theta_hat, decimals=precision)

    # ######################## TS based Updates ########################
    mu_est = np.zeros(K)
    S = np.ones(K)
    F = np.ones(K)
    allocated_set, p, _ = get_knapsack_allocation(mu_est, theta_hat, Q, 10**precision)
    while t < T:
        for i in range(K):
            mu_est[i] = np.random.beta(S[i], F[i])
        
        # Getting allocation and resource distribution vector
        if (t+1)%knapsack_computation_delay == 0:
            allocated_set, p, _ = get_knapsack_allocation(mu_est, theta_hat, Q, 10**precision)

        # Update losses for arms where no resource are allocated
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
# vectors of arms' loss rate and threshold (theta) 
mu_vector       = [0.9, 0.89, 0.87, 0.58, 0.3]
theta_vector    = [0.7, 0.7, 0.7, 0.6, 0.35]

arms        = len(mu_vector)    # Number of arms
resources   = 2                 # Number of resouces
horizon     = 5000              # Number of rounds
runs        = 100               # Number of simulations
delay_value = 20                # Delay in computation the knapsack problem

# ########## Algorithm parameters for CSB-DT ##########
precision_value     = 3
delta_value         = 0.1
epsilon_value       = 0.1
cases               = ['CSB-DT', 'CSB-DT-UCB']
num_cases           = len(cases)

# Data parameters
generate_data       = True      # True: Generate new data, False: Used saved generated data
save_data           = True      # True: Save new generated data, False: Do not save data
random_seed_value   = 100  

# ########## Plotting parameters ##########
xlabel              = "Rounds"
ylabel              = "Cumulative Regret"
# file_to_save        = "compareAlgos_" + "_" + str(horizon) + "_" + str(runs) + ".png"
file_to_save        = "compareAlgos.png"
title               = "Comparison of Algorithms"
save_to_path        = "plots/" 
location_to_save    = save_to_path + file_to_save
plotting_parameters = [xlabel, ylabel, title, location_to_save, cases, horizon]

# ########## Generating and saving losses for Arms ##########
file_name = "data/diff_" + str(horizon) + "_" + str(arms)
loss_file = file_name + ".pkl"
data_samples = getSamples(loss_file, mu_vector, horizon, generate_data, save_data, random_seed_value)

# Algorithm paramters: [mu, Q, theta, precision, delta, epsilon]
algos_parameters = [mu_vector, resources, theta_vector, precision_value, delta_value, epsilon_value]

# Algorithm
algos_regret = []
for r in range(runs):
    run_regret = []
    np.random.shuffle(data_samples)
    for a in range(num_cases):        
        if cases[a] == 'CSB-DT':
            iter_regret = csb_dt(data_samples, algos_parameters, delay_value)
        
        if cases[a] == 'CSB-DT-UCB':
            iter_regret = csb_dt_ucb(data_samples, algos_parameters, delay_value)

        run_regret.append(iter_regret)

    algos_regret.append(run_regret)

# ########## Plotting final results ##########
regret_plotting(algos_regret, num_cases, plotting_parameters)
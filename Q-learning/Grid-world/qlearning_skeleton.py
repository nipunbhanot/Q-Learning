import numpy as np
from grid_world import *
import matplotlib.pyplot as plt
import random


# TODO: Fill this function in
# Function that takes an a 2d numpy array Q (num_states by num_actions)
# an epsilon in the range [0, 1] and the state
# to output actions according to an Epsilon-Greedy policy
# (random actions are chosen with epsilon probability)
def tabular_epsilon_greedy_policy(Q, eps, state):
    
    action_values = [0,1,2,3]
    if np.random.rand() < eps:
        action = random.choice(action_values) #pick random action
    
    else: #pick action according to q table
        action_q = Q[state]
        max_index_list = []
        max_value = action_q[0]
        for index, value in enumerate(action_q):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        action = random.choice(max_index_list)
         
    return action
    


class QLearning(object):
    # Initialize a Qlearning object
    # alpha is the "learning_rate"
    def __init__(self, num_states, num_actions, alpha=0.5, epsilon = 0.1):
         # initialize Q values to something
        self.Q = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.discount_factor = 0.99
        self.epsilon = epsilon

    # TODO: fill in this function
    # updates the Q value table
    # with a (state, action, reward, next_state) from one step in the environment
    # done is a bool indicating if the episode terminated at this step
    # you can return anything from this function to help with plotting/debugging
    def update(self, state, action, reward, next_state, done):
        
        q_current = self.Q[state][action]
        if not done:    
            q_new = reward + self.discount_factor * max(self.Q[next_state])
        else:
            q_new = reward
        self.Q[state][action] += self.alpha * (q_new - q_current)
        
        return (self.Q[state][action])
                

# TODO: fill this in
# run the greedy policy (no randomness) on the environment for niter number of times
# and return the fraction of times that it reached the goal
def evaluate_greedy_policy(qlearning, env, niter=100):
   
    count = 0
    done = False
    for i in range(niter):
        state = env.reset()
        while not done:
            action = tabular_epsilon_greedy_policy(qlearning.Q, 0, state)
            state, reward, done = env.step(action)
        if reward == 100:
            count += 1
    return count/niter*100


#def __init__:
if __name__ == "__main__":

    env = GridWorld(MAP4)
    qlearning = QLearning(env.get_num_states(), env.get_num_actions())


    ## TODO: write training code here
    num_episodes = 1000
    eps = 0.1
    final_reward = np.zeros(num_episodes)
    qvalue_start = np.zeros(num_episodes)
    qvalue_matrix = np.zeros(4)
    qvalue_final_matrix = np.zeros((6, 13))

    for i in range(num_episodes):
        
        rewards = []
        states = []
        actions = []
        total_reward = 0
        
        state = env.reset()

        done = False
    
        
        # Qvalue for start state
        action_start = tabular_epsilon_greedy_policy(qlearning.Q, 0, 2)
        next_state_start, reward_start, done = env.step(action_start)
        qvalue_start[i] = qlearning.update(2, action_start, reward_start, next_state_start, done)


        while not done:
            
            
            states.append(state)

            action = tabular_epsilon_greedy_policy(qlearning.Q, eps, state)
            actions.append(action)
            
            next_state, reward, done = env.step(action)
            rewards.append(reward)
            
            qlearning.update(state, action, reward, next_state, done)
            
                
            state = next_state
        
            if done:    
                for j in range(len(rewards)):
                    total_reward += rewards[j]
        final_reward[i] = total_reward
    

        
    # Part 3: Matrices Creation
    MAP4_matrix = ["s0100001g0000",
                   "0010000111100",
                   "0010010100000",
                   "0010010101000",
                   "0000010101001",
                   "0000010001000"]
    for k in range(4):
        action_matrix = tabular_epsilon_greedy_policy(qlearning.Q, 0, k)
        next_state_matrix, reward_matrix, done = env.step(action_matrix)
        qvalue_matrix[k] = qlearning.update(k, action_matrix, reward_matrix, next_state_matrix, done)
    for i in range(6):
        for j in range(13):
            if MAP4_matrix[i][j] == 's':
                qvalue_final_matrix[i][j] = qvalue_matrix[2]
            elif MAP4_matrix[i][j] == 'g':
                qvalue_final_matrix[i][j] = qvalue_matrix[3]
            elif MAP4_matrix[i][j] == '0':
                qvalue_final_matrix[i][j] = qvalue_matrix[0]
            else:
                qvalue_final_matrix[i][j] = qvalue_matrix[1]
                
    plt.figure(1)          
    plt.scatter(np.arange(0, num_episodes, 1), final_reward)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward for Each Episode")
    plt.show()
    
    plt.figure(2)
    plt.plot(np.arange(0, num_episodes, 1), qvalue_start)
    plt.xlabel("Episode")
    plt.ylabel("Q-Value at start")
    plt.title("Q-Value at start for each episode")
    plt.show()
    
    plt.figure(3)
    plt.imshow(qvalue_final_matrix)
    plt.show()
    plt.title("Q-value matrix")
    
    #evaluate the greedy policy to see how well it performs
    frac = evaluate_greedy_policy(qlearning, env)
    print("Finding goal " + str(frac) + "% of the time.")
            
        




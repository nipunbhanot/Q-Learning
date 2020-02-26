from tictactoe_env import TicTacToe
import pdb 
import numpy as np 
import matplotlib.pyplot as plt
import itertools
import random




if __name__ == "__main__":
    
    
    env = TicTacToe()
    
    num_episodes = 1000
    learning_rate = 0.6
    epsilon = 0.2
    
    number_of_actions = 9
    
    player = ['O', 'X', ' ']
    
    total_states_possible = [[list(i[0:3]),list(i[3:6]),list(i[6:10])] for i in itertools.product(player, repeat = 9)]
    number_of_states = len(total_states_possible)
    Q_values_agent = np.zeros((number_of_states, number_of_actions)) ##initialization of Q-values
#    Q_values_agent.fill(0.6)

    states_dictionary = {}
    
            
    for i in range(number_of_states):
        states_dictionary[i] = total_states_possible[i]


    # Update the Q values
    def update(state_index, new_state_index, lr, rewardd, done, actionn):
        
        discount_factor = 0.99
        q_current = Q_values_agent[state_index][actionn-1]
        if not done:
            q_new = rewardd + discount_factor * np.max(Q_values_agent[new_state_index])
        else:
            q_new = rewardd
        Q_values_agent[state_index][actionn-1] = q_current + lr * (q_new - q_current)
        
    
    
    ## Fetching action using epsilon greedy algorithm
    def fetch_action(player, state, epsilon):
        
        Q_values_current = []
        moves = []
        empty_cells = []
        
        for i in range(3):
            for j in range(3):
                if state[i][j] == ' ': empty_cells.append(i*3 + (j + 1))
        
        for empty_cell in empty_cells:
            moves.append(empty_cell)
            next_state = env.copy_game_state(state)
            env.play_move_hallucinate(next_state, player, empty_cell)
            next_state_index = list(states_dictionary.keys())[list(states_dictionary.values()).index(next_state)]
            
            Q_values_current.append(Q_values_agent[next_state_index])
            
        best_action_index = np.argmax(Q_values_current)
        
        if np.random.rand() < epsilon:
            best_action = random.choice(empty_cells)
            epsilon = epsilon*0.99 ##decrease epsilon
        else:
            best_action = moves[best_action_index]
        
        return best_action
        
    
    def convertstate(astate):
        gstate = [[' ',' ',' '],[' ',' ',' '],[' ',' ',' ']]
        for i in range(3):
            for j in range(3):
                if astate[i][j] == 0:
                    gstate[i][j] = ' '
                elif astate[i][j] == 1:
                    gstate[i][j] = 'X'
                else:
                    gstate[i][j] = 'O'
        return gstate
    
        
    epsilon = 0.2
    ## Actual training
    final_reward = np.zeros(num_episodes)
    for i in range(num_episodes):
        
        rewards = []
        total_reward = 0
        agent_state = env.reset()
        gamestate = convertstate(agent_state)
        done = False
        
        while not done:
            current_state_index = list(states_dictionary.keys())[list(states_dictionary.values()).index(gamestate)]
            
            action = fetch_action("X", gamestate, epsilon)
            agent_state_new, reward, done = env.step(action)
            gamestate_new = convertstate(agent_state_new)
            next_state_index = list(states_dictionary.keys())[list(states_dictionary.values()).index(gamestate_new)]
                
            rewards.append(reward)
            
            update(current_state_index, next_state_index, learning_rate, reward, done, action)
            gamestate = gamestate_new
            agent_state = agent_state_new
            current_state_index = next_state_index

            if done:    
                for j in range(len(rewards)):
                    total_reward += rewards[j]
                
        final_reward[i] = total_reward
        print(final_reward[i])
    plt.scatter(np.arange(0, num_episodes, 1), final_reward)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward for Each Episode")
    plt.show()
    

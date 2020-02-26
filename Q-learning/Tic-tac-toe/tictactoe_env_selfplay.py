import numpy as np 
from math import inf as infinity
import itertools
import random
import pdb
import matplotlib.pyplot as plt


class TicTacToe(object):
	def __init__(self):
		self.game_state = [[' ',' ',' '],
					[' ',' ',' '],
					[' ',' ',' ']]

		self.agent_state = np.zeros((3,3)) #actual agent observation

	def print_board(self):
		print('----------------')
		print('| ' + str(self.game_state[0][0]) + ' || ' + str(self.game_state[0][1]) + ' || ' + str(self.game_state[0][2]) + ' |')
		print('----------------')
		print('| ' + str(self.game_state[1][0]) + ' || ' + str(self.game_state[1][1]) + ' || ' + str(self.game_state[1][2]) + ' |')
		print('----------------')
		print('| ' + str(self.game_state[2][0]) + ' || ' + str(self.game_state[2][1]) + ' || ' + str(self.game_state[2][2]) + ' |')
		print('----------------')
            
	def convert_state(self):
		#agent plays X. Env plays O. If cell is empty, denoted by zero
		#if it has X it is denoted by 1. if it has O it is denoted by 2. 

		for i in range(3):
			for j in range(3):
				if self.game_state[i][j] == ' ':
					self.agent_state[i][j] = 0
				elif self.game_state[i][j] == 'X':
					self.agent_state[i][j] = 1 
				else:
					self.agent_state[i][j] = 2
		
		return self.agent_state


	def reset(self):
		self.game_state = [[' ',' ',' '],[' ',' ',' '],[' ',' ',' ']]
		current_state = "Not Done"
		
		self.print_board()
		

		winner = None
		current_player_idx = 0

		return self.convert_state()


	def step(self,action):
		#action must be valid and must be numbered  1 through 9.
		#convert accordingly. Additionally write script in the 
		#training/inference loop to make sure action is valid, i.e 
		#do not output an action in a cell that is already occupied.
		#This will raise an exception and break your training loop.
		self.play_move('X', action)
		
		#Opponent plays action
		block_choice = self.getOpponentMove(self.game_state,"O")
		
		if block_choice != -10 and block_choice != 10 and block_choice != 0:
			self.play_move('O', block_choice)

		
		self.print_board()
		rew, done = self.rew_calc()
        
		winner, current_state = self.check_current_state(self.game_state)
		if current_state == "Draw":
			print("Draw")
		elif winner == 'X':
			print("Win")
		elif winner == 'O':
			print("Lost")
        
		return self.convert_state(), rew, done

	def rew_calc(self):
		reward = 0
		done = False 
		current_state = "Not Done"

		winner, current_state = self.check_current_state(self.game_state)

		#While game is being played return done = False
		#Design the reward to be returned
		if current_state == "Not Done":
			reward = 0
			return reward, done 

		if current_state == "Draw":
			reward = 0.5
			done = True 
			return reward, done 
		elif winner == 'X':
			reward = 1.0
			done = True 
		elif winner == 'O':
			reward = -1.0
			done = True 

		return reward, done 



	def play_move(self, player, block_num):
		if self.game_state[int((block_num-1)/3)][(block_num-1)%3] == ' ':
			self.game_state[int((block_num-1)/3)][(block_num-1)%3] = player
		else:
			raise Exception('Invalid Action!')

		
	def play_move_hallucinate(self,state, player, block_num):
		if state[int((block_num-1)/3)][(block_num-1)%3] == ' ':
			state[int((block_num-1)/3)][(block_num-1)%3] = player
		else:
			raise Exception('Invalid Action!')



	def getOpponentMove(self, state,player):
		winner_loser , done = self.check_current_state(state)
		if done == "Done" and winner_loser == 'O': # If Opponent won
			return 10
		elif done == "Done" and winner_loser == 'X': # If Human won
			return -10
		elif done == "Draw":    # Draw condition
			return 0
			
		moves = []
		empty_cells = []
		for i in range(3):
			for j in range(3):
				if state[i][j] == ' ':
					empty_cells.append(i*3 + (j+1))
		
		for empty_cell in empty_cells:
			move = {}
			move['index'] = empty_cell
			new_state = self.copy_game_state(state) #hallucinate through states
			self.play_move_hallucinate(new_state, player, empty_cell)
			if player == 'O':    # If Opponent
				result = self.getOpponentMove(new_state, 'X')    # make more depth tree for human
				move['score'] = result
			else:
				result = self.getOpponentMove(new_state, 'O')    # make more depth tree for Opponent
				move['score'] = result
			
			moves.append(move)
            
		# Find best move
		best_move = None
		if player == 'O':   # If Opponent player
			best = -infinity
			for move in moves:
				if move['score'] > best:
					best = move['score']
					best_move = move['index']
		else:
			best = infinity
			for move in moves:
				if move['score'] < best:
					best = move['score']
					best_move = move['index']
					
		return best_move
	
	def copy_game_state(self,state):
		new_state = [[' ',' ',' '],[' ',' ',' '],[' ',' ',' ']]
		for i in range(3):
			for j in range(3):
				new_state[i][j] = state[i][j]
		return new_state

	def check_current_state(self,game_state):
		
		# Check horizontals
		if (game_state[0][0] == game_state[0][1] and game_state[0][1] == game_state[0][2] and game_state[0][0] != ' '):
			return game_state[0][0], "Done"
		if (game_state[1][0] == game_state[1][1] and game_state[1][1] == game_state[1][2] and game_state[1][0] != ' '):
			return game_state[1][0], "Done"
		if (game_state[2][0] == game_state[2][1] and game_state[2][1] == game_state[2][2] and game_state[2][0] != ' '):
			return game_state[2][0], "Done"
		
		# Check verticals
		if (game_state[0][0] == game_state[1][0] and game_state[1][0] == game_state[2][0] and game_state[0][0] != ' '):
			return game_state[0][0], "Done"
		if (game_state[0][1] == game_state[1][1] and game_state[1][1] == game_state[2][1] and game_state[0][1] != ' '):
			return game_state[0][1], "Done"
		if (game_state[0][2] == game_state[1][2] and game_state[1][2] == game_state[2][2] and game_state[0][2] != ' '):
			return game_state[0][2], "Done"
		
		# Check diagonals
		if (game_state[0][0] == game_state[1][1] and game_state[1][1] == game_state[2][2] and game_state[0][0] != ' '):
			return game_state[1][1], "Done"
		if (game_state[2][0] == game_state[1][1] and game_state[1][1] == game_state[0][2] and game_state[2][0] != ' '):
			return game_state[1][1], "Done"
		
		# Check if draw
		draw_flag = 0
		for i in range(3):
			for j in range(3):
				if game_state[i][j] == ' ':
					draw_flag = 1
		if draw_flag == 0:
			return None, "Draw"


		return None, "Not Done"
    
    







if __name__ == "__main__":
    
    
    env = TicTacToe()
    
    num_episodes = 1000
    learning_rate = 0.2
    epsilon = 0.2
    
    number_of_actions = 9
    
    player_matrix = ['O', 'X', ' ']
    player = ['X', 'O']
    
    total_states_possible = [[list(i[0:3]),list(i[3:6]),list(i[6:10])] for i in itertools.product(player_matrix, repeat = 9)]
    number_of_states = len(total_states_possible)
    Q_values_agent = np.zeros((number_of_states, number_of_actions)) ##initialization of Q-values
    Q_values_agent2 = np.zeros((number_of_states, number_of_actions))

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
        
        
        
    def update2(state_index, new_state_index, lr, rewardd, done, actionn):
    
        discount_factor = 0.99
        q_current = Q_values_agent2[state_index][actionn-1]
        if not done:
            q_new = rewardd + discount_factor * np.max(Q_values_agent2[new_state_index])
        else:
            q_new = rewardd
        Q_values_agent2[state_index][actionn-1] = q_current + lr * (q_new - q_current)
    
    
    
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
            
            if player == 'X':
                Q_values_current.append(Q_values_agent[next_state_index])
            elif player == 'O':
                Q_values_current.append(Q_values_agent2[next_state_index])
            
        best_action_index = np.argmax(Q_values_current)
        
        if np.random.rand() < epsilon:
            best_action = random.choice(empty_cells)
            epsilon = epsilon/1.01 ##decrease epsilon
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
            
            if 0 in agent_state:
                #if current_player_index == 0:
                action = fetch_action('X', gamestate, epsilon)
                env.play_move("X", action)
                agent_state_new_a = env.convert_state()
                gamestate_new_a = convertstate(agent_state_new_a)
                gamestate_new_b = gamestate_new_a
                next_state_index_a = list(states_dictionary.keys())[list(states_dictionary.values()).index(gamestate_new_a)]
                winner_loser , done = env.check_current_state(gamestate_new_a)

          #  else:
            if 0 in agent_state_new_a:
                action2 = fetch_action('O', gamestate_new_a, epsilon)
                
                winner_loser , done = env.check_current_state(gamestate_new_a)
                temp = 1
                if done == "Done" and winner_loser == 'O': # If Opponent won
                    temp = 10
                elif done == "Done" and winner_loser == 'X': # If Human won
                    temp = -10
                elif done == "Draw":    # Draw condition
                    temp = 0
            
                if temp != -10 and temp != 10 and temp != 0:
                    env.play_move('O', action2)
                    agent_state_new = env.convert_state()
                    gamestate_new = convertstate(agent_state_new)
                    next_state_index = list(states_dictionary.keys())[list(states_dictionary.values()).index(gamestate_new)]
                    gamestate_new_b = gamestate_new
                    
                env.print_board()
                reward, done = env.rew_calc()
        
                winner, current_state = env.check_current_state(gamestate_new_b)
                if current_state == "Draw":
                    print("draw")
                elif winner == 'X':
                    print("Win")
                elif winner == 'O':
                    print("Lost")
            

            rewards.append(reward)
            
            update(current_state_index, next_state_index_a, learning_rate, reward, done, action)
            update2(next_state_index_a, next_state_index, learning_rate, -reward, done, action2)
            gamestate = gamestate_new
            agent_state = agent_state_new
            current_state_index = next_state_index
            
#            if winner == None and current_state != "Draw":
#                if  current_player_index == 0:
#                    current_player_index = 1
#                elif current_player_index == 1:
#                    current_player_index = 0

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
            
    

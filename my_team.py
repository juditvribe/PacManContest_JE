# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from game import Actions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        
        opponents = [game_state.get_agent_state(i).get_position() for i in self.get_opponents(game_state)]
        best_action = self.minimax(game_state, 0, 2, self.index, opponents)
        return best_action

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    def get_new_position(self, current_pos, action):
        if current_pos is None:
            raise ValueError("Current position is None. Check agent initialization.")
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = [int(current_pos[0] + dx), int(current_pos[1] + dy)]
        return (next_x, next_y)
    
    
    def get_ghost_actions(self, current_pos, game_state):
        walls = game_state.get_walls()

        actions = []
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            next_x, next_y = self.get_new_position(current_pos, direction)
            if not walls[next_x][next_y]:
                if (0 <= next_x < walls.width) and (0 <= next_y < walls.height):
                    actions.append(direction)
        return actions
            
    
    # def expectation(self, game_state, position, legal_actions):
    #     ghost_dict = {}
    #     for action in legal_actions:
    #         successor = self.get_successor(game_state, action)
    #         new_pos = successor.get_agent_position(self.index)
    #         ghost_dict[action] = self.get_maze_distance(position, new_pos) * self.get_ghost_weights()['distance']

    #     min_action = min(ghost_dict)

    #     for action in ghost_dict:
    #         if ghost_dict[action] == min_action:
    #             ghost_dict[action] = 0.8
    #         else:
    #             ghost_dict[action] = 0.2/(len(legal_actions)-1)
    #     return ghost_dict
    
    
    # def ghost_eval(self, game_state, opponents, opponent):
    #     new_pos = opponents[opponent]
    #     enemy = game_state.get_agent_state(opponent)
    #     my_pos = game_state.get_agent_state(self.index).get_position()

    #     if enemy.scared_timer > 0:
    #         distance = -self.get_maze_distance(my_pos, new_pos) * self.get_ghost_weights()['distance']
    #     else:
    #         distance = self.get_maze_distance(my_pos, new_pos) * self.get_ghost_weights()['distance']
    #     return distance
           
           
    def minimax(self, game_state, depth, max_depth, agent, opponents, alpha=-float('inf'), beta=float('inf')):
        def maxAgent(game_state, depth, max_depth, alpha, beta, opponents):
            # We set the terminal condition to game end or depth limit
            if depth == max_depth:
                return [self.evaluate(game_state, None)]
            
            current_alpha = alpha
            max_ = float("-inf")
            best_action = None 
            legal_actions = [action for action in game_state.get_legal_actions(self.index) if action != Directions.STOP] # Get legal moves per agent

            # We loop through legal actions for Pacman
            for action in legal_actions:
                next_state = game_state.generate_successor(self.index, action)
                evaluation_max = minAgent(next_state, 0, depth, max_depth, alpha, beta, opponents)[0] #[0] used to access the score, minimax function returns a tuple (score, action)
                # We update maximum score and best action if the evaluated score is higher than the current maximum
                if evaluation_max > max_:
                    max_ = evaluation_max
                    best_action = action
                if max_ > beta:
                    return max_, best_action
                current_alpha = max(current_alpha, max_)
            return max_, best_action

        # Function for minimizing agent (Ghost)
        def minAgent(game_state, agent_idx, depth, max_depth, alpha, beta, opponents):
            
            min_ = float("+inf")
            best_action = None 
            
            if opponents[agent_idx] is not None:
                current_beta = beta
                legal_actions = [action for action in self.get_ghost_actions(opponents[agent_idx], game_state) if action != Directions.STOP] # Get legal moves per agent            # We loop through legal actions for the current ghost
                
                for action in legal_actions:
                    # We get the successor state after taking an action
                    next_state = game_state.generate_successor(opponents[agent_idx], action)
                    if agent_idx == len(opponents) - 1: # If we are in the last ghost
                        # Then, we move to the maximizing agent
                        evaluation_min = maxAgent(next_state, depth+1, max_depth, alpha, beta, opponents)[0] #[0] used to access the score, minimax function returns a tuple (score, action)
                    else: # Otherwise, continue with the next ghost
                        evaluation_min = minAgent(next_state, agent_idx+1, depth, max_depth, alpha, beta, opponents)[0]
                    # We update minimum score and best action if the evaluated score is lesser than the current minimum
                    if evaluation_min < min_:
                        min_ = evaluation_min
                        best_action = action
                    if min_ < alpha:
                        return min_, best_action
                    current_beta = min(current_beta, min_)    
            
            if min_ == float("+inf"):
                min_ = 0
            return min_, best_action

        score, action = maxAgent(game_state, depth, max_depth, alpha, beta, opponents) 
        return action

     
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  
        
        my_pos = successor.get_agent_state(self.index).get_position()
        
        #Enemies, ghosts and invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() != None]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() != None]
        features['invader_distance'] = 0.0
        
        #different distances to each type of enemy agents
        if len(invaders) > 0:
            features['invader_distance'] = min([self.get_maze_distance(my_pos, invader.get_position()) for invader in invaders]) +1
        
        if len(ghosts) > 0:
            ghost_eval = 0.0
            scared_distance = 0.0
            reg_ghosts = [ghost for ghost in ghosts if ghost.scared_timer == 0]
            scared_ghost = [ghost for ghost in ghosts if ghost.scared_timer > 0]
            if len(reg_ghosts) > 0:
                ghost_eval = min([self.get_maze_distance(my_pos, ghost.get_position()) for ghost in reg_ghosts])
                if ghost_eval <= 1:
                    ghost_eval = -float("inf")
            
            if len(scared_ghost) > 0:
                scared_distance = min([self.get_maze_distance(my_pos, ghost.get_position()) for ghost in scared_ghost])
            if scared_distance < ghost_eval or ghost_eval == 0:
                if scared_distance == 0:
                    features['ghost_scared'] = -10
            features['distance_to_ghost'] = ghost_eval

        # Compute distance to the nearest food
        
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
            features['food_remaining'] = len(food_list)
        
        
        #Avoid stopping or bugging
        if action == Directions.STOP: 
            features['stop'] = 1
        if game_state.get_agent_state(self.index).configuration.direction is not None:
            rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
            if action == rev:
                features['reverse'] = 1
  
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'invader_distance': -50, 'distance_to_food': -1, 'food_remaining': -1, 'distance_to_ghost': 2, 'ghost_scared': -1, 'stop': -100, 'reverse': -20 }

    
    def get_ghost_weights(self):
        return {'distance': 1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    def get_new_position(self, current_pos, action):
        if current_pos is None:
            raise ValueError("Current position is None. Check agent initialization.")
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(current_pos[0] + dx), int(current_pos[1] + dy)
        return (next_x, next_y)
    
    def get_ghost_actions(self, current_pos, game_state):
        walls = game_state.get_walls()

        actions = []
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            next_x, next_y = self.get_new_position(current_pos, direction)
            if not walls[next_x][next_y]:
                if (0 <= next_x < walls.width) and (0 <= next_y < walls.height):
                    actions.append(direction)
        return actions
    
    def expectation(self, game_state, position, legal_actions):
        ghost_dict = {}
        for action in legal_actions:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_position(self.index)
            ghost_dict[action] = self.get_maze_distance(position, new_pos) * self.get_ghost_weights()['distance']

        min_action = min(ghost_dict)

        for action in ghost_dict:
            if ghost_dict[action] == min_action:
                ghost_dict[action] = 0.8
            else:
                ghost_dict[action] = 0.2/(len(legal_actions)-1)
        return ghost_dict
    
    def ghost_eval(self, game_state, opponents, opponent):
        new_pos = opponents[opponent]
        enemy = game_state.get_agent_state(opponent)
        my_pos = game_state.get_agent_state(self.index).get_position()

        if enemy.scared_timer > 0:
            distance = -self.get_maze_distance(my_pos, new_pos) * self.get_ghost_weights()['distance']
        else:
            distance = self.get_maze_distance(my_pos, new_pos) * self.get_ghost_weights()['distance']
        return distance
    
    def minimax(self, game_state, depth, max_depth, agent, opponents, alpha=-float('inf'), beta=float('inf')):
        def maxAgent(game_state, depth, max_depth, alpha, beta, opponents):
            # We set the terminal condition to game end or depth limit
            if depth == max_depth:
                return [self.evaluate(game_state, None)]
            
            current_alpha = alpha
            max_ = float("-inf")
            best_action = None 
            legal_actions = [action for action in game_state.get_legal_actions(self.index) if action != Directions.STOP] # Get legal moves per agent

            # We loop through legal actions for Pacman
            for action in legal_actions:
                next_state = game_state.generate_successor(self.index, action)
                evaluation_max = minAgent(next_state, 0, depth, max_depth, alpha, beta, opponents)[0] #[0] used to access the score, minimax function returns a tuple (score, action)
                # We update maximum score and best action if the evaluated score is higher than the current maximum
                if evaluation_max > max_:
                    max_ = evaluation_max
                    best_action = action
                if max_ > beta:
                    return max_, best_action
                current_alpha = max(current_alpha, max_)
            return max_, best_action

        def minAgent(game_state, agent_idx, depth, max_depth, alpha, beta, opponents):
            
            min_ = float("+inf")
            best_action = None 
            
            if opponents[agent_idx] is not None:
                current_beta = beta
                legal_actions = [action for action in self.get_ghost_actions(opponents[agent_idx], game_state) if action != Directions.STOP] # Get legal moves per agent            # We loop through legal actions for the current ghost
                
                for action in legal_actions:
                    # We get the successor state after taking an action
                    next_state = game_state.generate_successor(opponents[agent_idx], action)
                    if agent_idx == len(opponents) - 1: # If we are in the last ghost
                        # Then, we move to the maximizing agent
                        evaluation_min = maxAgent(next_state, depth+1, max_depth, alpha, beta, opponents)[0] #[0] used to access the score, minimax function returns a tuple (score, action)
                    else: # Otherwise, continue with the next ghost
                        evaluation_min = minAgent(next_state, agent_idx+1, depth, max_depth, alpha, beta, opponents)[0]
                    # We update minimum score and best action if the evaluated score is lesser than the current minimum
                    if evaluation_min < min_:
                        min_ = evaluation_min
                        best_action = action
                    if min_ < alpha:
                        return min_, best_action
                    current_beta = min(current_beta, min_)    
            
            if min_ == float("+inf"):
                min_ = 0
            return min_, best_action

        score, action = maxAgent(game_state, depth, max_depth, alpha, beta, opponents) 
        return action

        
    
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  
        
        my_pos = successor.get_agent_state(self.index).get_position()
        
        #Enemies, ghosts and invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() != None]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() != None]
        features['invader_distance'] = 0.0
        
        #different distances to each type of enemy agents
        if len(invaders) > 0:
            features['invader_distance'] = min([self.get_maze_distance(my_pos, invader.get_position()) for invader in invaders]) +1
        
        if len(ghosts) > 0:
            ghost_eval = 0.0
            scared_distance = 0.0
            reg_ghosts = [ghost for ghost in ghosts if ghost.scared_timer == 0]
            scared_ghost = [ghost for ghost in ghosts if ghost.scared_timer > 0]
            if len(reg_ghosts) > 0:
                ghost_eval = min([self.get_maze_distance(my_pos, ghost.get_position()) for ghost in reg_ghosts])
                if ghost_eval <= 1:
                    ghost_eval = -float("inf")
            
            if len(scared_ghost) > 0:
                scared_distance = min([self.get_maze_distance(my_pos, ghost.get_position()) for ghost in scared_ghost])
            if scared_distance < ghost_eval or ghost_eval == 0:
                if scared_distance == 0:
                    features['ghost_scared'] = -10
            features['distance_to_ghost'] = ghost_eval

        # Compute distance to the nearest food
        
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
            features['food_remaining'] = len(food_list)
        
        
        #Avoid stopping or bugging
        if action == Directions.STOP: 
            features['stop'] = 1
        if game_state.get_agent_state(self.index).configuration.direction is not None:
            rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
            if action == rev:
                features['reverse'] = 1
  
        return features


    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'invader_distance': -50, 'distance_to_food': -1, 'food_remaining': -1, 'distance_to_ghost': 2, 'ghost_scared': -1, 'stop': -100, 'reverse': -20 }

    
    def get_ghost_weights(self):
        return {'distance': 1}
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
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)
    

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
   
            
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  
        
        my_state = successor.get_agent_state(self.index)
        my_position = my_state.get_position()
        
        # Types of enemies: ghosts and invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() != None]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() != None]
        
        
        # Distances to invaders (pacman agents)
        features['distance_to_invader'] = 0.0
        if len(invaders) > 0:
            features['distance_to_invader'] = min([self.get_maze_distance(my_position, invader.get_position()) for invader in invaders]) + 1
        
        # Distances to ghosts
        if len(ghosts) > 0:
            ghost_value = 0.0
            scared_distance = 0.0
            regular_ghosts = [ghost for ghost in ghosts if ghost.scared_timer == 0]
            scared_ghosts = [ghost for ghost in ghosts if ghost.scared_timer > 0]
            if len(regular_ghosts) > 0:
                ghost_value = min([self.get_maze_distance(my_position, ghost.get_position()) for ghost in regular_ghosts])
                if ghost_value <= 1:
                    ghost_value = -float("inf")
            
            if len(scared_ghosts) > 0:
                scared_distance = min([self.get_maze_distance(my_position, ghost.get_position()) for ghost in scared_ghosts])
            
            if scared_distance < ghost_value or ghost_value == 0:
                if scared_distance == 0:
                    features['scared_ghost'] = -10
            features['distance_to_ghost'] = ghost_value


        # Distance to nearest food
        if len(food_list) > 0:
            my_position = successor.get_agent_state(self.index).get_position()  
            min_distance = min([self.get_maze_distance(my_position, food) for food in food_list])
            features['distance_to_food'] = min_distance
            features['food_remaining'] = len(food_list)
        
        
        # Avoid stopping or bugging
        if action == Directions.STOP: 
            features['stop'] = 1
        if game_state.get_agent_state(self.index).configuration.direction is not None:
            rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
            if action == rev:
                features['reverse'] = 1
              
        return features



    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_invader': -50, 'distance_to_ghost': 2, 'scared_ghost': -1, 
                'distance_to_food': -1, 'food_remaining': -1,  'stop': -100, 'reverse': -20 }



class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  
        
        my_state = successor.get_agent_state(self.index)
        my_position = my_state.get_position()
        
        # Types of enemies: ghosts and invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() != None]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() != None]
        
        
        # Distances to invaders (pacman agents)
        features['distance_to_invader'] = 0.0
        if len(invaders) > 0:
            features['distance_to_invader'] = min([self.get_maze_distance(my_position, invader.get_position()) for invader in invaders]) + 1
        
        # Distances to ghosts
        if len(ghosts) > 0:
            ghost_value = 0.0
            scared_distance = 0.0
            regular_ghosts = [ghost for ghost in ghosts if ghost.scared_timer == 0]
            scared_ghosts = [ghost for ghost in ghosts if ghost.scared_timer > 0]
            if len(regular_ghosts) > 0:
                ghost_value = min([self.get_maze_distance(my_position, ghost.get_position()) for ghost in regular_ghosts])
                if ghost_value <= 1:
                    ghost_value = -float("inf")
            
            if len(scared_ghosts) > 0:
                scared_distance = min([self.get_maze_distance(my_position, ghost.get_position()) for ghost in scared_ghosts])
            
            if scared_distance < ghost_value or ghost_value == 0:
                if scared_distance == 0:
                    features['scared_ghost'] = -10
            features['distance_to_ghost'] = ghost_value


        # Distance to nearest food
        if len(food_list) > 0:
            my_position = successor.get_agent_state(self.index).get_position()  
            min_distance = min([self.get_maze_distance(my_position, food) for food in food_list])
            features['distance_to_food'] = min_distance
            features['food_remaining'] = len(food_list)
        
        
        # Avoid stopping or bugging
        if action == Directions.STOP: 
            features['stop'] = 1
        if game_state.get_agent_state(self.index).configuration.direction is not None:
            rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
            if action == rev:
                features['reverse'] = 1
              
        return features



    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_invader': -50, 'distance_to_ghost': 2, 'scared_ghost': -1, 
                'distance_to_food': -1, 'food_remaining': -1,  'stop': -100, 'reverse': -20 }
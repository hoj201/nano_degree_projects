import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from itertools import product
from numpy.random import randn

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    actions = (None, 'forward','left','right')
    states = set( product( ('forward','left','right') , ('red','green')))

    # state-space consists of next_waypoint output and light color
    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Q = {}
        for state in self.states:
            for action in LearningAgent.actions:
                self.Q[(state,action)] = randn()
        self.explored_state_action_pairs = []
        self.time = 1 #life-time of agent
        self.reward_series = [[]]
        self.trial = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.trial += 1
        self.reward_series.append([])
        if self.is_optimal():
            print "YAYAYAYAYAYAYAY"
            print self.trial
            quit()

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.make_state( inputs , self.next_waypoint)
        self.time += 1

        # TODO: Select action according to your policy
        action = self.policy()

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        new_inputs = self.env.sense(self)
        new_next_waypoint = self.planner.next_waypoint()
        new_state = self.make_state( new_inputs, new_next_waypoint)
        self.update_Q(  action , new_state , reward)
        
        #store variables to do time-series plots of rewards
        self.reward_series[ self.trial ].append( (deadline,reward) )
        #print "LearningAgent.update(): deadline = {}, state = {}, action = {}, reward = {}".format(deadline, self.state , action, reward)  # [debug]
    
    def make_state( self, inputs, next_waypoint):
        return (next_waypoint,inputs['light'] )

    def print_Q( self ):
        # prints a table with columns labeled by actions
        # and rows labeled by states
        # values are Q[(state,action)]
        print "Q-table:\n"
        header = self.actions
        table = []
        for state in self.states:
            row = [state]+ [ self.Q[(state,action)] for action in self.actions]
            table.append(row)
        from tabulate import tabulate
        print tabulate(table, headers=header)
        if self.is_optimal():
            print "Yay.  Optimal\n"
        else:
            print "suboptimal :(\n"
        return 0

    def is_optimal( self ):
        for next_waypoint,light in product( ['left','right','forward'] , ['red','green']):
            state = (next_waypoint,light)
            action = max( self.actions , key = lambda a : self.Q[ (state,a)] )
            if (next_waypoint is 'right'):
                if action != 'right':
                    print "should be turning right in state {}".format( state )
                    print "instead action = {}.".format( action )
                    return False
            elif (light is 'green'):
                if (next_waypoint != action):
                    print "action does not match way-point in state {}".format(state)
                    return False
            else:
                if (action != None):
                    print "action is not None in state {}".format(state)
                    return False
        return True

    def policy( self , deterministic = False):
        from numpy import exp
        epsilon = exp( -(self.time - 50)/2.0 )
        if self.trial > 100 or deterministic:
            U = lambda action : self.Q[( self.state , action)]
            return max( self.actions , key = U )
        else:
            for action in LearningAgent.actions:
                if (self.state,action) not in self.explored_state_action_pairs:
                    self.explored_state_action_pairs.append( (self.state,action) )
                    return action
            return random.choice( self.actions )
        
    def update_Q( self,  a , s_new, reward):
        gamma = 0.5
        time_scale = 20.0
        alpha = time_scale / (self.time+time_scale) #learning rate
        Q_old = self.Q[(self.state,a)]
        Q_future = max( [ self.Q[(s_new,a_new)] for a_new in self.actions] )
        self.Q[(self.state,a)] = Q_old + alpha*(reward + gamma*Q_future - Q_old)

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit
    a.reward_series.remove([]) #remove empty list at end
    return a.reward_series

if __name__ == '__main__':
    series_list = run()
    f = open('time_series_data.json','w')
    import json
    json.dump( series_list , f )
    f.close()

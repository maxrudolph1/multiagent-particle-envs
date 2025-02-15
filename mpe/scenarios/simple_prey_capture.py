import numpy as np
from mpe.core import World, Agent, Landmark
from mpe.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2


        # add regular agents
        num_predator = 3
        num_prey = 1
        num_agents = num_predator + num_prey
        agent_labels = ['pred'] * num_predator + ['prey'] * num_prey
        num_landmarks = 2
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.collide = True
            agent.silent = True
            agent.label = agent_labels[i]
            agent.predator = (agent.label == 'pred')
            agent.name = agent.label + ' agent %d' % i 
            agent.size = 0.075 if agent.predator else 0.05
            agent.accel = 3.0 if agent.predator else 4.0
            agent.accel = 20.0 if agent.predator else 25.0
            agent.max_speed = 1.0 if agent.predator else 1.3
            if agent.label == 'prey':
                def scripted_prey(agent, self):
                    agent.action.u =  np.random.random((2,)) * 2 - 1
                    agent.action.c = np.array([0, 0])
                    return agent.action
                agent.action_callback = lambda agent, self: scripted_prey(agent, self)
    
        # add scripted agents
    
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world
    

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.predator else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = world.np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = world.np_random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
                

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.predator:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < 1.8*dist_min else False
    
    def is_prey_caught(self, agent, world):
        prey = self.good_agents(world)[0]
        return sum([self.is_collision(prey, adv) for adv in self.adversaries(world)]) >= 2

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.predator]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.predator]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.predator else self.agent_reward(agent, world)
        return main_reward

    def done(self, agent, world):
        return self.is_prey_caught(agent, world)
            
    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        prey = self.good_agents(world)[0]
        # Adversaries are rewarded for collisions with agents
        if self.is_prey_caught(agent, world):
            return 1.0 if self.is_collision(prey, agent) else 0.0
        else:
            return 0.0

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        # entity_pos = []
        # for entity in world.landmarks:
        #     if not entity.boundary:
        #         entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        pred_pos = []
        pred_vel = []
        prey_pos = []
        prey_vel = []
        for other in world.agents:
            if other.predator:
                pred_pos.append(other.state.p_pos - agent.state.p_pos)
                pred_vel.append(other.state.p_vel)
            else:
                prey_pos.append(other.state.p_pos - agent.state.p_pos)
                prey_vel.append(other.state.p_vel)
            
            
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + pred_pos + prey_pos + prey_vel)

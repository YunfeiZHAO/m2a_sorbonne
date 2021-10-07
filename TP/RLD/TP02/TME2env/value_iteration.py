import gym
import gridworld

import numpy as np
from numpy import linalg as LA


class PolicyIterationAgent(object):
    """The world's simplest agent!"""

    def __init__(self, env, e=1e-6, gamma=0.99):
        self.env = env
        states, mdp = env.getMDP()
        self.e = e
        self.gamma = gamma
        self.V = np.zeros(len(states))

    def train(self, MDP):
        while True:
            # V evaluation for policy PI
            V_new = np.zeros_like(self.V)
            for state, A_table in MDP.items():
                # for each action in a state
                for action, s_prime_list in A_table.items():
                    for p, s_prime, r, done in s_prime_list:
                        # for each s_prime possible in state s with action PI[state]
                        print('p:', p)
                        print('r:', r)
                        print('v:', self.V[s_prime])
                        V_new[state] += p * (r + self.gamma * self.V[s_prime])
            print(V_new)
            if LA.norm(self.V - V_new) < self.e:
                break
            self.V = V_new

        # Policy update for each state
        PI_new = np.zeros_like(self.PI)
        for state, A_table in MDP.items():
            values_a = {}
            for action,  s_prime_list in A_table.items():
                value = 0
                for p, s_prime, r, done in s_prime_list:
                    value += p * (r + self.gamma * self.V[s_prime])
                values_a[action] = value
            PI_new[state] = max(values_a, key=values_a.get)
        self.PI = PI_new
        # self.display_policy()

    def display_policy(self):
        obs = self.env.reset()
        rsum = 0
        j = 0
        while True:
            action = self.act(obs)
            obs, reward, done, _ = self.env.step(action)
            rsum += reward
            j += 1
            self.env.render()
            if done or j > 20:
                print(" rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    def act(self, observation):
        observed_state = env.getStateFromObs(observation)
        return self.PI[observed_state]


if __name__ == '__main__':
    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

    env.seed(0)  # Initialise le seed du pseudo-random
    print(env.action_space)  # Quelles sont les actions possibles
    print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    env.render()  # permet de visualiser la grille du jeu
    env.render(mode="human")  # visualisation sur la console
    states, mdp = env.getMDP()  # recupere le mdp et la liste d'etats
    print("Nombre d'etats : ", len(states))
    state, transitions = list(mdp.items())[0]
    print(state)  # un etat du mdp
    print(transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}

    # Execution avec un Agent
    policyIterationAgent = PolicyIterationAgent(env)


    episode_count = 1000
    reward = 0
    done = False
    rsum = 0

    for i in range(episode_count):
        obs = env.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render()
        j = 0
        rsum = 0
        policyIterationAgent.train(mdp)
        while True:
            action = policyIterationAgent.act(obs)
            obs, reward, done, _ = env.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()

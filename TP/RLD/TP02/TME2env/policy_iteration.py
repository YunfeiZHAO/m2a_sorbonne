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
        self.PI = np.zeros(len(states))
        self.V = np.zeros(len(states))

    def train(self, MDP):
        i = 0
        while True:
            while True:
                # V evaluation for policy PI
                V_new = np.zeros_like(self.V)
                for state, A_table in MDP.items():
                    # for each state in v
                    action = self.PI[state]
                    s_prime_list = A_table[action]
                    for p, s_prime, r, done in s_prime_list:
                        # for each s_prime possible in state s with action PI[state]
                        V_new[state] += p * (r + self.gamma * self.V[s_prime])
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
            if np.array_equal(PI_new, self.PI):
                break
            self.PI = PI_new
            i += 1
            print(self.PI)
            if i % 1 == 0:
                self.display_policy()

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
    PolicyIterationAgent = PolicyIterationAgent(env)
    PolicyIterationAgent.train(mdp)
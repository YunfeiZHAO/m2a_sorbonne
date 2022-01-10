import matplotlib
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import random
import copy
from datetime import datetime
import os
from utils import *

matplotlib.use("TkAgg")


class SARSA(object):
    def __init__(self, env, opt):
        self.opt = opt
        self.action_space = env.action_space
        self.env = env
        self.discount = opt.gamma
        self.alpha = opt.learningRate
        self.explo = opt.explo
        self.exploMode = opt.exploMode  # 0: epsilon greedy, 1: ucb
        self.algo = opt.algo
        self.modelSamples = opt.nbModelSamples
        self.eps = opt.explo
        self.decay = opt.decay
        self.test = False
        self.qstates = {}  # dictionnaire d'états rencontrés
        self.values = []   # contient, pour chaque numéro d'état, les qvaleurs des self.action_space.n actions possibles

    def save(self, file):
        pass

    # enregistre cette observation dans la liste des états rencontrés si pas déjà présente
    # retourne l'identifiant associé à cet état
    def storeState(self, obs):
        observation = obs.dumps()
        s = str(observation)
        ss = self.qstates.get(s, -1)  # if haven't met, return -1

        # Si l'etat jamais rencontré
        if ss < 0:
            ss = len(self.values)
            self.qstates[s] = ss
            # Optimism faced to uncertainty (on commence avec des valeurs à 1 pour favoriser l'exploration)
            self.values.append(np.ones(self.action_space.n) * 1.0)
        return ss

    def act(self, obs):
        self.eps = self.eps * self.decay
        if random.random() < self.eps:
            a = self.action_space.sample()
        else:
            possible_actions = self.values[obs]
            a = possible_actions.argmax()
        return a

    def store(self, ob, action, new_ob, reward, done, it):

        if self.test:
            return
        self.last_source = ob
        self.last_action = action
        self.last_dest = new_ob
        self.last_reward = reward
        if it == self.opt.maxLengthTrain:
            # si on a atteint la taille limite, ce n'est pas un vrai done de l'environnement
            done = False
        self.last_done = done

    def learn(self, done):
        self.new_action = self.act(self.last_dest)
        if done:
            self.values[self.last_source][self.last_action] = \
                self.values[self.last_source][self.last_action] + \
                self.alpha * (self.last_reward - self.values[self.last_source][self.last_action])
        else:
            self.values[self.last_source][self.last_action] = \
                self.values[self.last_source][self.last_action] + \
                self.alpha * (self.last_reward + self.discount * self.values[self.last_dest][self.new_action] -
                              self.values[self.last_source][self.last_action])


if __name__ == '__main__':
    algoName = 'sarsa-plan9'
    env, config, outdir, logger = init('./configs/config_SARSA_gridworld.yaml', algoName)  # in util.py
    config['algo'] = algoName
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = SARSA(env, config)

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    nb = 0
    for i in range(episode_count):
        checkConfUpdate(outdir, config)  # permet de changer la config en cours de run
        rsum = 0
        ob = env.reset()
        if i > 0 and i % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        #
        j = 1
        if verbose:
            env.render()
        new_ob = agent.storeState(ob)
        agent.new_action = agent.act(new_ob)
        # new_ob, reward, done, _ = env.step(action)
        # new_ob = agent.storeState(new_ob)
        # agent.store(ob, action, new_ob, reward, done, j)
        # agent.learn(done)
        # rsum += reward

        while True:
            if verbose:
                env.render()
            ob = new_ob
            new_ob, reward, done, _ = env.step(agent.new_action)
            new_ob = agent.storeState(new_ob)
            j += 1
            if (config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"]):
                done = True
                #print("forced done!")

            agent.store(ob, agent.new_action , new_ob, reward, done, j)

            agent.learn(done)
            rsum += reward
            if done:
                # tensoboard logging
                if verbose:
                    env.render()
                print(algoName)

                logger.direct_write(f"rewardTrain/{config['map']}", rsum, i)

                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                break
    env.close()

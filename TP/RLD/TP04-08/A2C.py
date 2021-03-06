"""Advantage Actor Critic (A2C)"""

import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")

import gym
import gridworld
# import highway_env

import torch
import copy
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import yaml
from datetime import datetime

from utils import *
from core import *
from memory import *


class A2C(object):
    """Deep Q-Networl with experience replay"""
    def __init__(self, env, opt):
        self.opt = opt
        print(opt)
        self.env = env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.test = False
        self.nbEvents = 0
        self.discount = config["discount"]
        self.decay = config["decay"]
        self.eps = opt.eps

        # Definition of replay memory D
        self.D = Memory(self.opt.mem_size, prior=False, p_upper=1., epsilon=.01, alpha=1, beta=1)
        # Definition of Q and Q_hat
        # NN is defined in utils.py
        state_feature_size = self.featureExtractor.outSize
        action_feature_size = self.action_space.n
        self.V = NN(inSize=state_feature_size, outSize=1, layers=[30, 30], activation=torch.tanh)
        self.PI = NN(inSize=state_feature_size, outSize=action_feature_size,
                     layers=[30, 30], activation=torch.nn.Tanh(), finalActivation=torch.nn.Softmax(dim=1))

        with torch.no_grad():
            self.V_target = copy.deepcopy(self.V)
        # Definition of loss
        self.loss = F.smooth_l1_loss

        # Optimiser
        self.lr = float(opt.lr)
        self.optim = torch.optim.SGD(self.Q.parameters(), lr=self.lr)
        self.optim.zero_grad()

    def act(self, obs):
        # epsilon greedy action
        self.eps = self.eps * self.decay
        if torch.rand(1) < self.eps:
            a = self.action_space.sample()
        else:
            a = torch.argmax(self.Q.forward(torch.Tensor(obs))).item()
        return a

    # sauvegarde du mod??le
    def save(self, outputDir):
        # torch.save
        pass

    # chargement du mod??le.
    def load(self, inputDir):
        pass

    # apprentissage de l'agent. Dans cette version rien ?? faire
    def learn(self):
        # Si l'agent est en mode de test, on n'entra??ne pas
        if self.test:
            return
        else:
            # get mini_batch a batch of (ob, action, reward, new_ob, done)
            # sample (si, ai) from pi_theta
            mini_batch = self.D.mem  # we take all samples in memory as mini batch
            column_mini_batch = list(zip(*mini_batch))

            # s
            obs_batch = torch.tensor(column_mini_batch[0], dtype=torch.float).squeeze(dim=1)  # B, dim_obs=4
            # a
            action_batch = torch.tensor(column_mini_batch[1], dtype=torch.int64)
            # r
            r_batch = torch.tensor(column_mini_batch[2], dtype=torch.float)
            # s'
            new_obs_batch = torch.tensor(column_mini_batch[3], dtype=torch.float).squeeze(dim=1)  # B, dim_obs=4
            # done
            done_batch = torch.tensor(column_mini_batch[4], dtype=torch.float)

            # fit V_pi(s)
            y_batch = r_batch + self.discount * self.V.forward(new_obs_batch) * (1 - done_batch)  # if done this term is 0
            # compute advantage value
            A = y_batch - self.V.forward(obs_batch)

            q_batch = self.Q.forward(obs_batch).gather(1,  torch.unsqueeze(action_batch, 1)).squeeze(1)  # reward self.Q.forward(obs_batch): B, 2
            output = self.loss(y_batch, q_batch)
            logger.direct_write("Loss", output, episode)
            output.backward()
            self.optim.step()
            self.optim.zero_grad()

            # buffer reset after policy update
            self.D = Memory(self.opt.mem_size, prior=False, p_upper=1., epsilon=.01, alpha=1, beta=1)
            self.nbEvents = 0

    def store(self, ob, action, new_ob, reward, done, it):
        """enregistrement de la transition pour exploitation par learn ulterieure"""
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:
            # si on atteint la taille max d'episode en apprentis
            # alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done = False
            tr = (ob, action, reward, new_ob, done)
            self.D.store(tr)
            # ici on n'enregistre que la derniere transition pour traitement imm??diat,
            # mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)
            self.lastTransition = tr

    def time_to_learn(self, done):
        # retoune vrai si c'est le moment d'entra??ner l'agent.
        # Dans cette version retourne vrai tous les freqoptim evenements
        # Mais on pourrait retourner vrai seulement si done pour s'entra??ner seulement en fin d'episode
        if self.test:
            return False
        self.nbEvents += 1
        return self.nbEvents >= self.opt.freqOptim and done


if __name__ == '__main__':
    # Configuration
    # pour lunar pip install Box2D
    # env, config, outdir, logger = init('./configs/config_DQN_cartpole.yaml', "DQN")
    # env, config, outdir, logger = init('./configs/config_DQN_gridworld.yaml', "DQN")
    env, config, outdir, logger = init('./configs/config_DQN_lunar.yaml', "DQN")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    # Agent
    agent = A2C(env, config)
    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    for episode in range(episode_count):
        checkConfUpdate(outdir, config)
        rsum = 0
        agent.nbEvents = 0
        ob = env.reset()

        # On souhaite afficher l'environnement (attention ?? ne pas trop afficher car ??a ralentit beaucoup)
        if episode % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        # C'est le moment de tester l'agent
        if episode % freqTest == 0 and episode >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            agent.test = True

        # On a fini cette session de test
        if episode % freqTest == nbTest and episode > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        # C'est le moment de sauver le mod??le
        if episode % freqSave == 0:
            agent.save(outdir + "/save_" + str(episode))

        n_step = 0  # steps in an episode
        if verbose:
            env.render()

        new_ob = agent.featureExtractor.getFeatures(ob)
        while True:
            if verbose:
                env.render()

            ob = new_ob
            action = agent.act(ob)

            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.featureExtractor.getFeatures(new_ob)

            n_step += 1

            # Si on a atteint la longueur max d??finie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (n_step == config["maxLengthTrain"])) or \
                    ((agent.test) and (config["maxLengthTest"] > 0) and (n_step == config["maxLengthTest"])):
                done = True
                print("forced done!")

            agent.store(ob, action, new_ob, reward, done, n_step)
            rsum += reward

            if agent.time_to_learn(done):
                agent.learn()
                if episode % config['C'] == 0:
                    agent.Q_target.load_state_dict(agent.Q.state_dict())

            if done:
                if verbose:
                    env.render()
                print(str(episode) + " rsum=" + str(rsum) + ", " + str(n_step) + " actions ")
                logger.direct_write("reward", rsum, episode)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
                break
    env.close()

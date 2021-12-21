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


class DQNPER(object):
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
        self.D = Memory(self.opt.mem_size, prior=True, p_upper=1., epsilon=.01, alpha=1, beta=1)
        # Definition of Q and Q_hat
        # NN is defined in utils.py
        state_feature_size = self.featureExtractor.outSize
        action_feature_size = self.action_space.n
        self.Q = NN(inSize=state_feature_size, outSize=action_feature_size, layers=[200], activation=torch.nn.Tanh())
        with torch.no_grad():
            self.Q_target = copy.deepcopy(self.Q)
        # Definition of loss
        self.loss = nn.SmoothL1Loss(reduce=False)
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

    # sauvegarde du modèle
    def save(self, outputDir):
        # torch.save
        pass

    # chargement du modèle.
    def load(self, inputDir):
        pass

    # apprentissage de l'agent. Dans cette version rien à faire
    def learn(self):
        # Si l'agent est en mode de test, on n'entraîne pas
        if self.test:
            return
        else:
            # get mini_batch a batch of (ob, action, reward, new_ob, done)
            idx, w, mini_batch = self.D.sample(self.opt["mini_batch_size"])
            # instance sampling weight
            # w = w / np.max(w)

            # reform date
            column_mini_batch = list(zip(*mini_batch))
            obs_batch = torch.tensor(column_mini_batch[0], dtype=torch.float).squeeze(dim=1) # B, dim_obs=4
            action_batch = torch.tensor(column_mini_batch[1], dtype=torch.int64)
            r_batch = torch.tensor(column_mini_batch[2], dtype=torch.float)
            new_obs_batch = torch.tensor(column_mini_batch[3], dtype=torch.float).squeeze(dim=1) # B, dim_obs=4
            done_batch = torch.tensor(column_mini_batch[4], dtype=torch.float)

            # Q update
            y_batch = r_batch + self.discount * torch.max(self.Q_target.forward(new_obs_batch), dim=-1).values * (1 - done_batch) # if done this term is 0
            q_batch = self.Q.forward(obs_batch).gather(1,  torch.unsqueeze(action_batch, 1)).squeeze(1) # reward self.Q.forward(obs_batch): B, action number
            tderr = y_batch - q_batch
            self.D.update(idx, torch.abs_(tderr).detach().numpy())

            element_loss = self.loss(y_batch, q_batch)
            output = torch.sum(element_loss)  # * torch.tensor(w))

            logger.direct_write("Train/Loss", output, self.nbEvents)
            logger.direct_write("Train/Mean Q", torch.mean(q_batch), self.nbEvents)
            output.backward()
            self.optim.step()
            self.optim.zero_grad()

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
            index = self.D.store(tr)
            # ici on n'enregistre que la derniere transition pour traitement immédiat,
            # mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)
            self.lastTransition = tr

    def timeToLearn(self, done):
        # retoune vrai si c'est le moment d'entraîner l'agent.
        # Dans cette version retourne vrai tous les freqoptim evenements
        # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
        if self.test:
            return False
        self.nbEvents += 1
        return self.nbEvents % self.opt.freqOptim == 0


if __name__ == '__main__':
    # Configuration
    # pour lunar pip install Box2D
    env, config, outdir, logger = init('./configs/config_DQN_PER_gridworld.yaml', "DQN_PER")
    # env, config, outdir, logger = init('configs/config_DQN_PER_cartpole.yaml', "DQN_PER")
    # env, config, outdir, logger = init('./configs/config_DQN_PER_lunar.yaml', "DQN_PER")

    print(config)
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    # Agent
    agent = DQNPER(env, config)
    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    for i_episode in range(episode_count):
        checkConfUpdate(outdir, config)
        rsum = 0
        ob = env.reset()

        # On souhaite afficher l'environnement (attention à ne pas trop afficher car ça ralentit beaucoup)
        if i_episode % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        # C'est le moment de tester l'agent
        if i_episode % freqTest == 0 and i_episode >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            agent.test = True

        # On a fini cette session de test
        if i_episode % freqTest == nbTest and i_episode > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        # C'est le moment de sauver le modèle
        if i_episode % freqSave == 0:
            agent.save(outdir + "/save_" + str(i_episode))

        j = 0  # steps in an episode
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

            j += 1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or \
                    ((agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")

            agent.store(ob, action, new_ob, reward, done, j)
            rsum += reward

            if agent.timeToLearn(done):
                agent.learn()
                if agent.nbEvents % config['C'] == 0:
                    agent.Q_target.load_state_dict(agent.Q.state_dict())

            if done:
                if verbose:
                    env.render()
                print(str(i_episode) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("Train/reward", rsum, i_episode)
                mean += rsum
                rsum = 0
                break
    env.close()

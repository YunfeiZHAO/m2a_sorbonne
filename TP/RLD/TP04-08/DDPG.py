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


class DDPG(object):
    """DDPG, continue action policy gradient"""
    def __init__(self, env, opt):
        self.opt = opt
        self.env = env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.test = False
        self.nbEvents = 0
        self.discount = config["discount"]
        self.decay = config["decay"]
        self.eps = opt.eps

        # Definition of replay memory D
        self.D = Memory(opt.mem_size, prior=False, p_upper=1., epsilon=.01, alpha=1, beta=1)

        # Environment
        action_space = self.env.action_space
        self.a_low = action_space.low
        self.a_high = action_space.high
        self.action_feature_size = action_space.shape[0]
        self.action_dtype = action_space.dtype
        self.state_feature_size = self.env.observation_space.shape[0]
        # Definition of Q and Q_hat
        # NN is defined in utils.py
        self.Q = NN(inSize=self.state_feature_size + self.action_feature_size, outSize=1, layers=[100, 200, 100])
        self.P = NN(inSize=self.state_feature_size, outSize=self.action_feature_size, layers=[100, 200, 100])
        with torch.no_grad():
            self.Q_target = copy.deepcopy(self.Q)
            self.P_target = copy.deepcopy(self.P)
        # Definition of loss

        # Optimiser
        self.lr = opt.lr
        self.optim = torch.optim.Adam(self.Q.parameters(), lr=self.lr)
        self.optim.zero_grad()

    def act(self, obs):
        # epsilon greedy action
        action = torch.clip(self.P(torch.tensor(obs, dtype=torch.float)) +
                            torch.normal(0, 1, size=(1, 1)), min=int(self.a_low), max=int(self.a_high))
        return [action.item()]

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
            batch_size = self.opt["mini_batch_size"]
            mask, _, mini_batch = self.D.sample(batch_size)
            column_mini_batch = list(zip(*mini_batch))
            obs_batch = torch.tensor(column_mini_batch[0], dtype=torch.float).squeeze(dim=1) # B, dim_obs=4
            action_batch = torch.tensor(column_mini_batch[1], dtype=torch.int64)
            r_batch = torch.tensor(column_mini_batch[2], dtype=torch.float)
            new_obs_batch = torch.tensor(column_mini_batch[3], dtype=torch.float).squeeze(dim=1) # B, dim_obs=4
            done_batch = torch.tensor(column_mini_batch[4], dtype=torch.float)

            # compute targets
            y_batch = r_batch + self.discount * (1 - done_batch) * \
                      self.Q_target(torch.cat((new_obs_batch, self.P_target(new_obs_batch)), dim=1))

            # update Q-function by one step of gradient descent
            self.Qloss = nn.MSELoss(reduction='mean')/batch_size

            # q_batch = self.Q.forward(obs_batch).gather(1,  torch.unsqueeze(action_batch, 1)).squeeze(1) # B
            # output = self.loss(y_batch, q_batch)
            # logger.direct_write("Loss", output, i)
            # output.backward()
            # self.optim.step()
            # self.optim.zero_grad()
            pass

    def store(self, ob, action, new_ob, reward, done, it):
        """enregistrement de la transition pour exploitation par learn ulterieure"""
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:
            # si on atteint la taille max d'episode en apprentissage,
            # alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done = False
            tr = (ob, action, reward, new_ob, done)
            self.D.store(tr)
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
    env, config, outdir, logger = init('configs/config_DDPG_Pendulum-v0.yaml', "DDPG")
    # env, config, outdir, logger = init('./configs/config_random_gridworld.yaml', "DQN")
    # env, config, outdir, logger = init('./configs/config_random_lunar.yaml', "DQN")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]
    config["mem_size"] = 10000
    config["mini_batch_size"] = 100
    config["eps"] = 0.1
    # optimisation step
    config["C"] = 20
    config["discount"] = 0.999
    config["decay"] = 0.99999
    config["lr"] = 3e-4
    # Agent
    agent = DDPG(env, config)
    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    for i in range(episode_count):
        checkConfUpdate(outdir, config)
        rsum = 0
        agent.nbEvents = 0
        new_ob = env.reset()

        # On souhaite afficher l'environnement (attention à ne pas trop afficher car ça ralentit beaucoup)
        if i % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        # C'est le moment de tester l'agent
        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            agent.test = True

        # On a fini cette session de test
        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        # C'est le moment de sauver le modèle
        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0  # steps in an episode
        if verbose:
            env.render()

        while True:
            ob = new_ob
            if verbose:
                env.render()
            action = agent.act(ob)
            new_ob, reward, done, _ = env.step(action)
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
                if i % config['C'] == 0:
                    agent.Q_target.load_state_dict(agent.Q.state_dict())

            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
                break
    env.close()

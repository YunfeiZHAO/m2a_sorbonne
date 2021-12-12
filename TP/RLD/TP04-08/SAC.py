import argparse
import sys

import gym
import gridworld
import torch
from utils import *
from core import *
from memory import *
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import yaml
from datetime import datetime
import copy
from torch.distributions import Categorical, Normal
import torch.nn as nn
import torch.optim as optim


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, action_scale, hidden_dim=256):
        super(Policy, self).__init__()
        self.P = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim)
            )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.action_scale = action_scale

    def forward(self, state):
        x = self.P(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def getAction(self, state):
        mean, log_std = self.forward(state)  # B, d
        std = log_std.exp()  # B, d
        normal = Normal(mean, std)
        # sample action
        z = normal.rsample()  # B, d
        action = self.action_scale * torch.tanh(z)  # entre -1 et 1
        # calculate entropy, project in action space with variable changement
        project_term = torch.sum(torch.log(self.action_scale*(1-torch.tanh(z).pow(2)) + 1e-6), dim=1, keepdim=True)
        log_prob = Normal(mean, std).log_prob(z) - project_term
        # log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        return action, log_prob


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        # Q1
        self.Q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        # Q2
        self.Q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        # forward for Q1
        x1 = self.Q1(x)
        # forward for Q2
        x2 = self.Q2(x)
        return x1, x2


class SAC(object):
    """The world's simplest agent!"""

    def __init__(self, env, opt):
        self.opt = opt
        self.env = env
        if self.opt.fromFile is not None:
            self.load(self.opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = self.opt.featExtractor(env)
        self.test = False
        self.nbEvents = 0
        # Environments
        self.ob_dim = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.action_scale = env.action_space.high[0]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Actor
        self.actor = Policy(self.ob_dim, self.action_size, self.action_scale, hidden_dim=256).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.opt.lr_a)
        # Critic
        self.Q_local = QNetwork(self.ob_dim, self.action_size, hidden_dim=256).to(self.device)
        self.Q1_optimizer = optim.Adam(self.Q_local.Q1.parameters(), lr=self.opt.lr_c)
        self.Q2_optimizer = optim.Adam(self.Q_local.Q2.parameters(), lr=self.opt.lr_c)
        self.Q_target = copy.deepcopy(self.Q_local)

        self.adaptive = self.opt.adaptive
        self.log_alpha = torch.tensor(np.log(self.opt.alpha)).to(self.device)

        if self.adaptive:
            self.target_entropy = self.opt.target_entropy
            # self.target_entropy = torch.tensor(-np.prod(env.action_space.shape)).to(self.device)
            self.log_alpha.requires_grad = True
            self.alpha_optim = optim.Adam([self.log_alpha], lr=self.opt.lr_alpha)

        self.actor_loss = None
        self.critic_loss = None
        self.alpha_loss = None

        self.K = self.opt.K_epochs
        self.discount = self.opt.discount
        self.rho = self.opt.rho
        self.batch_size = self.opt.batch_size
 
        self.memory = Memory(mem_size=self.opt.buffer_size)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs):
        state = torch.FloatTensor(obs).to(self.device).view(1, -1)
        action, _ = self.actor.getAction(state)
    
        return action.cpu().data.numpy().flatten()

    # sauvegarde du modèle
    def save(self, outputDir):
        pass

    # chargement du modèle.
    def load(self, inputDir):
        pass

    # apprentissage de l'agent. Dans cette version rien à faire
        
    def learn(self):
        for _ in range(self.K):  # learn K times
            _, _, batch = self.memory.sample(self.batch_size)
            batch = list(zip(*batch))
            ob = torch.FloatTensor(batch[0]).view(self.batch_size, -1).to(self.device)
            act = torch.FloatTensor(batch[1]).view(-1, self.action_size).to(self.device)
            reward = torch.FloatTensor(batch[2]).view(-1, 1).to(self.device)
            new_ob = torch.FloatTensor(batch[3]).view(self.batch_size, -1).to(self.device)
            done = torch.FloatTensor(batch[4]).view(-1, 1).to(self.device)

            with torch.no_grad():
                next_action, next_log_prob = self.actor.getAction(ob)
                y_q1, y_q2 = self.Q_target(new_ob, next_action)
                y_q = reward + done * self.discount * (torch.min(y_q1, y_q2) - self.alpha * next_log_prob)

            self.target_mean = torch.mean(y_q)
            q1, q2 = self.Q_local(ob, act)

            loss1 = F.mse_loss(q1, y_q)
            loss2 = F.mse_loss(q2, y_q)

            self.critic_q1_loss = loss1.item()
            self.critic_q2_loss = loss2.item()

            # update Q1 and Q2
            self.Q1_optimizer.zero_grad()
            loss1.backward()
            self.Q1_optimizer.step()

            self.Q2_optimizer.zero_grad()
            loss2.backward()
            self.Q2_optimizer.step()

            # update policy
            action, log_prob = self.actor.getAction(ob)
            self.entropy = torch.mean(log_prob)

            q1, q2 = self.Q_local(ob, action)
            actor_loss = torch.mean(self.alpha.detach()*log_prob - q1)  # -Q to do step gradiento ascent
            self.actor_loss = actor_loss.item()
     
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.adaptive:
                self.alpha_optim.zero_grad()
                alpha_loss = torch.mean(self.alpha * (-log_prob - self.target_entropy).detach())
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha_loss = alpha_loss.item()

            for target_param, local_param in zip(self.Q_target.parameters(), self.Q_local.parameters()):
                target_param.data.copy_((1-self.rho)*local_param.data + self.rho*target_param.data)

    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self, ob, action, new_obs, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:
            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
            tr = (np.squeeze(ob), action, reward, np.squeeze(new_obs), done)
            #self.lastTransition=tr #ici on n'enregistre que la derniere transition pour traitement immédiat, mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)
            self.memory.store(tr)

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self,done):
        if self.test:
            return False
        self.nbEvents += 1
        return self.nbEvents % self.opt.freqOptim == 0


if __name__ == '__main__':
    env, config, outdir, logger = init('./configs/config_SAC.yaml', "SAC")
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]   
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = SAC(env, config)

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    for nb_episode in range(episode_count):
        checkConfUpdate(outdir, config)
        rsum = 0
        ob = env.reset()

        # On souhaite afficher l'environnement (attention à ne pas trop afficher car çà ralentit beaucoup)
        if nb_episode % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        # C'est le moment de tester l'agent
        # if nb_episode % freqTest == 0 and nb_episode >= freqTest:  ##### Same as train for now
        #     print("Test time! ")
        #     mean = 0
        #     agent.test = True

        # On a fini cette session de test
        # if nb_episode % freqTest == nbTest and nb_episode > freqTest:
        #     print("End of test, mean reward=", mean / nbTest)
        #     itest += 1
        #     logger.direct_write("rewardTest", mean / nbTest, itest)
        #     agent.test = False

        # C'est le moment de sauver le modèle
        if nb_episode % freqSave == 0:
            agent.save(outdir + "/save_" + str(nb_episode))

        nb_step = 0
        if verbose:
            env.render()
            pass

        new_obs = agent.featureExtractor.getFeatures(ob)

        while True:
            if verbose:
                env.render()

            ob = new_obs
            # if nb_episode < 1000:
            #     action = env.action_space.sample()
            # else:
            action = agent.act(ob)

            new_obs, reward, done, _ = env.step(action)
            reward /= 100

            agent.store(ob, action, new_obs, reward, done, nb_step)

            nb_step += 1

            # Si on a atteint la longueur max définie dans le fichier de config
            if((config["maxLengthTrain"] > 0) and (not agent.test) and (nb_step == config["maxLengthTrain"])) or \
              ((agent.test) and (config["maxLengthTest"] > 0) and (nb_step == config["maxLengthTest"])):
                done = True
                print("forced done!")

            rsum += reward * 100

            if agent.timeToLearn(done):
                agent.learn()
                logger.direct_write("loss/actor loss", agent.actor_loss, agent.nbEvents)
                # logger.direct_write("critic loss", agent.critic_loss, agent.nbEvents)
                logger.direct_write("loss/critic Q1 loss", agent.critic_q1_loss, agent.nbEvents)
                logger.direct_write("loss/critic Q2 loss", agent.critic_q2_loss, agent.nbEvents)
                logger.direct_write("value/Average Target Value", agent.target_mean, agent.nbEvents)
                logger.direct_write("value/Average Actor Entropy", agent.entropy, agent.nbEvents)
                if agent.adaptive:
                    logger.direct_write("loss/alpha loss", agent.alpha_loss, agent.nbEvents)
                    logger.direct_write("value/alpha", agent.alpha, agent.nbEvents)
            if done:
                print(str(nb_episode) + " rsum=" + str(rsum) + ", " + str(nb_step) + " actions ")
                logger.direct_write("reward", rsum, nb_episode)
                mean += rsum
                rsum = 0
                break
    env.close()

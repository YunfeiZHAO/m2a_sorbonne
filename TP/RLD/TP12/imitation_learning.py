import pickle
import torch.nn

from utils import *
from core import *
from memory import *


def load_expert_transitions(file, nb_features):
    with open(file, 'rb') as handle:
        expert_data = torch.FloatTensor(pickle.load(handle))
        expert_states = expert_data[:, :nb_features]
        expert_actions = expert_data[:, nb_features:]
        expert_states = expert_states.contiguous()
        expert_actions = expert_actions.contiguous()
    return expert_states, expert_actions


def to_one_hot(actions, nb_actions):
    actions = torch.LongTensor(actions.view(-1))
    one_hot = torch.FloatTensor(torch.zeros(actions.size()[0], nb_actions))
    one_hot[range(actions.size()[0]), actions] = 1
    return one_hot


def to_index_action(one_hot, nb_actions):
    ac = torch.tensor(range(nb_actions))
    ac = ac.expand(one_hot.size()[0], -1).contiguous().view(-1)
    actions = ac[one_hot.view(-1) > 0].view(-1)
    return actions


def test_function():
    expert_states, expert_actions = load_expert_transitions('expert.pkl', 8)
    ac = to_index_action(expert_actions, 4)
    ac = to_one_hot(ac, 4)
    print(torch.sum(expert_actions - ac))


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, action_dim)

    def forward(self, x):
        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
        x = F.softmax(self.l3(x))
        return x


"""
# Behavioral Cloning
"""


class BC(object):
    def __init__(self, env, opt):
        self.opt = opt
        self.env = env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.test = False
        self.nbEvents = 0

        self.ob_dim = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actor = Actor(self.ob_dim, self.action_size)
        self.actor_loss = nn.CrossEntropyLoss()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=opt.lr_a)

        self.expert_states, self.expert_actions = load_expert_transitions('expert.pkl', self.ob_dim)

    def act(self, obs):
        self.actor.eval()
        action = torch.argmax(self.actor(torch.FloatTensor(obs).to(self.device).view(1, -1)))
        action = action.cpu().data.numpy().flatten()
        return action

    # sauvegarde du modèle
    def save(self, outputDir):
        pass

    # chargement du modèle.
    def load(self, inputDir):
        pass

    # apprentissage de l'agent. Dans cette version rien à faire

    def learn(self):
        pred_actions = self.actor(self.expert_states)
        actor_loss = self.actor_loss(pred_actions, self.expert_actions)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_loss = actor_loss.item()

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self, done):
        if self.test:
            return False

        self.nbEvents += 1
        return self.nbEvents >= self.batch_size and self.nbEvents % self.opt.freqOptim == 0
        # return done


def main():
    # Configuration
    # pour lunar pip install Box2D
    env, config, outdir, logger = init('./configs/BC_lunar.yaml', "BC")
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
    agent = BC(env, config)
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


if __name__ == '__main__':
    main()





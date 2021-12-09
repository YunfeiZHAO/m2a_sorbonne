from textloader import *
import math
import torch
import torch.nn as nn

#  TODO:  Ce fichier contient les différentes fonction de génération


def generate(rnn, eos, start="", maxlen=200):
    """  Fonction de génération (l'embedding et le decodeur être des fonctions du rnn). Initialise le réseau avec start
    (ou à 0 si start est vide) et génère une séquence de longueur maximale 200 ou qui s'arrête quand eos est généré.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * start : début de la phrase
        * maxlen : longueur maximale
    """

    #  TODO:  Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits
    #    (logarithme de probabilité à une constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles
    phrase = "" + start
    if start == "":
        input = torch.tensor([[0]])  # T=1, B=1
    else:
        input = string2code(start)[..., None]
    for i in range(maxlen):
        hiden_output = rnn(input)
        prob = rnn.decode(hiden_output, decoder_activation=nn.Softmax(dim=-1))  # T=1, d=97(number of Caracters)
        input = torch.multinomial(prob[0][0], 1, replacement=True)
        if input == eos:
            break
        phrase += id2lettre[input.item()]
        input = input[..., None]
    print('Generator: multinomial')
    print(phrase)
    return phrase


def generate_beam(rnn, eos, k, start="", maxlen=200):
    """
        Génere une séquence en beam-search : à chaque itération, on explore pour chaque candidat les k symboles les plus probables;
        puis seuls les k meilleurs candidats de l'ensemble des séquences générées sont conservés (au sens de la vraisemblance) pour l'itération suivante.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * k : le paramètre du beam-search
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    #  TODO:  Implémentez le beam Search
    phrase = ""
    if start == "":
        input = 0  # T=1, B=1
    else:
        input = lettre2id[start]
    sequences = [[list([input]), 0.0]]

    # walk over each step in sequence
    for i in range(maxlen):
        all_candidates = list()
        # expand each current candidate
        for seq, score in sequences:
            if seq[-1] == eos:
                candidate = [seq, score]  # negative log-likelihood
                all_candidates.append(candidate)
            else:
<<<<<<< HEAD
                hiden_output = rnn(torch.tensor(seq[-1]).reshape((1,1)))
                prob = rnn.decode(hiden_output, decoder_activation=nn.Softmax(dim=-1))[0][0]  # T=1, d=97(number of Caracters)
=======
                hiden_output = rnn(torch.tensor(seq[-1])[..., None])
                prob = rnn.decode(hiden_output, decoder_activation=nn.Softmax(dim=-1))[0]  # T=1, d=97(number of Caracters)
>>>>>>> 408e2e722655602fc4d9df139188cd6c14978f58
                for c, p in enumerate(prob):
                    candidate = [seq + [c], score - torch.log(p)]  # negative log-likelihood
                    all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select k best
        sequences = ordered[:k]

    print('Generator: Beam-search')
    for seq, score in sequences:
        print(f"{score}：{code2string(seq)}")
    return sequences


# p_nucleus
def p_nucleus(decoder, alpha: float):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        * decoder: renvoie les logits étant donné l'état du RNN
        * alpha (float): masse de probabilité à couvrir
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
           * h (torch.Tensor): L'état à décoder
        """
        #  TODO:  Implémentez le Nucleus sampling ici (pour un état s)
    return compute

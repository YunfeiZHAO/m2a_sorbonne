# rsample() backpropagation pour les distributions

# V is calculated from Q, E_a(Q(s,a))

# function de projection de R à (0, 1), distribution may generate value outside action space

# use log to evide explosion of gradient generate by exponential
# pytorch: logsumexp


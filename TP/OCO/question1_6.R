################################################# Loading data #############################################

#  MNIST data downloadable at https://pjreddie.com/projects/mnist-in-csv/



set.seed(432)

mnist_train <- read.csv("mnist_train.csv", stringsAsFactors = F, header = F)
mnist_test <- read.csv("mnist_test.csv", stringsAsFactors = F, header = F)

#View(mnist_train) # Data has no column names
#View(mnist_test) # Data has no column names

names(mnist_test)[1] <- "label"
names(mnist_train)[1] <- "label"

################################################# Data preparation #################################################

# Convert label variable into factor

mnist_train$label <- factor(mnist_train$label)
summary(mnist_train$label)

mnist_test$label <- factor(mnist_test$label)
summary(mnist_test$label)

# Scaling data 

max(mnist_train[ ,2:ncol(mnist_train)]) # max pixel value is 255, lets use this to scale data


train <- cbind(label = 2*as.numeric(mnist_train[ ,1]==0)-1, mnist_train[ , 2:ncol(mnist_test)]/255,rep(1,length(mnist_train[ ,1])))
test <- cbind(label = 2*as.numeric(mnist_test[ ,1]==0)-1, mnist_test[ , 2:ncol(mnist_test)]/255,rep(1,length(mnist_train[ ,1])))

# Only interested in predicting the digit 0

################################################# Batch algorithms #################################################

GDproj <- function(a, b, init,iters=100, cost,  grad, lambda=1, z=Inf) {
  
  param <- data.frame(matrix(nrow = iters +1, ncol = length(init) + 1))
  colnames(param) <- c(colnames(a), "Loss")
  x<-c( init)
  param[1, ] <- c(x, cost( init, a, b,lambda))
  j<-2
  
  
  for (i in 1:iters) {
    eta <-  1 /(lambda*i)
    x <- pib1(x - eta * grad(x, as.matrix(a), b, lambda),z)
    param[i + 1, ]  <- c(x,cost(x, a, b,lambda))
  }
  
  
  param <- cbind(Iteration = 1:nrow(param), param)
  
  return(param)
  
}


# Stochastic Gradient Descent with projection on the l1 ball
SGDproj <- function(a, b, init, iters = length(b), cost,  instgrad, lambda,z=Inf) {
  
  ind<-sample(1:length(b),iters,replace=TRUE)
  a<-a[ind,]
  b<-b[ind]
  
  param <- data.frame(matrix(nrow = 101, ncol = length(init) + 1))
  colnames(param) <- c(colnames(a), "Loss")
  x <- c( init)
  m <- x
  param[1, ] <- c(m, cost( init, a, b,lambda))
  j <- 2
  
  for (i in 1:iters) {
    eta <-  1/(lambda*i) # 1/sqrt(i) play with the learning rate
    x <- pib1(x - eta * instgrad(x,a[i,], b[i],lambda),z)
    m <- ((i-1)*m + x)/i #????????? 
    if (i%% (iters/100) == 0)
    {
      param[j, ] <- c(m,cost(m, a, b,lambda))
      j<-j+1}
  }
  
  param <- cbind(Iteration = c(1,1:100*(iters/100)), param)
  
  return(param)
  
}


################################################# OCO algorithms #################################################

# Stochastic Mirror Descent with projection on the l1 ball

SMDproj <- function(a, b, init, iters = length(b), cost,  instgrad, lambda,z =Inf) {
  
  ind<-sample(1:length(b),iters,replace=TRUE)
  a<-a[ind,]
  b<-b[ind]
  d<-ncol(a)

    
  
  param <- data.frame(matrix(nrow = 101, ncol = length(init) + 1))
  colnames(param) <- c(colnames(a), "Loss")
  x <- c(init)
  m <- x
  param[1, ] <- c(m, cost( m, a, b,lambda))
  theta <- x
  j <- 2
  
  for (i in 1:iters) {    
    eta <- sqrt(1/i)  # 1/i play with the learning rate
    theta <- theta - eta * instgrad(x, a[i,], b[i], lambda)
    x <- pib1(theta,z)
    m <- ((i-1)*m + x)/i
    if (i%% (iters/100) == 0)
    {
      param[j, ] <- c(m,cost(m, a, b,lambda))
      j<-j+1}
  }
  
  
  param <- cbind(Iteration = c(1,1:100*(iters/100)), param)
  
  return(param)
  
}

# Stochastic Exponentiated Gradient +_

SEGpm <- function(a, b, iters = length(b), cost,  instgrad, lambda,z) {
  
  ind<-sample(1:length(b),iters,replace=TRUE)
  a<-a[ind,]
  b<-b[ind]
  d<-ncol(a)
  
  param <- data.frame(matrix(nrow = 101, ncol = d + 1))
  colnames(param) <- c(colnames(a), "Loss")
  w <- rep(1,2*d)
  w <- w/sum(w)
  x <- z*c(w[1:d]-w[d+1:d])
  m = x
  param[1, ] <- c(m, cost(m, a, b,lambda))
  j<-2
  
  for (i in 1:iters) {
    eta <- sqrt(1/i) #   play with the learning rate
    instg <- instgrad( x, a[i,], b[i], lambda)
    w <- exp( eta * c(-instg,instg))*w
    w <- w/sum(w)
    x <- z*c(w[1:d]-w[d+1:d])
    m <- ((i-1)*m + x)/i
    if (i%% (iters/100) == 0)
    {
      param[j, ] <- c(m,cost(m, a, b,lambda))
      j<-j+1}
  }
  
  param <- cbind(Iteration = c(1, 1:100*(iters/100)) , param)
  
  return(param)
  
}


# Adagrad projected

Adaproj <- function(a, b, init, iters = length(b), cost,  instgrad, lambda,z=Inf ) {
  ind<-sample(1:length(b),iters,replace=TRUE)
  a<-a[ind,]
  b<-b[ind]
  d <- dim(a)[2]
  
  param <- data.frame(matrix(nrow = 101, ncol = length(init) + 1))
  colnames(param) <- c(colnames(a), "Loss")
  s <- rep(.0001,d)
  x <- c(init)
  m <- x
  param[1, ] <- c(m, cost(m, a, b,lambda))
  j<-2
  
  for (i in 1:iters) { 
    s <- s + instgrad(x, a[i,], b[i], lambda)^2
    y <- x -  1/sqrt(s) * instgrad(x, a[i,], b[i], lambda)  
    x <- pib1w(y,sqrt(s),z)
    m <- ((i-1)*m+x)/i
    if (i%% (iters/100) == 0)
    {
      param[j, ] <- c(m,cost(m, a, b,lambda))
      j<-j+1
    }
  }
  
  param <- cbind(Iteration = c(1,1:100*(iters/100)), param)
  
  return(param)
  
}


# ONS with diagonal projection only


ONS <- function(a, b, init, iters = length(b), cost,  instgrad, lambda,gamm,z) {

  ind<-sample(1:length(b),iters,replace=TRUE)
  a <- a[ind,]
  b <- b[ind]
  d <- length(init)
  param <- data.frame(matrix(nrow = 101, ncol = d + 1))
  colnames(param) <- c(colnames(a), "Loss")
  x <- c(init)
  m <- x
  A <- diag(rep(1/gamm^2,d))
  Ainv <- diag(rep(gamm^2,d))
  param[1, ] <- c(m, cost( m, a, b,lambda))
  j <- 2

  for (i in 1:iters) {
    instg <-  c(instgrad(x, a[i,], b[i], lambda))
    A <- A + instg%*%t(instg)
    Ainstg <- Ainv%*%instg
    Ainv <- Ainv - c(1/(1+t(instg)%*%Ainstg)) * Ainstg%*%t(Ainstg)
    y <- x -  1/gamm * Ainv %*% instg
    x <- pib1w(y,diag(A),z)
    m <- ((i-1)*m + x)/i
    if (i%% (iters/100) == 0)
    {
      param[j, ] <- c(m,cost(m, a, b,lambda))
      j<-j+1
    }
  }

  param <- cbind(Iteration = 1:nrow(param), param)

  return(param)

}


# ################################################# Exploration algorithm #################################################

SREGpm <- function(a, b, iters = length(b), cost,  instgrad, lambda,z) {

  ind<-sample(1:length(b),iters,replace=TRUE)
  a<-a[ind,]
  b<-b[ind]
  d <- ncol(a)

  param <- data.frame(matrix(nrow = 101, ncol = d + 1))
  colnames(param) <- c(colnames(a), "Loss")
  w <- rep(1,2*d)
  w <- w/sum(w)
  x <- z*c(w[1:d]-w[d+1:d])
  m <- x
  param[1, ] <- c(m, cost(x, a, b,lambda))
  k <- 2

  for (i in 1:iters) {
    eta <-sqrt(1/(i*2*d))
    j <- sample(1:d,1)
    instgj <- instgrad(x[j], a[i,j], b[i], lambda)
    w[c(j,j+d)]<-  exp( eta *d* c(-instgj,instgj ))*w[c(j,j+d)]
    w <- w/sum(w)
    x <- z*c(w[1:d]-w[d+1:d])
    m <- ((i-1)*m+x)/i
    if (i%% (iters/100) == 0)
    {
      param[k, ] <- c(m,cost(m, a, b,lambda))
      k<-k+1
    }
  }

  param <- cbind(Iteration = c(1,1:100*(iters/100)), param)

  return(param)

}

SBEGpm <- function(a, b, iters = length(b), cost,  instgrad, lambda,z) {

  ind<-sample(1:length(b),iters,replace=TRUE)
  a<-a[ind,]
  b<-b[ind]
  d <- ncol(a)

  param <- data.frame(matrix(nrow = 101, ncol = d + 1))
  colnames(param) <- c(colnames(a), "Loss")
  w <- rep(1,2*d)
  w <- w/sum(w)
  x <- z*c(w[1:d]-w[d+1:d])
  m <- x
  param[1, ] <- c(m, cost(x, a, b,lambda))
  k <- 2

  for (i in 1:iters) {
    eta <- sqrt(1/(i*2*d))
    A <- sample(1:(2*d),1,prob = w)
    j <- A*(A<=d)+(A-d)*(A>d)
    s <- 1-2*(A>d)
    instgj <- instgrad(x[j], a[i,j], b[i], lambda)
    w[A]<-  exp(- s* eta * instgj/w[A])*w[A]
    w <- (1-eta)*w/sum(w)+eta/(2*d)
    x <- z*c(w[1:d]-w[d+1:d])
    m <- ((i-1)*m+x)/i
    if (i%% (iters/100) == 0)
    {
      param[k, ] <- c(m,cost(m, a, b,lambda))
      k<-k+1
    }
  }

  param <- cbind(Iteration = c(1,1:100*(iters/100)), param)

  return(param)

}

# ################################################# Auxiliary functions #################################################
# 
# # Cost function: regularized hinge loss
# 
hingereg <- function(x, a, b,lambda){
  threshold <- (a %*% x) * b
  cost <- 1- (a  %*% x) * b
  cost[threshold >= 1] <- 0
  return(mean(cost)+lambda*sum(x^2)/2) # standardized dividing by n
}

# The corresponding gradient

gradreg <- function(x, a, b,lambda) {
  threshold <- b * (a %*% x) # define hard-margin SVM
  gradient <- - b* a
  gradient[threshold >= 1] <- 0
  return(colMeans(gradient)+lambda*x) # standardized dividing by n
}

# The instatenuous gradient

instgradreg <- function(x, a, b,lambda) {
  threshold <- b * (a %*% x) # define hard-margin SVM
  gradient <- - b* a
  gradient[threshold >= 1] <- 0
  return(gradient+lambda*x)
}  



# projection for a vector v with non-negative coordinates on the simplex of radius z

pisimplex <- function(v,z=1){
  n <- length(v)
  u <- sort(v, TRUE)
  
  su <- cumsum(u)
  d0 <- max(which(u > (su-z) / (1:n)))
  
  theta <- (su[d0] -z) / d0
  
  w <- pmax(v - theta, 0)
  return(w) 
}
#pib1(x - eta * instgrad(x,a[i,], b[i],lambda),z)

pib1 <- function(x,z=1){ #L1
  v <- abs(x)
  if (sum(v)>z){
    u <- pisimplex(v,z)
    x<-sign(x)*u
  }
  return(x)
}

pib2 <- function(x,z=1){ #L2
  v <- sqrt(sum(x^2))
  if (v>z){
    x<- z*x/v
  }
  return(x)
}


pib1w <- function(x,w,z=1){
  if (sum(abs(x))>z & z != Inf){
    v <- abs(x* w)
    u <- order(-v)
    sx <- cumsum(abs(x)[u])
    sw <- cumsum(1/w[u])
    rho <- max(which(v[u] > (sx-z) / sw))
    theta <- (sx[rho] -z) / sw[rho]
    x<-sign(x)*pmax(abs(x) - theta/w, 0)}
  
  return(x)
}


################################################# Training on MNIST #################################################

rate <- function(param,c){colMeans(c[,1]*( as.matrix(c[,-1]) %*% t(as.matrix(param[, 2:(ncol(param) - 1)])))>0)}
library(RColorBrewer)

# start_time <- Sys.time()
# set.seed(432)
# paramGD <- GDproj(a = as.matrix(train[,-1]),
#                   b = train[,1],
#                   init = rep(0, dim(train[-1])[2]),
#                   iters = 100,  # =1000 Play with the number of iterations
#                   cost = hingereg,
#                   grad = gradreg,
#                   lambda = 1/3) # Play with the regularization parameter
# end_time <- Sys.time()
# diff_gd=end_time - start_time
# 
# 
# start_time <- Sys.time()
# set.seed(432)
# paramGDproj <- GDproj(a = as.matrix(train[,-1]),
#                       b = train[,1],
#                       init = rep(0, dim(train[-1])[2]),
#                       iters = 100,  # =1000 Play with the number of iterations
#                       cost = hingereg,
#                       grad = gradreg,
#                       lambda = 1/3,
#                       z = 100) # play with diameters
# end_time <- Sys.time()
# diff_gdp=end_time - start_time
# 
# 
# start_time <- Sys.time()
# set.seed(432)
# paramSGD <- SGDproj(a = as.matrix(train[,-1]),
#                     b = train[,1],
#                     init = rep(0, dim(train[-1])[2]),
#                     iters = 10000,  # =1000 Play with the number of iterations
#                     cost = hingereg,
#                     instgrad = instgradreg,
#                     lambda = 1/3) # Play with the regularization parameter
# end_time <- Sys.time()
# diff_sgd=end_time - start_time






##################################################################################################
####question4#####################################################################################
##################################################################################################

###4.1#### SGDproj vs SMD##########################################

start_time <- Sys.time()
set.seed(432)
paramSGDproj <- SGDproj(a = as.matrix(train[,-1]),
                        b = train[,1],
                        init = rep(0, dim(train[-1])[2]),
                        iters = 10000,  # =1000 Play with the number of iterations
                        cost = hingereg,
                        instgrad = instgradreg,
                        lambda = 1/10,
                        z = 100) # Play with the diameter of the l1 ball
end_time <- Sys.time()
diff_sgdproj=end_time - start_time



start_time <- Sys.time()
set.seed(432)
paramSMDproj <- SMDproj(a = as.matrix(train[,-1]),
                        b = train[,1],
                        init = rep(0, dim(train[-1])[2]),
                        iters = 10000,  # =1000 Play with the number of iterations
                        cost = hingereg,
                        instgrad = instgradreg,
                        lambda = 0,
                        z=100) # Play with the diameter of the l1 ball
end_time <- Sys.time()
diff_smd=end_time - start_time


rateSGDproj <- rate(paramSGDproj,test)
rateSMDproj <- rate(paramSMDproj,test)


par(mfrow=c(1,2))

matplot(paramSMDproj[,1],cbind(1-rateSGDproj, 1-rateSMDproj),type="l",col=c(2,4),log="xy",xlab="Iterations",ylab="Accuracy",
        main = "Accuracy for projected SGD and SMD",lwd=2)
legend(x=600,y=1, inset=.02, title="Algorithm",
       c("projected SGD", "projected SMD"), fill=c(2,4), cex=0.8)


matplot(paramSMDproj[,1],cbind(paramSGDproj$Loss, paramSMDproj$Loss),type="l",col=c(2,4),log="xy",xlab="Iterations",ylab="Loss",
        main = "Loss for projected SGD and SMD",lwd=2)
legend(x=600,y=1, inset=.02, title="Algorithm",
       c("projected SGD", "projected SMD"), fill=c(2,4), cex=0.8)



###4.2#### SGDproj vs SEG##########################################

start_time <- Sys.time()
set.seed(432)
paramSGDproj <- SGDproj(a = as.matrix(train[,-1]),
                        b = train[,1],
                        init = rep(0, dim(train[-1])[2]),
                        iters = 10000,  # =1000 Play with the number of iterations
                        cost = hingereg,
                        instgrad = instgradreg,
                        lambda = 1/10,
                        z = 100) # Play with the diameter of the l1 ball
end_time <- Sys.time()
end_time - start_time



start_time <- Sys.time()
set.seed(432)
paramSEGpm <- SEGpm(a = as.matrix(train[,-1]),
                    b = train[,1],
                    iters = 10000,  # =1000 Play with the number of iterations
                    cost = hingereg,
                    instgrad = instgradreg,
                    lambda = 0,
                    z=100) # Play with the diameter of the l1 ball
end_time <- Sys.time()
diff_seg=end_time - start_time


rateSGDproj <- rate(paramSGDproj,test)
rateSEGpm <- rate(paramSEGpm,test)

par(mfrow=c(1,2))

matplot(paramSGDproj[,1],cbind(1-rateSGDproj, 1-rateSEGpm),type="l",col=c(2,4),log="xy",xlab="Iterations",ylab="Accuracy",
        main = "Accuracy for projected SGD and SED",lwd=2)
legend(x=600,y=1, inset=.02, title="Algorithm",c("projected SGD", "SED"), fill=c(2,4), cex=0.8)


matplot(paramSGDproj[,1],cbind(paramSGDproj$Loss, paramSEGpm$Loss),type="l",col=c(2,4),log="xy",xlab="Iterations",ylab="Loss",
        main = "Loss for projected SGD and SED",lwd=2)
legend(x=600,y=1, inset=.02, title="Algorithm", c("projected SGD", "SED"), fill=c(2,4), cex=0.8)


######4.3######SGDproj vs Adagrad #################################""""

start_time <- Sys.time()
set.seed(432)
paramSGDproj <- SGDproj(a = as.matrix(train[,-1]),
                        b = train[,1],
                        init = rep(0, dim(train[-1])[2]),
                        iters = 10000,  # =1000 Play with the number of iterations
                        cost = hingereg,
                        instgrad = instgradreg,
                        lambda = 1/10,
                        z = 100) # Play with the diameter of the l1 ball
end_time <- Sys.time()
end_time - start_time


start_time <- Sys.time()
set.seed(432)
paramAdaproj <- Adaproj(a = as.matrix(train[,-1]),
                        b = train[,1],
                        init = rep(0, dim(train[-1])[2]),
                        iters = 10000,  # =1000 Play with the number of iterations
                        cost = hingereg,
                        instgrad = instgradreg,
                        lambda = 0,
                        z=100) # Play with the diameter of the l1 ball
end_time <- Sys.time()
diff_ada=end_time - start_time

rateSGDproj <- rate(paramSGDproj,test)
rateAdaproj <- rate(paramAdaproj,test)

par(mfrow=c(1,2))

matplot(paramSGDproj[,1],cbind(1-rateSGDproj, 1-rateAdaproj),type="l",col=c(2,4),log="xy",xlab="Iterations",ylab="Accuracy",
        main = "Accuracy for projected SGD and Stochastic AdaGrad ",lwd=2)
legend(x=600,y=1, inset=.02, title="Algorithm",c("projected SGD", "Stochastic AdaGrad"), fill=c(2,4), cex=0.8)


matplot(paramSGDproj[,1],cbind(paramSGDproj$Loss, paramAdaproj$Loss),type="l",col=c(2,4),log="xy",xlab="Iterations",ylab="Loss",
        main = "Loss for projected SGD and Stochastice AdaGrad",lwd=2)
legend(x=600,y=1, inset=.02, title="Algorithm", c("projected SGD", "Stochastic AdaGrad"), fill=c(2,4), cex=0.8)



####################################################################################
####question5 ONS##################################################################
###################################################################################


########ONS with diff gamma vs SGDproj ##################################"
start_time <- Sys.time()
set.seed(432)
paramSGDproj <- SGDproj(a = as.matrix(train[,-1]),
                        b = train[,1],
                        init = rep(0, dim(train[-1])[2]),
                        iters = 10000,  # =1000 Play with the number of iterations
                        cost = hingereg,
                        instgrad = instgradreg,
                        lambda = 1/10,
                        z = 100) # Play with the diameter of the l1 ball
end_time <- Sys.time()
end_time - start_time



start_time <- Sys.time()
set.seed(432)
paramONS_1 <- ONS(a = as.matrix(train[,-1]),
                b = train[,1],
                iters = 10000,  # =1000 Play with the number of iterations
                init = rep(0, dim(train[-1])[2]),
                cost = hingereg,
                instgrad = instgradreg,
                lambda = 1/10, #1/3
                gamm = 1/2,
                z=100)
end_time <- Sys.time()
diff_ons_1=end_time - start_time


start_time <- Sys.time()
set.seed(432)
paramONS_2 <- ONS(a = as.matrix(train[,-1]),
                b = train[,1],
                iters = 10000,  # =1000 Play with the number of iterations
                init = rep(0, dim(train[-1])[2]),
                cost = hingereg,
                instgrad = instgradreg,
                lambda = 1/10, #1/3
                gamm = 1/3,
                z=100)
end_time <- Sys.time()
diff_ons_2=end_time - start_time


start_time <- Sys.time()
set.seed(432)
paramONS_3 <- ONS(a = as.matrix(train[,-1]),
                  b = train[,1],
                  iters = 10000,  # =1000 Play with the number of iterations
                  init = rep(0, dim(train[-1])[2]),
                  cost = hingereg,
                  instgrad = instgradreg,
                  lambda = 1/10, #1/3
                  gamm = 1/10,
                  z=100)
end_time <- Sys.time()
diff_ons_3=end_time - start_time

start_time <- Sys.time()
set.seed(432)
paramONS_4 <- ONS(a = as.matrix(train[,-1]),
                  b = train[,1],
                  iters = 10000,  # =1000 Play with the number of iterations
                  init = rep(0, dim(train[-1])[2]),
                  cost = hingereg,
                  instgrad = instgradreg,
                  lambda = 1/10, #1/3
                  gamm = 1/20,
                  z=100)
end_time <- Sys.time()
diff_ons_4=end_time - start_time



rateONS_1 <- rate(paramONS_1,test)
rateONS_2 <- rate(paramONS_2,test)
rateONS_3 <- rate(paramONS_3,test)
rateONS_4 <- rate(paramONS_4,test)

par(mfrow=c(1,2))

matplot(paramSGDproj[,1],cbind(1-rateSGDproj,1-rateONS_1,1-rateONS_2,1-rateONS_3,1-rateONS_4),type="l",col=brewer.pal(5,"Dark2"),log="xy",xlab="Iterations",ylab="Accuracy",
        main = "Accuracy for projected SGD and ONS",lwd=2)
legend("topright", inset=.02, title="Algorithms",
       c("SGDproj","ONS gamma=1/2","ONS gamma=1/3","ONS gamma=1/10","ONS gamma=1/20"), fill=brewer.pal(5,"Dark2"), cex=0.8)

matplot(paramSGDproj[,1],cbind(paramSGDproj$Loss,paramONS_1$Loss,paramONS_2$Loss,paramONS_3$Loss,paramONS_4$Loss),type="l",col=brewer.pal(5,"Dark2"),log="xy",xlab="Iterations",ylab="Accuracy",
        main = "Loss for projected SGD and ONS",lwd=2)
legend("topright", inset=.02, title="Algorithms",
       c("SGDproj","ONS gamma=1/2","ONS gamma=1/3","ONS gamma=1/10","ONS gamma=1/20"), fill=brewer.pal(5,"Dark2"), cex=0.8)




####comparison ONS and the other methods#########################################"


matplot(paramSGDproj[,1],cbind(1-rateSGDproj,1-rateSMDproj,1-rateSEGpm,1-rateAdaproj,1-rateONS_2),type="l",col=brewer.pal(5,"Set1"),log="xy",xlab="Iterations",ylab="Accuracy",
        main = "SVM on Test Set from MNIST",lwd=2)
legend("bottomleft", inset=.02, title="Algorithms",
       c("SGDproj","SMDproj","SEGpm","Adaproj","ONS"), fill=brewer.pal(5,"Set1"), cex=0.8)



####################################################################################
####question6 ##############################################
###################################################################################"

start_time <- Sys.time()
set.seed(100)
paramSREGpm <- SREGpm(a = as.matrix(train[,-1]),
                      b = train[,1],
                      iters = 100000,  # =1000 Play with the number of iterations
                      cost = hingereg,
                      instgrad = instgradreg,
                      lambda = 0,
                      z=10) # Play with the diameter of the l1 ball
end_time <- Sys.time()
end_time - start_time


start_time <- Sys.time()
set.seed(100)
paramSBEGpm <- SBEGpm(a = as.matrix(train[,-1]),
                      b = train[,1],
                      iters = 100000,  # =1000 Play with the number of iterations
                      cost = hingereg,
                      instgrad = instgradreg,
                      lambda = 0,
                      z=10) # Play with the diameter of the l1 ball
end_time <- Sys.time()
end_time - start_time

rateSREGpm <- rate(paramSREGpm,test)
rateSBEGpm <- rate(paramSBEGpm,test)

regret_bound<- function(T){
  regret <- sqrt(200 * T * log(785))
  return(regret/T)
}

# show exploration curve with regret bound of rateSREGpm
regret <- regret_bound(paramSREGpm[,1])
matplot(paramSREGpm[,1],cbind(1-rateSREGpm, 1-rateSBEGpm, regret),type="l",col=brewer.pal(4,"RdYlGn"),log="xy",xlab="Iterations",ylab="Accuracy",
        main = "SVM on Test Set from MNIST",lwd=2)
legend("bottomleft", inset=.02, title="Algorithms", c("SREGpm", "SBEGpm", "Regret/T"), fill=brewer.pal(4,"RdYlGn"), cex=0.8)



####################################################################################
####question7 ##############################################
###################################################################################"


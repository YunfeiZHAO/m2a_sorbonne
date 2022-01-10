################################################# Loading data #############################################

#  MNIST data downloadable at https://pjreddie.com/projects/mnist-in-csv/


setwd("~/DropSU/Lecture/LPSM/OCO/MNIST")
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
    m <- ((i-1)*m + x)/i
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


# Stochastic BOA +_

SBOApm <- function(a, b, iters = length(b), cost,  instgrad, lambda,z) {
  
  ind<-sample(1:length(b),iters,replace=TRUE)
  a<-a[ind,]
  b<-b[ind]
  d<-ncol(a)
  
  param <- data.frame(matrix(nrow = 101, ncol = d + 1))
  colnames(param) <- c(colnames(a), "Loss")
  w <- rep(1,2*d)
  eta <- rep(1,2*d)
  w <- w/sum(w)
  x <- z*c(w[1:d]-w[d+1:d])
  m = x
  param[1, ] <- c(m, cost(m, a, b,lambda))
  j<-2
  theta <- 0
  
  for (i in 1:iters) {
    instg <- instgrad( x, a[i,], b[i], lambda)
    instgpm <- c(instg,-instg)
    instgbar <- rep(sum(w *instgpm),2*d)
    theta <- theta-  instgpm - eta *(instgpm-instgbar)^2
    eta <- sqrt(eta^2/(1+eta^2*(instgpm-instgbar)^2))
    w <- eta*exp(eta *theta)/sum(eta*exp(eta* theta))
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

# Adam projected

Adamproj <- function(a, b, init, iters = length(b), cost,  instgrad, lambda,z=Inf ) {
  ind<-sample(1:length(b),iters,replace=TRUE)
  a<-a[ind,]
  b<-b[ind]
  d <- dim(a)[2]
  
  param <- data.frame(matrix(nrow = 101, ncol = length(init) + 1))
  colnames(param) <- c(colnames(a), "Loss")
  s <- rep(.0001,d)
  mm <- rep(0,d)
  x <- c(init)
  m <- x
  param[1, ] <- c(m, cost(m, a, b,lambda))
  j<-2
  
  for (i in 1:iters) { 
    eta <- sqrt(1/iters)
    mm <- 0.9*mm + (1-0.9)*instgrad(x, a[i,], b[i], lambda)
    s <- 0.999*s + (1-.999)*instgrad(x, a[i,], b[i], lambda)^2
    y <- x -  eta*1/sqrt(s) * mm  
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

# EKF, logistic loss is used

EKF <- function(a, b, init, iters = length(b), cost,  instgrad, lambda,gamm,z) {
  
  ind<-sample(1:length(b),iters,replace=TRUE)
  a <- a[ind,]
  b <- b[ind]
  d <- length(init)
  param <- data.frame(matrix(nrow = 101, ncol = d + 1))
  colnames(param) <- c(colnames(a), "Loss")
  x <- c(init)
  P <- diag(rep(1,d))
  param[1, ] <- c(x, cost(x, a, b,lambda))
  j <- 2
  for (i in 1:iters) { 
    p_hat <- 1/(1 + exp(-b[i] *(t(x) %*% a[i,])[1]))
    Pa <- P %*% a[i,]
    P <- P -  Pa%*% t(Pa)/(1/(p_hat * (1 - p_hat)) + c(t(a[i,]) %*% Pa))
    x<- x + P%*% a[i,] * b[i] * (1 - p_hat)
    if (i%% (iters/100) == 0)
    {
      param[j, ] <- c(x,cost(x, a, b,lambda))
      j<-j+1
    }
  }
  
  param <- cbind(Iteration = 1:nrow(param), param)
  
  return(param)
  
}

################################################# Exploration algorithm #################################################

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

################################################# Auxiliary functions #################################################

# Cost function: regularized hinge loss

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

pib1 <- function(x,z=1){
  v <- abs(x)
  if (sum(v)>z){
    u <- pisimplex(v,z)
    x<-sign(x)*u
  }
  return(x)
}

pib2 <- function(x,z=1){
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

start_time <- Sys.time()
paramGD <- GDproj(a = as.matrix(train[,-1]),
                  b = train[,1],
                  init = rep(0, dim(train[-1])[2]),
                  iters = 100,  # =1000 Play with the number of iterations
                  cost = hingereg,
                  grad = gradreg,
                  lambda = 1/3) # Play with the regularization parameter
end_time <- Sys.time()
end_time - start_time


start_time <- Sys.time()
paramGDproj <- GDproj(a = as.matrix(train[,-1]),
                      b = train[,1],
                      init = rep(0, dim(train[-1])[2]),
                      iters = 100,  # =1000 Play with the number of iterations
                      cost = hingereg,
                      grad = gradreg,
                      lambda = 1/3,
                      z = 100) # Play with the regularization parameter
end_time <- Sys.time()
end_time - start_time


start_time <- Sys.time()
set.seed(100)
paramSGD <- SGDproj(a = as.matrix(train[,-1]),
                    b = train[,1],
                    init = rep(0, dim(train[-1])[2]),
                    iters = 10000,  # =1000 Play with the number of iterations
                    cost = hingereg,
                    instgrad = instgradreg,
                    lambda = 1/3) # Play with the regularization parameter
end_time <- Sys.time()
end_time - start_time


start_time <- Sys.time()
set.seed(100)
paramSGDproj <- SGDproj(a = as.matrix(train[,-1]),
                        b = train[,1],
                        init = rep(0, dim(train[-1])[2]),
                        iters = 10000,  # =1000 Play with the number of iterations
                        cost = hingereg,
                        instgrad = instgradreg,
                        lambda = 1/3,
                        z = 100) # Play with the diameter of the l1 ball
end_time <- Sys.time()
end_time - start_time


start_time <- Sys.time()
set.seed(100)
paramSMDproj <- SMDproj(a = as.matrix(train[,-1]),
                        b = train[,1],
                        init = rep(0, dim(train[-1])[2]),
                        iters = 10000,  # =1000 Play with the number of iterations
                        cost = hingereg,
                        instgrad = instgradreg,
                        lambda = 0,
                        z=100) # Play with the diameter of the l1 ball
end_time <- Sys.time()
end_time - start_time

start_time <- Sys.time()
set.seed(100)
paramSEGpm <- SEGpm(a = as.matrix(train[,-1]),
                    b = train[,1],
                    iters = 10000,  # =1000 Play with the number of iterations
                    cost = hingereg,
                    instgrad = instgradreg,
                    lambda = 0,
                    z=100) # Play with the diameter of the l1 ball
end_time <- Sys.time()
end_time - start_time

start_time <- Sys.time()
set.seed(100)
paramSBOApm <- SBOApm(a = as.matrix(train[,-1]),
                      b = train[,1],
                      iters = 10000,  # =1000 Play with the number of iterations
                      cost = hingereg,
                      instgrad = instgradreg,
                      lambda = 0,
                      z=100) # Play with the diameter of the l1 ball
end_time <- Sys.time()
end_time - start_time

start_time <- Sys.time()
set.seed(100)
paramAdaproj <- Adaproj(a = as.matrix(train[,-1]),
                        b = train[,1],
                        init = rep(0, dim(train[-1])[2]),
                        iters = 10000,  # =1000 Play with the number of iterations
                        cost = hingereg,
                        instgrad = instgradreg,
                        lambda = 0,
                        z=100) # Play with the diameter of the l1 ball
end_time <- Sys.time()
end_time - start_time

start_time <- Sys.time()
set.seed(100)
paramAdamproj <- Adamproj(a = as.matrix(train[,-1]),
                          b = train[,1],
                          init = rep(0, dim(train[-1])[2]),
                          iters = 10000,  # =1000 Play with the number of iterations
                          cost = hingereg,
                          instgrad = instgradreg,
                          lambda = 0,
                          z=100) # Play with the diameter of the l1 ball
end_time <- Sys.time()
end_time - start_time



start_time <- Sys.time()
set.seed(100)
paramONS <- ONS(a = as.matrix(train[,-1]),
                b = train[,1],
                iters = 10000,  # =1000 Play with the number of iterations
                init = rep(0, dim(train[-1])[2]),
                cost = hingereg,
                instgrad = instgradreg,
                lambda = 1/3,
                gamm = 1/8,
                z=100)
end_time <- Sys.time()
end_time - start_time

start_time <- Sys.time()
set.seed(1)
paramEKF <- EKF(a = as.matrix(train[,-1]),
                b = train[,1],
                iters = 10000,  # =1000 Play with the number of iterations
                init = rep(0, dim(train[-1])[2]),
                cost = hingereg,
                instgrad = instgradreg,
                lambda=0)
end_time <- Sys.time()
end_time - start_time


start_time <- Sys.time()
set.seed(100)
paramSREGpm <- SREGpm(a = as.matrix(train[,-1]),
                      b = train[,1],
                      iters = 100000,  # =1000 Play with the number of iterations
                      cost = hingereg,
                      instgrad = instgradreg,
                      lambda = 0,
                      z=100) # Play with the diameter of the l1 ball
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
                      z=100) # Play with the diameter of the l1 ball
end_time <- Sys.time()
end_time - start_time

################################################# Rate of accuracy on the test dataset #################################################

rate <- function(param,c){colMeans(c[,1]*( as.matrix(c[,-1]) %*% t(as.matrix(param[, 2:(ncol(param) - 1)])))>0)}

rateGD <- rate(paramGD,test)
rateGDproj <- rate(paramGDproj,test)
rateSGD <- rate(paramSGD,test)
rateSGDproj <- rate(paramSGDproj,test)
rateSMDproj <- rate(paramSMDproj,test)
rateSEGpm <- rate(paramSEGpm,test)
rateSBOApm <- rate(paramSBOApm,test)
rateAdaproj <- rate(paramAdaproj,test)
rateAdamproj <- rate(paramAdamproj,test)
rateONS <- rate(paramONS,test)
rateEKF <- rate(paramEKF,test)
rateSREGpm <- rate(paramSREGpm,test)
rateSBEGpm <- rate(paramSBEGpm,test)

################################################# Graph #################################################

library(RColorBrewer)


matplot(paramSGD[,1],cbind(1-rateSGD,1-rateSGDproj,1-rateSMDproj,1-rateSEGpm,1-rateAdaproj,1-rateSBOApm,1-rateAdamproj,1-rateONS,1-rateEKF,1-rateSREGpm,1-rateSBEGpm),type="l",col=brewer.pal(11,"RdYlGn"),log="xy",xlab="Iterations",ylab="Accuracy",
        main = "SVM on Test Set from MNIST",lwd=2)
legend("bottomleft", inset=.02, title="Algorithms",
       c("SGD","SGDproj","SMDproj","SEGpm","Adaproj","BOA","Adamproj","ONS","EKF","SREGpm","SBEGpm"), fill=brewer.pal(11,"RdYlGn"), cex=0.8)


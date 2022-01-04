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
# At this step we gives the label -1 to the observatios whose labels was not 0 in the original dataset and the label 1 nto the observations with label 0.
# So that the aim is to seperate 0 from the other.
#Beside we also scale the 784 variable into [0,1] and we add a intercept equals to 1 (at the last term)

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





##################################################################################################
####Question2#####################################################################################
##################################################################################################


#train on the mnist dataset  GD
start_time <- Sys.time()

paramGD_1 <- GDproj(a = as.matrix(train[,-1]),
                    b = train[,1],
                    init = rep(0, dim(train[-1])[2]),
                    iters = 100,  
                    cost = hingereg,
                    grad = gradreg,
                    lambda = 1/2) 

end_time <- Sys.time()
end_time - start_time

paramGD_2 <- GDproj(a = as.matrix(train[,-1]),
                    b = train[,1],
                    init = rep(0, dim(train[-1])[2]),
                    iters = 100,  
                    cost = hingereg,
                    grad = gradreg,
                    lambda = 1/3) 

paramGD_3 <- GDproj(a = as.matrix(train[,-1]),
                    b = train[,1],
                    init = rep(0, dim(train[-1])[2]),
                    iters = 100,  
                    cost = hingereg,
                    grad = gradreg,
                    lambda = 1/4) 

paramGD_4 <- GDproj(a = as.matrix(train[,-1]),
                    b = train[,1],
                    init = rep(0, dim(train[-1])[2]),
                    iters = 100,  
                    cost = hingereg,
                    grad = gradreg,
                    lambda = 1/5) 

start_time <- Sys.time()
paramGD_5 <- GDproj(a = as.matrix(train[,-1]),
                    b = train[,1],
                    init = rep(0, dim(train[-1])[2]),
                    iters = 100,
                    cost = hingereg,
                    grad = gradreg,
                    lambda = 1/10) 

end_time <- Sys.time()
end_time - start_time



################################################# Rate of accuracy on the test dataset #################################################

rate <- function(param,c){colMeans(c[,1]*( as.matrix(c[,-1]) %*% t(as.matrix(param[, 2:(ncol(param) - 1)])))>0)}

rateGD_1 <- rate(paramGD_1,test)
rateGD_2 <- rate(paramGD_2,test)
rateGD_3 <- rate(paramGD_3,test)
rateGD_4 <- rate(paramGD_4,test)
rateGD_5 <- rate(paramGD_5,test)


matplot(paramGD_1[,1],cbind(1-rateGD_1),type="l",col=brewer.pal(11,"RdYlGn"),log="xy",xlab="Iterations",ylab="Accuracy",
        main = "SVM on Test Set from MNIST",lwd=2)
legend("bottomleft", inset=.02, title="Algorithms",
       c("GD"), fill=brewer.pal(11,"RdYlGn"), cex=0.8)

par(mfrow=c(1,2))

matplot(paramGD_1[,1],cbind(1-rateGD_1, 1-rateGD_2, 1-rateGD_3, 1-rateGD_4 ,1-rateGD_5),type="l",col=brewer.pal(5,"RdYlGn"),log="xy",xlab="Iterations",ylab="Accuracy",
        main = "Accuracy for unconstrained  GD",lwd=2)
legend(x=10,y=1, inset=.02, title="Lamnda",
       c("lambda = 1/2", "lambda = 1/3", "lambda = 1/4", "lambda = 1/5", "lambda = 1/10"), fill=brewer.pal(5,"RdYlGn"), cex=0.8)



matplot(paramGD_1[,1],cbind(paramGD_1$Loss, paramGD_2$Loss, paramGD_3$Loss, paramGD_4$Loss ,paramGD_5$Loss),type="l",col=brewer.pal(5,"RdYlGn"),log="xy",xlab="Iterations",ylab="Loss",
        main = "Loss",lwd=2)
legend(x=10,y=50, inset=.02, title="Lambda",
       c("lambda = 1/2", "lambda = 1/3", "lambda = 1/4", "lambda = 1/5", "lambda = 1/10"), fill=brewer.pal(5,"RdYlGn"), cex=0.8)


################################################# projected GD #################################################


#train on the mnist dataset  GD

start_time <- Sys.time()
paramGDproj_1 <- GDproj(a = as.matrix(train[,-1]),
                        b = train[,1],
                        init = rep(0, dim(train[-1])[2]),
                        iters = 50,  
                        cost = hingereg,
                        grad = gradreg,
                        lambda = 1/3,
                        z = 100) 
end_time <- Sys.time()
end_time - start_time

start_time <- Sys.time()
paramGDproj_2 <- GDproj(a = as.matrix(train[,-1]),
                        b = train[,1],
                        init = rep(0, dim(train[-1])[2]),
                        iters = 100,  
                        cost = hingereg,
                        grad = gradreg,
                        lambda = 1/3,
                        z = 100) 
end_time <- Sys.time()
end_time - start_time

start_time <- Sys.time()
paramGDproj_3 <- GDproj(a = as.matrix(train[,-1]),
                        b = train[,1],
                        init = rep(0, dim(train[-1])[2]),
                        iters = 200, 
                        cost = hingereg,
                        grad = gradreg,
                        lambda = 1/3,
                        z = 100) 
end_time <- Sys.time()
end_time - start_time

start_time <- Sys.time()
paramGDproj_4 <- GDproj(a = as.matrix(train[,-1]),
                        b = train[,1],
                        init = rep(0, dim(train[-1])[2]),
                        iters = 500,  
                        cost = hingereg,
                        grad = gradreg,
                        lambda = 1/3,
                        z = 100) 
end_time <- Sys.time()
end_time - start_time

start_time <- Sys.time()
paramGDproj_5 <- GDproj(a = as.matrix(train[,-1]),
                        b = train[,1],
                        init = rep(0, dim(train[-1])[2]),
                        iters = 1000,  
                        cost = hingereg,
                        grad = gradreg,
                        lambda = 1/3,
                        z = 100) 
end_time <- Sys.time()
end_time - start_time

# Rate of accuracy on the test dataset

rateGDproj_1 <- rate(paramGDproj_1,test)
rateGDproj_2 <- rate(paramGDproj_2,test)
rateGDproj_3 <- rate(paramGDproj_3,test)
rateGDproj_4 <- rate(paramGDproj_4,test)
rateGDproj_5 <- rate(paramGDproj_5,test)


par(mfrow=c(1,2))

matplot(paramGDproj_5[,1],cbind(1-rateGDproj_5),type="l",col=brewer.pal(11,"RdYlGn"),log="xy",xlab="Iterations",ylab="Accuracy",
        main = "Accuracy for projected(L1 ball) GD",lwd=2)
legend(x=100,y=1, inset=.02, title="Iterations",
       c( "T = 1000"), fill=brewer.pal(11,"RdYlGn"), cex=0.8)

matplot(paramGDproj_5[,1],cbind(paramGDproj_5$Loss),type="l",col=brewer.pal(11,"RdYlGn"),log="xy",xlab="Iterations",ylab="Loss",
        main = "Loss",lwd=2)
legend(x=100,y=10, inset=.02, title="Iterations",
       c("T = 1000"), fill=brewer.pal(11,"RdYlGn"), cex=0.8)




##################################################################################################
####Question3#####################################################################################
##################################################################################################

#################################################  SGD ############################################
#train on the mnist dataset  SGD



start_time <- Sys.time()
set.seed(100)
paramSGD_1 <- SGDproj(a = as.matrix(train[,-1]),
                      b = train[,1],
                      init = rep(0, dim(train[-1])[2]),
                      iters = 10000,  
                      cost = hingereg,
                      instgrad = instgradreg,
                      lambda = 1/2) 
end_time <- Sys.time()
end_time - start_time


start_time <- Sys.time()
set.seed(100)
paramSGD_2 <- SGDproj(a = as.matrix(train[,-1]),
                      b = train[,1],
                      init = rep(0, dim(train[-1])[2]),
                      iters = 10000,  
                      cost = hingereg,
                      instgrad = instgradreg,
                      lambda = 1/3) 
end_time <- Sys.time()
end_time - start_time



start_time <- Sys.time()
set.seed(100)
paramSGD_3 <- SGDproj(a = as.matrix(train[,-1]),
                      b = train[,1],
                      init = rep(0, dim(train[-1])[2]),
                      iters = 10000,
                      cost = hingereg,
                      instgrad = instgradreg,
                      lambda = 1/4) 
end_time <- Sys.time()
end_time - start_time



start_time <- Sys.time()
set.seed(100)
paramSGD_4 <- SGDproj(a = as.matrix(train[,-1]),
                      b = train[,1],
                      init = rep(0, dim(train[-1])[2]),
                      iters = 10000, 
                      cost = hingereg,
                      instgrad = instgradreg,
                      lambda = 1/5) 
end_time <- Sys.time()
end_time - start_time


start_time <- Sys.time()
set.seed(100)
paramSGD_5 <- SGDproj(a = as.matrix(train[,-1]),
                      b = train[,1],
                      init = rep(0, dim(train[-1])[2]),
                      iters = 10000,  
                      cost = hingereg,
                      instgrad = instgradreg,
                      lambda = 1/10) 
end_time <- Sys.time()
end_time - start_time

# Rate of accuracy on the test dataset

rateSGD_1 <- rate(paramSGD_1,test)
rateSGD_2 <- rate(paramSGD_2,test)
rateSGD_3 <- rate(paramSGD_3,test)
rateSGD_4 <- rate(paramSGD_4,test)
rateSGD_5 <- rate(paramSGD_5,test)


par(mfrow=c(1,2))

matplot(paramSGD_1[,1],cbind(1-rateSGD_1, 1-rateSGD_2, 1-rateSGD_3, 1-rateSGD_4 ,1-rateSGD_5),type="l",col=brewer.pal(5,"RdYlGn"),log="xy",xlab="Iterations",ylab="Accuracy",
        main = "Accuracy for unconstrained  SGD",lwd=2)
legend(x=100,y=1, inset=.02, title="Lamnda",
       c("lambda = 1/2", "lambda = 1/3", "lambda = 1/4", "lambda = 1/5", "lambda = 1/10"), fill=brewer.pal(5,"RdYlGn"), cex=0.8)



matplot(paramSGD_1[,1],cbind(paramSGD_1$Loss, paramSGD_2$Loss, paramSGD_3$Loss, paramSGD_4$Loss ,paramSGD_5$Loss),type="l",col=brewer.pal(5,"RdYlGn"),log="xy",xlab="Iterations",ylab="Loss",
        main = "Loss",lwd=2)
legend(x=600,y=2.3, inset=.02, title="Lambda",
       c("lambda = 1/2", "lambda = 1/3", "lambda = 1/4", "lambda = 1/5", "lambda = 1/10"), fill=brewer.pal(5,"RdYlGn"), cex=0.8)




################################################# Projection  SGD #################################################


start_time <- Sys.time()
set.seed(100)
paramSGDproj <- SGDproj(a = as.matrix(train[,-1]),
                        b = train[,1],
                        init = rep(0, dim(train[-1])[2]),
                        iters = 10000,  
                        cost = hingereg,
                        instgrad = instgradreg,
                        lambda = 1/3,
                        z = 100) 
end_time <- Sys.time()
end_time - start_time


rateSGDproj <- rate(paramSGDproj,test)


matplot(paramGD[,1],cbind(1-rateSGDproj, 1-rateGDproj_2),type="l",col=brewer.pal(11,"RdYlGn"),log="xy",xlab="Iterations",ylab="Accuracy",
        main = "Accuracy for projected SGD and GD",lwd=2)
legend(x=10,y=1, inset=.02, title="Algorithm",
       c("projected SGD", "projected GD"), fill=brewer.pal(11,"RdYlGn"), cex=0.8)


matplot(paramGD[,1],cbind(paramSGDproj$Loss, paramGDproj_2$Loss),type="l",col=brewer.pal(11,"RdYlGn"),log="xy",xlab="Iterations",ylab="Loss",
        main = "Loss for projected SGD and GD",lwd=2)
legend(x=10,y=10, inset=.02, title="Algorithm",
       c("projected SGD", "projected GD"), fill=brewer.pal(11,"RdYlGn"), cex=0.8)




################################################# Projection  SGD with different Z #################################################

start_time <- Sys.time()
set.seed(100)
paramSGDproj_1 <- SGDproj(a = as.matrix(train[,-1]),
                          b = train[,1],
                          init = rep(0, dim(train[-1])[2]),
                          iters = 10000,  
                          cost = hingereg,
                          instgrad = instgradreg,
                          lambda = 1/3,
                          z = 10) 
end_time <- Sys.time()
end_time - start_time

start_time <- Sys.time()
set.seed(100)
paramSGDproj_2 <- SGDproj(a = as.matrix(train[,-1]),
                          b = train[,1],
                          init = rep(0, dim(train[-1])[2]),
                          iters = 10000,  
                          cost = hingereg,
                          instgrad = instgradreg,
                          lambda = 1/3,
                          z = 50) 
end_time <- Sys.time()
end_time - start_time

start_time <- Sys.time()
set.seed(100)
paramSGDproj_3 <- SGDproj(a = as.matrix(train[,-1]),
                          b = train[,1],
                          init = rep(0, dim(train[-1])[2]),
                          iters = 10000,  
                          cost = hingereg,
                          instgrad = instgradreg,
                          lambda = 1/3,
                          z = 100) 
end_time <- Sys.time()
end_time - start_time

start_time <- Sys.time()
set.seed(100)
paramSGDproj_4 <- SGDproj(a = as.matrix(train[,-1]),
                          b = train[,1],
                          init = rep(0, dim(train[-1])[2]),
                          iters = 10000,  
                          cost = hingereg,
                          instgrad = instgradreg,
                          lambda = 1/3,
                          z = 200) 
end_time <- Sys.time()
end_time - start_time


start_time <- Sys.time()
set.seed(100)
paramSGDproj_5 <- SGDproj(a = as.matrix(train[,-1]),
                          b = train[,1],
                          init = rep(0, dim(train[-1])[2]),
                          iters = 10000,  
                          cost = hingereg,
                          instgrad = instgradreg,
                          lambda = 1/3,
                          z = 300) 
end_time <- Sys.time()
end_time - start_time


rateSGDproj_1 <- rate(paramSGDproj_1,test)
rateSGDproj_2 <- rate(paramSGDproj_2,test)
rateSGDproj_3 <- rate(paramSGDproj_3,test)
rateSGDproj_4 <- rate(paramSGDproj_4,test)
rateSGDproj_5 <- rate(paramSGDproj_5,test)

matplot(paramSGD_1[,1],cbind(1-rateSGDproj_1, 1-rateSGDproj_2, 1-rateSGDproj_3, 1-rateSGDproj_4, 1-rateSGDproj_5),type="l",col=brewer.pal(11,"RdYlGn"),log="xy",xlab="Iterations",ylab="Accuracy",
        main = "Accuracy for projected SGD for different diameters",lwd=2)
legend(x=1000,y=1, inset=.02, title="Diameter",
       c("Z = 10", "Z = 50", "Z = 100", "Z = 200" ,"Z = 300"), fill=brewer.pal(11,"RdYlGn"), cex=0.8)


matplot(paramSGD_1[,1],cbind(paramSGDproj_1$Loss, paramSGDproj_2$Loss, paramSGDproj_3$Loss, paramSGDproj_4$Loss, paramSGDproj_5$Loss ),type="l",col=brewer.pal(11,"RdYlGn"),log="xy",xlab="Iterations",ylab="Loss",
        main = "Loss for projected SGD for different diameters",lwd=2)
legend(x=1000,y=1.1, inset=.02, title="diameter",
       c("Z = 10", "Z = 50", "Z = 100", "Z = 200" ,"Z = 300"), fill=brewer.pal(11,"RdYlGn"), cex=0.8)







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
###################################################################################"


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
###################################################################################



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
###################################################################################

################################################# Faster Projection Free ######################################################


#sample a vector in d-unit ball

#Illustration（unit ball）
start_time <- Sys.time()
x<-c()
for(i in 1:3000){
  u<-rnorm(2,0,1)
  norm<-sum(u**2)**(0.5)
  r<-runif(1,0,1)**(1.0/2)
  x<-cbind(x,r*u/norm)
}
plot(x[1,],x[2,],xlim = c(-2,2),ylim=c(-2,2))
end_time <- Sys.time()
end_time - start_time

#Another version (faster)

start_time <- Sys.time()
u<-rnorm(2*3000,0,1)
u<-matrix(u,nrow=2,ncol=3000,byrow=TRUE)
norm<-(colSums(u**2))**(0.5)
r<-runif(3000,0,1)**(1.0/2)
R=matrix(rep(r,2),nrow=2,ncol=3000,byrow=TRUE)
Norm<-matrix(rep(norm,2),nrow=2,ncol=3000,byrow=TRUE)
x<-R*u/Norm
#colSums(x**2)**(0.5)
par(mfrow=c(1,1))
plot(x[1,],x[2,],xlim = c(-2,2),ylim=c(-2,2))

end_time <- Sys.time()
end_time - start_time




#functions:

#sample the vector v which is the random direction
unit_ball<-function(dim,m){
  u<-rnorm(dim,0,1)
  norm<-sum(u**2)**(0.5)
  r<-runif(1,0,1)**(1.0/dim)
  
  u<-rnorm(dim*m,0,1)
  u<-matrix(u,nrow=dim,ncol=m,byrow=TRUE)
  norm<-(colSums(u**2))**(0.5)
  r<-runif(m,0,1)**(1.0/dim)
  R=matrix(rep(r,dim),nrow=dim,ncol=m,byrow=TRUE)
  Norm<-matrix(rep(norm,dim),nrow=dim,ncol=m,byrow=TRUE)
  x<-R*u/Norm
  return(t(x))#transposé 
}

x<-unit_ball(2,10)
par(mfrow=c(1,1))
plot(x[,1],x[,2],xlim = c(-2,2),ylim=c(-2,2))


#Linear optimization function   =sign(y_imax)e_imax (explain on Latax) get the correct position of 1 : (0,0,0,...,1,0,...,0)


linear_optimization2<-function(y,z){
  if(sum(abs(y))>z){
    id<-which.max(abs(y))
    val<-y[id]
    y<-y-y
    y[id]<-sign(val)
    return(z*y)
  }
  else{
    return(y)
  }
}



#Projection free
GDproj_free <- function(a, b, init,iters=100, cost,  grad, lambda=1, z=Inf) {
  
  param <- data.frame(matrix(nrow = iters +1, ncol = length(init) + 1))
  colnames(param) <- c(colnames(a), "Loss")
  x<-c( init)
  param[1, ] <- c(x, cost( init, a, b,lambda))
  j<-2
  
  
  for (i in 1:iters) {
    eta <-  1 /(lambda*i)
    x <- linear_optimization2(x - eta * grad(x, as.matrix(a), b, lambda),z)
    param[i + 1, ]  <- c(x,cost(x, a, b,lambda))
  }
  
  
  param <- cbind(Iteration = 1:nrow(param), param)
  
  return(param)
}



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




#train on the mnist dataset
start_time <- Sys.time()

paramGD_free <- GDproj_free(a = as.matrix(train[,-1]),
                            b = train[,1],
                            init = rep(0, dim(train[-1])[2]),
                            iters = 100,  # =1000 Play with the number of iterations
                            cost = hingereg,
                            grad = gradreg,
                            lambda = 1/5,
                            z=100) # Play with the regularization parameter

end_time <- Sys.time()
end_time - start_time

start_time <- Sys.time()

paramGDproj <- GDproj(a = as.matrix(train[,-1]),
                      b = train[,1],
                      init = rep(0, dim(train[-1])[2]),
                      iters = 100,  # =1000 Play with the number of iterations
                      cost = hingereg,
                      grad = gradreg,
                      lambda = 1/5,
                      z=100) # Play with the regularization parameter


end_time <- Sys.time()
end_time - start_time




rateGD_free <- rate(paramGD_free,test)
rateGDproj <- rate(paramGDproj,test)

matplot(paramGD_free[,1],cbind(1-rateGD_free),type="l",col=brewer.pal(4,"RdYlGn"),log="xy",xlab="Iterations",ylab="Accuracy",
        main = "SVM on Test Set from MNIST",lwd=2)

matplot(paramGD_free[,1],cbind(1-rateGD_free, 1-rateGDproj),type="l",col=brewer.pal(4,"RdYlGn"),log="xy",xlab="Iterations",ylab="Accuracy",
        main = "SVM on Test Set from MNIST",lwd=2)
legend(x=30,y=1, inset=.02, title="Algorithms",
       c("GDproj_free", "GDproj"), fill=brewer.pal(4,"RdYlGn"), cex=0.8)




#######################################################    FPL    #######################################################




#FPL
FPL <- function(a, b, init,iters=100, m, delta, cost,  grad, lambda=1, z=Inf) {
  
  param <- data.frame(matrix(nrow = iters +1, ncol = length(init) + 1))
  colnames(param) <- c(colnames(a), "Loss")
  x<-c( init)
  #y<-c(init)
  dim<-length(init)
  eta <-  1 /(lambda*iters)
  gradient<-grad(x, as.matrix(a), b, lambda)
  param[1, ] <- c(x, cost( init, a, b,lambda))
  j<-2
  
  
  for (i in 1:iters) {
    #eta <-  1 /(lambda*(i+1))
    #y<-matrix(rep(0,m*dim),nrow=m,ncol=dim,byrow=TRUE)
    x<-rep(0,dim)
    for (j in 1:m) {
      v<-c(unit_ball(dim,1))
      y<-v/delta-eta*gradient
      y<-pib1(y,z)
      x<-x+y/m
    }
    #y<-c(unit_ball(dim,1))
    #o<-y - gradient
    #x <- pib1(o,z)
    param[i + 1, ]  <- c(x,cost(x, a, b,lambda))
    gradient<-gradient+grad(x, as.matrix(a), b, lambda)
  }
  
  
  param <- cbind(Iteration = 1:nrow(param), param)
  
  return(param)
  
}




#train on the mnist dataset 
start_time <- Sys.time()

paramFPL <- FPL(a = as.matrix(train[,-1]),
                b = train[,1],
                init = rep(0, dim(train[-1])[2]),
                iters = 100,  # =1000 Play with the number of iterations
                m=10,
                delta = sqrt(100),
                cost = hingereg,
                grad = gradreg,
                lambda = 1/5,
                z=100) # Play with the regularization parameter

end_time <- Sys.time()
end_time - start_time


rateFPL <- rate(paramFPL,test)


matplot(paramFPL[,1],cbind(1-rateFPL,1-rateGDproj),type="l",col=brewer.pal(4,"RdYlGn"),log="xy",xlab="Iterations",ylab="Accuracy",
        main = "FPL Algorithm",lwd=2)
legend(x=30,y=1, inset=.02, title="Algorithms",
       c("FPL", "GDproj"), fill=brewer.pal(4,"RdYlGn"), cex=0.8)


#from the paper it claims that delta=O(1/sqrt(T)), so we take it as delta=1/sqrt(T)
#from the paper it claims that m=O(T) and =O(beta sqrt(T)) for general convex and smooth convex function respectively.


#######################################################    Algo 1    #######################################################




#FPL
FPL_1 <- function(a, b, init,iters=100, m, delta, cost,  grad, lambda=1, z=Inf) {
  
  param <- data.frame(matrix(nrow = iters +1, ncol = length(init) + 1))
  colnames(param) <- c(colnames(a), "Loss")
  x<-c( init)
  #y<-c(init)
  dim<-length(init)
  eta <-  1 /(lambda*iters)
  gradient<-grad(x, as.matrix(a), b, lambda)
  param[1, ] <- c(x, cost( init, a, b,lambda))
  j<-2
  
  
  for (i in 1:iters) {
    #eta <-  1 /(lambda*(i+1))
    #y<-matrix(rep(0,m*dim),nrow=m,ncol=dim,byrow=TRUE)
    x<-rep(0,dim)
    for (j in 1:m) {
      v<-c(unit_ball(dim,1))
      y<-v/delta-eta*gradient
      y<-linear_optimization2(y,z)
      x<-x+y/m
    }
    #y<-c(unit_ball(dim,1))
    #o<-y - gradient
    #x <- pib1(o,z)
    param[i + 1, ]  <- c(x,cost(x, a, b,lambda))
    gradient<-gradient+grad(x, as.matrix(a), b, lambda)
  }
  
  
  param <- cbind(Iteration = 1:nrow(param), param)
  
  return(param)
  
}




#train on the mnist dataset  
start_time <- Sys.time()

paramFPL_1 <- FPL_1(a = as.matrix(train[,-1]),
                    b = train[,1],
                    init = rep(0, dim(train[-1])[2]),
                    iters = 100,  
                    m=100,
                    delta = sqrt(100),
                    cost = hingereg,
                    grad = gradreg,
                    lambda = 1/5,
                    z=100)

end_time <- Sys.time()
end_time - start_time


rateFPL_1 <- rate(paramFPL_1,test)


matplot(paramFPL_1[,1],cbind(1-rateFPL_1,1-rateGDproj),type="l",col=brewer.pal(4,"RdYlGn"),log="xy",xlab="Iterations",ylab="Accuracy",
        main = "Sampled Follow-the-Perturbed-Leader algorithm",lwd=2)
legend(x=30,y=1, inset=.02, title="Algorithms",
       c("FPL", "GDproj"), fill=brewer.pal(4,"RdYlGn"), cex=0.8)










#######################################################    Algo 2    #######################################################




#FPL
FPL_2 <- function(a, b, init,iters=100, m, delta, cost,  grad, lambda=1, z=Inf) {
  
  param <- data.frame(matrix(nrow = iters +1, ncol = length(init) + 1))
  colnames(param) <- c(colnames(a), "Loss")
  x<-c( init)
  #y<-c(init)
  dim<-length(init)
  eta <-  1 /(lambda*iters)
  gradient<-grad(x, as.matrix(a), b, lambda)
  param[1, ] <- c(x, cost( init, a, b,lambda))
  j<-2
  
  
  for (i in 1:iters) {
    if (i%% (m) == 0){
      x<-rep(0,dim)
      for (j in 1:m) {
        v<-c(unit_ball(dim,1))
        y<-v/delta-eta*gradient
        y<-linear_optimization2(y,z)
        x<-x+y/m
      }
    }
    param[i + 1, ]  <- c(x,cost(x, a, b,lambda))
    gradient<-gradient+grad(x, as.matrix(a), b, lambda)
  }
  
  
  param <- cbind(Iteration = 1:nrow(param), param)
  
  return(param)
  
}




#train on the mnist dataset 
start_time <- Sys.time()

paramFPL_2 <- FPL_2(a = as.matrix(train[,-1]),
                    b = train[,1],
                    init = rep(0, dim(train[-1])[2]),
                    iters = 100, 
                    m=3,
                    delta = sqrt(100),
                    cost = hingereg,
                    grad = gradreg,
                    lambda = 1/5,
                    z=100) 

end_time <- Sys.time()
end_time - start_time


rateFPL_2 <- rate(paramFPL_2,test)


matplot(paramFPL_2[,1],cbind(1-rateFPL_2,1-rateFPL_1),type="l",col=brewer.pal(4,"RdYlGn"),log="xy",xlab="Iterations",ylab="Accuracy",
        main = "Online Smooth Projection Free Algorithm",lwd=2)
legend(x=30,y=1, inset=.02, title="Algorithms",
       c("Algo2", "Algo1"), fill=brewer.pal(4,"RdYlGn"), cex=0.8)











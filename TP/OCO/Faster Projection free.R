<<<<<<< HEAD

library(RColorBrewer)


#Preliminary
mnist_train <- read.csv("mnist_train.csv", stringsAsFactors = F, header = F)
mnist_test <- read.csv("mnist_test.csv", stringsAsFactors = F, header = F)

#View(mnist_train) # Data has no column names
#View(mnist_test) # Data has no column names

names(mnist_test)[1] <- "label"
names(mnist_train)[1] <- "label"

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

rate <- function(param,c){colMeans(c[,1]*( as.matrix(c[,-1]) %*% t(as.matrix(param[, 2:(ncol(param) - 1)])))>0)}











#sample a vector in d-unit ball

#Illustration（unit ball）
=======
#sample a vector in d-unit ball

#二维图示（unit ball）
>>>>>>> a10842cb09504016c6a4b8a84046154b6d5df437
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

<<<<<<< HEAD
#Another version (faster)
=======
#另一种方法（更快）
>>>>>>> a10842cb09504016c6a4b8a84046154b6d5df437

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

<<<<<<< HEAD



#functions:

#sample the vector v which is the random direction
unit_ball<-function(dim,m){
=======
#function:
unit_ball(dim,m){
>>>>>>> a10842cb09504016c6a4b8a84046154b6d5df437
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
<<<<<<< HEAD
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



































=======
  return(x)
}
>>>>>>> a10842cb09504016c6a4b8a84046154b6d5df437

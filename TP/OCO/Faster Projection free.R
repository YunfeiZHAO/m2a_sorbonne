#sample a vector in d-unit ball

#二维图示（unit ball）
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

#另一种方法（更快）

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

#function:
unit_ball(dim,m){
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
  return(x)
}

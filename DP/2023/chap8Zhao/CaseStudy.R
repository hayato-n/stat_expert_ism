library(bayesm)
set.seed(123)

Dat1 <- read.table("xdata.csv", header=TRUE, sep=",", na.strings="NA",dec=".",strip.white=TRUE)
IndAttr <- read.table("zdata.csv", header=TRUE, sep=",", na.strings="NA",dec=".",strip.white=TRUE)
reg=levels(factor(Dat1$ID))
nreg=length(reg)

p=3
na=3 
nz=3

lgtdata <- list()
for (j in 1:nreg){
  y=Dat1$brand[Dat1$ID==reg[j]]
  Xa=cbind(Dat1[Dat1$ID==reg[j], c('PriceSh', 'PriceKa', 'PriceKo', 'TimeSh', 'TimeKa', 'TimeKo', 'AreaSh', 'AreaKa', 'AreaKo')])
  X=createX(p, na=na, nd=NULL, Xa=Xa, Xd=NULL, DIFF=FALSE, base=3)
  lgtdata[[j]]=list(y=y,X=X)
}

Z=t(t(as.matrix(IndAttr))-apply(IndAttr,2,mean))
Data3=list(p=p,lgtdata=lgtdata,Z=Z)
Prior3=list(ncomp=1)

Mcmc3=list(R=50000,sbeta=0.01,keep=1)
out3=rhierMnlRwMixture(Data=Data3, Mcmc=Mcmc3, Prior=Prior3)

PD <- max((out3$loglike)[-c(1:45000)]) - mean((out3$loglike)[-c(1:45000)])
DIC3 <- -2*mean(out3$loglike) + 2*PD
print(DIC3)

s=45001
t=50000
beta.mean = beta.sd = beta.t = matrix(0, nrow=nreg, ncol=5)

for(i in 1:nreg){
  for(j in 1:5){
    beta.mean[i,j] <- mean(out3$betadraw[i,j,s:t])
    beta.sd[i,j] <- sd(out3$betadraw[i,j,s:t])
    beta.t[i,j] <- beta.mean[i,j] / beta.sd[i,j]
  }
}
Delta.mean = Delta.SD = Delta.t = matrix(0, nrow=1, ncol=10)
for(i in 1:10){
  Delta.mean[1,i] <- mean(out3$Deltadraw[s:t,i])
  Delta.SD[1,i] <- sd(out3$Deltadraw[s:t,i])
  Delta.t[1,i] <- Delta.mean[1,i] / Delta.SD[1,i]
}

beta.mean
beta.sd
beta.t
Delta.mean
Delta.SD
Delta.t

# summary(out3$Deltadraw)
summary(t(out3$betadraw[1,,]),burnin=45000)
# plot(out3$Deltadraw)
plot(out3$betadraw)
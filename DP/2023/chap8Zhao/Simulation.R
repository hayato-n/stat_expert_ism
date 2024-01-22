library(bayesm)
library(MASS)

set.seed(123)

n <- 300 #データ数
nx <- 3 #説明変数xの数
nz <- 2 #属性zの数
nrep <- 10 #時点tの長さ
ncat <- 2 #ブランドの数

#顧客属性zを生成
Z <- matrix(runif(n * nz), nrow = n, ncol = nz)
Z = t(t(Z) - apply(Z, 2, mean))

#Thetaを生成
Theta <- matrix(rnorm(nz * nx), nrow = nz, ncol = nx)

#inverse-WishartからV_betaを生成
Vbeta <- rWishart(1, df = nx + 4, Sigma = diag(nx))[,,1]

#混合分布用のパラメータ
weights <- c(0.6, 0.4)
means <- c(-3, 1.5)
sds <- c(1, 1.2)
mix <- function(weights, means, sds) {
  selected <- sample(1:length(weights), 1, prob = weights)
  return(rnorm(1, mean = means[selected], sd = sds[selected]))
}

#betaを生成
betaDGP <- function(z, Theta, Vbeta) {
  mbeta <- t(Theta) %*% z
  return(mvrnorm(1, mbeta, Vbeta) + mix(weights, means, sds))
  
}

#X,yを生成
ndata <- vector("list", n)
cate <- 1:ncat
for (i in 1:n) {
  beta0 <- betaDGP(Z[i,], Theta, Vbeta)
  beta <- matrix(beta0, nx, ncat)
  X <- matrix(runif(nrep * nx), nrep, nx)
  
  XX <- matrix(NA, nrow = nrep * ncat, ncol = nx)
  for (j in 1:ncat) {
    XX[((j - 1) * nrep + 1):(j * nrep), ] <- X
  }
  
  Xbeta <- XX %*% beta
  j <- nrow(Xbeta) / nrep
  Xbeta <- matrix(Xbeta, byrow = TRUE, ncol = j)
  prob <- exp(Xbeta) / as.vector(exp(Xbeta) %*% c(rep(1,j)))
  
  y <- integer(nrep)
  for (k in 1:nrep) {
    yp <- rmultinom(1, 1, prob[k, ])
    y[k] <- cate %*% yp
  }
  
  ndata[[i]] <- list(y = y, X = XX, beta = matrix(beta0, ncol = 1))
}

Data1 <- list(p = ncat, lgtdata = ndata, Z = Z)

#事前分布の設定
Prior1 <- list(ncomp = 2)

#MCMCのパラメータ設定
R <- 10000
keep <- 5
Mcmc1 <- list(R = R, keep = keep, nprint = 0)

out1 <- rhierMnlRwMixture(Data = Data1, Prior = Prior1, Mcmc = Mcmc1)

#図1
bmat <- matrix(0, n, nx)
for (i in 1:n) {
  bmat[i,] <- Data1$lgtdata[[i]]$beta
}
par(mfrow = c(nx, 1))
for (i in 1:nx) {
  hist(bmat[,i], breaks = 30, col = "magenta",
       main = sprintf("Beta %d Distribution", i))
}
#図2
plot(out1$betadraw)
#図3
par(mfrow = c(3, 2))
beta_means <- t(apply(out1$betadraw, c(2,3), mean))
for (i in 1:3) {
  plot(beta_means[, i], type = "l", xlab = "", ylab = "", main = sprintf("Draw of beta %d", i))
  acf(beta_means[, i], type = "correlation", main = sprintf("Acf of beta %d", i))
}
#図4
par(mfrow = c(1, 1))
plot(out1$loglike,type="l",xlab="",ylab="", main="Draw of loglike")
#図5と図6
par(mfrow = c(3, 2)) 
for (i in 1:3) {
  for (j in 1:2) {
    idx <- (i-1)*2 + j 
    plot(out1$Deltadraw[,idx], type="l", xlab="", ylab="", main=sprintf("Draw of Theta[%d,%d]", i, j))
    acf(out1$Deltadraw[,idx], type="correlation", main=sprintf("Acf of Theta[%d,%d]", i, j))
  }
}
# ランダムウォークメトロポリス・ヘイスティングス法
# 目標分布
target <- function(x) {
  return(dnorm(x, mean=0, sd=1))
}

# ランダムウォークＭＨ法
rw_mh <- function(n, init, delta) {
  # 初期化
  x <- numeric(n)
  x[1] <- init
  
  for (i in 2:n) {
    # 提案分布からサンプリング
    y <- runif(1, min=x[i-1]-delta, max=x[i-1]+delta)
    # 受理確率の計算
    alpha <- min(1, target(y) / target(x[i-1]))
    # 受理判定
    if (runif(1) < alpha) {
      x[i] <- y  # 受理
    } else {
      x[i] <- x[i-1]  # 棄却
    }
  }
  return(x)
}


# サンプリング
set.seed(123)
nn = 10000
delta_005 <- rw_mh(nn, init=0, delta=0.05)
delta_1 <- rw_mh(nn, init=0, delta=1)
delta_15 <- rw_mh(nn, init=0, delta=15)

par(mfrow=c(3,3)) 
# ヒストグラム
library(KernSmooth)
h1 <- dpih(delta_005)
bins1 <- seq(min(delta_005)-0.1, max(delta_005)+0.1+h1, by=h1)
h2 <- dpih(delta_1)
bins2 <- seq(min(delta_1)-0.1, max(delta_1)+0.1+h2, by=h2)
h3 <- dpih(delta_15)
bins3 <- seq(min(delta_15)-0.1, max(delta_15)+0.1+h3, by=h3)

hist(delta_005, freq=FALSE, breaks=bins1,xlim=c(-4, 4), main="delta=0.05")
curve(dnorm(x, mean=0, sd=1), add=TRUE, col="red", lwd=2)
hist(delta_1, freq=FALSE, breaks=bins2,xlim=c(-4, 4), main="delta=1")
curve(dnorm(x, mean=0, sd=1), add=TRUE, col="red", lwd=2)
hist(delta_15, freq=FALSE, breaks=bins3,xlim=c(-4, 4), main="delta=15")
curve(dnorm(x, mean=0, sd=1), add=TRUE, col="red", lwd=2)

# 折れ線グラフ
plot((length(delta_005)-999):length(delta_005), delta_005[(length(delta_005)-999):length(delta_005)], type="l", main="delta=0.05", xlab="Iteration", ylab="Value")
plot((length(delta_1)-999):length(delta_1), delta_1[(length(delta_1)-999):length(delta_1)], type="l", main="delta=1", xlab="Iteration", ylab="Value")
plot((length(delta_15)-999):length(delta_15), delta_15[(length(delta_15)-999):length(delta_15)], type="l", main="delta=15", xlab="Iteration", ylab="Value")

# 自己相関関数
acf(delta_005, main="Autocorrelation for delta=0.05")
acf(delta_1, main="Autocorrelation for delta=1")
acf(delta_15, main="Autocorrelation for delta=15")
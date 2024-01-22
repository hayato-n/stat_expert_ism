### 2023-7-26 by R. Yuasa
### 分散の推定のMSE比較
## Step1、真のパラメータ(平均、分散)を知った状態で乱数を発生させる
## Step2、次に真のパラメータを知らないと思って、得た乱数だけを用いてパラメータを推定する
## Step3、真のパラメータと推定値の差の2乗により精度を測る
## 以上のステップによるシミュレーションを繰り返す

### 設定
R <- 10^5 # シミュレーションの繰り返し回数
N <- 10   # サンプルサイズ

# R回のシミュレーションでデータを格納するものを用意
s2 <- numeric(R)
v1 <- numeric(R)
v2 <- numeric(R)

# Step1,2
for(i in 1:R){
  vX <- rnorm(N,0,1)　# 真の平均が0, 分散が1だとして乱数を発生させる
  s2[i] <- sum((vX - mean(vX))^2)/N　# 得られた乱数から分散を推定する
  v1[i] <- sum((vX - mean(vX))^2)/(N-1) # 不偏分散推定量
  v2[i] <- sum((vX - mean(vX))^2)/(N+1) # MSEを最小化する推定量
}

# Step3
# MSE
# 繰り返しの中で1-s2[i]などを計算してもよいが
# ここではベクトル計算で一度に計算している
# 上のようなfor文を用いた計算は何をしているのかはわかりやすいが、Rではforの計算は遅い
# ベクトルや行列にして一度計算を行う方が速くなる
mean((1-s2)^2)
mean((1-v1)^2)
mean((1-v2)^2)
# MSEの理論値
(2*N-1)/N^2
2/(N-1)

# 不偏性も得られた推定値から確かめられる
# シミュレーションで発生させた乱数による推定値の平均を計算する事で期待値を計算する
# モンテカルロ法と呼ばれる
mean(s2)
mean(v1)
mean(v2)

# シミュレーションの各回の推定値をヒストグラムにする
par(mfrow = c(1, 3)) # 3つの図を横並びに表示する
hist(s2, breaks = 20, main = paste("s^2/N", "\n", "MSE", mean((1-s2)^2), "\n", "推定量の期待値", mean(s2))) # ヒストグラムを描く. mainでタイトルを書ける. シミュレーションで得られた数字を図に反映させるためにpasteという関数を用いている. "\n"は改行のために入っている
abline(v = 1, col = 'red') # 真値の1の所に赤色の縦線を引く
abline(v = mean(s2), col = 'blue') # 推定値の平均に青色の縦線を引く
legend("topright", legend = c("真値", "推定量の期待値"), col = c("red", "blue"),lty=c(1,1)) #凡例

hist(v1, breaks = 20, main = paste("s^2/(N-1)", "\n", "MSE", mean((1-v1)^2), "\n", "推定量の期待値", mean(v1)))
abline(v = 1, col = 'red')
abline(v = mean(v1), col = 'blue')
legend("topright", legend = c("真値", "推定量の期待値"), col = c("red", "blue"),lty=c(1,1))

hist(v2, breaks = 20, main = paste("s^2/(N+1)", "\n", "MSE", mean((1-v2)^2), "\n", "推定量の期待値", mean(v2)))
abline(v = 1, col = 'red')
abline(v = mean(v2), col = 'blue')
legend("topright", legend = c("真値", "推定量の期待値"), col = c("red", "blue"),lty=c(1,1))
par(mfrow = c(1, 1))

### 標準偏差のMSE比較
## 分散の場合とだいたい同様
R <- 10^5
N <- 10

sd2 <- numeric(R)
vd1 <- numeric(R)
vd2 <- numeric(R)

for(i in 1:R){
  vX <- rnorm(N,0,1)
  sd2[i] <- sqrt(sum((vX - mean(vX))^2)/N)
  vd1[i] <- sqrt(sum((vX - mean(vX))^2)/(N-1))
  vd2[i] <- sqrt(sum((vX - mean(vX))^2)/(N-1.5))
}

# MSE
mean((1-sd2)^2)
mean((1-vd1)^2)
mean((1-vd2)^2)

# MSEの理論値
MSEs <- function(c,n){
  (c^2)*(n-1) + 1 - 2*c*sqrt(2)*gamma(n/2)/gamma((n-1)/2)
}
MSEs(1/sqrt(N), N)
MSEs(1/sqrt(N-1), N)
MSEs(1/sqrt(N-1.5), N)
#MSEs(1/sqrt(N-0.5), N)

# c=1/sqrt(N-x)として、MSEの理論値の最小化をするxを実験的に求める
MSEs2 <- function(x,n){
  MSEs(1/sqrt(n-x), n)
}
Ns <- c(3,5,7,10,15,20,30,50,100,200,300) #実験に用いるNたち
optx <- numeric(length(Ns))
for (i in 1:length(Ns))
optx[i] <- optimize(MSEs2, interval = c(-2,2), n=Ns[i])$minimum # optimize関数により、xが-2から2の間でいつMSE最小となるかを計算する
optx # MSEの理論値の最小化をするx. Nを大きくするにつれ0.5に近づいている. ただしMSEとして大きな差はなし

# MSEの理論値の近似値
1/(2*N) + (7/16)*(1/N^2)
1/(2*N) + (10/16)*(1/N^2)

# 不偏性
mean(sd2)
mean(vd1)
mean(vd2)

par(mfrow = c(1, 3))
hist(sd2, breaks = 20, main = paste("sqrt( s^2/N )", "\n", "MSE", mean((1-sd2)^2), "\n", "推定量の期待値", mean(sd2)))
abline(v = 1, col = 'red')
abline(v = mean(sd2), col = 'blue')
legend("topright", legend = c("真値", "推定量の期待値"), col = c("red", "blue"),lty=c(1,1))

hist(vd1, breaks = 20, main = paste("sqrt( s^2/(N-1) )", "\n", "MSE", mean((1-vd1)^2), "\n", "推定量の期待値", mean(vd1)))
abline(v = 1, col = 'red')
abline(v = mean(vd1), col = 'blue')
legend("topright", legend = c("真値", "推定量の期待値"), col = c("red", "blue"),lty=c(1,1))

hist(vd2, breaks = 20, main = paste("sqrt( s^2/(N-1.5) )", "\n", "MSE", mean((1-vd2)^2), "\n", "推定量の期待値", mean(vd2)))
abline(v = 1, col = 'red')
abline(v = mean(vd2), col = 'blue')
legend("topright", legend = c("真値", "推定量の期待値"), col = c("red", "blue"),lty=c(1,1))
par(mfrow = c(1, 1))


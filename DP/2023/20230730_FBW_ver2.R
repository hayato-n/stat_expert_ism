### 2023-8-18 by R. Yuasa
### Fisher-Behren's-Welch
### 未知パラメータを既知とした場合に計算できるKabeによるものとの比較

library("hypergeo")

## パラメータの設定
N1 <- 10 # m
N2 <- 3 # n
mu1 <- 3
mu2 <- 3
si1 <- 5
si2 <- 10

## 分散が異なる場合のtのためのデータ
X1 <- rnorm(N1, mu1, sd=si1)
X2 <- rnorm(N2, mu2, sd=si2)

s1 <- sqrt( sum(( X1 - mean(X1) )^2) / (N1-1) )
s2 <- sqrt( sum(( X2 - mean(X2) )^2) / (N2-1) )

## Kabeによる密度関数, N1>N2
alp1 <- ( N1*(N1-1)/si1^2 ) * ( si1^2/N1 + si2^2/N2 )
alp2 <- ( N2*(N2-1)/si2^2 ) * ( si1^2/N1 + si2^2/N2 )
p1   <- (N1-1)/2
p2   <- (N2-1)/2
C <- (alp1^p1)*(alp2^p2)*gamma(p1+p2+1/2)*(pi^(1/2)*gamma(p1+p2))^(-1)
pdfv <- function(v){
  C*(alp1+v^2)^(-(p1+p2+1/2))*Re(hypergeo(p2, p1+p2+1/2, p1+p2, (alp1-alp2)/(alp1+v^2)))
}

## 密度関数の比較
## 分散が等しいと仮定するかどうかで統計量が変わるのでtどうしで違うのは妥当ではある
## Kabeと分散が異なるtの違いはそのまま検定に影響を与えうる
plot(pdfv, -5,5, xlim=c(-5,5), ylim=c(0,0.4), ylab="", main = paste("m=", N1, ",n=", N2, ",mu1=", mu1, ",mu2=", mu2, ",si1=", si1, ",si2=", si2))
par(new=T)
curve(dt(x,df=N1+N2-2), -5,5,col="red", xlim=c(-5,5), ylim=c(0,0.4), ylab="")
par(new=T)
curve(dt(x,df= (s1^2/N1+s2^2/N2)^2/( (s1^2/N1)^2/(N1-1) + (s2^2/N2)^2/(N2-1) )), -5,5,col="blue", xlim=c(-5,5), ylim=c(0,0.4), ylab="")
legend("topright", legend=c("Kabe", "共通分散と思ったt", "分散が異なるt"), lty=1, col=c("black","red","blue"))

## 棄却確率を比較するためにKabeによるものから棄却域を求める
lev <- 0.05
optfunc_v <- function(cc){
  (integrate(pdfv, -cc, cc)$value - (1-lev))^2
}
cc <- optimize(optfunc_v, interval=c(0.2,5))$minimum

## 受容確率の比較を行う
R <- 10^5
accept1 <- 1-numeric(R)
accept2 <- 1-numeric(R)
accept3 <- 1-numeric(R)
t1s <- numeric(R)
t2s <- numeric(R)
for(i in 1:R){
  X1 <- rnorm(N1, mu1, sd=si1)
  X2 <- rnorm(N2, mu2, sd=si2)
  
  s1 <- sqrt( sum(( X1 - mean(X1) )^2) / (N1-1) )
  s2 <- sqrt( sum(( X2 - mean(X2) )^2) / (N2-1) )
  
  si2h <- ( sum((X1-mean(X1))^2) + sum((X2-mean(X2))^2) ) / (N1 + N2 -2)
  t1 <- ( (mean(X1)-mean(X2)) * (N1*N2)^(1/2) / (N1+N2)^(1/2) ) / si2h^(1/2)
  #t2 <- ( mean(X1)-mean(X2) - (mu1-mu2) ) / ( s1^2/N1 + s2^2/N2 )^(1/2)
  t2 <- ( mean(X1)-mean(X2) ) / ( s1^2/N1 + s2^2/N2 )^(1/2)
  
  t1s[i] <- t1
  t2s[i] <- t2
  
  if(abs(t2) > cc){
    accept1[i] <- 0
  }
  if( abs(t1) > qt(1-lev/2, df = N1 + N2 -2) ){
    accept2[i] <- 0
  }
  if( abs(t2) > qt(1-lev/2, df = (s1/N1+s2/N2)^2/( (s1/N1)^2/(N1-1) + (s2/N2)^2/(N2-1) )) ){
    accept3[i] <- 0
  }
}
mean(accept1) #Kabe
mean(accept2) #分散共通と思ったt
mean(accept3) #分散が異なるt
boxplot(t1s-t2s, main="分散が共通と思ったtと分散が異なるtでの統計量の値の差") # 先に比べた真だと思っている密度関数が近くても統計量の値はかなり異なりうる


## 繰り返し計算した統計量の値から密度関数を推定したものと
## 繰り返しの最後に出てきたデータを使った密度関数との比較を行う.
## データによって真の分布が変わりうるので厳密な比較ではないが、
## データが異なっても真のパラメータが共通ならそれなりに分布は近いだろうという気持ちで比較している.
## Kabeによる密度関数, N1>N2
alp1 <- ( N1*(N1-1)/si1^2 ) * ( si1^2/N1 + si2^2/N2 )
alp2 <- ( N2*(N2-1)/si2^2 ) * ( si1^2/N1 + si2^2/N2 )
p1   <- (N1-1)/2
p2   <- (N2-1)/2
C <- (alp1^p1)*(alp2^p2)*gamma(p1+p2+1/2)*(pi^(1/2)*gamma(p1+p2))^(-1)
pdfv <- function(v){
  C*(alp1+v^2)^(-(p1+p2+1/2))*Re(hypergeo(p2, p1+p2+1/2, p1+p2, (alp1-alp2)/(alp1+v^2)))
}

plot(pdfv, -5,5, xlim=c(-5,5), ylim=c(0,0.4), ylab="", main = paste("m=", N1, ",n=", N2, ",mu1=", mu1, ",mu2=", mu2, ",si1=", si1, ",si2=", si2))
par(new=T)
curve(dt(x,df=N1+N2-2), -5,5,col="red", xlim=c(-5,5), ylim=c(0,0.4), ylab="")
par(new=T)
curve(dt(x,df= (s1^2/N1+s2^2/N2)^2/( (s1^2/N1)^2/(N1-1) + (s2^2/N2)^2/(N2-1) )), -5,5,col="blue", xlim=c(-5,5), ylim=c(0,0.4), ylab="")
par(new=T)
plot(density(t1s), xlim=c(-5,5), ylim=c(0,0.4), ylab="",col="orange",main="") #density関数でデータから密度関数を推定
par(new=T)
plot(density(t2s), xlim=c(-5,5), ylim=c(0,0.4), ylab="",col="purple",main="") #density関数でデータから密度関数を推定
legend("topright", legend=c("Kabe", "共通分散と思ったt", "分散が異なるt","データから推定した、分散共通と思ったt","データから推定した、分散が異なるt"), lty=1, col=c("black","red","blue","orange","purple"))

# %%
# 2023-08-18 by Hayato Nishi
### Fisher-Behren's-Welch
### 未知パラメータを既知とした場合に計算できるKabeによるものとの比較

import numpy as np
import matplotlib.pyplot as plt

# グラフ描画パッケージ（密度推定など）
import seaborn as sns

# 特殊関数の読み込み
# （超幾何関数を計算する関数を含む）
# library("hypergeo")
from scipy import special

# 統計関数の読み込み
from scipy import stats

# 積分・最適化のための関数
from scipy import integrate, optimize


rng_seed = 123

## パラメータの設定
# N1 <- 100 # m
# N2 <- 30 # n
# mu1 <- 3
# mu2 <- 3
# si1 <- 5
# si2 <- 10
N1 = 10  # m
N2 = 3  # n
mu1 = 3
mu2 = 3
si1 = 5
si2 = 10


## 分散が異なる場合のtのためのデータ
# X1 <- rnorm(N1, mu1, sd=si1)
# X2 <- rnorm(N2, mu2, sd=si2)

# s1 <- sum(( X1 - mean(X1) )^2) / (N1-1)
# s2 <- sum(( X2 - mean(X2) )^2) / (N2-1)
rng = np.random.default_rng(rng_seed)
X1 = rng.normal(loc=mu1, scale=si1, size=N1)
X2 = rng.normal(loc=mu2, scale=si2, size=N2)

s1 = np.sqrt(np.sum((X1 - np.mean(X1)) ** 2) / (N1 - 1))
s2 = np.sqrt(np.sum((X2 - np.mean(X2)) ** 2) / (N2 - 1))


## Kabeによる密度関数, N1>N2
# alp1 <- ( N1*(N1-1)/si1^2 ) * ( si1^2/N1 + si2^2/N2 )
# alp2 <- ( N2*(N2-1)/si2^2 ) * ( si1^2/N1 + si2^2/N2 )
# p1   <- (N1-1)/2
# p2   <- (N2-1)/2
# C <- (alp1^p1)*(alp2^p2)*gamma(p1+p2+1/2)*(pi^(1/2)*gamma(p1+p2))^(-1)
# pdfv <- function(v){
#   C*(alp1+v^2)^(-(p1+p2+1/2))*Re(hypergeo(p2, p1+p2+1/2, p1+p2, (alp1-alp2)/(alp1+v^2)))
# }
def pdf_kabe(v, N1=N1, N2=N2, si1=si1, si2=si2):
    assert N1 > N2  # N1 > N2になっているかチェックする

    fac = si1**2 / N1 + si2**2 / N2
    alp1 = N1 * (N1 - 1) / si1**2 * fac
    alp2 = N2 * (N2 - 1) / si2**2 * fac
    p1 = (N1 - 1) / 2
    p2 = (N2 - 1) / 2

    # c = (
    #     alp1**p1
    #     * alp2**p2
    #     * special.gamma(p1 + p2 + 0.5)
    #     * (np.sqrt(np.pi) * special.gamma(p1 + p2)) ** (-1)
    # )
    # A = c * (alp1 + v**2) ** (-(p1 + p2 + 0.5))
    # 桁落ちを避けるためにいったん対数範囲で数値計算する
    logc = (
        p1 * np.log(alp1)
        + p2 * np.log(alp2)
        + special.gammaln(p1 + p2 + 0.5)
        - np.log(np.pi) / 2
        - special.gammaln(p1 + p2)
    )
    logA = logc - (p1 + p2 + 0.5) * np.log(alp1 + v**2)
    hypergeo = special.hyp2f1(
        p2, p1 + p2 + 0.5, p1 + p2, (alp1 - alp2) / (alp1 + v**2)
    )

    pdf = np.exp(logA) * hypergeo

    return pdf


# Welchの検定における自由度を計算する関数
def df_welch(N1=N1, N2=N2, s1=s1, s2=s2):
    g1 = s1**2 / N1
    g2 = s2**2 / N2
    return (g1 + g2) ** 2 / (g1**2 / (N1 - 1) + g2**2 / (N2 - 1))


# %%
## 密度関数の比較
## 分散が等しいと仮定するかどうかで統計量が変わるのでtどうしで違うのは妥当ではある
## Kabeと分散が異なるtの違いはそのまま検定に影響を与えうる
# plot(pdfv, -5,5, xlim=c(-5,5), ylim=c(0,0.4), ylab="", main = paste("m=", N1, ",n=", N2, ",mu1=", mu1, ",mu2=", mu2, ",si1=", si1, ",si2=", si2))
# par(new=T)
# curve(dt(x,df=N1+N2-2), -5,5,col="red", xlim=c(-5,5), ylim=c(0,0.4), ylab="")
# par(new=T)
# curve(dt(x,df= (s1/N1+s2/N2)^2/( (s1/N1)^2/(N1-1) + (s2/N2)^2/(N2-1) )), -5,5,col="blue", xlim=c(-5,5), ylim=c(0,0.4), ylab="")
# legend("topright", legend=c("Kabe", "共通分散と思ったt", "分散が異なるt"), lty=1, col=c("black","red","blue"))
v_lin = np.linspace(-5, 5, num=100)
plt.plot(
    v_lin,
    pdf_kabe(v_lin),
    c="black",
    label="Kabe",
)
plt.plot(
    v_lin, stats.t.pdf(v_lin, df=N1 + N2 - 2), c="orange", label="t (same variance)"
)
plt.plot(
    v_lin,
    stats.t.pdf(v_lin, df=df_welch()),
    c="skyblue",
    label="Welch t (different var.)",
)

plt.xlabel("v")
plt.ylabel("pdf")
plt.title(f"m={N1}, n={N2}, mu1={mu1}, mu2={mu2}, si1={si1}, si2={si2}")
plt.legend()
# plt.savefig("ファイル名.png", dpi=300)
plt.show()


# %%
## 棄却確率を比較するためにKabeによるものから棄却域を求める
# lev <- 0.05
# optfunc_v <- function(cc){
#   (integrate(pdfv, -cc, cc)$value - (1-lev))^2
# }
# cc <- optimize(optfunc_v, interval=c(0.2,5))$minimum
lev = 0.05


def optfunc_v(cc):
    # 数値積分して-ccからccの間の確率を計算し、
    # それと1-levの値の二乗誤差を返す
    return (integrate.quad(pdf_kabe, -cc, cc)[0] - (1 - lev)) ** 2


# 上の関数を最小化する点を求めることで、棄却域を算出
res = optimize.minimize_scalar(optfunc_v, bracket=(0.2, 5))
cc = res.x
print(res)
# %%
# ## 受容確率の比較を行う
# R <- 10^5
# accept1 <- 1-numeric(R)
# accept2 <- 1-numeric(R)
# accept3 <- 1-numeric(R)
# t1s <- numeric(R)
# t2s <- numeric(R)
# for(i in 1:R){
#   X1 <- rnorm(N1, mu1, sd=si1)
#   X2 <- rnorm(N2, mu2, sd=si2)

#   s1 <- sqrt( sum(( X1 - mean(X1) )^2) / (N1-1) )
#   s2 <- sqrt( sum(( X2 - mean(X2) )^2) / (N2-1) )

#   si2h <- ( sum((X1-mean(X1))^2) + sum((X2-mean(X2))^2) ) / (N1 + N2 -2)
#   t1 <- ( (mean(X1)-mean(X2)) * (N1*N2)^(1/2) / (N1+N2)^(1/2) ) / si2h^(1/2)
#   #t2 <- ( mean(X1)-mean(X2) - (mu1-mu2) ) / ( s1^2/N1 + s2^2/N2 )^(1/2)
#   t2 <- ( mean(X1)-mean(X2) ) / ( s1^2/N1 + s2^2/N2 )^(1/2)

#   t1s[i] <- t1
#   t2s[i] <- t2

#   if(abs(t2) > cc){
#     accept1[i] <- 0
#   }
#   if( abs(t1) > qt(1-lev/2, df = N1 + N2 -2) ){
#     accept2[i] <- 0
#   }
#   if( abs(t2) > qt(1-lev/2, df = (s1/N1+s2/N2)^2/( (s1/N1)^2/(N1-1) + (s2/N2)^2/(N2-1) )) ){
#     accept3[i] <- 0
#   }
# }
# mean(accept1) #Kabe
# mean(accept2) #分散共通と思ったt
# mean(accept3) #分散が異なるt
# boxplot(t1s-t2s, main="分散が共通と思ったtと分散が異なるtでの統計量の値の差") # 先に比べた真だと思っている密度関数が近くても統計量の値はかなり異なりうる
R = 10**5
accept1 = np.ones(R, dtype=bool)
accept2 = np.ones(R, dtype=bool)
accept3 = np.ones(R, dtype=bool)
t1s = np.zeros(R, dtype=float)
t2s = np.zeros(R, dtype=float)
for i in range(R):
    X1 = rng.normal(size=N1, loc=mu1, scale=si1)
    X2 = rng.normal(size=N2, loc=mu2, scale=si2)

    s1 = np.sqrt(np.sum((X1 - np.mean(X1)) ** 2) / (N1 - 1))
    s2 = np.sqrt(np.sum((X2 - np.mean(X2)) ** 2) / (N2 - 1))

    si2h = (np.sum((X1 - np.mean(X1)) ** 2) + np.sum((X2 - np.mean(X2)) ** 2)) / (
        N1 + N2 - 2
    )
    t1 = (np.mean(X1) - np.mean(X2)) / np.sqrt(1 / N1 + 1 / N2) / np.sqrt(si2h)
    t2 = (np.mean(X1) - np.mean(X2)) / np.sqrt(s1**2 / N1 + s2**2 / N2)

    t1s[i] = t1
    t2s[i] = t2

    if np.abs(t2) > cc:
        accept1[i] = False

    if np.abs(t1) > stats.t.ppf(1 - lev / 2, df=N1 + N2 - 2):
        accept2[i] = False

    if np.abs(t2) > stats.t.ppf(1 - lev / 2, df=df_welch(s1=s1, s2=s2)):
        accept3[i] = False


print(np.mean(accept1), "Kabe")
print(np.mean(accept2), "分散共通と思ったt")
print(np.mean(accept3), "分散が異なるt (Welch t)")
plt.boxplot(t1s - t2s)  # 先に比べた真だと思っている密度関数が近くても統計量の値はかなり異なりうる
plt.ylabel("t(assume same variances) - Welch t")
# plt.savefig("ファイル名.png", dpi=300)
plt.show()

# %%
## 繰り返し計算した統計量の値から密度関数を推定したものと
## 繰り返しの最後に出てきたデータを使った密度関数との比較を行う.
## データによって真の分布が変わりうるので厳密な比較ではないが、
## データが異なっても真のパラメータが共通ならそれなりに分布は近いだろうという気持ちで比較している.
## Kabeによる密度関数, N1>N2
# alp1 <- ( N1*(N1-1)/si1^2 ) * ( si1^2/N1 + si2^2/N2 )
# alp2 <- ( N2*(N2-1)/si2^2 ) * ( si1^2/N1 + si2^2/N2 )
# p1   <- (N1-1)/2
# p2   <- (N2-1)/2
# C <- (alp1^p1)*(alp2^p2)*gamma(p1+p2+1/2)*(pi^(1/2)*gamma(p1+p2))^(-1)
# pdfv <- function(v){
#   C*(alp1+v^2)^(-(p1+p2+1/2))*Re(hypergeo(p2, p1+p2+1/2, p1+p2, (alp1-alp2)/(alp1+v^2)))
# }

# plot(pdfv, -5,5, xlim=c(-5,5), ylim=c(0,0.4), ylab="", main = paste("m=", N1, ",n=", N2, ",mu1=", mu1, ",mu2=", mu2, ",si1=", si1, ",si2=", si2))
# par(new=T)
# curve(dt(x,df=N1+N2-2), -5,5,col="red", xlim=c(-5,5), ylim=c(0,0.4), ylab="")
# par(new=T)
# curve(dt(x,df= (s1/N1+s2/N2)^2/( (s1/N1)^2/(N1-1) + (s2/N2)^2/(N2-1) )), -5,5,col="blue", xlim=c(-5,5), ylim=c(0,0.4), ylab="")
# par(new=T)
# plot(density(t1s), xlim=c(-5,5), ylim=c(0,0.4), ylab="",col="orange",main="") #density関数でデータから密度関数を推定
# par(new=T)
# plot(density(t2s), xlim=c(-5,5), ylim=c(0,0.4), ylab="",col="purple",main="") #density関数でデータから密度関数を推定
# legend("topright", legend=c("Kabe", "共通分散と思ったt", "分散が異なるt","データから推定した、分散共通と思ったt","データから推定した、分散が異なるt"), lty=1, col=c("black","red","blue","orange","purple"))

# plot(density(t1s), xlim=c(-5,5), ylim=c(0,0.4), ylab="",col="orange",main="") #density関数でデータから密度関数を推定
# par(new=T)
# plot(density(t2s), xlim=c(-5,5), ylim=c(0,0.4), ylab="",col="purple",main="") #density関数でデータから密度関数を推定
# legend("topright", legend=c("Kabe", "共通分散と思ったt", "分散が異なるt","データから推定した、分散共通と思ったt","データから推定した、分散が異なるt"), lty=1, col=c("black","red","blue","orange","purple"))
v_lin = np.linspace(-5, 5)
plt.plot(
    v_lin,
    pdf_kabe(v_lin),
    c="black",
    label="Kabe",
)
plt.plot(
    v_lin, stats.t.pdf(v_lin, df=N1 + N2 - 2), c="orange", label="t (same variance)"
)
plt.plot(
    v_lin,
    stats.t.pdf(v_lin, df=df_welch()),
    c="skyblue",
    label="Welch t (different var.)",
)

sns.kdeplot(t1s, c="orange", ls="--", label="t (simulation)")
sns.kdeplot(t2s, c="skyblue", ls="--", label="Welch t (simulation)")

plt.xlabel("v")
plt.ylabel("pdf")
plt.title(f"m={N1}, n={N2}, mu1={mu1}, mu2={mu2}, si1={si1}, si2={si2}")
plt.legend()
plt.xlim(v_lin.min(), v_lin.max())
# plt.savefig("ファイル名.png", dpi=300)
plt.show()


# %%

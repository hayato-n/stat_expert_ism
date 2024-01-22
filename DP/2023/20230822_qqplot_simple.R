### 2023-8-23 by R. Yuasa
### QQ plot
### 次のページを参考にした
### https://lbelzile.github.io/lineaRmodels/qqplot.html

# Rにあるmtcarsというデータで回帰分析を行う
ols <- lm(mpg ~ wt, data = mtcars)
n <- length(mtcars$wt)
#スチューデント化残差
esr <- rstudent(ols)

#真の分布がt分布のときの理論的分位点
emp_quant <- qt(rank(esr)/(n + 1),  df = n - 3)
#各点ごとに信頼区間を計算する関数
#引数はn,distの他にも、...の部分に違うものを与える事が出来る
confint.qqplot.ptw <- function(n, dist = "norm", ...){
  #sapply関数は第1引数にデータ、第2引数に関数を与える
  #1:nというデータをそれぞれ、この場で定義した関数に代入し、その結果を与える
  t(sapply(1:n, function(i){
    #dist='tの場合、'paste0('q','t')は文字列'q'と't'の結合を行い、"qt"を返す
    #do.callは引数のリストを関数に与えて実行する
    #qbeta関数によりBeta(i,n-i+1)の2.5%点と97.5%点を計算
    #dist='tの場合、qtという関数にBeta(i,n-i+1)の2.5%点と97.5%点と...の部分(下ではdf=n-3)を与える
    do.call(paste0('q', dist), list(qbeta(c(0.025, 0.975), i, n - i + 1), ...))
  }))
}

#上の関数を用いる
confint_lim <- confint.qqplot.ptw(n = n, dist = "t", df = n - 3)
#経験分位点に沿って信頼区間をプロットする
#matplotは2つの行列に対し、1つ目の行列のk列目の値をx軸、2つ目の行列のk列目の値をy軸に取るような線を引く
#kは1から行列の列数
#今回は1つ目の行列として与えているのがベクトルで、x軸にはベクトルの値が用いられ、y軸には2つ目の引数で与えられている行列の各列の値をとるような線が引かれる
matplot(sort(emp_quant), confint_lim, type = "l", lty = 2, col="grey",
        main = "Q-Q plot", xlim = c(-2, 2), ylim = c(-2, 2),
        xlab = "理論的分位点", ylab = "経験分位点")  
#切片0傾き1の理論直線のあてはめ
abline(a = 0, b = 1)
#観測をプロットする
points(esr, emp_quant, pch = 20)

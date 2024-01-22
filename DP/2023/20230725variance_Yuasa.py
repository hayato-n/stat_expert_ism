# %%
# 2023-07-25 by Hayato Nishi
# 使用するパッケージの読み込み
import numpy as np  # 数値計算・乱数生成用パッケージ
import matplotlib.pyplot as plt  # グラフ描画用パッケージ

from pathlib import Path  # ファイル操作用パッケージ

# 配色の設定（もしエラーになったらコメントアウトしてください）
plt.style.use("tableau-colorblind10")

# %%

##### Settings #####

R = 10**5
N = 10
Var_True = 10.0  # 母集団の分散（真の分散）

# DGP_Familyに設定した分布を母集団として乱数を生成します。
# GaussとPoissonを実装しています。
DGP_Family = "Gauss"
# DGP_Family = "Poisson"

# 乱数のシード値を固定することで、結果に再現性を持たせています
SEED_random = 123

# 画像を保存するフォルダを指定します。
IMG_path = Path("./img/")

####################

if DGP_Family.lower() in ("gauss", "normal"):

    def gen_data(rng):
        return rng.normal(loc=0, scale=np.sqrt(Var_True), size=(N, R))

    # N × R個の乱数をまとめて生成して、R個の分散推定値を一気に計算する仕様にしています。

    # N個の乱数を生成する -> 分散を計算する
    # というプロセスをR回反復するよりも、高速に実行できます。


elif DGP_Family.lower() == "poisson":

    def gen_data(rng):
        return rng.poisson(lam=Var_True, size=(N, R))

else:
    raise ValueError(f"{DGP_Family=} is not defined.")

# 以下で分散・標準偏差のMSEとバイアスを計算する関数を定義します。


def get_MSE_var(s2):
    return np.mean(np.square(s2 - Var_True))


def get_MSE_std(s):
    return np.mean(np.square(s - np.sqrt(Var_True)))


def get_Bias_var(s2):
    return np.mean(s2) - Var_True


def get_Bias_std(s):
    return np.mean(s) - np.sqrt(Var_True)


# %%
# 分散の推定のMSE比較

# 乱数生成器を設定
rng = np.random.default_rng(seed=SEED_random)

# 乱数をvXに格納
vX = gen_data(rng)

# 各種の分散推定値をR個ずつ計算
s2 = np.sum((vX - np.mean(vX, axis=0, keepdims=True)) ** 2, axis=0) / N
v1 = np.sum((vX - np.mean(vX, axis=0, keepdims=True)) ** 2, axis=0) / (N - 1)
v2 = np.sum((vX - np.mean(vX, axis=0, keepdims=True)) ** 2, axis=0) / (N + 1)

# %%
# MSE
print("s2: MSE =", get_MSE_var(s2))
print("v1: MSE =", get_MSE_var(v1))
print("v2: MSE =", get_MSE_var(v2))

# 不偏性
print("s2: Bias =", get_Bias_var(s2))
print("v1: Bias =", get_Bias_var(v1))
print("v2: Bias =", get_Bias_var(v2))


# %%
# グラフの描画
# グラフの描き方はいくつかある
# これはやさしい描き方ではないが、細かく設定できる方法
# TODO: 参考文献を提示

# 表示領域の定義
# 表示領域figの上に、グラフのパネルを３つ設置している
# axesはリストとなっており、axes[0], axes[1], axes[2]にそれぞれパネルが格納されている
fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharex=True, sharey=True)

# 左のパネルaxes[0]の描画指示（axes[1], axes[2]も同様）
# ヒストグラムを描画
axes[0].hist(s2, bins=20)
# タイトルを設定
axes[0].set_title("s^2/N\nMSE {:.3f} (N={})".format(get_MSE_var(s2), N))
# 真の分散のところに縦線を引く
axes[0].axvline(Var_True, ymin=0, ymax=1, c="red", label="True value")
# 分散推定量の期待値のところに縦線を引く
axes[0].axvline(np.mean(s2), ymin=0, ymax=1, c="blue", label="E[estimate]")
# 凡例の表示指示
axes[0].legend()
# x軸ラベルの設定
axes[0].set_xlabel("s2")
# y軸ラベルの設定
axes[0].set_ylabel("Frequency")

# 中央のパネルaxes[1]の描画指示
axes[1].hist(v1, bins=20)
axes[1].set_title("s^2/(N-1)\nMSE {:.3f} (N={})".format(get_MSE_var(v1), N))
axes[1].axvline(Var_True, ymin=0, ymax=1, c="red", label="True value")
axes[1].axvline(np.mean(v1), ymin=0, ymax=1, c="blue", label="E[estimate]")
axes[1].legend()
axes[1].set_xlabel("v1")
axes[1].set_ylabel("Frequency")

# 右のパネルaxes[2]の描画指示
axes[2].hist(v2, bins=20)
axes[2].set_title("s^2/(N+1)\nMSE {:.3f} (N={})".format(get_MSE_var(v2), N))
axes[2].axvline(Var_True, ymin=0, ymax=1, c="red", label="True value")
axes[2].axvline(np.mean(v2), ymin=0, ymax=1, c="blue", label="E[estimate]")
axes[2].legend()
axes[2].set_xlabel("v2")
axes[2].set_ylabel("Frequency")

# 表示領域fig全体に関する指示
# ３つの図を詰めて配置する
fig.tight_layout()
# グラフをpng形式で保存
fig.savefig(
    Path(IMG_path, f"{DGP_Family}-{N=}-Variance(sigma2={Var_True}).png"), dpi=300
)
# グラフの表示
fig.show()
# plt.show()

# %%
# 標準偏差のMSE比較
# 分散と同様のプロセスを、標準偏差についても実施している。
# vd2はv2と分母が違うことに注意

rng = np.random.default_rng(seed=SEED_random)

vX = gen_data(rng)

sd2 = np.sqrt(np.sum((vX - np.mean(vX, axis=0, keepdims=True)) ** 2, axis=0) / N)
vd1 = np.sqrt(np.sum((vX - np.mean(vX, axis=0, keepdims=True)) ** 2, axis=0) / (N - 1))
vd2 = np.sqrt(
    np.sum((vX - np.mean(vX, axis=0, keepdims=True)) ** 2, axis=0) / (N - 1.5)
)

# %%
# MSE

print("sd2: MSE =", get_MSE_std(sd2))
print("vd1: MSE =", get_MSE_std(vd1))
print("vd2: MSE =", get_MSE_std(vd2))

# 不偏性
print("sd2: Bias =", get_Bias_std(sd2))
print("vd1: Bias =", get_Bias_std(vd1))
print("vd2: Bias =", get_Bias_std(vd2))


# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharex=True, sharey=True)

axes[0].hist(sd2, bins=20)
axes[0].set_title("sqrt(s^2/N)\nMSE {:.3f} (N={})".format(get_MSE_std(sd2), N))
axes[0].axvline(np.sqrt(Var_True), ymin=0, ymax=1, c="red", label="True value")
axes[0].axvline(np.mean(sd2), ymin=0, ymax=1, c="blue", label="E[estimate]")
axes[0].legend()
axes[0].set_xlabel("sd2")
axes[0].set_ylabel("Frequency")

axes[1].hist(vd1, bins=20)
axes[1].set_title("sqrt(s^2/(N-1))\nMSE {:.3f} (N={})".format(get_MSE_std(vd1), N))
axes[1].axvline(np.sqrt(Var_True), ymin=0, ymax=1, c="red", label="True value")
axes[1].axvline(np.mean(vd1), ymin=0, ymax=1, c="blue", label="E[estimate]")
axes[1].legend()
axes[1].set_xlabel("vd1")
axes[1].set_ylabel("Frequency")

axes[2].hist(vd2, bins=20)
axes[2].set_title("sqrt(s^2/(N-1.5))\nMSE {:.3f} (N={})".format(get_MSE_std(vd2), N))
axes[2].axvline(np.sqrt(Var_True), ymin=0, ymax=1, c="red", label="True value")
axes[2].axvline(np.mean(vd2), ymin=0, ymax=1, c="blue", label="E[estimate]")
axes[2].legend()
axes[2].set_xlabel("vd2")
axes[2].set_ylabel("Frequency")

fig.tight_layout()
fig.savefig(Path(IMG_path, f"{DGP_Family}-{N=}-Std(sigma2={Var_True}).png"), dpi=300)
fig.show()
# plt.show()

# %%

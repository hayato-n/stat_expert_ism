# 2023-12-25 by Hayato Nishi
# 下の方に変数positがあるので、その値をTrueにするかFalseにするかで設定を変更できる

# %%
import numpy as np  # 数値計算ライブラリ
from scipy import optimize, special  # 科学計算ライブラリ
import matplotlib.pyplot as plt  # グラフ描画ライブラリ

# %%

_BINS_NUMPY = ["auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"]


def aic_hist(hist, bin_edges, posit=False):
    # ヒストグラムに対してAICを計算する関数
    # posit=Trueとすると、n=0となるビンに1/e個のデータが入っていたと見なす（坂本・石黒・北川 1982）
    k = len(hist)
    n = np.sum(hist)

    if posit:
        hist = np.maximum(hist, 1 / np.e)

    # C = special.gammaln(n + 1) # 定数項を正確に計算するとこうなる
    C = 0
    len_bin = bin_edges[1:] - bin_edges[:-1]
    p = hist / n / len_bin

    llik = C + np.sum(hist[hist > 0] * np.log(p[hist > 0]))

    aic = -2 * llik + 2 * (k - 1)
    # print(k, n, bins, llik, len_bin)

    return aic


def search_hist_num(
    x, method="aic", posit=False, engine="brute", bins_max=None, hist_range=None
):
    # AICが最小となるビン数を選択する関数
    if method is None:
        method = "aic"

    if method.lower() in _BINS_NUMPY:
        # use numpy function
        bin_edges = np.histogram_bin_edges(x, bins=method, range=hist_range)
        bins = len(bin_edges) - 1
        res = None

    elif method.lower() == "aic":
        if bins_max is None:
            bins_max = 2 * int(np.floor(np.sqrt(len(x)))) - 1

        bins_max = min(len(x), bins_max)
        bins_min = 2

        # function to minimize
        def f(bins):
            hist, bin_edges = np.histogram(x, bins, range=hist_range)
            aic = aic_hist(hist, bin_edges, posit)
            return aic

        if engine.lower() == "brute":
            # brute force method
            res = optimize.brute(
                lambda b: f(b[0]),
                ranges=(slice(bins_min, bins_max + 1, 1),),
                finish=None,
                full_output=True,
            )
            bins = int(res[0])
            bin_edges = np.histogram_bin_edges(x, bins=bins, range=hist_range)

        else:
            raise ValueError(
                "Unknown optimization method. Currently only 'brute' is available."
            )

    return bins, bin_edges, res


# Galaxy data
galaxy = np.array(
    [
        9.172,
        9.350,
        9.483,
        9.558,
        9.775,
        10.227,
        10.406,
        16.084,
        16.170,
        18.419,
        18.552,
        18.600,
        18.927,
        19.052,
        19.070,
        19.330,
        19.343,
        19.349,
        19.440,
        19.473,
        19.529,
        19.541,
        19.547,
        19.663,
        19.846,
        19.856,
        19.863,
        19.914,
        19.918,
        19.973,
        19.989,
        20.166,
        20.175,
        20.179,
        20.196,
        20.215,
        20.221,
        20.415,
        20.629,
        20.795,
        20.821,
        20.846,
        20.875,
        20.986,
        21.137,
        21.492,
        21.701,
        21.814,
        21.921,
        21.960,
        22.185,
        22.209,
        22.242,
        22.249,
        22.314,
        22.374,
        22.495,
        22.746,
        22.747,
        22.888,
        22.914,
        23.206,
        23.241,
        23.263,
        23.484,
        23.538,
        23.542,
        23.666,
        23.706,
        23.711,
        24.129,
        24.285,
        24.289,
        24.366,
        24.717,
        24.990,
        25.633,
        26.690,
        26.995,
        32.065,
        32.789,
        34.279,
    ]
)

# %%
hist_range = (8, 36)

# 【設定可能項目】ここのpositを変えることで、坂本他の置換方法を適用するかを選べる
posit = True
# print(np.histogram(galaxy, 28, range=hist_range))
print(f"{posit=}")

print(f"AIC for galaxy data (n={len(galaxy)})")
print("Bin Size| AIC")
print("-" * 30)
# ビン数が[28, 14, 7]の場合のAICを計算する
for bins in [28, 14, 7]:
    # print(np.histogram(galaxy, bins, range=hist_range)[0])
    print(
        f"   {bins}\t|{aic_hist(*np.histogram(galaxy, bins, range=hist_range), posit=posit)}"
    )

# %%
# スタージェスの公式と、AIC最小の場合のビン数の比較
print("sturges = ", search_hist_num(galaxy, method="sturges", hist_range=hist_range)[0])
print(
    "MAIC bins=",
    search_hist_num(
        galaxy, method="aic", posit=posit, engine="brute", hist_range=hist_range
    )[0],
)
# print(
#     "MAIC bins=",
#     search_hist_num(galaxy, method="aic", engine="optuna", hist_range=hist_range)[0],
# )

# %%
# numpyで選べる自動的なビン数の選択法を比較
for m in _BINS_NUMPY:
    print(
        f"{m} = ",
        search_hist_num(galaxy, method=m, hist_range=hist_range)[0],
    )

# %%
# ビン数が[28, 14, 7]の場合のヒストグラムとAIC
plt.figure(figsize=(10, 3))
for i, b in enumerate([28, 14, 7]):
    plt.subplot(1, 3, i + 1)
    plt.hist(galaxy, bins=b, range=hist_range)
    plt.title(
        "bins={}, AIC={:.2f}".format(
            b, aic_hist(*np.histogram(galaxy, b, range=hist_range), posit=posit)
        )
    )
plt.tight_layout()
plt.show()

# %%
# ビン数を変化させたときのAICの変化をグラフに表示
bins = list(range(2, 29))
plt.plot(
    bins,
    [aic_hist(*np.histogram(galaxy, b, range=hist_range), posit=posit) for b in bins],
)
plt.xticks(bins)
plt.xlabel("bins")
plt.title(f"AIC({posit=})")
plt.show()

# %%
# AIC最小となるビン数のときのヒストグラム
b = search_hist_num(
    galaxy, method="aic", posit=posit, engine="brute", hist_range=hist_range
)[0]
plt.hist(galaxy, bins=b, range=hist_range)
plt.title(
    f"bins={b}, AIC={aic_hist(*np.histogram(galaxy, b, range=hist_range), posit=posit)}"
)
plt.show()

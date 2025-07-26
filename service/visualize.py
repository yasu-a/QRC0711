from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from core.score import r2_score_2d
from model.prediction_result import PredictionResult


def plot_state_series(
        result: PredictionResult,
        *,
        x_label: Literal["time", "index"] = "index",
        ax=None,
):
    """
    PredictionResultの状態系列（state series）をプロットします。

    Args:
        result (PredictionResult): プロット対象のPredictionResultインスタンス。
        x_label (Literal["time", "index"], optional): x軸に"時間"（"time"）または"インデックス"（"index"）を使用するかを指定します。デフォルトは"index"。
        ax: matplotlibのAxes。Noneの場合は現在のAxesを使用。

    Returns:
        bool: プロットが正常に完了した場合はTrueを返します。
    """
    assert isinstance(result, PredictionResult), type(result)

    # データ取得
    t_arr = result.t_arr
    states_array = np.array(result.states)

    # x軸データの決定
    x_data = t_arr if x_label == "time" else range(len(t_arr))

    if ax is None:
        ax = plt.gca()

    # 状態系列のプロット
    for i in range(states_array.shape[1]):
        ax.plot(x_data, states_array[:, i], label=f"State #{i}", lw=1)

    # ラベル・タイトル設定
    ax.set_title(f"States Time Series (n_states={result.states.n_states})")
    ax.set_xlabel("Time" if x_label == "time" else "Index")
    ax.set_ylabel("State value")
    ax.legend()

    # レイアウト調整・表示
    plt.tight_layout()
    if ax.figure is plt.gcf():
        plt.show()

    return True


def plot_prediction_time_series(
        result: PredictionResult,
        *,
        x_label: Literal["time", "index"] = "index",
        out_index: int | list[int] | Literal["all"] | None = None,
        ax=None,
):
    """
    PredictionResultの予測時系列（出力系列）をプロットします。

    Args:
        result (PredictionResult): プロット対象のPredictionResultインスタンス。
        x_label (Literal["time", "index"], optional): x軸に"時間"（"time"）または"インデックス"（"index"）を使用するかを指定します。デフォルトは"index"。
        out_index (int | list[int] | Literal["all"] | None, optional): プロットする出力次元。intの場合はその次元、list[int]の場合は複数次元、"all"の場合は全次元、Noneの場合はn_out=1なら0、そうでなければエラー。
        ax: matplotlibのAxes。Noneの場合は現在のAxesを使用。

    Returns:
        bool: プロットが正常に完了した場合はTrueを返します。

    Raises:
        ValueError: 出力次元が複数でout_indexが指定されていない場合、または不正な型の場合。
    """
    assert isinstance(result, PredictionResult), type(result)

    # データ取得
    t_arr = result.t_arr
    u_arr = result.u_mlt_arr
    y_true = result.y_mlt_arr
    y_pred = result.y_pred_mlt_arr
    n_out = y_true.shape[1]

    # 出力次元の選択
    if out_index is None:
        if n_out == 1:
            out_index = 0
        else:
            raise ValueError("出力の次元が複数の場合は`out_index`を指定してください")
    if out_index == "all":
        out_indexes = list(range(n_out))
    elif isinstance(out_index, int):
        out_indexes = [out_index]
    elif isinstance(out_index, list):
        out_indexes = out_index
    else:
        raise ValueError(f"不正な`out_index`型: {type(out_index)}")

    # x軸データの決定
    x_data = t_arr if x_label == "time" else range(len(t_arr))

    if ax is None:
        ax = plt.gca()

    # 入力系列のプロット
    for i in range(u_arr.shape[1]):
        ax.plot(
            x_data,
            u_arr[:, i],
            label=f'u (#{i})',
            alpha=0.7,
            linewidth=1,
            color="gray",
            ls="--"
        )

    # 出力系列（真値・予測値）のプロット
    for i in out_indexes:
        line, = ax.plot(
            x_data,
            y_true[:, i],
            label=f'y_true (#{i})',
            lw=0.7,
            ls="--"
        )
        ax.plot(
            x_data,
            y_pred[:, i],
            label=f'y_pred (#{i})',
            lw=0.7,
            color=line.get_color()
        )

    # ラベル・タイトル・レイアウト調整
    ax.set_title("Prediction time series")
    ax.set_xlabel("Time" if x_label == "time" else "Index")
    ax.set_ylabel("Output")
    ax.legend()

    return True


def plot_prediction_vs_ground_truth(
        result: PredictionResult,
        *,
        out_index: int | list[int] | Literal["all"] | None = None,
        ax=None,
):
    """
    PredictionResultの予測値と真値の散布図（予測vs真値）をプロットします。

    Args:
        result (PredictionResult): プロット対象のPredictionResultインスタンス。
        out_index (int | list[int] | Literal["all"] | None, optional): プロットする出力次元。intの場合はその次元、list[int]の場合は複数次元、"all"の場合は全次元、Noneの場合はn_out=1なら0、そうでなければエラー。
        ax: matplotlibのAxes。Noneの場合は現在のAxesを使用。

    Returns:
        bool: プロットが正常に完了した場合はTrueを返します。

    Raises:
        ValueError: 出力次元が複数でout_indexが指定されていない場合、または不正な型の場合。
    """
    assert isinstance(result, PredictionResult), type(result)

    # データ取得
    y_true_arr = result.y_mlt_arr
    y_pred_arr = result.y_pred_mlt_arr
    n_out = y_true_arr.shape[1]

    # 出力次元の選択
    if out_index is None:
        if n_out == 1:
            out_index = 0
        else:
            raise ValueError("出力の次元が複数の場合は`out_index`を指定してください")
    if out_index == "all":
        out_indexes = list(range(n_out))
    elif isinstance(out_index, int):
        out_indexes = [out_index]
    elif isinstance(out_index, list):
        out_indexes = out_index
    else:
        raise ValueError(f"不正な`out_index`型: {type(out_index)}")

    # R2スコア計算
    r2 = r2_score_2d(y_true=y_true_arr, y_pred=y_pred_arr)
    if isinstance(r2, (list, tuple, float)) or (hasattr(r2, 'shape') and r2.shape == ()):  # scalar
        r2_str = f"$R^2={float(r2):.3f}$"
    else:
        r2_str = f"$R^2={float(np.mean(r2)):.3f}$ avg."

    if ax is None:
        ax = plt.gca()

    # 散布図の作成
    for i in out_indexes:
        ax.scatter(y_true_arr[:, i], y_pred_arr[:, i], alpha=0.3, s=2, label=f"out#{i}")

    # y=xの補助線の描画
    y_min = min(y_true_arr.min(), y_pred_arr.min())
    y_max = max(y_true_arr.max(), y_pred_arr.max())
    ax.plot([y_min, y_max], [y_min, y_max], "k--", label="y_pred = y_true")

    # タイトル・ラベル・レイアウト調整
    ax.set_title("Prediction vs Ground Truth")
    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")
    ax.legend()
    # r2スコアを透明なボックスで表示
    ax.text(
        0.98, 0.02, r2_str,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5, edgecolor='none')
    )
    ax.set_aspect("equal", adjustable="box")

    return True


def plot_prediction(
        results: list[PredictionResult],
        *,
        x_label: Literal["time", "index"] = "index",
        out_index: int | list[int] | Literal["all"] | None = None,
):
    """
    PredictionResultのリストを受け取り、各行に各結果の時系列・散布図を描画します。

    Args:
        results (list[PredictionResult]): プロット対象のPredictionResultインスタンスのリスト。
        x_label (Literal["time", "index"], optional): x軸に"時間"（"time"）または"インデックス"（"index"）を使用するかを指定します。デフォルトは"index"。
        out_index (int | list[int] | Literal["all"] | None, optional): 各サンプルに同じout_indexを適用します。intの場合はその次元、list[int]の場合は複数次元、"all"の場合は全次元、Noneの場合はn_out=1なら0、そうでなければエラー。

    Returns:
        bool: プロットが正常に完了した場合はTrueを返します。

    Raises:
        AssertionError: resultsがPredictionResultのリストでない場合。
    """
    assert isinstance(results, list) and all(
        isinstance(r, PredictionResult) for r in results), "resultsはPredictionResultのリストである必要があります"
    n_samples = len(results)

    # サブプロットの作成
    fig, axs = plt.subplots(n_samples, 2, figsize=(16, 4 * n_samples), sharex=False,
                            gridspec_kw={'width_ratios': [7, 2]})
    if n_samples == 1:
        axs = [axs]

    # 各サンプルごとに描画
    for row, result in enumerate(results):
        # 1列目: 予測時系列
        ax_time = axs[row][0]
        plot_prediction_time_series(
            result,
            x_label=x_label,
            out_index=out_index,
            ax=ax_time,
        )
        ax_time.set_title(f'Prediction time series (sample {row})')
        ax_time.legend()

        # 2列目: 散布図
        ax_scatter = axs[row][1]
        plot_prediction_vs_ground_truth(
            result,
            out_index=out_index,
            ax=ax_scatter,
        )
        ax_scatter.set_title(f'Prediction vs Ground Truth (sample {row})')
        ax_scatter.legend()
        ax_scatter.set_aspect('equal', adjustable='box')

    # レイアウト調整・表示
    plt.tight_layout()
    plt.show()
    return True

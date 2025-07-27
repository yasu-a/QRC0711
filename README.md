# Quantum Reservoir Computing (QRC) Research Project

## 概要

本プロジェクトは、量子リザバーコンピューティング（Quantum Reservoir Computing,
QRC）の研究・実装を目的とした先端的な機械学習プロジェクトです。量子系の特性を利用した革新的な計算手法の開発と検証を行っています。

## 研究テーマ

- 量子リザバーコンピューティングにおける状態推定
- 時間発展モデリング
- パラメータ最適化
- 機械学習アルゴリズムの量子拡張

## 主要機能

- 量子状態の時間発展シミュレーション
- 高度な状態推定アルゴリズム
- パラメータ探索（遺伝的アルゴリズム、グリッドサーチ）
- 機械学習モデルの評価と可視化

## 必要環境

- Python 3.10+
- NumPy
- QuTiP (Quantum Toolbox in Python)
- Matplotlib
- scikit-learn

## インストール

1. リポジトリをクローン

```bash
git clone https://github.com/your-username/QRC0711.git
cd QRC0711
```

2. 依存ライブラリをインストール

```bash
pip install -r requirements.txt
```

## プロジェクト構造

```
QRC0711/
│
├── cache/                  # 一時データ・キャッシュ
│   ├── state_test.pickle   # テスト用状態データ
│   └── state_train.pickle  # トレーニング用状態データ
│
├── search/                 # パラメータ探索モジュール
│   ├── __init__.py
│   ├── base.py             # 探索アルゴリズムの基本抽象クラス
│   ├── ga.py               # 遺伝的アルゴリズム
│   └── grid.py             # グリッドサーチ
│
├── service/                # データ処理・可視化サービス
│   ├── dataset.py          # データセット生成と管理
│   └── visualize.py        # 実験結果の可視化
│
├── tests/                  # 単体テストモジュール
│   ├── conftest.py         # pytest設定
│   ├── test_axis.py
│   ├── test_dataset_v1.py
│   ├── test_dataset_v2.py
│   ├── test_fullstate.py
│   ├── test_score.py
│   ├── test_search.py
│   ├── test_seed_or_rng.py
│   └── test_utils_fullgate.py
│
├── utils/                  # ユーティリティモジュール
│   ├── __init__.py
│   ├── app_logging.py      # カスタムロギング
│   ├── axis.py             # 軸関連ユーティリティ
│   ├── dataset_v1.py       # データセット実装（v1）
│   ├── dataset_v2.py       # データセット実装（v2）
│   ├── fullgate.py         # 量子ゲート関連ユーティリティ
│   ├── fullstate.py        # 量子状態操作ユーティリティ
│   ├── score.py            # スコア計算
│   └── seed_or_rng.py      # 再現可能な乱数生成
│
├── model/                  # モデル関連クラス
│   ├── prediction_result.py    # 予測結果管理
│   └── state_array.py          # 量子状態配列抽象化
│
├── results/                # 実験結果
│   └── qrc_param_search_*.csv  # パラメータサーチ結果
│
├── main.py                 # エントリーポイント
├── experiment.py           # 主要な実験ロジック
├── model.py                # モデル定義
├── physical_system.py      # 物理システム定義
├── time_evol_solver.py     # 時間発展ソルバー
├── requirements.txt        # 依存ライブラリ
└── README.md               # プロジェクト説明書
```

### ディレクトリ詳細

#### 1. `cache/`

- 一時的な計算結果や状態データを保存
- 実験の中間結果をキャッシュ

#### 2. `search/`

- パラメータ探索アルゴリズムの実装
- 遺伝的アルゴリズム（GA）
- グリッドサーチ
- 探索戦略の抽象基底クラス

#### 3. `service/`

- データセット管理
- 実験結果の可視化
- 時系列データの処理

#### 4. `tests/`

- 各モジュールの単体テスト
- pytest設定
- コードの信頼性確保

#### 5. `utils/`

- 汎用的なユーティリティ関数
- ロギング
- スコア計算
- 乱数生成
- データセット関連ユーティリティ

#### 6. `model/`

- 予測結果の管理
- 量子状態配列の抽象化
- モデルの核心的なクラス

#### 7. `results/`

- 実験結果の保存
- パラメータサーチ結果のCSVファイル

### メインスクリプト

- `main.py`: プログラムのエントリーポイント
- `experiment.py`: 実験ロジックの中心
- `model.py`: モデル定義
- `physical_system.py`: 物理システムのモデル化
- `time_evol_solver.py`: 量子系の時間発展計算

## 使用方法

### 実験の実行

```bash
python main.py
```

### テストの実行

```bash
python -m pytest tests/
```

## 主要モジュール

- `search/`: パラメータ探索（遺伝的アルゴリズム、グリッドサーチ）
- `service/`: データセット管理、結果可視化
- `model/`: 予測結果、状態配列の管理
- `utils/`: スコア計算、データ処理、ロギング

## 研究背景

量子リザバーコンピューティングは、量子系の複雑な動的特性を機械学習の計算モデルに活用する革新的なアプローチです。従来の古典的な機械学習手法とは異なり、量子系の非線形性と高次元性を利用して、より高度な情報処理を目指します。

## 貢献方法

1. Issueをチェック
2. フォーク
3. 新しいブランチを作成 (`git checkout -b feature/amazing-feature`)
4. 変更をコミット (`git commit -m 'Add some amazing feature'`)
5. ブランチにプッシュ (`git push origin feature/amazing-feature`)
6. プルリクエストを作成

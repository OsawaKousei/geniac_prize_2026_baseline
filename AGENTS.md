# Codex / repository instructions

作業開始時に [共通の実行ルール](docs/execution_policy.md) と [CONTRIBUTING.md](CONTRIBUTING.md) を読み、従う。
チュートリアルの手順は [ローカル開発と Slurm 実行](docs/tutorial/local_remote.md) を参照する。

- 最新の依頼と許可の範囲を守る。「ローカルまで」「実行しない」という依頼をリモート実行へ広げない。
  許可済みの操作について、段階ごとの再確認は不要。
- ログインノードではアプリケーションのインストール・ダウンロードを一切行わない。
- リモートの受信・展開・環境構築・実験は、CPU 処理も含め Slurm の計算ノード内で実行する。
- 共有モデルは固定版を参照し、本人の領域に環境・キャッシュ・結果を保存する。他メンバーの領域を変更しない。
- 通常の入口は `python3 scripts/remote.py`。リモート操作の前に `prepare` と `plan` を確認する。

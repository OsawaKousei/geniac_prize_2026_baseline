# ローカル開発と Slurm 実行

ローカルの VS Code で編集し、コードを計算ノードで受け取り、環境構築と実験を Slurm ジョブで実行する。
リモートへの VS Code Server、Codex、常駐同期エージェントの導入は不要。
ログインノードは SSH の中継と Slurm の操作に使い、ファイルの受信・展開・ダウンロード・インストールは計算ノード内で行う。

初期対象は **Robosuite `cube_restack` → 少数試行 → 成功データの変換 → SFT**。
LIBERO、DMPO、提出は既存の各章を参照する。この少数試行の設定は動作確認用で、ベースライン全体の再現条件ではない。

## AI エージェントで操作する場合

共通の注意事項は [エージェント共通の実行ルール](../execution_policy.md) にまとめている。
各メンバーのローカル checkout に、次の形式で配置する。

| ツール | 指示ファイル | 共通ルールの読み込み方 |
| --- | --- | --- |
| Codex | `AGENTS.md` | 本文で共通文書の読み込みを指示。重要な実行制約も本文に記載 |
| Claude Code | `CLAUDE.md` | `@docs/execution_policy.md` による import |
| Antigravity | `GEMINI.md` | 指示ファイル内の相対パスによる include で共通文書を展開 |
| Cursor | `.cursor/rules/remote-execution.mdc` | `alwaysApply: true`。重要な実行制約と共通文書を読む指示を本文に記載 |

ルール変更時は共通文書を更新し、Codex / Cursor 本文の重要制約も矛盾しないように揃える。
これらのファイルを置いても、転送やジョブが自動で実行されることはない。
指示ファイルの形式と参照先は静的に確認できるが、実際の読み込みは使用するツール上でも確認する。
既に開いているセッションで反映されない場合は、新しいセッションで適用ルールを確認する。
リモートにエージェントを導入する必要はない。

形式の出典は [Codex の AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md)、
[Claude Code の memory / import](https://code.claude.com/docs/en/memory)、
[Antigravity の rules / include](https://antigravity.google/docs/rules/)、
[Cursor の project rules](https://cursor.com/docs/rules)。

## 1. 各メンバーのローカル設定

ローカルに Python 3.10 以上、Git、OpenSSH が必要。最初に SSH 設定とホスト鍵の確認を済ませておく。
次のコマンドはローカル設定を作るだけで、SSH 接続しない。

```bash
python3 scripts/remote.py init \
  --host '<SSH の接続名>' \
  --account '<Slurm アカウント>' \
  --partition '<計算パーティション>' \
  --shared-root '<共用ストレージの絶対パス>' \
  --remote-user '<自分のクラスタユーザー名>' \
  --port-offset <チーム内で調整した整数>
```

値は `.remote.local.json` に保存される。Git と転送の対象外で、既存ファイルは上書きしない。
認証情報は設定に書かず、SSH/Git 自身の認証機能を使う。
共有モデルの場所は既定で `<共用ストレージ>/models`。異なる場合は `--models-root` を指定する。
既存の個人環境を整理して移行する場合は、削除対象を確認して別途実施する。このスクリプトは削除しない。

初期値は次のとおり。変更は個人設定を編集してから `prepare` を実行する。

| 項目 | 初期値 |
| --- | --- |
| 個人作業領域 | `<共用ストレージ>/<自分のユーザー名>/baseline-tutorial` |
| 収集 | `cube_restack`、train seed 301–302、validation seed 391–392、worker 1 |
| 評価 | seed 401–402、学習前と SFT 後で同じ seed |
| SFT | 最大 2 optimizer steps、1 epoch、W&B 無効、validation 必須 |
| CPU ジョブ | 4 CPU、16 GB、1 時間、GPU 要求なし |
| GPU ジョブ | B300 相当の 1 GPU、8 CPU、128 GB、4 時間（SFT のホストメモリは 256 GB） |

`port_offset` は 1–57413。既定の 8110–8122 番に加算する。
これはポート予約ではない。実行直前に待受ポートを検査し、同じ個人領域の収集・評価はロックで直列化する。
他メンバーとの番号調整も必要。時間・CPU・メモリは割当条件や実際の所要時間に合わせて設定する。

## 2. ローカルで転送物を作る

```bash
python3 scripts/remote.py prepare --source '<ローカルの geniac_prize_2026_code checkout>'
python3 scripts/remote.py plan
```

`--source` で指定した checkout は変更しない。ローカルの管理用 clone を作り、既存の
`patches/geniac_prize/setup.sh` が指定する commit を Git bundle にする。
元の checkout にある未コミット変更は CaP-X の bundle に含めない。
`capx_source` を個人設定に保存すると、次回以降は `prepare` だけでよい。
指定しない場合は、ローカルの Git 認証を使って `capx_url` から clone する。
private リポジトリで HTTPS が使えない場合は、この URL を SSH 形式に変更する。

ベースライン側は現在の作業ファイルを収録する。未コミットの編集と新規コードも含む。
対象は実行に必要なソース・設定・テスト・ドキュメントで、`tmp/`、個人設定、モデル、データ、実験結果、venv、
`.git`、`.aws`、`.codex`、`.env*`、`configs/my_*` などは除外する。転送対象内の symlink は拒否する。
独自の秘密ファイルをソースの配下に置かない。

`.remote-build/<release>/` に次を生成する。

- `payload.tar`: ソースと固定 CaP-X の bundle。仮想環境やモデル本体は含まない。
- `manifest.json`: ファイルごとの SHA256、設定、CaP-X の pin、ローカル HEAD。
- `plan.json`: 転送先、サイズ、チェックサム、計算ノードでの受信と各ジョブのコマンド。
- `state.json`: この版で投入したジョブ ID。再度同じ `prepare` をしても保持する。

コードと設定から版 ID を作り、版ごとのディレクトリに配置する。既存の版は上書きしない。
コードや設定を変えたら再度 `prepare` → `plan`。実行中の版の Python パスは固定される。

## 3. リモート環境を構築・検査する

**ここから SSH 接続・転送・ジョブ投入が発生する。**

```bash
python3 scripts/remote.py setup
python3 scripts/remote.py status
python3 scripts/remote.py logs
```

`setup` は CPU の `srun` で受信・SHA256 検証・安全な展開を行い、その後 CPU の `sbatch` で環境を構築する。
受信にも計算ノードの割当が必要なので、空き状況によってコマンドが待機する。
共有モデルの取得は行わず、公開済みの固定スナップショットを参照する。
計算ノードにはインターネット接続、Git、Python 3、Bash、tar 展開に必要な標準 Python 機能が必要。
実験には `curl` と `ss` も使う。CaP-X 本体はローカルから送るため、計算ノードに private Git 認証情報を置く必要はない。
選択した公開 submodule と Python 配布物は計算ノードで取得する。

| 環境 | 内容 | 固定方法 |
| --- | --- | --- |
| baseline `.venv` | Python 3.12、変換・学習・テスト | `remote_tutorial/envs/base/pyproject.toml` と `uv.lock` |
| CaP-X `.venv` | Python 3.10、Robosuite | 固定 CaP-X の `uv.lock`、`--frozen --extra robosuite` |
| CaP-X `.venv-serve` | Python 3.12、vLLM 0.26.0 | `remote_tutorial/envs/serve/pyproject.toml` と `uv.lock` |

学習用と vLLM 用は独立した uv プロジェクトで、共通の workspace にはしない。
異なる PyTorch の版をそれぞれの lock に固定し、CUDA 13.0 の取得元も `tool.uv.sources` に明示する。
学習用の固定条件は既存の `requirements-train.txt` に合わせる。
両方とも Linux x86_64 / Python 3.12 用で、依存を `pyproject.toml` で編集し、`uv.lock` も Git に含める。
ローカル開発用のルート環境とこれらの GPU 用環境は別々に管理する。

セットアップでは `uv sync --locked` を使用し、設定と lock が不整合なら停止する。
`UV_PROJECT_ENVIRONMENT` で表の最終配置先を指定するため、プロジェクト内に別の venv は作らない。
学習用はベースライン本体を相対パスの editable 依存として含むので、別途 `pip install -e` は不要。
Robosuite は固定 CaP-X の既存設定を `uv sync --frozen --extra robosuite` で利用する。

Python・uv・NVRTC の版は `remote_tutorial/versions.json` に固定する。
B300 対策として NVRTC 12.9 を個人領域の別 target に置いて preload し、シミュレータの固定 Torch 依存を変更しない。
LIBERO 用の追加 venv は作らない。既存のセットアップスクリプトと同じ公開 7 submodule を初期化する。
venv は最終配置先で作るため、生成後にパスを移動しない。
個人キャッシュは版をまたいで再利用し、モデル共有領域には書き込まない。

構築中は固定 lock の依存検査、実際の freeze 保存、CPU テスト、chat template 作成と既存 preflight を実行する。
Robosuite 環境には上流の固定依存に由来する既知の 4 警告
（PyOpenGL、NumPy/headless OpenCV、opencv-python 未導入、decord の wheel platform）がある。
これらを監査ファイルに記録し、その他の依存不整合は失敗として扱う。
メタデータ上の無警告を保証する構成ではない。

`setup` が成功したら GPU 検査を投入する。

```bash
python3 scripts/remote.py check
python3 scripts/remote.py status
python3 scripts/remote.py logs
```

GPU 上で Torch の計算、vLLM のモデル登録、実際の `cube_restack` の reset と EGL 描画を確認する。
この検査ではモデルサーバーや trial は開始しない。モデル間通信を含む全経路は次の `collect` で確認する。

## 4. 各段階を明示して実行する

それぞれの成功とログを確認してから次に進む。

```bash
python3 scripts/remote.py submit collect
# status / logs で成功を確認
python3 scripts/remote.py submit convert
# 成功した train / validation 例があることを確認
python3 scripts/remote.py submit sft
```

`convert` は成功 trial だけを既存 CLI の rows → tag → split に渡す。
両方の split を検証し、SFT の dry-run まで行う。4 trial では成功例が不足する可能性がある。
その場合は、個人設定の `collect_seeds` を許可範囲内で増やし、新しい版として収集し直す。
train は 301–390、validation は 391–400 を使い、401–450 の評価 seed は混ぜない。
初期スクリプトには複数版のデータを自動結合する機能はない。

必要なら学習前後を同じ seed で比較する。

```bash
python3 scripts/remote.py submit eval-base
# 成功を確認
python3 scripts/remote.py submit eval-sft
```

短い SFT で epoch が終わらない場合、評価には最終 `sft/adapter` を使う。
epoch の選択結果がある場合は `best_epoch.json` に従う。
各評価の `score.json` は結果ディレクトリに保存する。2 trial の結果から性能改善を判断しない。

## 状態・再実行・保存先

ジョブはこの版の直前のジョブに `afterok` 依存を付ける。同じ段階の二重投入は既定で行わない。
失敗した最後のジョブだけは `setup --retry`、`check --retry`、`submit <stage> --retry` で再投入できる。
まだ待機・実行中なら再投入しない。後続ジョブを先に投入していた場合は、その状態の整理を別途行う。

環境構築は完了済みの段階から再開する。収集・評価の既存出力や SFT の既存出力は上書きしない。
それらの途中失敗では、必要なログを確認して個人設定の `experiment` を変更し、`prepare` から新しい版を作る。
SFT の手動 resume は [学習の説明](../train.md) を参照する。

```text
<共用ストレージ>/<自分のユーザー名>/baseline-tutorial/
  releases/<release>/             コード、3 venv、_tutorial/config/ と監査記録
  runs/<release>/                collect/、data/、sft/、eval-base/、eval-sft/
  logs/                          Slurm の stdout / stderr
  cache/                         個人の再利用キャッシュ
  tools/                         固定版 uv、Python、追加 NVRTC
```

ローカルの `.remote-build` とリモートの版は自動削除しない。不要になった版は実行中でないことを確認して整理する。
結果は必要なファイルだけ SSH 経由でローカルに取り出せる。コードと大容量成果物の双方向同期は行わない。

## 固定依存の更新

通常のセットアップでは lock を解き直さない。保守時に、各プロジェクトの `pyproject.toml` を編集して、
`versions.json` に指定した版の uv で次を実行し、`uv.lock` の差分も確認する。
`uv lock` はメタデータ解決で、GPU 用の ML ライブラリをローカル環境へインストールしない。
Python 3.12 がローカルに必要。実際の計算ノード用 Python の patch 版は `versions.json` で指定する。

```bash
uv lock --project remote_tutorial/envs/base --python 3.12
uv lock --project remote_tutorial/envs/serve --python 3.12
uv lock --project remote_tutorial/envs/base --check
uv lock --project remote_tutorial/envs/serve --check
```

特定パッケージを更新する場合は `uv lock --project <対象> --upgrade-package <名前>` を使う。
既存の root `requirements-train.txt` は通常のインストール手順用として残す。
チュートリアルで追加していた `requirements-base/serve.in` と `requirements-base/serve.lock` は不要になる。
uv の動作は [lock と sync](https://docs.astral.sh/uv/concepts/projects/sync/) と
[PyTorch の設定](https://docs.astral.sh/uv/guides/integration/pytorch/) を参照する。

ローカルでの設定生成・梱包・テストは、計算ノード上の構築成功や GPU 実行成功の証明にはならない。
新しい版は `setup`、`check`、少数試行で順に検証する。

# .dat 設定差分比較（H8）

## 結果（2026-08-20）

**Remove Static Clutter のフラグは `0x72`（1byte, 0=OFF / 1=ON）と特定した。**
fs・サンプル数・PRI の変更可否は**未検証**（今回はclutterのみ変えたため）。

| 検証 | 使用ファイル | 結果 |
|---|---|---|
| clutter OFF/ON 比較 | `atlas_log_20260820_155904.dat`（OFF）/ `..._155929.dat`（ON） | 既知フィールド0件変化。未同定領域は `0x70` の int32のみ変化 |

`0x70` の int32 を byte 分解すると変化は `0x72` の1byteのみ（他3byteは両方 `0x01` で共通）。

| offset | OFF | ON |
|---|---|---|
| 0x70 | 0x01 | 0x01 |
| 0x71 | 0x01 | 0x01 |
| **0x72** | **0x00** | **0x01** |
| 0x73 | 0x01 | 0x01 |

→ `0x64〜0xB3` は int32 の並びではなく、byte単位の設定値が詰まった領域だと分かった。
他の設定は変えていないため誤検出の可能性は低い。

次のアクション・優先度判断は `meeting/2026-09-08.md` の「進捗」セクションを参照
（fs・サンプル数・PRIの変更可否は必要性低下により見送り済み）。

---

`meeting/2026-09-08.md` H8「設定を振って`.dat`を差分比較」の実装。
`0727/atlas_dat_format.md` で未同定のまま残っているヘッダ 0x64〜0xB3 の意味と、
fs・サンプル数・PRI がアプリの設定で変更可能かを、条件を1つだけ変えた録画のヘッダ差分から特定する。

## 目的

| # | 論点 | 出所 |
|---|---|---|
| 1 | Remove Static Clutter の ON/OFF フラグが 0x64〜0xB3 のどこにあるか | `atlas_dat_format.md` §2 |
| 2 | fs・サンプル数・PRI がアプリの設定で変更可能か | `atlas_dat_format.md` §8.4（H1で最も知りたかった項目） |

## 撮影条件

Profiling設定は一切変えず、**Remove Static Clutter だけ** OFF→ON に切り替えて録画する
（default は OFF）。ターゲットは無人でよい（ヘッダしか見ないため）。

| ファイル | 条件 |
|---|---|
| `raw/*_clutteroff.dat` | Remove Static Clutter OFF（default） |
| `raw/*_clutteron.dat` | Remove Static Clutter ON 以外は同一設定 |

カメラ映像（`.cam`/`.mp4`）も同時に回した場合は `raw/` に一緒に置いてよいが、
このスクリプトが見るのは `.dat` のヘッダのみ。

## 使い方

```
python header_diff.py raw/xxx_clutteroff.dat raw/xxx_clutteron.dat
```

既知フィールド（`0727/atlas_dat_parse.py` の `parse_header` と同一定義で読む 0x0C〜0x5C）と、
未同定領域 0x64〜0xB3（int32, 20個）をそれぞれ offset ごとに比較し、
値が変わった offset に `<- CHANGED` を付ける。

## 期待する結果

- 未同定領域のうち **ちょうど1箇所だけ** が 0/1 のような2値で変化していれば、
  そこが Remove Static Clutter フラグの位置だと確定できる
- 複数箇所が変化した場合は、他の設定が連動して変わっている可能性があるため要注意
- fs・サンプル数・PRI（既知フィールド側の n_sample / sampling_ksps / chirp_interval_us）が
  UI操作で実際に変わるかどうかは、別途その設定を変えた録画を用意して同じスクリプトで確認する

# YOLO/ByteTrack の実データでの気づき（2026-08-19）

`0817/detect/detect_yolo.py` + `merge_cyclist()` を実測映像（`0817/data/atlas_log_20260816_172841.mp4`）
にかけて`../../400MHz変更用/0908/comparison/`でbefore/after比較をした際に見つかったもの。対応済みと未対応が混在する。

## 1. bicycle/motorcycle の二重検出（対応済み）

低信頼度のフレームでは、同一の自転車が bicycle と motorcycle の両方として検出されることがある
（ほぼ同じ bbox 座標で cls だけ違う2件）。ultralytics の NMS はクラスごとに独立して行われるため、
別クラス間の重複は自動では消えない。overlay 動画上でbboxが重なって見えていたのはこれが原因。

→ `0817/lib/detections.py` の `merge_cyclist()` に `_dedup_by_conf()` を追加して対応済み
（コミット `2362e82`）。

## 2. track_id が単発で -1 になるフレームがある（未対応）

`0817/detect/detect_yolo.py` の該当箇所:

```python
tid = (b.id.cpu().numpy().astype(int) if b.id is not None
       else [-1] * len(cls))            # 追跡が付かなかったフレーム
```

ByteTrack が **そのフレームだけ** 内部的にどの既存トラックにも対応付けられなかった場合、
`res.boxes.id` がフレーム全体で `None` になり、コード側が機械的に `-1` を割り当てる。

実測データ（frame=24, 43）で確認したところ、bbox 座標は前後フレーム（track_id=1）と
ほぼ同じ位置だった。**物理的には同じ自転車で、トラッカーが一瞬 ID 紐付けに失敗しただけ**。

### なぜ気にする必要があるか

track_id でグルーピングする処理（`overlay_detections.py` の「短いtrack_id」診断、将来の
S3 レーダー対応付け）は、`-1` を挟むと「別トラック」として扱われ、連続性が途切れて見える。
1フレームだけ挟まる分にはtrack自体は大きくは崩れないが、`-1` が複数フレームに渡って出る
非連続なフレームでは `groupby("track_id")` が本来無関係なフレーム同士を1つの「track_id=-1」
としてまとめてしまう副作用もある（今回は2フレームだけなので実害は小さい）。

### 対応方針（未着手）

`detect_yolo.py` 自体を直すのではなく、後処理側（`merge_cyclist` か、`../../400MHz変更用/0908/postprocess/track.py`
のような追跡ロジック）で `-1` を「不明」として扱い、前後の track_id と位置が近ければ
同一トラックとして繋ぎ直す、という形が良さそう。優先度は週末の撮影・データ収集より低い。

---
name: weekly-slides
description: 週次ミーティング報告用のパワポ（日本語・英語）を、meeting/ の議事メモ・週次フォルダのREADME・docs/experiments.md から生成する
---

# weekly-slides

研究室の週次ミーティング報告スライド（.pptx、日本語版・英語版の2本）を作るスキル。

内容の要約・翻訳はこのSKILL.mdの指示に従ってClaude自身が行う。pptxへの流し込みは
`scripts/build_pptx.py`（決定的なコード、文章判断はしない）が担当する。

## 全体の流れ

1. 対象週を決める（ユーザーが日付を指定しなければ `meeting/` 内の最新の `YYYY-MM-DD.md` を使う）
2. 対応する週次フォルダを解決する（下記「週次フォルダの解決」参照）
3. 資料を読む（下記「読む資料」参照）
4. `slides_content.json` を対象週の週次フォルダ直下に作成する（下記「JSONスキーマ」参照）
5. スクリプトを実行してpptxを生成する:
   ```
   python .claude/skills/weekly-slides/scripts/build_pptx.py \
     --content <週次フォルダ>/slides_content.json \
     --outdir <週次フォルダ> \
     --base-dir .
   ```
   （リポジトリルートから実行すること。`--template` は省略可、`templates/` 配下の唯一の
   .pptx を自動で使う）
6. `<週次フォルダ>/slides_ja.pptx` と `slides_en.pptx` が生成される。ユーザーに報告し、
   内容を直したい場合は `slides_content.json` を編集して手順5だけ再実行すればよいと伝える
   （スクリプトは決定的なので、文章の手直しにClaudeを呼び直す必要はない）

## 週次フォルダの解決

`meeting/YYYY-MM-DD.md` のファイル名からMMDDを取り出し、リポジトリルート直下で
その4桁から始まるディレクトリ（`0908/`, `0705meeting/`, `0713poster/` など、接尾辞の
有無は問わない）をglobで探す。

- 一致が1件 → それを使う
- 一致が0件 or 複数件 → どのフォルダを使うかユーザーに確認する

## 読む資料

| 資料 | 使いみち |
|---|---|
| `meeting/<対象週>.md` | 「前回のおさらい」「課題・論点」「次回に向けて」の主な材料 |
| 週次フォルダの `README.md` | **「今フェーズの目的・手法」の主な材料**。このファイルは既に「このフェーズが何を目指し、各部品がどう繋がるか」をトップダウンでまとめてある |
| 週次フォルダ内サブフォルダの `*.md`・既存の `*.png` | 「今週の進捗・結果」の具体的な作業内容と図表 |
| `docs/experiments.md` | 現在の最良結果の数値・「今後の検討事項」の補足 |

**画像は週次フォルダ内に既に保存されているPNGのみを使う**。notebookの埋め込み出力を
新規に抽出することはしない（必要な図があれば、事前にnotebook側でPNG保存してもらう）。

## スライド構成（固定）

1. 表紙
2. 前回のおさらい — 前回/今回のmdの「決まったこと」節など
3. **今フェーズの目的・手法** — 週次フォルダのREADME.mdのトップダウン要約から。
   **研究全体の背景（`docs/overview.md`）には触れない**。聴衆は既に知っている前提。
   このフェーズで具体的に何に取り組み、どういう手法（パイプライン）を使っているかに絞る
4. 今週の進捗・結果（図表つき、1〜2枚/スライド。多ければスライドを複数に分ける）
5. 課題・論点
6. 次回に向けて

## JSONスキーマ (`slides_content.json`)

```jsonc
{
  "cover": {
    "title":      {"ja": "...", "en": "..."},
    "event_name": {"ja": "研究室ミーティング", "en": "Lab Meeting"},
    "authors":    {"ja": "山久保浩介", "en": "Kousuke Yamakubo"},
    "date":       {"ja": "2026年9月8日", "en": "September 8, 2026"},
    "year":       {"ja": "2026", "en": "2026"}   // 必須。省略不可
    // "affiliation" は省略可（省略時はデフォルトの所属表記を使う）
  },
  "sections": [
    {
      "id": "recap",              // 自由な識別子。同じidを複数スライドに分けて使ってよい
      "layout": "OBJECT",         // "OBJECT"（タイトル+本文） or "2カラム"（タイトル+本文+画像）
      "title":   {"ja": "前回のおさらい", "en": "Recap of Last Meeting"},
      "bullets": [
        {"level": 0, "text": {"ja": "...", "en": "..."}},
        {"level": 1, "text": {"ja": "...", "en": "..."}}   // levelでインデント階層
      ]
    },
    {
      "id": "progress",
      "layout": "2カラム",
      "title": {"ja": "今週の進捗", "en": "This Week's Progress"},
      "bullets": [ /* 左カラムの箇条書き */ ],
      "images": [
        {
          "path": "0908/matching/example.png",  // リポジトリルートからの相対パス
          "caption": {"ja": "...", "en": "..."}   // 省略可
        }
      ]
    }
  ]
}
```

- `layout: "2カラム"` のスライドだけ `images` を指定できる（右カラムに縦積みで配置される）
- 本文・キャプションは全てja/enの両方を自然な文章で書く（直訳ではなく、英語として読める文章にする）
- 表紙の `affiliation` を省略した場合のデフォルトは大阪大学情報科学研究科＋理研R-CCS
  （`build_pptx.py` 内 `DEFAULT_AFFILIATION` を参照。所属が変わったらそこを書き換える）

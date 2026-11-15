# discord-numeragon-ai-bot
ぬめらごんの分身を作ろうプロジェクト

サーバーの会話の雰囲気（口調・スラング・絵文字・よく出る話題）を学習し、
人の会話に自然なタイミングで混ざるBot。

## 仕組み
- チャンネルの発言をサーバーごとに直近2,000件まで保存（data/<guild_id>/messages.jsonl）
- 150件たまるごとにLLMが「雰囲気プロファイル」を要約・更新（profile.json）
- 返信時は 直近25件の会話 + 雰囲気プロファイル をLLMへ渡して生成
- 会話参加ルール: メンション/返信は即応答。通常時は「2人以上の会話が3件以上続いた後、
  20〜55秒の間が空いた時」に確率35%で参加、チャンネルごとに最低4分のクールダウン
- 人間らしく typing 表示を入れてから送信

## 頭脳の切り替え（config.json の llm_backend）
- `openai`: OpenAI互換APIを直接使用（.env の OPENAI_API_KEY）。高速・利用枠ゼロ
- `relay`: data/relay/inbox にリクエストを置き、外部エージェント（Muse）が
  outbox に回答を返す方式。APIキー不要だが応答は遅め

## 起動
1. Discord開発者ポータルでアプリ作成 → Botトークン発行 →
   Privileged Gateway Intents の **MESSAGE CONTENT INTENT** をON
2. OAuth2 URL Generator で bot スコープ（権限: メッセージ送信/閲覧/履歴閲覧）を発行しサーバーへ招待
3. `.env` に DISCORD_TOKEN（方式がopenaiなら OPENAI_API_KEY も）を記入
4. 常駐: systemd ユーザーサービス `discord-vibe-bot.service` を有効化

## オーナーコマンド（config.json の owner_id を設定した場合）
- `!botpause` / `!botresume` : そのチャンネルでの自動参加を停止/再開
- `!botprofile` : 雰囲気プロファイルを即時更新して表示

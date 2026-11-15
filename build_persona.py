#!/usr/bin/env python3
"""Build the 'numeragon' persona (style profile + example lines) from the
backfilled corpus, using the same LLM relay as the bot."""
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bot  # noqa: E402

BASE = Path(__file__).resolve().parent
GUILD = "1354424178194776105"
GDIR = BASE / "data" / GUILD


async def main():
    corpus = [json.loads(l) for l in open(GDIR / "persona_corpus.jsonl", encoding="utf-8")]
    texts = [c["content"] for c in corpus if c.get("content")]
    print(f"corpus: {len(texts)} messages")
    # examples: short, self-contained lines
    random.seed(42)
    ex_pool = sorted({t for t in texts
                      if 4 <= len(t) <= 50 and "http" not in t and not t.startswith("<@")})
    examples = random.sample(ex_pool, min(80, len(ex_pool)))
    # sample for the style summary: recent 350 + random 350 across history
    sample = texts[-350:] + random.sample(texts, min(350, len(texts)))
    joined = "\n".join(sample)
    system = ("あなたは話し方の分析者です。ある人物のDiscord発言サンプルから、"
              "その人物になりきるための『話し方プロファイル』を日本語で作ってください。")
    user = ("次の観点を具体的に、再現可能なレベルでまとめてください:\n"
            "- 文の長さと改行・分割の癖（短文連投か、一文完結か）\n"
            "- 語尾・口調（タメ口の種類、よく使う語尾や言い回し）\n"
            "- 笑いの表現（w、草、絵文字、顔文字などの使い方と頻度）\n"
            "- よく使う単語・スラング・口癖（具体例を多く挙げる）\n"
            "- 相手への絡み方（ツッコミ、煽り、自虐、質問の仕方）\n"
            "-テンションの特徴（テンションが上がる場面、下がる場面）\n"
            "- 話題の傾向\n"
            "箇条書き中心で、1200文字以内。\n\n--- 発言サンプル ---\n" + joined[-14000:])
    profile = await bot.llm(system, user)
    out = {"profile": profile, "examples": examples,
           "source_count": len(texts)}
    (GDIR / "persona.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PERSONA_PROFILE:\n", profile[:900])
    print("EXAMPLES_SAMPLE:", examples[:8])


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Backfill past messages of the target user (Moriya's persona source) and
general channel history from the guild, via the Discord REST API."""
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE = Path(__file__).resolve().parent
GUILD = "1354424178194776105"
TARGET_IDS = {"1180881836398870543", "733629007291088898"}  # 2428 (main) + alt, both "ぬめらごん"
MAX_PAGES = 15

env = {}
for line in open(BASE / ".env"):
    if "=" in line:
        k, v = line.strip().split("=", 1)
        env[k] = v
TOK = env["DISCORD_TOKEN"]


def get(path):
    req = urllib.request.Request("https://discord.com/api/v10" + path,
                                 headers={"Authorization": "Bot " + TOK,
                                          "User-Agent": "DiscordBot (https://discord.com, 1.0)"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=25))
    except urllib.error.HTTPError as e:
        return {"_error": e.code}
    except Exception as e:
        return {"_error": str(e)}


def main():
    channels = get(f"/guilds/{GUILD}/channels")
    texts = [c for c in channels if c.get("type") == 0]
    persona = []
    general = []
    for c in texts:
        cid = c["id"]
        before = None
        fetched = 0
        for _ in range(MAX_PAGES):
            q = f"/channels/{cid}/messages?limit=100"
            if before:
                q += f"&before={before}"
            msgs = get(q)
            if not isinstance(msgs, list) or not msgs:
                break
            for m in msgs:
                fetched += 1
                a = m["author"]
                content = (m.get("content") or "").strip()
                rec = {"id": m["id"], "author_id": a["id"],
                       "author": (m.get("member", {}).get("nick") or a.get("global_name") or a["username"]),
                       "content": content, "channel_id": int(cid),
                       "ts": time.mktime(time.strptime(m["timestamp"][:19], "%Y-%m-%dT%H:%M:%S"))}
                if not a.get("bot"):
                    general.append(rec)
                    if a["id"] in TARGET_IDS and content:
                        persona.append(rec)
            before = msgs[-1]["id"]
            if len(msgs) < 100:
                break
            time.sleep(0.35)
        print(f"channel {c['name']}: fetched {fetched}", flush=True)
    gdir = BASE / "data" / GUILD
    gdir.mkdir(parents=True, exist_ok=True)
    with open(gdir / "persona_corpus.jsonl", "w", encoding="utf-8") as f:
        for r in sorted(persona, key=lambda x: x["ts"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(gdir / "backfill_general.jsonl", "w", encoding="utf-8") as f:
        for r in sorted(general, key=lambda x: x["ts"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"PERSONA_MESSAGES: {len(persona)}")
    print(f"GENERAL_MESSAGES: {len(general)}")


if __name__ == "__main__":
    main()

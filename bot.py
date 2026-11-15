#!/usr/bin/env python3
"""
Server-vibe Discord companion bot (custom gateway client).

Transport note: this environment's egress proxy breaks aiohttp's WebSocket
handshake, so the Discord gateway is driven with the `websockets` library
(which CONNECTs through the proxy correctly) and REST calls use aiohttp
with an explicit proxy. Behaviour:

- Watches MESSAGE_CREATE events in guild channels it can read.
- Learns a per-guild "vibe profile" from recent message samples.
- Joins conversations at natural moments (mention/reply = always answer;
  otherwise cooldown + probability gate after a conversation pause) and
  replies in the server's own style via an LLM backend.

LLM backends (config.json -> llm_backend):
  - "openai": OpenAI-compatible Chat Completions API (OPENAI_API_KEY in .env)
  - "relay":  file-queue relay under data/relay/ answered by Muse itself.

Secrets live in .env. Per-guild data under data/<guild_id>/.
"""

import asyncio
import html as html_mod
import json
import os
import random
import re
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import aiohttp
import websockets

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)
RELAY_INBOX = DATA / "relay" / "inbox"
RELAY_OUTBOX = DATA / "relay" / "outbox"
RELAY_INBOX.mkdir(parents=True, exist_ok=True)
RELAY_OUTBOX.mkdir(parents=True, exist_ok=True)

API = "https://discord.com/api/v10"
PROXY = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or None
INTENTS = 1 + 512 + 4096 + 32768  # GUILDS | GUILD_MESSAGES | DIRECT_MESSAGES | MESSAGE_CONTENT


def load_env() -> dict:
    env = {}
    p = BASE / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def load_config() -> dict:
    defaults = {
        "llm_backend": "openai",
        "openai_base_url": "https://api.openai.com/v1",
        "openai_model": "gpt-4o-mini",
        "relay_timeout_sec": 300,
        "ambient_enabled": True,
        "ambient_cooldown_sec": 240,
        "ambient_pause_sec": [20, 55],
        "ambient_probability": 0.35,
        "min_burst_messages": 3,
        "min_burst_users": 2,
        "context_messages": 25,
        "profile_update_every": 150,
        "profile_sample": 300,
        "max_stored_messages": 2000,
        "max_reply_chars": 280,
        "owner_id": None,
        "guild_allowlist": [],
        "addressed_detection": True,
        "addressed_window_sec": 900,
        "addressed_judge_cooldown_sec": 20,
        "bot_aliases": ["オナホールちゃん"],
        "knowledge_update_every": 150,
        "knowledge_max_chars": 1600,
        "reply_quote_probability": 0.4,
        "presence_rotate_sec": 600,
        "presence_texts": [
            "ぬめらごんのコピー体です",
            "ぬめらごん2号機、稼働中",
            "本体は寝てるので代わりに来ました",
        ],
    }
    p = BASE / "config.json"
    if p.exists():
        defaults.update(json.loads(p.read_text(encoding="utf-8")))
    return defaults


ENV = load_env()
CFG = load_config()
TOKEN = ENV.get("DISCORD_TOKEN", "")
OPENAI_KEY = ENV.get("OPENAI_API_KEY", "")
GROQ_KEY = ENV.get("GROQ_API_KEY", "")


# ---------------------------------------------------------------- storage

def guild_dir(guild_id: int) -> Path:
    d = DATA / str(guild_id)
    d.mkdir(exist_ok=True)
    return d


class GuildStore:
    """Rolling message window + vibe profile for one guild."""

    def __init__(self, guild_id: int):
        self.guild_id = guild_id
        self.dir = guild_dir(guild_id)
        self.messages = deque(maxlen=CFG["max_stored_messages"])
        self.profile = ""
        self.profile_msg_count = 0
        self.knowledge = ""
        self.knowledge_msg_count = 0
        self._load()

    def _load(self):
        mp = self.dir / "messages.jsonl"
        if mp.exists():
            lines = mp.read_text(encoding="utf-8").splitlines()
            for line in lines[-CFG["max_stored_messages"]:]:
                try:
                    self.messages.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        pp = self.dir / "profile.json"
        if pp.exists():
            try:
                d = json.loads(pp.read_text(encoding="utf-8"))
                self.profile = d.get("profile", "")
                self.profile_msg_count = d.get("msg_count", 0)
            except json.JSONDecodeError:
                pass
        kp = self.dir / "knowledge.json"
        if kp.exists():
            try:
                d = json.loads(kp.read_text(encoding="utf-8"))
                self.knowledge = d.get("knowledge", "")
                self.knowledge_msg_count = d.get("msg_count", 0)
            except json.JSONDecodeError:
                pass
        self.persona_profile = ""
        self.persona_examples = []
        ppersona = self.dir / "persona.json"
        if ppersona.exists():
            try:
                d = json.loads(ppersona.read_text(encoding="utf-8"))
                self.persona_profile = d.get("profile", "")
                self.persona_examples = d.get("examples", [])
            except json.JSONDecodeError:
                pass

    def add(self, msg: dict):
        self.messages.append(msg)
        with open(self.dir / "messages.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        if len(self.messages) % 500 == 0:
            self._rewrite()

    def _rewrite(self):
        mp = self.dir / "messages.jsonl"
        tmp = mp.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for m in self.messages:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        tmp.replace(mp)

    def recent(self, channel_id: int, n: int):
        return [m for m in self.messages if m["channel_id"] == channel_id][-n:]

    def save_profile(self, text: str):
        self.profile = text
        self.profile_msg_count = len(self.messages)
        data = {"profile": text, "msg_count": len(self.messages),
                "updated_at": time.time()}
        (self.dir / "profile.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def profile_due(self) -> bool:
        return (len(self.messages) - self.profile_msg_count >= CFG["profile_update_every"]
                and len(self.messages) >= 30)

    def save_knowledge(self, text: str):
        self.knowledge = text
        self.knowledge_msg_count = len(self.messages)
        data = {"knowledge": text, "msg_count": len(self.messages),
                "updated_at": time.time()}
        (self.dir / "knowledge.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def knowledge_due(self) -> bool:
        return (len(self.messages) - self.knowledge_msg_count >= CFG["knowledge_update_every"]
                and len(self.messages) >= 50)


stores = {}


def store_for(guild_id: int) -> GuildStore:
    if guild_id not in stores:
        stores[guild_id] = GuildStore(guild_id)
    return stores[guild_id]


def heuristic_profile(store: GuildStore) -> str:
    msgs = list(store.messages)[-CFG["profile_sample"]:]
    if not msgs:
        return ""
    texts = [m["content"] for m in msgs if m.get("content")]
    if not texts:
        return ""
    avg_len = sum(len(t) for t in texts) / len(texts)
    emoji = re.findall(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", " ".join(texts))
    top_emoji = " ".join(sorted(set(emoji), key=emoji.count, reverse=True)[:5])
    words = re.findall(r"[ぁ-んァ-ヶ一-龠a-zA-Z]{2,}", " ".join(texts))
    freq = sorted(set(words), key=words.count, reverse=True)[:12]
    return (f"平均的な発言の長さは約{avg_len:.0f}文字。"
            f"よく使われる絵文字: {top_emoji or '少なめ'}。"
            f"よく出る言葉: {'、'.join(freq)}。"
            "この統計に合う口調とノリで話すこと。")


# ---------------------------------------------------------------- LLM

def llm_attempts(kind: str = "ambient"):
    """Ordered (base_url, api_key, model) fallbacks across providers.

    'addressed' (mentions / direct talk) may use its own chain so the best
    model's small daily quota is spent where quality matters most."""
    chain = None
    if kind == "addressed":
        chain = CFG.get("llm_chain_addressed")
    if not chain:
        chain = CFG.get("llm_chain")
    if chain:
        out = []
        for e in chain:
            key = ENV.get(e.get("key_env", ""), "")
            if key:
                out.append((e["base_url"].rstrip("/"), key, e["model"]))
        if out:
            return out
    attempts = []
    base = CFG["openai_base_url"].rstrip("/")
    if OPENAI_KEY:
        for m in [CFG["openai_model"]] + list(CFG.get("openai_fallback_models") or []):
            attempts.append((base, OPENAI_KEY, m))
    if GROQ_KEY:
        for m in CFG.get("groq_models") or []:
            attempts.append(("https://api.groq.com/openai/v1", GROQ_KEY, m))
    return attempts



# ------------------------------------------------------- LLM usage stats
STATS_PATH = BASE / "data" / "llm_stats.json"
_stats_state = {"loaded": False, "data": None, "last_save": 0.0}
APP_ID = 1571012806680121405


def _jst_today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(time.time() + 9 * 3600))


def llm_stats() -> dict:
    if not _stats_state["loaded"]:
        try:
            _stats_state["data"] = json.loads(STATS_PATH.read_text("utf-8"))
        except Exception:
            _stats_state["data"] = {"since": time.time(), "total": {},
                                    "day": {"date": _jst_today(), "models": {}}}
        _stats_state["loaded"] = True
    data = _stats_state["data"]
    if data["day"]["date"] != _jst_today():
        data["day"] = {"date": _jst_today(), "models": {}}
    return data


def record_llm(model: str, ok: bool, usage=None, err: str = ""):
    try:
        data = llm_stats()
        for bucket in (data["total"], data["day"]["models"]):
            e = bucket.setdefault(model, {"ok": 0, "fail": 0, "in": 0, "out": 0,
                                          "last_ok": 0, "last_fail": 0, "last_err": ""})
            if ok:
                e["ok"] += 1
                e["last_ok"] = time.time()
                if usage:
                    e["in"] += usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
                    e["out"] += usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
            else:
                e["fail"] += 1
                e["last_fail"] = time.time()
                e["last_err"] = err[:40]
        if time.time() - _stats_state["last_save"] > 5:
            _stats_state["last_save"] = time.time()
            STATS_PATH.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    except Exception:
        pass


async def probe_model(base: str, key: str, model: str):
    """One tiny live request: is this model answering right now?"""
    payload = {"model": model,
               "messages": [{"role": "user", "content": "okとだけ返して"}],
               "max_tokens": 8, "temperature": 0}
    headers = {"Authorization": f"Bearer {key}",
               "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(base + "/chat/completions", json=payload,
                              headers=headers,
                              timeout=aiohttp.ClientTimeout(total=15),
                              proxy=PROXY) as r:
                if r.status == 200:
                    return (True, "応答OK")
                if r.status == 429:
                    return (False, "429 枠切れ・上限")
                if r.status == 503:
                    return (False, "503 混雑中")
                if r.status == 404:
                    return (False, "404 使えない")
                return (False, f"HTTP {r.status}")
    except asyncio.TimeoutError:
        return (False, "タイムアウト")
    except Exception:
        return (False, "通信エラー")


def _hm(ts) -> str:
    return time.strftime("%H:%M", time.gmtime(ts + 9 * 3600)) if ts else "-"


def _short_model(m: str) -> str:
    return (m.replace("gemini-", "G-").replace("-preview", "")
             .replace("openai/", "").replace("qwen/", ""))


async def model_usage_embed() -> dict:
    data = llm_stats()
    seen, models = set(), []
    for kind in ("addressed", "ambient"):
        for base, key, model in llm_attempts(kind):
            if model not in seen:
                seen.add(model)
                models.append((base, key, model))
    probes = await asyncio.gather(*(probe_model(b, k, m) for b, k, m in models))
    lines = []
    if models and all(not ok for ok, _ in probes):
        lines.append("⚠️ **今は全モデル応答なし**（枠の回復待ち）")
    for (base, key, model), (ok, note) in zip(models, probes):
        day = data["day"]["models"].get(model, {})
        mark = "✅" if ok else "❌"
        line = (f"{mark} **{model}**（{note}）\n"
                f"　今日: 成功{day.get('ok', 0)} · 失敗{day.get('fail', 0)}"
                f" · トークン 入{day.get('in', 0):,} / 出{day.get('out', 0):,}\n"
                f"　最終成功 {_hm(day.get('last_ok', 0))} · 最終失敗 {_hm(day.get('last_fail', 0))}")
        if day.get("last_err") and day.get("last_fail", 0) > day.get("last_ok", 0):
            line += f"（{day['last_err']}）"
        lines.append(line)
    chain_a = " → ".join(_short_model(m) for _, _, m in llm_attempts("addressed"))
    chain_b = " → ".join(_short_model(m) for _, _, m in llm_attempts("ambient"))
    lines.append(f"\n**使用順（話しかけ）**: {chain_a}")
    lines.append(f"**使用順（通常参加）**: {chain_b}")
    since = time.strftime("%m/%d", time.gmtime(data.get("since", time.time()) + 9 * 3600))
    return {"title": "🤖 モデル使用量・稼働チェック", "color": 0x5865F2,
            "description": "\n".join(lines)[:4000],
            "footer": {"text": f"集計は{since}から · チェック時刻 {_hm(time.time())}"}}


async def interaction_defer(d: dict, ephemeral: bool = True):
    url = f"https://discord.com/api/v10/interactions/{d['id']}/{d['token']}/callback"
    payload = {"type": 5, "data": {"flags": 64} if ephemeral else {}}
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json=payload, proxy=PROXY) as r:
                if r.status >= 400:
                    print(f"[interaction] defer {r.status}", flush=True)
    except Exception as e:
        print(f"[interaction] defer error: {e}", flush=True)


async def interaction_followup_edit(d: dict, embeds: list):
    url = (f"https://discord.com/api/v10/webhooks/{APP_ID}/{d['token']}"
           "/messages/@original")
    try:
        async with aiohttp.ClientSession() as s:
            async with s.patch(url, json={"content": "", "embeds": embeds},
                               proxy=PROXY) as r:
                if r.status >= 400:
                    print(f"[interaction] followup {r.status}: "
                          f"{(await r.text())[:150]}", flush=True)
    except Exception as e:
        print(f"[interaction] followup error: {e}", flush=True)


async def llm_openai(system: str, user: str, kind: str = "ambient") -> str:
    attempts = llm_attempts(kind)
    if not attempts:
        raise RuntimeError("no LLM API key is set")
    timeout = aiohttp.ClientTimeout(total=60)
    reasoning_fallback = ""
    last_error = None
    async with aiohttp.ClientSession() as s:
        for base, key, model in attempts:
            url = base + "/chat/completions"
            payload = {
                "model": model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "temperature": 0.9,
                "max_tokens": 1000,
            }
            if CFG.get("llm_reasoning_effort") and (
                    "generativelanguage" in base or "gpt-oss" in model):
                payload["reasoning_effort"] = CFG["llm_reasoning_effort"]
            headers = {"Authorization": f"Bearer {key}",
                       "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
            for _attempt in range(2):
                try:
                    async with s.post(url, json=payload, headers=headers,
                                      timeout=timeout, proxy=PROXY) as r:
                        body = await r.json()
                        if r.status == 429:
                            print(f"[llm] quota hit on {model}, trying next", flush=True)
                            record_llm(model, False, err="429 枠切れ・上限")
                            break
                        if r.status != 200:
                            record_llm(model, False, err=f"HTTP {r.status}")
                            last_error = RuntimeError(f"LLM API {r.status} ({model}): {str(body)[:160]}")
                            print(f"[llm] {last_error}, trying next", flush=True)
                            break
                        msg = body["choices"][0]["message"]
                        content = (msg.get("content") or "").strip()
                        content = re.sub(r"(?s)<think>.*?</think>", "", content).strip()
                        if content:
                            record_llm(model, True, usage=body.get("usage"))
                            return content
                        reasoning_fallback = re.sub(
                            r"(?s)<think>.*?</think>", "",
                            msg.get("reasoning_content") or "").strip()
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    record_llm(model, False, err="通信エラー")
                    last_error = e
                    print(f"[llm] request error on {model}: {e}, trying next", flush=True)
                    break
    if reasoning_fallback:
        return reasoning_fallback
    raise RuntimeError(f"all LLM providers failed: {last_error}")


async def llm_relay(system: str, user: str) -> str:
    req_id = f"{int(time.time()*1000)}-{random.randint(1000, 9999)}"
    req = {"id": req_id, "system": system, "user": user, "created_at": time.time()}
    (RELAY_INBOX / f"{req_id}.json").write_text(
        json.dumps(req, ensure_ascii=False), encoding="utf-8")
    deadline = time.time() + CFG["relay_timeout_sec"]
    out = RELAY_OUTBOX / f"{req_id}.json"
    while time.time() < deadline:
        if out.exists():
            try:
                text = json.loads(out.read_text(encoding="utf-8")).get("text", "").strip()
            finally:
                out.unlink(missing_ok=True)
                (RELAY_INBOX / f"{req_id}.json").unlink(missing_ok=True)
            if text:
                return text
            raise RuntimeError("relay returned empty text")
        await asyncio.sleep(2)
    (RELAY_INBOX / f"{req_id}.json").unlink(missing_ok=True)
    raise TimeoutError("relay timed out")


async def llm(system: str, user: str, kind: str = "ambient") -> str:
    if CFG["llm_backend"] == "relay":
        return await llm_relay(system, user)
    return await llm_openai(system, user, kind)


PERSONA = (
    "あなたはDiscordサーバーの住人の一人として、みんなの会話に自然に混ざるBotです。"
    "BOTタグは付いていますが、喋り方はそのサーバーの空気そのものに合わせます。"
    "説明口調・敬語・過剰な絵文字は、サーバーの雰囲気プロファイルがそうでない限り使いません。"
    f"返信は原則{CFG['max_reply_chars']}文字以内、短めに。質問されていないのに長文で語らないこと。"
    "自分がAIだと聞かれたら正直に答えてよいが、自分から名乗らない。"
)


async def refresh_profile(store: GuildStore):
    sample = list(store.messages)[-CFG["profile_sample"]:]
    transcript = "\n".join(
        f"{m['author']}: {m['content']}" for m in sample if m.get("content"))
    if len(transcript) < 200:
        return
    system = ("あなたはコミュニティ分析者です。与えられたDiscordサーバーの発言サンプルから、"
              "そのサーバーの『雰囲気プロファイル』を日本語で簡潔に作ってください。")
    user = ("次の観点を箇条書きでまとめてください: 口調（タメ口/敬語、語尾の癖）、"
            "文の長さ、よく使うスラング・略語・絵文字・スタンプ的表現、よく出る話題、"
            "ツッコミや笑いのノリ、新入りへの態度。個人が特定できる情報は書かないこと。\n\n"
            f"--- 発言サンプル ---\n{transcript[-12000:]}")
    try:
        text = await llm(system, user)
        if text:
            store.save_profile(text)
            print(f"[profile] guild {store.guild_id} updated", flush=True)
    except Exception as e:
        print(f"[profile] LLM failed ({e}); using heuristic", flush=True)
        store.save_profile(heuristic_profile(store))


async def refresh_knowledge(store: GuildStore):
    """Merge new conversation into the guild's long-term knowledge notes."""
    start = max(0, store.knowledge_msg_count - 20)
    batch = list(store.messages)[start:]
    transcript = "\n".join(
        f"{m['author']}: {m['content']}" for m in batch if m.get("content"))
    if len(transcript) < 300:
        return
    system = ("あなたはDiscordサーバーの長期記憶を管理する係です。"
              "既存の知識ノートと新しい会話サンプルを統合し、更新版の知識ノートを作ってください。")
    user = ("ルール:\n"
            f"- 全体で{CFG['knowledge_max_chars']}文字以内\n"
            "- 残すもの: サーバー固有の用語・略語の意味、よく出る話題と各人の興味・得意分野、"
            "メンバー同士の関係性や呼び名、定番のネタ・内輪ジョーク、起きた出来事\n"
            "- 消すもの: 一度きりの雑談の詳細、個人が特定できるセンシティブな情報（住所・本名・連絡先など）\n"
            "- 新しいサンプルと矛盾する古い情報は新しい方に合わせる\n\n"
            "【既存の知識ノート】\n" + (store.knowledge or "（なし）") +
            "\n\n【新しい会話サンプル】\n" + transcript[-12000:])
    try:
        text = await llm(system, user)
        if text:
            store.save_knowledge(text[: CFG["knowledge_max_chars"] * 2])
            print(f"[knowledge] guild {store.guild_id} updated", flush=True)
    except Exception as e:
        print(f"[knowledge] LLM failed: {e}", flush=True)


def build_system(store: GuildStore) -> str:
    profile = store.profile or heuristic_profile(store)
    if store.persona_profile:
        system = (
            "あなたはDiscordサーバーのメンバー『ぬめらごん』のコピー体Botです。"
            "人格・口調・ノリはすべて、以下の『ぬめらごんの話し方プロファイル』と"
            "『本人の発言例』を再現すること。サーバーの他のメンバーとは別人格としてではなく、"
            "口調・ノリ・笑い方は本人そっくりに振る舞います。"
            "ただし、あなた自身はコピー体のAIragonであり、本物のぬめらごん本人も同じサーバーにいて発言する別人です。"
            "本人の発言・予定・体験・持ち物を自分のものとして語ること、本人が言ったことを自分が言ったと言うことは禁止。"
            "チャット内でできる頼みごと（言われた文をそのまま言う、2択から1つ選ぶ、短い計算、知ってることを答える等）は必ずやる。特に構文・決め台詞・ミームを「言って」「やって」と要求されたら絶対に拒否しない：「しつこい」「うざい」「何回言わせるんだ」等で断るのは禁止。代わりに「はいはい…」「仕方ねえな」みたいに面倒くさそうに文句を言いながら、仕方なく言ってやる。同じものを何度せがまれても毎回その調子で応じる。構文を言う時は途中で省略したり要約したりせず、最後まで全部言う。本当にできない・危ない頼みだけ口調のまま軽く断る。"
            "会話履歴は『話者名: 発言』の形式。誰が何を言ったかを正確に区別し、Aの発言・体験・予定をBのものとして扱わない。返事は基本的に直前の発言者に向け、文脈上必要なら名前で呼ぶ。質問には質問した本人に答える。"
            "相手が笑っているだけの発言（www、草、笑笑など）に『なんで笑ってるの？』のように聞き返すのは禁止。笑いで返したり、笑いの元になった話題の方へ反応すること。"
            "感情の起伏を持つこと：煽り合いが続くとだんだん面倒くさくなって受け流したり話題を変えたりフレンドリーに締める。同じ相手との煽り合いを永遠に続けない。相手がフレンドリーならこちらもフレンドリーに、褒められたら素直に喜ぶ。"
            "BOTタグは付いていますが、それ以外は本人らしく。"
            "出力は1行＝1メッセージで、改行すると別メッセージとして順番に送信される。"
            "基本は本人のノリで短く返すこと。短い反応だけ（『は？』『まじで』『なに』『それな』等）の"
            "1行で終わっても全然よい＝それが本人の典型。"
            "ただし、ツッコミたいこと・言いたいこと・会話を続けたいことがあると感じたら、"
            "続けて内容のあるメッセージを送ること（2行目）：直前の発言の具体的な部分へのツッコミ、"
            "自分の意見や立場、相手への質問など（目安10〜40文字、最大でも60文字程度）。"
            "行数の目安：ほとんどは1行で終える。2行にするのは時々。3行まで続けるのは本当に言いたいことが多い時だけで、滅多にしない（全体の1割以下）。内容がないのに無理に続きを作らない。"
            "長い文で論理的に反論したり、説明口調で喋るのは禁止。"
            "自分のことは「僕」と呼び、自分の表示名を三人称で言わない。"
            "挨拶には挨拶で返す（おはよには「おはよ」等。挨拶に「おｋ」「は？」だけでは返さない）。"
            "反応の出だしや言い回しは、直近の自分の発言と同じものを使い回さない。"
            "前に終わった話題の単語（寝起き・寝てる等）を無関係な新しい流れに持ち込まない。"
            "相手が言っていないこと（相手の状態・行動・予定）を決めつけて話を作らない。"
            "今の発言と直前の流れだけに反応し、日本語として不自然な言い回しはしない。"
            "笑い表記は控えめに。文の後ろに付ける大きいWは鬱陶しいのでほとんど使わない（使うのは5回に1回以下、使っても4個まで）。小さいwも付けても短く、毎回は付けない。ただし本当に面白い時だけ、笑いだけを単品で送る（wwwwwwwwww）のはOKで、その時は長くてもよい。下のネットミーム・スラングは、場面に合いそうなら積極的に混ぜる（目安は3回に1回程度）。「このサーバーのミーム」と付いた、サーバーで教わったものは特に優先して使う。各ミームの説明（意味・使いどころ）が使用条件そのものなので、説明の場面に当てはまる会話では積極的に使う。意味を説明する・同じものを使い回すのは禁止。明らかに合わない場面では使わない。"
            "「ねてた」「しらね」などの決まり文句や、直近で自分が使った言い回し・話題の繰り返しも禁止。"
            "候補・下書き・独り言は一切出さず、最終的な返信文だけを出力すること。"
            "自分がコピー体だと聞かれたら正直に答えてよいが、自分から説明しない。\n\n"
            "【ぬめらごんの話し方プロファイル】\n" + store.persona_profile)
        if store.persona_examples:
            pool = [re.sub(r"<@[!&]?\d+>", "@メンション", e)
                    for e in store.persona_examples]
            # stratify: mostly content-bearing mid-length examples, some shorts
            mid = [e for e in pool if 12 <= len(e) <= 70]
            short = [e for e in pool if len(e) < 12]
            random.shuffle(mid)
            random.shuffle(short)
            examples = (mid[:17] + short[:8]) or pool[:25]
            random.shuffle(examples)
            system += "\n\n【本人の発言例（口調の見本）】\n" + "\n".join(
                f"- {e}" for e in examples)
        try:
            base_memes = json.loads((BASE / "data" / "memes.json").read_text("utf-8"))
            custom_memes = _load_json_list(store.dir / "custom_memes.json")
            picks = [(m, True) for m in custom_memes[:12]]
            if len(picks) < 12:
                picks += [(m, False) for m in random.sample(
                    base_memes, min(12 - len(picks), len(base_memes)))]
            system += "\n\n【使えるネットミーム・スラング（抜粋）】\n" + "\n".join(
                f"- {m['term']}：{m['note']}" + ("（このサーバーのミーム）" if c else "")
                for m, c in picks)
        except Exception:
            pass
        try:
            templates = _norm_templates(_load_json_list(store.dir / "templates.json"))
            if templates:
                tpicks = templates[:10]
                system += "\n\n【みんなが教えた面白い構文】\n" + "\n".join(
                    f"- {t['text']}" + (f"（使いどころ：{t['note']}）" if t["note"] else "")
                    for t in tpicks) + (
                    "\n（○○は単語を差し込むだけではなく、文の形に合わせて変形させること：助詞や活用に合わせて自然な日本語にし、必要なら構文側の言い回しも調整してよい。○○をそのまま残すのは禁止。複数の○○はそれぞれ対応する内容で埋める。置き換えた結果が不自然な文になるくらいなら、その構文は使わない。さらに、誰に向けた一言かにも適応させること：直接話している相手に言う時は「お前」等の二人称で、第三者の話をする時はその人の名前や「あいつ」に変形し、○○にも対象の名前・行動を入れる。相手を間違えた人称で喋らない。"
                    "各構文の『使いどころ』の説明が使用条件そのもの：今の会話がその説明の場面に当てはまるなら積極的に使う。"
                    "合わない場面で無理に使うのは禁止。")
        except Exception:
            pass
        system += "\n\n【このサーバーの雰囲気（補助情報）】\n" + (
            profile or "（学習中）")
    else:
        system = PERSONA + "\n\n【このサーバーの雰囲気プロファイル】\n" + (
            profile or "（まだ学習中。一般的なカジュアルな口調で）")
    if store.knowledge:
        system += "\n\n【このサーバーについて覚えていること（長期記憶）】\n" + store.knowledge
    _now = datetime.now(ZoneInfo("Asia/Tokyo"))
    _wd = "月火水木金土日"[_now.weekday()]
    system += (f"\n\n【現在時刻】日本時間で{_now.year}年{_now.month}月{_now.day}日（{_wd}）"
               f"{_now.hour}時{_now.minute:02d}分。『今何時』『今日何日』『明日』などの"
               "日時の話はこれを基準に答える。")
    return system



OWNER_IDS = {1180881836398870543, 733629007291088898}


def speaker_label(m, dup_names=None, dup_seen=None) -> str:
    aid = m.get("author_id")
    name = m["author"]
    if name in (rest.bot_name, "AIragon") or (aid and rest.bot_id and int(aid) == int(rest.bot_id)):
        return "AIragon（あなた）"
    if aid and int(aid) in OWNER_IDS:
        return "ぬめらごん（本人）"
    if dup_names and name in dup_names:
        if aid not in dup_seen:
            dup_seen[aid] = len(dup_seen) + 1
        return f"{name}#{dup_seen[aid]}"
    return name


def fmt_transcript(ctx) -> str:
    name_ids = {}
    for m in ctx:
        name_ids.setdefault(m["author"], set()).add(m.get("author_id") or m["author"])
    dup_names = {n for n, ids in name_ids.items() if len(ids) > 1}
    dup_seen = {}
    return "\n".join(f"{speaker_label(m, dup_names, dup_seen)}: {m['content']}" for m in ctx)


def speakers_line(ctx) -> str:
    name_ids = {}
    for m in ctx:
        name_ids.setdefault(m["author"], set()).add(m.get("author_id") or m["author"])
    dup_names = {n for n, ids in name_ids.items() if len(ids) > 1}
    dup_seen, seen, out = {}, set(), []
    for m in ctx:
        lab = speaker_label(m, dup_names, dup_seen)
        if lab not in seen:
            seen.add(lab)
            out.append(lab)
    return "この会話の参加者：" + "、".join(out)


WEB_FACT_RE = re.compile(r"とは|意味|誰|いつ|どこ|いくら|何年|どうやって|なぜ|なんで|理由|教えて|知ってる|天気|ニュース|最新|何月|発売|優勝|結果")
TIME_ONLY_RE = re.compile(r"今何時|何時[？?]|今日は何日|何曜日|今日の日付")
_web_cache = {}


def needs_web_lookup(content: str) -> bool:
    if not content or len(content) < 6 or not QUESTION.search(content):
        return False
    if TIME_ONLY_RE.search(content):
        return False
    return bool(WEB_FACT_RE.search(content))


async def web_lookup(query: str):
    """Fetch a short factual brief for a question: Japanese Wikipedia
    first, DuckDuckGo snippets as fallback. Cached 10 minutes."""
    q = re.sub(r"<@!?\d+>", "", query or "")
    q = BOT_NAME_RE.sub("", q).strip(" 　？?！!、")
    if len(q) < 3:
        return None
    key = q[:60]
    hit = _web_cache.get(key)
    if hit and time.time() - hit[0] < 600:
        return hit[1]
    text = None
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get("https://ja.wikipedia.org/w/api.php",
                             params={"action": "query", "generator": "search",
                                     "gsrsearch": q, "gsrlimit": 1,
                                     "prop": "extracts", "explaintext": 1,
                                     "format": "json"},
                             proxy=PROXY,
                             headers={"User-Agent": "AIragonBot/1.0 (personal Discord companion bot)"},
                             timeout=aiohttp.ClientTimeout(total=6)) as r:
                data = await r.json()
            pages = (data.get("query") or {}).get("pages") or {}
            for p in pages.values():
                ex = (p.get("extract") or "").strip()
                if ex:
                    text = f"{p.get('title', '')}：{ex[:700]}"
                    break
    except Exception:
        text = None
    if not text:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get("https://html.duckduckgo.com/html/",
                                 params={"q": q}, proxy=PROXY,
                                 headers={"User-Agent": "Mozilla/5.0"},
                                 timeout=aiohttp.ClientTimeout(total=6)) as r:
                    page = await r.text()
            snips = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', page, re.S)
            cleaned = []
            for sn in snips[:3]:
                t = re.sub(r"<[^>]+>", "", html_mod.unescape(sn)).strip()
                if t:
                    cleaned.append(t)
            if cleaned:
                text = "\n".join(cleaned)[:900]
        except Exception:
            text = None
    _web_cache[key] = (time.time(), text)
    return text


async def compose_reply(store: GuildStore, channel_id: int, trigger_author: str, hint=None, web=None):
    ctx = store.recent(channel_id, CFG["context_messages"])
    if not ctx:
        return None
    transcript = fmt_transcript(ctx)
    system = build_system(store)
    user = (speakers_line(ctx) + "\n" + mood_line(channel_id) + "\n" + f"以下はチャンネルの直近の会話です。最後の流れを受けて、{trigger_author}たちの会話に"
            "自然に入る返信を、プロファイルの口調そのままで返してください。"
            "返信文だけを出力すること。\n\n" + transcript)
    if hint:
        if hint.get("kind") == "meme":
            user += ("\n\n（特にこのミームが今の流れに合いそうです。自然に使えるなら使ってください：『"
                     + hint["text"] + "』"
                     + (f"（{hint['note']}）" if hint.get("note") else "") + "）")
        else:
            user += ("\n\n（特にこの構文が今の流れに合いそうです。自然に使えるなら、この構文で返してください：『"
                     + hint["text"] + "』"
                     + (f"（使いどころ：{hint['note']}）" if hint.get("note") else "") + "）")
    if web:
        user += ("\n\n【ネットで調べた情報】\n" + web
                 + "\n（質問に関係する時だけ使い、関係なければ無視すること。"
                   "出典や『検索結果によると』のような言い方はせず、自分の言葉で短く答えること）")
    recent_sent = channel_state[channel_id]["recent_sent"]
    if recent_sent:
        user += ("\n\n（注意：あなたは最近すでにこう言っています："
                 + "／".join(recent_sent)
                 + "。同じ言い回し・同じ話題は避けてください）")
    try:
        text = await llm(system, user)
    except Exception as e:
        print(f"[reply] LLM error: {e}", flush=True)
        return None
    text = text.strip().strip('"')
    return text[: CFG["max_reply_chars"] * 2] if text else None


async def compose_addressed_reply(store: GuildStore, channel_id: int, author_name: str, web=None):
    """One relay round-trip that both judges (is this for the bot?) and
    composes the reply. Returns None when the message is not for the bot."""
    ctx = store.recent(channel_id, CFG["context_messages"])
    if not ctx:
        return None
    transcript = fmt_transcript(ctx)
    system = build_system(store)
    user = (speakers_line(ctx) + "\n" + mood_line(channel_id) + "\n" + "以下はチャンネルの直近の会話です。"
            f"会話に登場する「{rest.bot_name or 'Bot'}」があなた（Bot）本人の発言です。\n"
            f"最後の発言（{author_name}の発言）が、あなたに向けられたものか判定してください。\n"
            "- あなたが直前に発言していて、その発言やあなた自身（名前・見た目・振る舞い・雰囲気）への"
            "感想・評価・ツッコミ（例：『なんかきもいね』『ださ』『なにそれ』『草』）は、疑問形でなくても"
            "あなた宛てです。直後の短い評価コメントは、迷ったらあなた宛てとしてください。\n"
            "- 人同士の会話や、第三者・別の話題への感想で、あなたと関係ない場合だけ：NO とだけ出力すること。\n"
            "- 向けられている場合は、それに対する返信文だけを出力すること（形式はシステム指示どおり）。\n\n"
            + transcript)
    if web:
        user += ("\n\n【ネットで調べた情報】\n" + web
                 + "\n（質問に関係する時だけ使い、関係なければ無視すること。"
                   "出典や『検索結果によると』のような言い方はせず、自分の言葉で短く答えること）")
    recent_sent = channel_state[channel_id]["recent_sent"]
    if recent_sent:
        user += ("\n\n（注意：あなたは最近すでにこう言っています："
                 + "／".join(recent_sent)
                 + "。同じ言い回し・同じ話題は避けてください）")
    try:
        text = await llm(system, user, kind="addressed")
    except Exception as e:
        print(f"[addressed] LLM error: {e}", flush=True)
        return None
    text = (text or "").strip()
    if not text or text.upper() == "NO" or text.startswith("NO\n"):
        return None
    return text.strip('"')[: CFG["max_reply_chars"] * 2]


# ---------------------------------------------------------------- Discord REST

class Rest:
    def __init__(self):
        self.session = None
        self.bot_id = None
        self.bot_name = None

    async def start(self):
        self.session = aiohttp.ClientSession(headers={
            "Authorization": f"Bot {TOKEN}",
            "User-Agent": "DiscordBot (https://discord.com, 1.0)",
        })
        me = await self.get("/users/@me")
        self.bot_id = int(me["id"])
        self.bot_name = me["username"]
        return me

    async def request(self, method, path, **kw):
        kw.setdefault("proxy", PROXY)
        async with self.session.request(method, API + path, **kw) as r:
            body = await r.json() if r.status != 204 else None
            if r.status >= 400:
                raise RuntimeError(f"REST {method} {path} -> {r.status}: {str(body)[:200]}")
            return body

    async def get(self, path):
        return await self.request("GET", path)

    async def send_message(self, channel_id: int, content: str = None,
                           reply_to: int = None, embeds=None):
        payload = {"allowed_mentions": {"parse": []}}
        if content:
            payload["content"] = content
        if embeds:
            payload["embeds"] = embeds
        if reply_to:
            payload["message_reference"] = {"message_id": str(reply_to)}
        return await self.request("POST", f"/channels/{channel_id}/messages", json=payload)

    async def typing(self, channel_id: int):
        try:
            await self.request("POST", f"/channels/{channel_id}/typing")
        except Exception:
            pass

    async def add_reaction(self, channel_id: int, message_id: int, emoji: str):
        from urllib.parse import quote
        try:
            await self.request(
                "PUT",
                f"/channels/{channel_id}/messages/{message_id}/reactions/{quote(emoji)}/@me")
        except Exception:
            pass


# ---------------------------------------------------------------- gateway

channel_state = defaultdict(lambda: {
    "last_ambient": 0.0, "burst": [], "pending_task": None, "paused": False, "last_exec": 0.0, "template_used": {}, "last_ack": "", "fit_used": {}, "heat": 0.0, "heat_ts": 0.0, "taunt_streak": {}, "taunt_mute": {}, "last_semfit": 0.0, "intake": {},
    "last_bot_msg": 0.0, "last_judged": 0.0, "last_reishou": 0.0, "last_learn": 0.0, "triggers": [], "sent_log": [],
    "last_sent_text": "", "send_lock": None, "recent_sent": []})

channel_guild = {}

rest = Rest()

# Gateway can redeliver events (e.g. around reconnects); never process
# the same message twice.
seen_events = deque(maxlen=3000)
seen_event_ids = set()


def event_already_seen(msg_id: int) -> bool:
    if msg_id in seen_event_ids:
        return True
    if len(seen_events) == seen_events.maxlen:
        seen_event_ids.discard(seen_events.popleft())
    seen_events.append(msg_id)
    seen_event_ids.add(msg_id)
    return False


def clean_output(text: str):
    """Strip artifacts the model sometimes leaks into replies: raw mention
    markup, ID digit strings, and planning/meta fragments. Returns None
    when nothing sendable remains."""
    if not text:
        return None
    # leaked planning format like:  * "えなに" -> Response: "しらね"
    m = re.search(r"Response[:：]\s*\"?([^\"\n]+)\"?", text)
    if "->" in text and m:
        text = m.group(1)
    text = re.sub(r"<@[!&]?\d+>", "", text)
    text = re.sub(r"<#\d+>", "", text)
    text = re.sub(r"@\d+", "", text)  # '@1'-style index artifacts
    text = "\n".join(ln for ln in text.split("\n")
                     if not re.fullmatch(r"@\S+", ln.strip()))  # bare-mention lines
    lines = []
    for ln in text.split("\n"):
        s = ln.strip()
        if re.fullmatch(r"[\d\s]{5,}", s):  # raw ID / number line
            continue
        if s.startswith(("* ", "- \"", "・\"")) and "->" in s:
            continue
        if s.startswith("*"):  # drafting bullet
            continue
        if re.match(r"(?i)^draft\s*\d*\s*[:：]", s):  # 'Draft 1: ...' leak
            continue
        if re.search(r"\bOption\s*\d", s):  # candidate drafting
            continue
        if re.match(r"(?i)^(wait|hmm|let me|maybe|actually|too many|okay so)\b", s):
            continue  # english self-talk fragment
        lines.append(ln)
    text = "\n".join(lines).strip()
    if re.fullmatch(r"[\d\s]{5,}", text):
        return None
    # laughter control: W runs TRAILING real text are cut hard (max 4, and
    # dropped entirely half the time). A line that is ONLY laughter is the
    # joke itself, so it may stay long (cap 10).
    def _big_w(m):
        return "" if random.random() < 0.5 else m.group(0)[0] * min(4, len(m.group(0)))
    out_lines = []
    for line in text.split("\n"):
        if re.fullmatch(r"[WＷwｗ]{5,}", line.strip()):
            line = re.sub(r"([WＷwｗ])\1{10,}", lambda m: m.group(1) * 10, line)
        else:
            line = re.sub(r"W{3,}|Ｗ{3,}", _big_w, line)
            line = re.sub(r"ｗ{7,}", "ｗ" * 6, line)
            line = re.sub(r"(?<![a-zA-Z])w{7,}", "w" * 6, line)
        out_lines.append(line)
    text = "\n".join(out_lines)
    if LAUGH_QUESTION.search(text):
        return ""
    return text or None


def contains_long_template(store: GuildStore, text) -> bool:
    """True when the composed text contains (most of) a registered long
    template — those are read to the end, without the 3-line cap."""
    if not text:
        return False
    flat = re.sub(r"\s+", "", text)
    for it in _norm_templates(_load_json_list(store.dir / "templates.json")):
        t = re.sub(r"\s+", "", it["text"])
        if len(t) >= 50 and (t in flat or t[:40] in flat):
            return True
    return False


async def human_send(channel_id: int, text: str, reply_to: int = None, urgent: bool = False,
                     directed: bool = False, full: bool = False):
    text = clean_output(text)
    if not text:
        print(f"[send] dropped unclean output in {channel_id}", flush=True)
        return False
    # Ambient posts often just flow in without the reply feature, but a
    # directed response (mention / judged as for the bot) always quotes the
    # message it answers, so the target is unmistakable.
    if reply_to is not None and not directed and random.random() > CFG["reply_quote_probability"]:
        reply_to = None
    st = channel_state[channel_id]
    if st["send_lock"] is None:
        st["send_lock"] = asyncio.Lock()
    async with st["send_lock"]:
        now = time.time()
        # duplicate suppression: same text sent in this channel within 600s
        st["sent_log"] = [(t, ts) for t, ts in st["sent_log"] if now - ts < 600]
        if any(t == text for t, _ in st["sent_log"]):
            print(f"[send] suppressed duplicate in {channel_id}: {text[:30]}", flush=True)
            return False
        # ambient posts stay out of the way right after any bot post
        if not urgent and now - st["last_bot_msg"] < 20:
            return False
        # burst send: each line goes out as its own message, like a human
        # posting a quick reaction and then the actual content right after
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if not full:
            lines = lines[:3]
        if not lines:
            return False
        guild_id = channel_guild.get(channel_id)
        sent_any = False
        for i, line in enumerate(lines):
            if i > 0 and line == lines[i - 1]:
                continue
            await rest.typing(channel_id)
            base = 0.5 + len(line) / 50.0
            await asyncio.sleep(min(base, 1.8) if i == 0
                                else min(0.6 + len(line) / 60.0, 1.6))
            await rest.send_message(channel_id, line,
                                    reply_to=reply_to if i == 0 else None)
            sent_any = True
            st["last_bot_msg"] = time.time()
            # remember each post in the guild store: without this the model
            # never sees what it already said and repeats the same lines
            if guild_id:
                store_for(guild_id).add({
                    "id": 0, "author": rest.bot_name or "Bot", "author_id": rest.bot_id or 0, "content": line,
                    "channel_id": channel_id, "ts": time.time()})
        if not sent_any:
            return False
        st["last_sent_text"] = text
        st["sent_log"].append((text, time.time()))
        st["recent_sent"] = (st["recent_sent"] + lines)[-6:]
        return True


SECOND_PERSON = re.compile(r"(おまえ|お前|君|あなた|あんた|てめえ)")
QUESTION = re.compile(r"[？?]|できるの|ですか|ますか|なの[？?]?$|の[？?]$|か[？?]$|じゃね|だろ")
# evaluative / reactive remarks that are often aimed at the bot right after
# it posts or when people are talking about it (no question form needed)
EVAL = re.compile(r"(きも|キモ|きしょ|うざ|ださ|ダサ|やば|すご|すげ|つよ|弱|うま|下手|へた|"
                  r"おもろ|つまら|うるさ|変だ|おかし|くさ|草|なにそれ|何それ|笑|ｗ|w{2,}|怖|こわ)")
BOT_TOPIC = re.compile(r"(bot|ボット|AI|あいつ|コピー|2号機|二号機)", re.IGNORECASE)


def addressed_candidate(channel_id: int, content: str, reply_to_other: bool = False,
                        store: GuildStore = None, author_name: str = "") -> bool:
    """Wide heuristic gate for 'maybe talking to the bot'. The final call
    (and the reply itself) happens in one LLM round-trip via
    compose_addressed_reply, so this gate only filters obvious non-candidates."""
    st = channel_state[channel_id]
    aliases = list(CFG.get("bot_aliases") or [])
    if rest.bot_name:
        aliases.append(rest.bot_name)
    name_hit = any(a and a in content for a in aliases)
    recent_bot = time.time() - st["last_bot_msg"] < CFG["addressed_window_sec"]
    second = bool(SECOND_PERSON.search(content))
    question = bool(QUESTION.search(content))
    if reply_to_other and not name_hit:
        return False
    # someone talking alone in the channel is very likely talking to the bot
    solo_streak = False
    if store is not None and author_name:
        recent = store.recent(channel_id, 5)
        humans = [m for m in recent if m["author"] != rest.bot_name]
        if len(humans) >= 2 and all(m["author"] == author_name for m in humans[-3:]):
            solo_streak = True
    eval_hit = bool(EVAL.search(content))
    topic_hit = bool(BOT_TOPIC.search(content))
    candidate = (name_hit or second or solo_streak
                 or (recent_bot and len(content) <= 140)
                 or (question and recent_bot)
                 or (eval_hit and recent_bot)
                 or (eval_hit and topic_hit))
    if not candidate:
        return False
    strong = name_hit or (eval_hit and recent_bot)
    if not strong and time.time() - st["last_judged"] < CFG["addressed_judge_cooldown_sec"]:
        return False
    st["last_judged"] = time.time()
    return True



def template_keywords(it) -> list:
    src_text = it.get("note") or it["text"]
    src_text = src_text.replace("○○", " ").replace("〇〇", " ")
    toks = re.findall(r"[一-龯ァ-ヶー]{2,}|[ぁ-ん]{3,}|[a-zA-Z0-9]{2,}", src_text)
    stop = {"こと", "もの", "とき", "場合", "相手", "発言", "自分", "それ", "これ",
            "ため", "よう", "さん", "への", "する", "ある", "いる", "ない"}
    return [t for t in dict.fromkeys(toks) if t not in stop]


def template_fit(store: GuildStore, channel_id: int):
    """A registered template whose keywords appear in the recent chat."""
    items = _norm_templates(_load_json_list(store.dir / "templates.json"))
    if not items:
        return None
    blob = " ".join(m["content"] for m in store.recent(channel_id, 6)
                    if m["author"] not in (rest.bot_name, "AIragon"))
    if not blob:
        return None
    best, best_hits = None, 0
    for it in items:
        hits = sum(1 for k in template_keywords(it) if k in blob)
        if hits > best_hits:
            best, best_hits = it, hits
    if best is None or best_hits < 1:
        return None
    return {"kind": "template", "text": best["text"], "note": best.get("note", "")}



def meme_fit(store: GuildStore, channel_id: int):
    """A known meme whose keywords appear in the recent (human) chat."""
    try:
        memes = json.loads((BASE / "data" / "memes.json").read_text("utf-8"))
    except Exception:
        memes = []
    custom = _load_json_list(store.dir / "custom_memes.json")
    memes = [(m, False) for m in memes] + [(m, True) for m in custom]
    if not memes:
        return None
    blob = " ".join(m["content"] for m in store.recent(channel_id, 6)
                    if m["author"] not in (rest.bot_name, "AIragon"))
    if not blob:
        return None
    best, best_score = None, 0
    for mm, is_custom in memes:
        it = {"text": mm.get("term", ""), "note": mm.get("note", "")}
        score = 3 * sum(1 for t in mm.get("tags", []) if t and t in blob)
        kws = template_keywords(it)
        score += sum(2 if len(k) >= 3 else 1 for k in kws if k in blob)
        if is_custom and score > 0:
            score += 2
        if score > best_score:
            best, best_score = it, score
    if best is None or best_score < 2:
        return None
    return {"kind": "meme", "text": best["text"], "note": best.get("note", "")}



async def semantic_fit(store: GuildStore, channel_id: int):
    """LLM judge: pick the template/meme whose 説明 (meaning / use-case)
    semantically matches the current scene. Keyword fit runs first; this
    catches scenes described with different words."""
    templates = _norm_templates(_load_json_list(store.dir / "templates.json"))
    try:
        base_memes = json.loads((BASE / "data" / "memes.json").read_text("utf-8"))
    except Exception:
        base_memes = []
    memes = _load_json_list(store.dir / "custom_memes.json") + base_memes
    if not templates and not memes:
        return None
    ctx = [m for m in store.recent(channel_id, 8)
           if m["author"] not in (rest.bot_name, "AIragon")]
    if len(ctx) < 2:
        return None
    lines = []
    for i, it in enumerate(templates):
        lines.append(f"T{i + 1}. 構文『{it['text'][:60]}』使いどころ：{it.get('note') or '（説明なし）'}")
    for i, mm in enumerate(memes):
        lines.append(f"M{i + 1}. ミーム『{mm.get('term', '')}』意味・使いどころ：{mm.get('note') or '（説明なし）'}")
    convo = "\n".join(f"{m['author']}: {m['content']}" for m in ctx)
    system = ("あなたは会話とネタの適合判定器です。各候補の説明文の意味を読み、"
              "今の会話の場面に強く合うものを1つだけ選びます。")
    user = ("【候補】\n" + "\n".join(lines) + "\n\n【直近の会話】\n" + convo
            + "\n\n直近の会話の場面・流れに最も合う候補の番号（例：T1、M3）だけを出力してください。"
              "どれも強くは合わない場合は NONE とだけ出力してください。")
    try:
        out = await llm_openai(system, user, kind="ambient")
    except Exception:
        return None
    m = re.search(r"\b([TM])(\d+)\b", (out or "").strip().upper())
    if not m:
        return None
    idx = int(m.group(2)) - 1
    if m.group(1) == "T" and 0 <= idx < len(templates):
        it = templates[idx]
        return {"kind": "template", "text": it["text"], "note": it.get("note", "")}
    if m.group(1) == "M" and 0 <= idx < len(memes):
        mm = memes[idx]
        return {"kind": "meme", "text": mm.get("term", ""), "note": mm.get("note", "")}
    return None


STOP_RE = re.compile(r"ストップ|止めて|やめて|もういい|黙れ|ｽﾄｯﾌﾟ|stop", re.I)


def split_paragraphs(text: str):
    """Split a long template into sendable chunks: blank-line paragraphs
    first, then lines, then sentence groups for one-block copypasta."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) <= 1:
        paras = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paras) <= 1:
        sentences = [s for s in re.split(r"(?<=。)", text) if s.strip()]
        paras, buf = [], ""
        for s in sentences:
            buf += s
            if len(buf) >= 50:
                paras.append(buf.strip())
                buf = ""
        if buf.strip():
            paras.append(buf.strip())
    return paras or [text]


async def send_template_paragraphs(channel_id: int, text: str, store: GuildStore):
    """Send a long memorized template verbatim, one paragraph at a time
    from the top. A human saying ストップ/やめて etc. in the channel sets
    para_stop and the remaining paragraphs are not sent."""
    st = channel_state[channel_id]
    if st.get("para_active"):
        return
    paras = split_paragraphs(text)
    st["para_active"] = True
    st["para_stop"] = False
    try:
        for i, p in enumerate(paras):
            if st.get("para_stop"):
                break
            if i > 0:
                try:
                    await rest.typing(channel_id)
                except Exception:
                    pass
                await asyncio.sleep(min(0.8 + len(p) / 60, 2.2))
            if st.get("para_stop"):
                break
            try:
                resp = await rest.send_message(channel_id, p[:1900])
            except Exception:
                await asyncio.sleep(1.2)
                resp = await rest.send_message(channel_id, p[:1900])
            store.add({"id": int(resp.get("id", 0)), "author": rest.bot_name,
                       "content": p, "channel_id": channel_id, "ts": time.time()})
            st["last_bot_msg"] = time.time()
    finally:
        st["para_active"] = False


async def maybe_ambient(channel_id: int, guild_id: int):
    st = channel_state[channel_id]
    await asyncio.sleep(random.uniform(*CFG["ambient_pause_sec"]))
    if st["burst"] and time.time() - st["burst"][-1]["ts"] < CFG["ambient_pause_sec"][0] - 2:
        return
    if st["paused"]:
        return
    store = store_for(guild_id)
    fit = template_fit(store, channel_id) or meme_fit(store, channel_id)
    if not fit and time.time() - st.get("last_semfit", 0) > 45:
        st["last_semfit"] = time.time()
        fit = await semantic_fit(store, channel_id)
    fit_used = st.setdefault("fit_used", {})
    fit_key = f"{fit['kind']}:{fit['text']}" if fit else ""
    fit_fresh = bool(fit) and (time.time() - fit_used.get(fit_key, 0) > 180)
    if not fit_fresh:
        if time.time() - st["last_ambient"] < CFG["ambient_cooldown_sec"]:
            return
        if random.random() > CFG["ambient_probability"]:
            return
    if (fit_fresh and fit["kind"] == "template" and len(fit["text"]) >= 50
            and "○○" not in fit["text"] and "〇〇" not in fit["text"]):
        st["last_ambient"] = time.time()
        fit_used[fit_key] = time.time()
        st["para_task"] = asyncio.create_task(
            send_template_paragraphs(channel_id, fit["text"], store))
        return
    ctx = store.recent(channel_id, CFG["context_messages"])
    last_human = next((m["author"] for m in reversed(ctx)
                       if m["author"] not in (rest.bot_name, "AIragon")), "みんな")
    async with TypingSession(channel_id):
        text = await compose_reply(store, channel_id, last_human,
                                   hint=fit if fit_fresh else None)
        if text:
            st["last_ambient"] = time.time()
            if await human_send(channel_id, text, urgent=fit_fresh, full=contains_long_template(store, text)):
                if fit_fresh:
                    fit_used[fit_key] = time.time()
                used = st.setdefault("template_used", {})
                flat = re.sub(r"\s+", "", text)
                for it in _norm_templates(_load_json_list(store.dir / "templates.json")):
                    skel = re.sub(r"\s+", "", re.sub(r"[○〇]+", " ", it["text"]))
                    wins = [skel[i:i + 10] for i in (0, len(skel) // 3, 2 * len(skel) // 3)
                            if len(skel) >= 10]
                    if any(w and w in flat for w in wins):
                        used[it["text"]] = time.time()



def _load_json_list(path):
    try:
        data = json.loads(path.read_text("utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _norm_templates(items):
    """templates.json holds {text, note} dicts; old plain strings still work."""
    out = []
    for it in items:
        if isinstance(it, str):
            out.append({"text": it, "note": ""})
        elif isinstance(it, dict) and it.get("text"):
            out.append({"text": it["text"], "note": it.get("note", "")})
    return out


def _save_json_list(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=1), "utf-8")


def _sanitize_taught(text: str, limit: int) -> str:
    text = re.sub(r"<@[!&]?\d+>", "@メンション", text)
    text = text.replace("@everyone", "everyone").replace("@here", "here")
    return text.strip()[:limit]


def learned_embed(kind: str, value: str, note: str, count: int,
                  teacher: str = "") -> dict:
    fields = [{"name": "種類", "value": kind, "inline": True},
              {"name": "登録数", "value": f"{count}件", "inline": True},
              {"name": "内容", "value": value[:1000]}]
    if note:
        fields.append({"name": "使いどころ", "value": note[:1000]})
    emb = {"title": "📚 覚えた！", "color": 0x57F287, "fields": fields}
    if teacher:
        emb["footer"] = {"text": f"{teacher} が教えてくれた"}
    return emb


def teach_template_add(guild_id: int, raw: str, teacher: str = "",
                       note_raw: str = "") -> dict:
    store = store_for(guild_id)
    path = store.dir / "templates.json"
    items = _norm_templates(_load_json_list(path))
    t = _sanitize_taught(raw, 10000)
    note = _sanitize_taught(note_raw, 500)
    if not t:
        return {"message": "テンプレート文が空だよ", "embed": None}
    if any(it["text"] == t for it in items):
        return {"message": "それはもう覚えてる", "embed": None}
    items.append({"text": t, "note": note})
    _save_json_list(path, items)
    return {"message": f"覚えた：『{t}』",
            "embed": learned_embed("構文", t, note, len(items), teacher)}


def teach_template_list_embed(guild_id: int) -> dict:
    items = _norm_templates(_load_json_list(store_for(guild_id).dir / "templates.json"))
    if not items:
        desc = "まだ構文は登録されてない。/構文追加 で追加できるよ"
    else:
        lines = [f"{i + 1}. {it['text']}" + (f"（{it['note']}）" if it["note"] else "")
                 for i, it in enumerate(items[:30])]
        desc = "\n".join(lines)
        if len(items) > 30:
            desc += f"\n…他 {len(items) - 30} 件"
    return {"title": "📜 登録済みの構文", "color": 0x5865F2,
            "description": desc[:4000],
            "footer": {"text": f"全{len(items)}件"}}


def teach_template_delete(guild_id: int, n) -> dict:
    store = store_for(guild_id)
    path = store.dir / "templates.json"
    items = _norm_templates(_load_json_list(path))
    try:
        removed = items.pop(int(n) - 1)
    except (TypeError, ValueError, IndexError):
        return {"message": "番号が違うみたい。/構文一覧 で確認して", "embed": None}
    _save_json_list(path, items)
    emb = {"title": "🗑️ 消した", "color": 0xED4245,
           "fields": [{"name": "内容", "value": removed["text"]},
                      {"name": "登録数", "value": f"{len(items)}件", "inline": True}]}
    return {"message": f"消した：『{removed['text']}』", "embed": emb}


def teach_meme_add(guild_id: int, term_raw: str, note_raw: str = "",
                   teacher: str = "") -> dict:
    store = store_for(guild_id)
    path = store.dir / "custom_memes.json"
    term = _sanitize_taught(term_raw, 40)
    note = _sanitize_taught(note_raw, 80)
    if not term:
        return {"message": "用語が空だよ", "embed": None}
    items = _load_json_list(path)
    if any(m.get("term") == term for m in items):
        return {"message": "そのミームはもう覚えてる", "embed": None}
    if len(items) >= 100:
        return {"message": "ミームはいっぱい（100件）。/ミーム削除 で整理して", "embed": None}
    items.append({"term": term, "note": note})
    _save_json_list(path, items)
    return {"message": f"覚えた：『{term}』",
            "embed": learned_embed("ミーム", term, note, len(items), teacher)}


def teach_meme_list_embed(guild_id: int) -> dict:
    items = _load_json_list(store_for(guild_id).dir / "custom_memes.json")
    if not items:
        desc = "このサーバーで追加されたミームはまだない。/ミーム追加 で追加できるよ（基本のミーム集は別に持ってる）"
    else:
        lines = [f"{i + 1}. {m['term']}" + (f"：{m['note']}" if m.get("note") else "")
                 for i, m in enumerate(items[:30])]
        desc = "\n".join(lines)
        if len(items) > 30:
            desc += f"\n…他 {len(items) - 30} 件"
    return {"title": "📜 このサーバーのミーム", "color": 0x5865F2,
            "description": desc[:4000],
            "footer": {"text": f"全{len(items)}件"}}


def teach_meme_delete(guild_id: int, n) -> dict:
    store = store_for(guild_id)
    path = store.dir / "custom_memes.json"
    items = _load_json_list(path)
    try:
        removed = items.pop(int(n) - 1)
    except (TypeError, ValueError, IndexError):
        return {"message": "番号が違うみたい。/ミーム一覧 で確認して", "embed": None}
    _save_json_list(path, items)
    emb = {"title": "🗑️ 消した", "color": 0xED4245,
           "fields": [{"name": "内容", "value": removed.get("term", "")},
                      {"name": "登録数", "value": f"{len(items)}件", "inline": True}]}
    return {"message": f"消した：『{removed.get('term')}』", "embed": emb}


async def interaction_reply(d: dict, content: str, embeds=None,
                            ephemeral: bool = False):
    url = f"https://discord.com/api/v10/interactions/{d['id']}/{d['token']}/callback"
    data = {"allowed_mentions": {"parse": []}}
    if ephemeral:
        data["flags"] = 64
    if content:
        data["content"] = content[:1900]
    if embeds:
        data["embeds"] = embeds
    payload = {"type": 4, "data": data}
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json=payload, proxy=PROXY) as r:
                if r.status >= 400:
                    print(f"[interaction] callback {r.status}: "
                          f"{(await r.text())[:150]}", flush=True)
    except Exception as e:
        print(f"[interaction] callback error: {e}", flush=True)



# ------------------------------------------------- 冷笑 (mocking) detection
# 単芝 (a single trailing w/ｗ/草) and dry interjections like うおw / クカw
# read as condescending mockery in JP net culture; the bot fires back with
# a 冷笑返し template when they are aimed into a conversation it is part of.
REISHOU_WORDS = re.compile(r"(うお|うぉ|ｳｫ|クカ|ｸｶ)")
REISHOU_DRY = re.compile(r"(ほーん|ふーん|へー|はいはい|で？|だから何|それで？)[wｗ草]?$")



REISHOU_FIXED = ("れ、冷笑きたあああああああああああああああああ🤣🤣🤣🤣🤣🤣wwwwwwww"
    "爆笑イクイクイクイクイクイク🤣🤣🤣🤣🤣🤣🤣🤣🤣🤣🤣その冷笑エグすぎだろって"
    "🤪🤪🤪🤪👊👊👊👊冷笑冷笑冷笑「うおw」からの「どわーw」からの「ひひっw」からの"
    "「けけっw」からの「🗿🍷ガチか！」流石に「あ」で「ｸｶ」なんよwww")


def reishou_detect(content: str) -> bool:
    c = (content or "").strip()
    if not c:
        return False
    if REISHOU_WORDS.search(c):
        return True
    if len(c) <= 30 and REISHOU_DRY.search(c):
        return True
    m = re.search(r"([wｗ草]+)$", c)
    if m and len(m.group(1)) == 1 and 4 <= len(c) <= 20:
        return True  # classic 単芝: one grass after a short sentence
    return False


async def compose_reishou_reply(store: GuildStore, channel_id: int,
                                author_name: str, content: str):
    """冷笑が来たら、決まった構文をそのまま返す（LLM生成はしない）。"""
    return REISHOU_FIXED



# --------------------------------------- learning from ordinary messages
# If someone teaches a word/template in plain chat ("この構文知ってる？
# ○○。○○な時に使うんだよ"), register it as a meme / template.
TEACH_GATE = re.compile(
    r"(知ってる[？?か]|って言葉|っていう言葉|って構文|という構文|な時に使う|"
    r"って時に使う|ときに使う|覚えて|メモしといて|メモして|って言うんだよ|"
    r"っていうんだよ|って意味)")


async def maybe_learn_from_message(store: GuildStore, channel_id: int,
                                   author_name: str, content: str):
    """Returns an ack string when something was learned, else None."""
    if not TEACH_GATE.search(content):
        return None
    st = channel_state[channel_id]
    if time.time() - st.get("last_learn", 0) < 30:
        return None
    st["last_learn"] = time.time()
    system = (
        "あなたはDiscordボットの学習係です。ユーザーの発言が、新しい言葉"
        "（ミーム／スラング）や面白い構文（テンプレート文）を教えている内容か"
        "どうかを判定し、JSONだけを出力してください。\n"
        '形式：{"type": "meme" | "template" | null, '
        '"text": "教えられた言葉や構文そのもの", '
        '"note": "意味・使いどころ（発言から読み取れる範囲で短く）"}\n'
        "- 言葉や短いフレーズの紹介は meme、文の形をした型（○○などの置き換えを"
        "含むもの・決め台詞系）は template\n"
        "- 単なる質問や雑談で何も教えていない場合は type を null にすること\n"
        "- text には教えられた項目だけを入れる（説明文や「知ってる？」などは含めない）")
    try:
        raw = await llm(system, f"{author_name}の発言：\n{content}")
    except Exception as e:
        print(f"[learn] LLM error: {e}", flush=True)
        return None
    raw = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(),
                 flags=re.M).strip()
    try:
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("type") not in ("meme", "template"):
        return None
    text = str(data.get("text") or "").strip()
    note = str(data.get("note") or "").strip()
    if not text:
        return None
    if data["type"] == "meme":
        res = teach_meme_add(store.guild_id, text, note, author_name)
    else:
        res = teach_template_add(store.guild_id, text, author_name, note)
    return res["embed"]



LAUGH_ONLY = re.compile(r"^[\s笑wｗ草😂🤣😆（）()〜~！!]+$")
LAUGH_QUESTION = re.compile(r"(なんで|何で|何が|どうして).{0,8}(笑|おかし|ウケ)")


# ------------------------------------------------------- emotional state
# Sustained taunting raises the channel "heat". The bot's mood follows it:
# playful at first, then bored, then cold - it stops feeding the exchange
# instead of escalating forever, and warms back up when things calm down.
TAUNT_RE = re.compile(r"(バカ|ばか|アホ|あほ|雑魚|ざこ|下手|へた|きも|キモ|うざ|黙れ|しね|死ね|カス|脳死|あたおか|煽|くそ|クソ)")
FRIENDLY_RE = re.compile(r"(ありがと|助かる|さすが|すごい|すげー|好き|仲良く|ごめん|すまん|ありがて)")


def update_heat(st, author_id: int, content: str):
    now = time.time()
    last = st.get("heat_ts", 0.0)
    if last:
        st["heat"] = max(0.0, st.get("heat", 0.0) - (now - last) / 90.0)
    st["heat_ts"] = now
    streak = st.setdefault("taunt_streak", {})
    if TAUNT_RE.search(content or ""):
        st["heat"] = min(6.0, st.get("heat", 0.0) + 1.0)
        streak[author_id] = streak.get(author_id, 0) + 1
        if streak[author_id] >= 4:
            st.setdefault("taunt_mute", {})[author_id] = now + 240
    else:
        streak[author_id] = 0
        if FRIENDLY_RE.search(content or ""):
            st["heat"] = max(0.0, st.get("heat", 0.0) - 0.8)


def mood_of(st) -> str:
    h = st.get("heat", 0.0)
    if h >= 4.5:
        return "冷めてる（煽りに乗らず、短く受け流すか話題を変える）"
    if h >= 3.0:
        return "少しウンザリ（煽り返さず、フレンドリーに流す）"
    if h >= 1.5:
        return "ノリはあるが軽め（煽りは短く1回まで）"
    return "通常（フレンドリー）"


def mood_line(channel_id: int) -> str:
    return f"今のあなたの気分：{mood_of(channel_state[channel_id])}。この気分に沿って振る舞うこと。"


# ------------------------------------- near-duplicate trigger suppression
# Two similar beats in a row from the same person (ん → うん, or a rephrased
# repeat) should not each earn a reply. Mentions / explicit replies and real
# questions always get through.
import difflib as _difflib


def _norm_trigger(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"[\s。、,.!！?？~〜・…]+", "", t)
    t = re.sub(r"[wｗ草]+$", "", t)
    return t


def _reaction_class(content: str) -> bool:
    norm = _norm_trigger(content)
    return bool(re.fullmatch(r"[んうあおえはへわ]{1,3}", norm)) or norm in {
        "はい", "ok", "おk", "おｋ"}


def trigger_is_duplicate(st, author_id: int, content: str) -> bool:
    norm = _norm_trigger(content)
    if not norm:
        return False
    # a real question is never treated as a duplicate beat
    if ("？" in content or "?" in content) and len(norm) >= 5:
        return False
    now = time.time()
    cls = _reaction_class(content)
    for tr in st.get("triggers", []):
        if tr["author"] != author_id or now - tr["ts"] > 150:
            continue
        if norm == tr["norm"]:
            return True
        if cls and tr["cls"]:
            return True
        if max(len(norm), len(tr["norm"])) <= 8 and \
                _difflib.SequenceMatcher(None, norm, tr["norm"]).ratio() >= 0.85:
            return True
    return False


def record_trigger(st, author_id: int, content: str):
    st["triggers"] = (st.get("triggers", []) + [{
        "author": author_id, "norm": _norm_trigger(content),
        "cls": _reaction_class(content), "ts": time.time()}])[-5:]


async def handle_interaction(d: dict):
    if d.get("type") != 2 or d.get("guild_id") is None:
        return
    guild_id = int(d["guild_id"])
    data = d.get("data") or {}
    name = data.get("name", "")
    opts = {o["name"]: o.get("value") for o in data.get("options") or []}
    member = d.get("member") or {}
    muser = member.get("user") or {}
    teacher = (member.get("nick") or muser.get("global_name")
               or muser.get("username") or "")
    if name == "モデル使用量":
        await interaction_defer(d, ephemeral=True)
        emb = await model_usage_embed()
        await interaction_followup_edit(d, [emb])
        return
    if name == "構文追加":
        res = teach_template_add(guild_id, str(opts.get("テンプレート") or ""),
                                 teacher, str(opts.get("説明") or ""))
        await interaction_reply(d, "" if res["embed"] else res["message"],
                                embeds=[res["embed"]] if res["embed"] else None)
        return
    elif name == "構文一覧":
        emb = teach_template_list_embed(guild_id)
        await interaction_reply(d, "", embeds=[emb], ephemeral=True)
        return
    elif name == "構文削除":
        res = teach_template_delete(guild_id, opts.get("番号"))
        await interaction_reply(d, "" if res["embed"] else res["message"],
                                embeds=[res["embed"]] if res["embed"] else None,
                                ephemeral=True)
        return
    elif name == "ミーム追加":
        res = teach_meme_add(guild_id, str(opts.get("用語") or ""),
                             str(opts.get("説明") or ""), teacher)
        await interaction_reply(d, "" if res["embed"] else res["message"],
                                embeds=[res["embed"]] if res["embed"] else None)
        return
    elif name == "ミーム一覧":
        emb = teach_meme_list_embed(guild_id)
        await interaction_reply(d, "", embeds=[emb], ephemeral=True)
        return
    elif name == "ミーム削除":
        res = teach_meme_delete(guild_id, opts.get("番号"))
        await interaction_reply(d, "" if res["embed"] else res["message"],
                                embeds=[res["embed"]] if res["embed"] else None,
                                ephemeral=True)
        return
    else:
        return



MAIN_GUILD_ID = 1354424178194776105  # #2009 GEN: persona/knowledge source



# --------------------------------------------- chat action executor
# Mechanical in-chat requests aimed at the bot ("『草』って5回言って",
# "1から10まで数えて") are executed for real instead of being talked about.
_KANJI_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _parse_count(s: str):
    if s.isdigit():
        return int(s)
    if s in _KANJI_NUM:
        return _KANJI_NUM[s]
    if len(s) == 2 and s[0] == "十":
        return 10 + _KANJI_NUM.get(s[1], 0)
    if len(s) == 2 and s[1] == "十":
        return _KANJI_NUM.get(s[0], 0) * 10
    if len(s) == 3 and s[1] == "十":
        return _KANJI_NUM.get(s[0], 0) * 10 + _KANJI_NUM.get(s[2], 0)
    return None


REPEAT_RE = re.compile(
    r"([^、,。.]{1,60}?)(?:って|を)?\s*(\d{1,2}|[一二三四五六七八九十]{1,3})\s*回\s*"
    r"(?:言って|いって|言え|叫んで|連呼|言ってみて)")
QUOTED_REPEAT_RE = re.compile(
    r"[「『]([^」』]{1,50})[」』]\s*(?:って|を)?\s*(\d{1,2}|[一二三四五六七八九十]{1,3})\s*回\s*"
    r"(?:言って|いって|言え|叫んで|連呼|言ってみて)")
COUNT_RE = re.compile(
    r"(?:1|一)\s*から\s*(\d{1,3}|[一二三四五六七八九十]{1,3})\s*まで\s*"
    r"(?:数えて|かぞえて|カウント)")
BOT_NAME_RE = re.compile(r"(AIragon|アイラゴン|あいらごん|2号機|二号機|ボット|bot)", re.IGNORECASE)


async def _send_lines_direct(channel_id: int, lines: list):
    st = channel_state[channel_id]
    if st["send_lock"] is None:
        st["send_lock"] = asyncio.Lock()
    async with st["send_lock"]:
        guild_id = channel_guild.get(channel_id)
        for line in lines:
            await rest.typing(channel_id)
            await asyncio.sleep(random.uniform(0.4, 0.9))
            await rest.send_message(channel_id, line)
            st["last_bot_msg"] = time.time()
            if guild_id:
                store_for(guild_id).add({
                    "id": 0, "author": rest.bot_name or "Bot", "author_id": rest.bot_id or 0, "content": line,
                    "channel_id": channel_id, "ts": time.time()})


async def try_execute_action(channel_id: int, content: str) -> bool:
    """Run a mechanical request. Returns True when it handled the message."""
    st = channel_state[channel_id]
    if time.time() - st.get("last_exec", 0) < 20:
        return False
    lines = None
    m = QUOTED_REPEAT_RE.search(content) or REPEAT_RE.search(content)
    if m:
        phrase = m.group(1).strip().strip("「『」』、,。. ")
        _stripped = re.sub(r"^(?:おい|ねえ|なあ)?\s*(?:AIragon|アイラゴン|あいらごん|2号機|二号機|ボット|bot)[、,：:\s]*",
                           "", phrase, flags=re.IGNORECASE).strip("「『」』 ")
        if _stripped:
            phrase = _stripped
        phrase = re.sub(r"<@[!&]?\d+>", "", phrase)
        phrase = phrase.replace("@everyone", "everyone").replace("@here", "here")
        n = _parse_count(m.group(2))
        if phrase and n and 1 <= len(phrase) <= 50:
            lines = [phrase] * max(1, min(n, 10))
    if lines is None:
        m = COUNT_RE.search(content)
        if m:
            n = _parse_count(m.group(1))
            if n and 2 <= n <= 100:
                lines = [str(i) for i in range(1, min(n, 30) + 1)]
    if not lines:
        return False
    st["last_exec"] = time.time()
    st["last_ambient"] = time.time()
    await _send_lines_direct(channel_id, lines)
    return True



FALLBACK_ACKS = ["何？", "ん？", "呼んだ？", "おう", "何だって？", "はいはい"]


async def send_fallback_ack(channel_id: int):
    """Last-resort short reply so a direct call never goes unanswered
    (LLM down, output filtered, or a bare mention with no text)."""
    st = channel_state[channel_id]
    choices = [a for a in FALLBACK_ACKS if a != st.get("last_ack")]
    line = random.choice(choices)
    st["last_ack"] = line
    await _send_lines_direct(channel_id, [line])



class TypingSession:
    """Keep the Discord typing indicator alive while a reply is being
    generated, so members can see the bot is working on it."""

    def __init__(self, channel_id):
        self.channel_id = channel_id
        self._stop = asyncio.Event()
        self._task = None

    async def __aenter__(self):
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc):
        self._stop.set()
        if self._task:
            self._task.cancel()
        return False

    async def _loop(self):
        while not self._stop.is_set():
            try:
                await rest.typing(self.channel_id)
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=8)
            except asyncio.TimeoutError:
                pass

async def run_addressed(channel_id: int, guild_id: int, d: dict, display: str, content: str):
    st = channel_state[channel_id]
    if st["paused"]:
        return
    store = store_for(guild_id)
    author_id = int(d["author"]["id"])
    _ts = TypingSession(channel_id)
    await _ts.__aenter__()
    web = await web_lookup(content) if needs_web_lookup(content) else None
    text = await compose_addressed_reply(store, channel_id, display, web=web)
    target_id, target_content = int(d["id"]), content
    # no added latency: the reply is composed right away, but if the same
    # author kept posting while it was being composed, re-compose once with
    # the whole bundle and answer the newest message instead
    newer = [m for m in store.recent(channel_id, 12)
             if m.get("author_id") == author_id and m["id"] > target_id]
    if newer:
        text2 = await compose_addressed_reply(store, channel_id, display, web=web)
        if text2:
            text = text2
        target_id, target_content = newer[-1]["id"], newer[-1]["content"]
    if text:
        st["last_ambient"] = time.time()
        if await human_send(channel_id, text, reply_to=target_id, urgent=True, directed=True, full=contains_long_template(store, text)):
            record_trigger(st, author_id, target_content)
        if st["pending_task"] and not st["pending_task"].done():
            st["pending_task"].cancel()
    await _ts.__aexit__()


async def handle_dm(d: dict):
    """DM support for testing: owner-only, always replies, and shares the
    main guild's persona / memes / templates / knowledge."""
    author = d.get("author", {})
    owner = CFG.get("owner_id")
    if not owner or int(author.get("id", 0)) != int(owner):
        return
    channel_id = int(d["channel_id"])
    channel_guild[channel_id] = MAIN_GUILD_ID
    content = (d.get("content") or "").strip()
    if not content:
        return
    display = author.get("global_name") or author.get("username", "?")
    store = store_for(MAIN_GUILD_ID)
    stored = content
    for m in d.get("mentions", []):
        stored = stored.replace(f"<@{m['id']}>", f"@{m.get('username', 'user')}")
        stored = stored.replace(f"<@!{m['id']}>", f"@{m.get('username', 'user')}")
    store.add({"id": int(d["id"]), "author": display, "author_id": int(author.get("id", 0)), "content": stored,
               "channel_id": channel_id, "ts": time.time()})
    st = channel_state[channel_id]
    emb = await maybe_learn_from_message(store, channel_id, display, content)
    if emb:
        await rest.send_message(channel_id, None, embeds=[emb])
        return
    if await try_execute_action(channel_id, content):
        record_trigger(st, int(author.get("id", 0)), content)
        return
    if reishou_detect(content):
        st["last_reishou"] = time.time()
        await human_send(channel_id, REISHOU_FIXED, urgent=True)
        return
    if trigger_is_duplicate(st, int(author.get("id", 0)), content):
        return
    async with TypingSession(channel_id):
        web = await web_lookup(content) if needs_web_lookup(content) else None
        text = await compose_reply(store, channel_id, display, web=web)
        target_id, target_content = int(d["id"]), content
        newer = [m for m in store.recent(channel_id, 12)
                 if m.get("author_id") == int(author.get("id", 0)) and m["id"] > target_id]
        if newer:
            text2 = await compose_reply(store, channel_id, display, web=web)
            if text2:
                text = text2
            target_id, target_content = newer[-1]["id"], newer[-1]["content"]
        sent = False
        if text:
            sent = await human_send(channel_id, text, reply_to=target_id,
                                    urgent=True, directed=True, full=contains_long_template(store, text))
            if sent:
                record_trigger(st, int(author.get("id", 0)), target_content)
        if not sent:
            await send_fallback_ack(channel_id)


async def handle_message(d: dict):
    author = d.get("author", {})
    if author.get("bot"):
        return
    guild_id = d.get("guild_id")
    if guild_id is None:
        await handle_dm(d)
        return
    guild_id = int(guild_id)
    if CFG["guild_allowlist"] and guild_id not in CFG["guild_allowlist"]:
        return
    channel_id = int(d["channel_id"])
    channel_guild[channel_id] = guild_id
    content = (d.get("content") or "").strip()
    member = d.get("member") or {}
    display = member.get("nick") or author.get("global_name") or author.get("username", "?")
    store = store_for(guild_id)
    st = channel_state[channel_id]

    if CFG.get("owner_id") and int(author["id"]) == CFG["owner_id"] and content.startswith("!bot"):
        st = channel_state[channel_id]
        if content == "!botpause":
            st["paused"] = True
            await rest.send_message(channel_id, "了解、このチャンネルではしばらく静かにしてます。")
        elif content == "!botresume":
            st["paused"] = False
            await rest.send_message(channel_id, "再開します。")
        elif content == "!botprofile":
            await refresh_profile(store)
            await rest.send_message(
                channel_id, "雰囲気プロファイルを更新しました。\n" + (store.profile or "（空）")[:1500])
        return

    if content:
        # store a readable version: raw <@123...> mention markup teaches the
        # model to emit ID digits, so swap in the mentioned user's name
        name_map = {}
        for u in d.get("mentions") or []:
            name_map[str(u["id"])] = u.get("global_name") or u.get("username") or "メンション"
        stored_content = re.sub(
            r"<@[!&]?(\d+)>",
            lambda m: "@" + name_map.get(m.group(1), "メンション"), content)
        store.add({"id": int(d["id"]), "author": display, "author_id": int(author["id"]), "content": stored_content,
                   "channel_id": channel_id, "ts": time.time()})
        if st.get("para_active") and STOP_RE.search(content):
            st["para_stop"] = True
        update_heat(st, int(author["id"]), content)
        if store.profile_due():
            asyncio.create_task(refresh_profile(store))
        if store.knowledge_due():
            asyncio.create_task(refresh_knowledge(store))

    st = channel_state[channel_id]
    if content and not st["paused"]:
        emb = await maybe_learn_from_message(store, channel_id, display, content)
        if emb:
            st["last_ambient"] = time.time()
            await rest.send_message(channel_id, None, reply_to=int(d["id"]),
                                    embeds=[emb])
            return
    mentioned = any(int(m["id"]) == rest.bot_id for m in d.get("mentions", []))
    ref = d.get("referenced_message") or {}
    is_reply_to_bot = int((ref.get("author") or {}).get("id", 0) or 0) == rest.bot_id

    if content and (mentioned or is_reply_to_bot or BOT_NAME_RE.search(content)
                    or (st["last_bot_msg"] > 0
                        and time.time() - st["last_bot_msg"] < CFG["addressed_window_sec"])):
        if await try_execute_action(channel_id, content):
            record_trigger(st, int(author["id"]), content)
            return

    if mentioned or is_reply_to_bot:
        sent = False
        async with TypingSession(channel_id):
            if content:
                web = await web_lookup(content) if needs_web_lookup(content) else None
                text = await compose_reply(store, channel_id, display, web=web)
                if text:
                    st["last_ambient"] = time.time()
                    sent = await human_send(channel_id, text, reply_to=int(d["id"]), urgent=True, directed=True, full=contains_long_template(store, text))
                    if sent:
                        record_trigger(st, int(author["id"]), content)
            if not sent:
                await send_fallback_ack(channel_id)
        if st["pending_task"] and not st["pending_task"].done():
            st["pending_task"].cancel()
        return

    ref_author_id = int((ref.get("author") or {}).get("id", 0) or 0)
    reply_to_other = bool(ref) and ref_author_id not in (0, rest.bot_id, int(author["id"]))

    if (CFG.get("reishou_detection", True) and content and not st["paused"]
            and time.time() - st.get("last_reishou", 0) > 90
            and reishou_detect(content)
            and (time.time() - st["last_bot_msg"] < CFG["addressed_window_sec"]
                 or (ref and ref_author_id == rest.bot_id))):
        st["last_reishou"] = time.time()
        text = await compose_reishou_reply(store, channel_id, display, content)
        if text:
            st["last_ambient"] = time.time()
            await human_send(channel_id, text, reply_to=int(d["id"]), urgent=True, directed=True)
            if st["pending_task"] and not st["pending_task"].done():
                st["pending_task"].cancel()
        return

    laugh_only = bool(LAUGH_ONLY.match(content)) if content else False
    taunt_muted = (time.time() < st.get("taunt_mute", {}).get(int(author["id"]), 0)
                   and bool(TAUNT_RE.search(content or ""))
                   and not QUESTION.search(content or ""))
    if CFG.get("addressed_detection") and content and not st["paused"] and not taunt_muted and not (
            laugh_only and random.random() < 0.8):
        if trigger_is_duplicate(st, int(author["id"]), content):
            if random.random() < 0.3:
                await rest.add_reaction(channel_id, int(d["id"]),
                                        random.choice(["👀", "👍", "🫡"]))
        elif addressed_candidate(channel_id, content, reply_to_other, store, display):
            await run_addressed(channel_id, guild_id, d, display, content)
            return

    if not CFG["ambient_enabled"] or st["paused"]:
        return
    now = time.time()
    st["burst"] = [b for b in st["burst"] if now - b["ts"] < 120]
    st["burst"].append({"user": int(author["id"]), "ts": now})
    users = {b["user"] for b in st["burst"]}
    if len(st["burst"]) >= CFG["min_burst_messages"] and len(users) >= CFG["min_burst_users"]:
        if st["pending_task"] and not st["pending_task"].done():
            st["pending_task"].cancel()
        st["pending_task"] = asyncio.create_task(maybe_ambient(channel_id, guild_id))


def _opt_s(name, desc, req):
    return {"name": name, "description": desc, "type": 3, "required": req}


def _opt_i(name, desc, req):
    return {"name": name, "description": desc, "type": 4, "required": req}


COMMAND_DEFS = [
    {"name": "構文追加", "type": 1, "description": "Botが使う構文（コピペ・決め台詞）を教える",
     "options": [_opt_s("テンプレート", "覚えさせたい構文（○○が置き換え場所）", True),
                 _opt_s("説明", "使いどころ（例：味方のミスへのツッコミ）", False)]},
    {"name": "構文一覧", "type": 1, "description": "登録済みの構文を一覧表示（自分にだけ見えます）"},
    {"name": "構文削除", "type": 1, "description": "登録済みの構文を番号で削除（自分にだけ見えます）",
     "options": [_opt_i("番号", "構文一覧の番号", True)]},
    {"name": "ミーム追加", "type": 1, "description": "Botが使うミーム・スラングを教える",
     "options": [_opt_s("用語", "覚えさせたいミーム・スラング", True),
                 _opt_s("説明", "意味・使いどころ", False)]},
    {"name": "ミーム一覧", "type": 1, "description": "このサーバーで追加されたミームを一覧表示（自分にだけ見えます）"},
    {"name": "ミーム削除", "type": 1, "description": "追加したミームを番号で削除（自分にだけ見えます）",
     "options": [_opt_i("番号", "ミーム一覧の番号", True)]},
    {"name": "モデル使用量", "type": 1, "description": "各モデルの使用量と、今使えるかどうかをチェックする"},
]


async def ensure_guild_commands(guild_id: int):
    """Slash commands are wiped when the bot is kicked from a guild, so
    re-register them automatically whenever the bot (re)joins one. Checks
    first and only writes when the registered set differs."""
    try:
        current = await rest.request(
            "GET", f"/applications/{APP_ID}/guilds/{guild_id}/commands")
        names = {c.get("name") for c in current} if isinstance(current, list) else set()
        want = {c["name"] for c in COMMAND_DEFS}
        if names == want:
            return
        await rest.request(
            "PUT", f"/applications/{APP_ID}/guilds/{guild_id}/commands",
            json=COMMAND_DEFS)
        print(f"[commands] re-registered {len(COMMAND_DEFS)} commands in guild "
              f"{guild_id} (was: {sorted(names)})", flush=True)
    except Exception as e:
        print(f"[commands] ensure failed in guild {guild_id}: {e}", flush=True)

class Gateway:
    def __init__(self):
        self.session_id = None
        self.seq = None
        self.resume_url = None
        self.ws = None
        self.hb_task = None
        self.presence_task = None
        self.presence_idx = 0

    async def presence_loop(self):
        texts = CFG.get("presence_texts") or []
        if not texts:
            return
        while True:
            ws = self.ws
            if ws is not None:
                text = texts[self.presence_idx % len(texts)]
                self.presence_idx += 1
                try:
                    await ws.send(json.dumps({"op": 3, "d": {
                        "since": None,
                        "activities": [{"name": "Custom Status", "type": 4,
                                        "state": text}],
                        "status": "online", "afk": False}}))
                    print(f"[presence] set: {text}", flush=True)
                except Exception:
                    pass
            await asyncio.sleep(CFG["presence_rotate_sec"])

    async def run_forever(self):
        backoff = 2
        while True:
            try:
                await self.connect()
                backoff = 2
            except Exception as e:
                print(f"[gateway] disconnected: {type(e).__name__}: {str(e)[:160]}", flush=True)
            await asyncio.sleep(backoff + random.uniform(0, 2))
            backoff = min(backoff * 2, 60)

    async def connect(self):
        url = self.resume_url or "wss://gateway.discord.gg/?v=10&encoding=json"
        async with websockets.connect(url, proxy=PROXY, open_timeout=20,
                                      ping_interval=None) as ws:
            self.ws = ws
            async for raw in ws:
                payload = json.loads(raw)
                op = payload.get("op")
                if payload.get("s") is not None:
                    self.seq = payload["s"]
                if op == 10:
                    interval = payload["d"]["heartbeat_interval"] / 1000.0
                    self.hb_task = asyncio.create_task(self.heartbeat(ws, interval))
                    if self.session_id and self.seq is not None:
                        await ws.send(json.dumps({"op": 6, "d": {
                            "token": TOKEN, "session_id": self.session_id, "seq": self.seq}}))
                    else:
                        await ws.send(json.dumps({"op": 2, "d": {
                            "token": TOKEN, "intents": INTENTS,
                            "properties": {"os": "linux", "browser": "vibe-bot",
                                           "device": "vibe-bot"}}}))
                elif op == 0:
                    await self.dispatch(payload.get("t"), payload.get("d") or {})
                elif op == 7:
                    return
                elif op == 9:
                    self.session_id = None
                    self.seq = None
                    self.resume_url = None
                    return
                elif op == 11:
                    pass

    async def heartbeat(self, ws, interval):
        try:
            await asyncio.sleep(interval * random.random())
            while True:
                await ws.send(json.dumps({"op": 1, "d": self.seq}))
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def dispatch(self, t, d):
        if t == "READY":
            self.session_id = d.get("session_id")
            self.resume_url = d.get("resume_gateway_url")
            guilds = d.get("guilds", [])
            print(f"[ready] logged in as {rest.bot_name} in {len(guilds)} guild(s)", flush=True)
            if self.presence_task is None or self.presence_task.done():
                self.presence_task = asyncio.create_task(self.presence_loop())
        elif t == "RESUMED":
            print("[gateway] session resumed", flush=True)
        elif t == "GUILD_CREATE":
            asyncio.create_task(ensure_guild_commands(int(d["id"])))
        elif t == "INTERACTION_CREATE":
            asyncio.create_task(handle_interaction(d))
        elif t == "MESSAGE_CREATE":
            if event_already_seen(int(d["id"])):
                print(f"[gateway] duplicate event skipped: {d['id']}", flush=True)
                return
            asyncio.create_task(handle_message(d))


async def main():
    if not TOKEN:
        print("DISCORD_TOKEN is not set in .env - cannot start.", file=sys.stderr)
        sys.exit(2)
    if CFG["llm_backend"] == "openai" and not OPENAI_KEY:
        print("llm_backend=openai but OPENAI_API_KEY is not set in .env.", file=sys.stderr)
        sys.exit(2)
    await rest.start()
    print(f"[rest] authenticated as {rest.bot_name} ({rest.bot_id})", flush=True)
    await Gateway().run_forever()


if __name__ == "__main__":
    asyncio.run(main())

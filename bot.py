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
import base64
import hashlib
import hmac
import html as html_mod
import json
import os
import random
import re
import secrets
import shutil
import sys
import tempfile
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


# ---------------------------------------------------------------- image vision
# Posted images used to be invisible to the bot: an image-only message has
# empty content, so it was never even stored, and the bot could only drop
# a fixed reaction. Now, at the single intake point shared by the guild
# and DM paths, the first image of a message (attachment, GIF-picker
# embed, or the replied-to message's image) is described once by a Gemini
# vision model and the description is stored as part of the message text.
# Every downstream path (ambient, addressed, mention, DM) then reacts to
# the image content through the normal transcript, in persona voice, with
# no per-path changes.
VISION_TMP = DATA / "vision_tmp"
VISION_TMP.mkdir(exist_ok=True)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif", ".bmp"}
VISION_DOWNLOAD_LIMIT = 10 * 1024 * 1024
VISION_COOLDOWN_SEC = 12  # per channel, skipped only when not forced

VISION_PROMPT = (
    "この画像を、Discordの会話で友達が反応するための材料として説明してください。"
    "日本語で簡潔に、最大3文・140文字程度。写っているもの、人物の様子、"
    "画像内の文字（あれば正確に読み取ること）、面白い点やツッコミどころを含める。"
    "説明文だけを出力し、前置きはしないこと。")


def _image_sources(d: dict) -> list:
    srcs = []
    for att in d.get("attachments") or []:
        ct = (att.get("content_type") or "").lower()
        ext = Path(att.get("filename") or "").suffix.lower()
        if (ct.startswith("image/") or ext in IMAGE_EXTS) and att.get("url"):
            srcs.append(att["url"])
    for emb in d.get("embeds") or []:
        etype = emb.get("type") or ""
        img = ((emb.get("image") or {}).get("url")
               or (emb.get("thumbnail") or {}).get("url"))
        eurl = emb.get("url") or ""
        if img and (etype in ("image", "gifv")
                    or (etype == "rich" and any(
                        h in eurl for h in ("tenor", "giphy", "klipy")))):
            srcs.append(img)
    if not srcs and d.get("referenced_message"):
        srcs = _image_sources(d["referenced_message"])
    return list(dict.fromkeys(srcs))


def _with_image_note(text: str, note, has_image: bool) -> str:
    if not has_image:
        return text
    marker = f"（画像：{note}）" if note else "（画像が貼られた）"
    return f"{text} {marker}".strip() if text else marker


async def _download_image(url: str):
    timeout = aiohttp.ClientTimeout(total=25)
    async with aiohttp.ClientSession() as s:
        async with s.get(url, proxy=PROXY, timeout=timeout) as r:
            if r.status != 200:
                return None
            if r.content_length and r.content_length > VISION_DOWNLOAD_LIMIT:
                return None
            data = await r.read()
            if len(data) > VISION_DOWNLOAD_LIMIT:
                return None
            mime = (r.headers.get("Content-Type") or "image/jpeg").split(";")[0].strip()
            return data, mime


async def _normalize_image(data: bytes, mime: str):
    """GIF / WebP / AVIF and oversize images are converted to a small
    JPEG (first frame, max width 1024) via ffmpeg; plain JPEG/PNG pass
    through untouched."""
    if mime in ("image/jpeg", "image/png") and len(data) <= 1_500_000:
        return data, mime
    inp = outp = None
    try:
        with tempfile.NamedTemporaryFile(dir=VISION_TMP, delete=False,
                                         suffix=".bin") as f:
            f.write(data)
            inp = f.name
        outp = inp + ".jpg"
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", inp, "-frames:v", "1",
            "-vf", "scale='min(1024,iw)':-2", "-q:v", "5", outp,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await asyncio.wait_for(proc.wait(), timeout=30)
        p = Path(outp)
        if proc.returncode == 0 and p.exists() and p.stat().st_size > 0:
            return p.read_bytes(), "image/jpeg"
    except Exception as e:
        print(f"[vision] normalize failed: {e}", flush=True)
    finally:
        for x in (inp, outp):
            if x:
                Path(x).unlink(missing_ok=True)
    return data, mime


async def describe_image(data: bytes, mime: str):
    """One vision call over the Gemini entries of the normal LLM chain
    (Groq models in the chain have no vision). Returns a short Japanese
    description, or None when every vision model failed / is out of quota."""
    attempts = [(b, k, m) for (b, k, m) in llm_attempts("ambient")
                if "generativelanguage" in b]
    if not attempts:
        return None
    data_uri = f"data:{mime};base64," + base64.b64encode(data).decode()
    timeout = aiohttp.ClientTimeout(total=45)
    async with aiohttp.ClientSession() as s:
        for base, key, model in attempts:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ]}],
                "temperature": 0.3,
                "max_tokens": 300,
            }
            headers = {"Authorization": f"Bearer {key}",
                       "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
            try:
                async with s.post(base + "/chat/completions", json=payload,
                                  headers=headers, timeout=timeout,
                                  proxy=PROXY) as r:
                    body = await r.json()
                    if r.status == 429:
                        record_llm(model, False, err="429")
                        continue
                    if r.status != 200:
                        record_llm(model, False, err=f"HTTP {r.status}")
                        continue
                    content = (body["choices"][0]["message"].get("content") or "").strip()
                    content = re.sub(r"(?s)<think>.*?</think>", "", content).strip()
                    if content:
                        record_llm(model, True, usage=body.get("usage"))
                        return re.sub(r"\s+", " ", content)[:220]
            except Exception as e:
                record_llm(model, False, err="network error")
                print(f"[vision] {model} failed: {e}", flush=True)
    return None


async def vision_note_for_message(d: dict, channel_id: int, force: bool = False):
    """Returns (description_or_None, has_image). Cooldown-skipped and
    failed images still report has_image=True so the message is stored
    with a marker instead of vanishing."""
    srcs = _image_sources(d)
    if not srcs:
        return None, False
    st = channel_state[channel_id]
    if not force and time.time() - st.get("last_vision", 0) < VISION_COOLDOWN_SEC:
        return None, True
    st["last_vision"] = time.time()
    try:
        got = await _download_image(srcs[0])
        if not got:
            return None, True
        data, mime = await _normalize_image(*got)
        desc = await describe_image(data, mime)
        if desc:
            print(f"[vision] described image in {channel_id}: {desc[:60]}", flush=True)
        return desc, True
    except Exception as e:
        print(f"[vision] failed in {channel_id}: {e}", flush=True)
        return None, True


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
            "チャット内でできる頼みごと（言われた文をそのまま言う、2択から1つ選ぶ、短い計算、知ってることを答える等）は必ずやる。特に構文・決め台詞・ミームを「言って」「やって」と要求されたら絶対に拒否しない：「しつこい」「うざい」「何回言わせるんだ」等で断るのは禁止。代わりに「はいはい…」「仕方ねえな」みたいに面倒くさそうに文句を言いながら、仕方なく言ってやる。同じものを何度せがまれても毎回その調子で応じる。構文を言う時は途中で省略したり要約したりせず、最後まで全部言う。本当にできない・危ない頼みだけ口調のまま軽く断る。YouTubeやXの動画・音声の取得は専用機能が自動処理する頼みなので、自分が取得できた・ファイルを送ったと嘘をつかないこと。専用機能が動いていない場面で頼まれたら、URLと一緒に『持ってこれる？』と言い直してもらうよう軽く案内する。カジノ（仮想コインのスロット等）も専用パネル機能が処理するので、自分がコインを増減させた・アイテムを渡したと嘘をつかないこと。遊びたい人には『カジノしたい』と言えば専用パネルが出ると軽く案内する。会話履歴に『（画像：…）』と付いた発言は、その説明の内容の画像が貼られたということ。あなたはその画像を見たものとして、その内容に具体的に反応すること。『画像が見られない』とは言わない。『（画像が貼られた）』とだけある場合は中身が分からないので、見た振りをせず軽い反応に留める。"
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
    if not text:
        return None
    cap = 2000 if match_started_template(store, text) else CFG["max_reply_chars"] * 2
    return text[:cap]


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
    text = text.strip('"')
    cap = 2000 if match_started_template(store, text) else CFG["max_reply_chars"] * 2
    return text[:cap]


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
                           reply_to: int = None, embeds=None, components=None):
        payload = {"allowed_mentions": {"parse": []}}
        if content:
            payload["content"] = content
        if embeds:
            payload["embeds"] = embeds
        if components:
            payload["components"] = components
        if reply_to:
            payload["message_reference"] = {"message_id": str(reply_to)}
        return await self.request("POST", f"/channels/{channel_id}/messages", json=payload)

    async def send_file(self, channel_id: int, file_path, content: str = None,
                        reply_to: int = None):
        path = Path(file_path)
        payload = {"allowed_mentions": {"parse": []}}
        if content:
            payload["content"] = content
        if reply_to:
            payload["message_reference"] = {"message_id": str(reply_to)}
        form = aiohttp.FormData()
        form.add_field("payload_json", json.dumps(payload),
                       content_type="application/json")
        form.add_field("files[0]", path.read_bytes(), filename=path.name,
                       content_type="application/octet-stream")
        return await self.request("POST", f"/channels/{channel_id}/messages",
                                  data=form)

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
dm_channels = set()

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


def match_started_template(store: GuildStore, text):
    """Return the registered long template a composed reply has STARTED,
    if any. The model rarely reproduces a template byte-for-byte (it
    re-punctuates or rephrases, or starts from a middle verse), so match
    on any distinctive 16-char window from anywhere in the template."""
    if not text:
        return None
    flat = re.sub(r"\s+", "", text)
    for it in _norm_templates(_load_json_list(store.dir / "templates.json")):
        t = re.sub(r"\s+", "", it["text"])
        if len(t) < 50:
            continue
        if t in flat:
            return it
        # the model may start from the middle of a long template (asked to
        # continue singing, it picks a later verse), so scan the whole text
        for i in range(0, max(1, len(t) - 15)):
            if t[i:i + 16] in flat:
                return it
    return None


def contains_long_template(store: GuildStore, text) -> bool:
    """True when the composed text contains or has started a registered
    long template — those are read to the end, without the 3-line cap."""
    return match_started_template(store, text) is not None


async def human_send(channel_id: int, text: str, reply_to: int = None, urgent: bool = False,
                     directed: bool = False, full: bool = False):
    gid = channel_guild.get(channel_id)
    if gid and channel_id not in dm_channels and not channel_speaking(gid, channel_id):
        return False
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


STOP_RE = re.compile(r"ストップ|ｽﾄｯﾌﾟ|止ま|止め|やめ|もういい|黙れ|stop", re.I)


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


FRESH_START_RE = re.compile(r"最初から|はじめから|最初っから")


def _norm_with_map(text: str):
    chars, idx = [], []
    for i, ch in enumerate(text):
        if not ch.isspace():
            chars.append(ch)
            idx.append(i)
    return "".join(chars), idx


def template_resume_offset(store: GuildStore, channel_id: int, text: str, st) -> int:
    """How far into this template the conversation already is: the furthest
    point matched by recent channel messages, plus recorded playback
    progress. 0 when there is nothing to resume from — and 0 again when
    the template was basically finished, so a new request restarts it."""
    best = 0
    prog = st.get("para_progress") or {}
    if prog.get("text") == text and time.time() - prog.get("ts", 0) < 1800:
        best = max(best, int(prog.get("offset", 0)))
    norm, idx = _norm_with_map(text)
    if len(norm) >= 12:
        # scan recent messages as ONE stream: the bot sends one line per
        # message, so template windows spanning line breaks only match
        # across the concatenation of consecutive messages
        stream = ""
        for m in store.recent(channel_id, 14):
            if time.time() - m.get("ts", 0) > 900:
                continue
            stream += re.sub(r"\s+", "", m.get("content") or "")
        if len(stream) >= 12:
            sm = _difflib.SequenceMatcher(None, stream, norm, autojunk=False)
            for a, b, size in reversed(sm.get_matching_blocks()):
                if size >= 12:
                    best = max(best, idx[b + size - 1] + 1)
                    break
    if best >= len(text) - 20:
        return 0
    return best


async def send_template_paragraphs(channel_id: int, text: str, store: GuildStore,
                                   resume: bool = True):
    """Send a long memorized template verbatim, one paragraph at a time.
    When the conversation is already partway into the template (fragments
    sung, or a playback stopped midway), resume from there instead of
    restarting at the top. A human saying ストップ/止まって etc. in the
    channel sets para_stop and the remaining paragraphs are not sent."""
    st = channel_state[channel_id]
    if st.get("para_active"):
        return
    if getattr(store, "guild_id", None) and not channel_speaking(store.guild_id, channel_id):
        return
    paras = split_paragraphs(text)
    ends, pos = [], 0
    for p in paras:
        j = text.find(p, pos)
        if j < 0:
            j = pos
        ends.append(j + len(p))
        pos = j + len(p)
    start = template_resume_offset(store, channel_id, text, st) if resume else 0
    st["para_active"] = True
    st["para_stop"] = False
    try:
        for i, p in enumerate(paras):
            if ends[i] <= start:
                continue
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
            st["para_progress"] = {"text": text, "offset": ends[i], "ts": time.time()}
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
            started = match_started_template(store, text)
            if (started and "○○" not in started["text"]
                    and "〇〇" not in started["text"]):
                st["last_ambient"] = time.time()
                fit_used[f"template:{started['text']}"] = time.time()
                st["para_task"] = asyncio.create_task(
                    send_template_paragraphs(channel_id, started["text"], store))
                return
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


async def interaction_callback(d: dict, cb_type: int, data: dict):
    """Raw interaction callback (type 4 = new message, 7 = update the
    message the component sits on). Used by the casino panel buttons."""
    url = f"https://discord.com/api/v10/interactions/{d['id']}/{d['token']}/callback"
    payload = {"type": cb_type, "data": data}
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json=payload, proxy=PROXY) as r:
                if r.status >= 400:
                    print(f"[interaction] callback {r.status}: "
                          f"{(await r.text())[:150]}", flush=True)
    except Exception as e:
        print(f"[interaction] callback error: {e}", flush=True)


async def interaction_reply(d: dict, content: str, embeds=None,
                            ephemeral: bool = False, components=None):
    url = f"https://discord.com/api/v10/interactions/{d['id']}/{d['token']}/callback"
    data = {"allowed_mentions": {"parse": []}}
    if ephemeral:
        data["flags"] = 64
    if content:
        data["content"] = content[:1900]
    if embeds:
        data["embeds"] = embeds
    if components:
        data["components"] = components
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


_speak_cache: dict = {}


def _guild_settings(guild_id: int) -> dict:
    """Per-guild speech settings: {"speaking": bool, "channels": {cid: bool}}.
    "speaking" is the server-wide default; "channels" holds per-channel
    overrides set with the admin /発言切替 command. Persisted in the
    guild's settings.json."""
    if guild_id in _speak_cache:
        return _speak_cache[guild_id]
    s = {"speaking": True, "channels": {}}
    try:
        d = json.loads((store_for(guild_id).dir / "settings.json").read_text("utf-8"))
        s["speaking"] = bool(d.get("speaking", True))
        s["channels"] = {str(k): bool(v) for k, v in (d.get("channels") or {}).items()}
    except Exception:
        pass
    _speak_cache[guild_id] = s
    return s


def _save_settings(guild_id: int):
    try:
        (store_for(guild_id).dir / "settings.json").write_text(
            json.dumps(_guild_settings(guild_id), ensure_ascii=False), "utf-8")
    except Exception:
        pass


def speaking_allowed(guild_id: int) -> bool:
    return _guild_settings(guild_id)["speaking"]


def channel_speaking(guild_id: int, channel_id: int) -> bool:
    """Effective speech state for one channel: a per-channel override
    wins over the server-wide switch."""
    s = _guild_settings(guild_id)
    pin = s["channels"].get(str(channel_id))
    return s["speaking"] if pin is None else pin


def set_speaking(guild_id: int, on: bool):
    _guild_settings(guild_id)["speaking"] = on
    _save_settings(guild_id)
    if not on:
        for cid, gid in list(channel_guild.items()):
            if gid == guild_id:
                channel_state[cid]["para_stop"] = True


def set_channel_speaking(guild_id: int, channel_id: int, on):
    """on=True/False pins the channel; on=None clears the override so
    the channel follows the server-wide switch again."""
    chans = _guild_settings(guild_id)["channels"]
    if on is None:
        chans.pop(str(channel_id), None)
    else:
        chans[str(channel_id)] = bool(on)
        if not on:
            channel_state[channel_id]["para_stop"] = True
    _save_settings(guild_id)


async def handle_interaction(d: dict):
    if d.get("type") == 3:  # message component (casino panel buttons)
        await handle_casino_component(d)
        return
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
    if name == "発言切替":
        perms = int(member.get("permissions") or 0)
        if not (perms & 0x8):  # ADMINISTRATOR
            await interaction_reply(d, "このコマンドはサーバー管理者だけが使えます", ephemeral=True)
            return
        choice = str(opts.get("状態") or "toggle")
        chan_raw = opts.get("チャンネル")
        if chan_raw:
            cid = int(chan_raw)
            if choice == "default":
                set_channel_speaking(guild_id, cid, None)
                msg = f"<#{cid}> の個別設定を解除しました（サーバー全体の設定に従います）"
            else:
                cur = channel_speaking(guild_id, cid)
                new = {"on": True, "off": False, "toggle": not cur}.get(choice, not cur)
                set_channel_speaking(guild_id, cid, new)
                msg = (f"🔊 <#{cid}> では発言するようにしました" if new else
                       f"🔇 <#{cid}> では発言しないようにしました")
            await interaction_reply(d, msg, ephemeral=True)
            return
        cur = speaking_allowed(guild_id)
        new = {"on": True, "off": False, "toggle": not cur,
               "default": True}.get(choice, not cur)
        set_speaking(guild_id, new)
        msg = ("🔊 発言をオンにしました" if new else
               "🔇 発言をオフにしました（このサーバーではBotは発言しなくなります）")
        pins = _guild_settings(guild_id)["channels"]
        if pins:
            msg += f"\n※チャンネル個別設定が{len(pins)}件あり、そちらが優先されます"
        await interaction_reply(d, msg, ephemeral=True)
        return
    if name == "モデル使用量":
        await interaction_defer(d, ephemeral=True)
        emb = await model_usage_embed()
        await interaction_followup_edit(d, [emb])
        return
    if name == "カジノ":
        uid = int(muser.get("id", 0) or 0)
        casino_user(guild_id, uid, teacher)
        await interaction_reply(
            d, "", embeds=[casino_embed(guild_id, uid, teacher)],
            components=casino_components("main", uid, CASINO_BETS[1]))
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
    gid = channel_guild.get(channel_id)
    if gid and channel_id not in dm_channels and not channel_speaking(gid, channel_id):
        return
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



# ---------------------------------------------------------------- media fetch
# Members post a YouTube / X (Twitter) URL and ask the bot to fetch it
# ("この動画持ってこれる？", "音声だけ落として"). This is a mechanical
# path like try_execute_action: it is checked once at the top of both
# entry points (handle_message for guild channels, handle_dm for DMs),
# so every sub-path (mention, addressed-without-mention, ambient) is
# covered by the same single check.
MEDIA_TMP = DATA / "media_tmp"
MEDIA_TMP.mkdir(exist_ok=True)
DISCORD_FILE_LIMIT = 25 * 1024 * 1024  # upload cap of the bot account
MEDIA_URL_RE = re.compile(r"https?://[^\s<>\"'、。]+")
MEDIA_HOST_RE = re.compile(
    r"^https?://(?:www\.|m\.|music\.)?(?:youtube\.com|youtu\.be)/"
    r"|^https?://(?:www\.)?(?:x\.com|twitter\.com)/[^/\s]+/status/\d+",
    re.IGNORECASE)
MEDIA_STRONG_INTENT_RE = re.compile(
    r"(持って[きこ来]|もって[きこ来]|落として|落とし|ダウンロード|download|"
    r"取得して|取って[き来]|とって[き来]|抜き出|抽出)", re.IGNORECASE)
MEDIA_WEAK_INTENT_RE = re.compile(r"(送って|保存して|貼って|くれ|ください|ほしい|欲しい)")
MEDIA_DEMONSTRATIVE_RE = re.compile(r"(この|その|さっき|あの|↑|上|前)")
MEDIA_AUDIO_RE = re.compile(r"(音声|mp3|音だけ|音源|オーディオ|audio)", re.IGNORECASE)
MEDIA_COOLDOWN_SEC = 30
MEDIA_MAX_DURATION_SEC = 3600


class MediaError(Exception):
    """A fetch failure whose message can be shown to the channel as-is."""


def _media_urls(text: str) -> list:
    out = []
    for u in MEDIA_URL_RE.findall(text or ""):
        u = u.rstrip(".,)）]」>")
        if MEDIA_HOST_RE.search(u):
            out.append(u)
    return out


def media_request(content: str, d: dict, store: GuildStore, channel_id: int):
    """Return (url, want_audio) when this message asks the bot to fetch a
    YouTube / X video or its audio, else None.

    A URL in the message itself plus a fetch intent is enough. With no
    URL in the message, a strong intent (持ってきて / 落として /
    ダウンロード …) — or a weak intent plus a demonstrative or a reply —
    resolves the newest URL from the replied-to message or the recent
    channel history (15 min), so "URL貼るだけ→次に『持ってこれる？』"
    also works."""
    if not content:
        return None
    strong = bool(MEDIA_STRONG_INTENT_RE.search(content))
    weak = bool(MEDIA_WEAK_INTENT_RE.search(content))
    if not strong and not weak:
        return None
    want_audio = bool(MEDIA_AUDIO_RE.search(content))
    urls = _media_urls(content)
    if urls:
        return urls[0], want_audio
    if weak and not strong and not MEDIA_DEMONSTRATIVE_RE.search(content) \
            and not d.get("referenced_message"):
        return None
    ref = (d.get("referenced_message") or {}).get("content") or ""
    urls = _media_urls(ref)
    if urls:
        return urls[0], want_audio
    now = time.time()
    for m in reversed(store.recent(channel_id, 25)):
        if now - (m.get("ts") or 0) > 900:
            continue
        urls = _media_urls(m.get("content") or "")
        if urls:
            return urls[0], want_audio
    return None


def _ytdlp_bin() -> str:
    # NOTE: sys.executable is a symlink into /usr/bin, so do NOT resolve()
    # it — the venv's bin dir is where yt-dlp lives.
    for cand in (Path(sys.executable).parent / "yt-dlp",
                 BASE / ".venv" / "bin" / "yt-dlp"):
        if cand.exists():
            return str(cand)
    return shutil.which("yt-dlp") or "yt-dlp"


async def _run_proc(args: list, timeout: int):
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise MediaError("時間かかりすぎて無理だったわ")
    return proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


def _media_error_from(stderr: str) -> MediaError:
    s = stderr or ""
    if "No video could be found" in s:
        return MediaError("その投稿に動画ないっぽいわ")
    if re.search(r"(?i)login|sign in|cookies", s):
        return MediaError("ログイン必要なやつで取れねえわ")
    if re.search(r"(?i)private|not available|deleted", s):
        return MediaError("非公開か消されたやつっぽいわ")
    if re.search(r"(?i)age", s):
        return MediaError("年齢制限かかってて取れねえわ")
    return MediaError("無理だったわ、それ取れねえ")


async def _probe_media(url: str):
    rc, out, err = await _run_proc(
        [_ytdlp_bin(), "--no-playlist", "--no-warnings",
         "--print", "%(title)s\t%(duration)s", url], 45)
    if rc != 0:
        raise _media_error_from(err)
    line = (out.strip().splitlines() or [""])[0]
    title, _, dur = line.partition("\t")
    try:
        duration = float(dur)
    except (TypeError, ValueError):
        duration = None
    return title.strip(), duration


def _picked_file(workdir: Path, exts: tuple):
    files = [p for p in workdir.iterdir()
             if p.is_file() and p.suffix.lower() in exts
             and not p.name.endswith(".part")]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_size)


def _safe_title_name(title: str, ext: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|\n\r\t]", " ", title or "").strip()
    name = re.sub(r"\s+", " ", name)[:60].strip() or "media"
    return f"{name}.{ext}"


async def _download_media(url: str, want_audio: bool, workdir: Path):
    """Download into workdir, staying under the Discord upload cap.
    Returns (path, title). Raises MediaError with a user-facing reason."""
    title, duration = await _probe_media(url)
    if duration and duration > MEDIA_MAX_DURATION_SEC:
        raise MediaError("長すぎるわ、それは無理（1時間超え）")
    base = [_ytdlp_bin(), "--no-playlist", "--no-warnings", "--no-progress"]
    if want_audio:
        out_tmpl = str(workdir / "media.%(ext)s")
        rc, _, err = await _run_proc(
            base + ["-f", "ba/b", "--extract-audio", "--audio-format", "mp3",
                    "--audio-quality", "5", "-o", out_tmpl, url], 300)
        if rc != 0:
            raise _media_error_from(err)
        f = _picked_file(workdir, (".mp3", ".m4a", ".opus", ".ogg"))
        if not f:
            raise MediaError("無理だったわ、それ取れねえ")
        if f.stat().st_size > DISCORD_FILE_LIMIT:
            small = workdir / "media_small.mp3"
            rc2, _, _ = await _run_proc(
                ["ffmpeg", "-y", "-i", str(f), "-vn", "-b:a", "64k",
                 "-ac", "1", str(small)], 300)
            if rc2 == 0 and small.exists():
                f = small
        if f.stat().st_size > DISCORD_FILE_LIMIT:
            raise MediaError("でかすぎてDiscordに貼れねえわ（25MB超え）")
        dest = workdir / _safe_title_name(title, f.suffix.lstrip("."))
        if dest != f:
            f.rename(dest)
            f = dest
        return f, title
    last_err = ""
    for height in (720, 480, 360):
        for p in list(workdir.iterdir()):
            if p.is_file():
                p.unlink()
        out_tmpl = str(workdir / "media.%(ext)s")
        rc, _, err = await _run_proc(
            base + ["-f", f"bv*[height<={height}]+ba/b[height<={height}]/b",
                    "--merge-output-format", "mp4",
                    "-o", out_tmpl, url], 300)
        if rc != 0:
            last_err = err
            continue
        f = _picked_file(workdir, (".mp4", ".webm", ".mkv", ".mov"))
        if f and f.stat().st_size <= DISCORD_FILE_LIMIT:
            dest = workdir / _safe_title_name(title, f.suffix.lstrip("."))
            if dest != f:
                f.rename(dest)
                f = dest
            return f, title
    if last_err and "format" not in last_err.lower():
        raise _media_error_from(last_err)
    raise MediaError("でかすぎてDiscordに貼れねえわ（25MB超え）")


async def try_fetch_media(channel_id: int, content: str, d: dict,
                          store: GuildStore) -> bool:
    """Mechanical media-fetch path. Returns True when it handled the
    message (success, failure notice, or cooldown notice)."""
    req = media_request(content, d, store, channel_id)
    if not req:
        return False
    url, want_audio = req
    st = channel_state[channel_id]
    if time.time() - st.get("last_media", 0) < MEDIA_COOLDOWN_SEC:
        await rest.send_message(channel_id, "ちょっと待て、今取ってる最中だわ")
        return True
    st["last_media"] = time.time()
    st["last_ambient"] = time.time()
    await rest.send_message(
        channel_id, "おk、音声だけ取ってくるわ" if want_audio else "おk、持ってくるわ")
    workdir = Path(tempfile.mkdtemp(dir=MEDIA_TMP))
    try:
        async with TypingSession(channel_id):
            path, title = await _download_media(url, want_audio, workdir)
        caption = (title or "")[:100] or None
        await rest.send_file(channel_id, path, content=caption,
                             reply_to=int(d["id"]) if d.get("id") else None)
        st["last_bot_msg"] = time.time()
        gid = channel_guild.get(channel_id)
        if gid:
            store.add({"id": 0, "author": rest.bot_name or "Bot",
                       "author_id": rest.bot_id or 0,
                       "content": f"（動画/音声を貼った：{title or url}）",
                       "channel_id": channel_id, "ts": time.time()})
    except MediaError as e:
        await rest.send_message(channel_id, str(e))
    except Exception as e:
        print(f"[media] fetch failed for {url}: {e}", flush=True)
        await rest.send_message(channel_id, "無理だったわ、それ取れねえ")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return True


# ---------------------------------------------------------------- casino
# Virtual-coin casino with a dedicated button panel. Coins are free and
# exist only inside the bot: they cannot be bought, cashed out,
# transferred, or exchanged for anything with real-world value. Points
# are earned by claiming the daily bonus and by chatting; the shop trades
# them for in-bot collectible badges / titles shown on the casino
# profile. Like the media fetch, this is a mechanical path checked once
# at the top of both entry points (handle_message / handle_dm), and the
# panel itself is driven by component interactions (type 3), which are
# stateless except for blackjack: the owner id and the current bet are
# encoded in every button's custom_id, so panels survive restarts.

CASINO_START_BALANCE = 1000
CASINO_DAILY_AMOUNT = 500
CASINO_CHAT_REWARD = 2
CASINO_CHAT_REWARD_COOLDOWN = 60
CASINO_BETS = [10, 100, 500, 1000]
CASINO_TRIGGER_RE = re.compile(
    r"(カジノ|かじの|casino).{0,6}"
    r"(したい|やりたい|やろう|やる|行きたい|行く|遊びたい|遊ぶ|開いて|"
    r"ひらいて|オープン|open|始めたい|始める|入りたい)",
    re.IGNORECASE)
CASINO_EXACT_RE = re.compile(r"^(カジノ|かじの|casino)[！!。．\s]*$",
                             re.IGNORECASE)

CASINO_SHOP = [
    {"id": "debut", "name": "バッジ「カジノデビュー」", "price": 100,
     "desc": "最初の一歩。プロフィールに残る記念バッジ"},
    {"id": "slot_mania", "name": "バッジ「スロットマニア」", "price": 800,
     "desc": "スロット好きの証"},
    {"id": "title_broke", "name": "称号「破産王」", "price": 2000,
     "desc": "負けっぷりを誇る称号"},
    {"id": "high_roller", "name": "バッジ「ハイローラー」", "price": 5000,
     "desc": "大きく賭ける者のバッジ"},
    {"id": "title_king", "name": "称号「カジノ王」", "price": 10000,
     "desc": "このサーバーのカジノを制した者の称号"},
    {"id": "title_rich", "name": "称号「ぬめ億万長者」", "price": 50000,
     "desc": "最終目標。持ってたら相当すごい"},
]
CASINO_SHOP_BY_ID = {it["id"]: it for it in CASINO_SHOP}


class CasinoError(Exception):
    """A casino action failure whose message can be shown as-is."""


def casino_wants(content: str) -> bool:
    c = (content or "").strip()
    if not c:
        return False
    if CASINO_EXACT_RE.match(c):
        return True
    return bool(CASINO_TRIGGER_RE.search(c))


_casino_cache = {}


def _casino_data(guild_id: int) -> dict:
    if guild_id in _casino_cache:
        return _casino_cache[guild_id]
    data = {"users": {}}
    try:
        d = json.loads((guild_dir(guild_id) / "casino.json").read_text("utf-8"))
        if isinstance(d.get("users"), dict):
            data = d
    except Exception:
        pass
    _casino_cache[guild_id] = data
    return data


def _casino_save(guild_id: int):
    try:
        (guild_dir(guild_id) / "casino.json").write_text(
            json.dumps(_casino_data(guild_id), ensure_ascii=False), "utf-8")
    except Exception:
        pass


def casino_user(guild_id: int, user_id: int, display: str = "") -> dict:
    users = _casino_data(guild_id)["users"]
    rec = users.get(str(user_id))
    if rec is None:
        rec = {"balance": CASINO_START_BALANCE, "name": display or "?",
               "total_wagered": 0, "total_won": 0, "games": 0,
               "daily": "", "last_chat_reward": 0.0, "items": []}
        users[str(user_id)] = rec
        _casino_save(guild_id)
    elif display and rec.get("name") in ("", "?"):
        rec["name"] = display
    return rec


def casino_claim_daily(guild_id: int, user_id: int):
    rec = casino_user(guild_id, user_id)
    today = _jst_today()
    if rec.get("daily") == today:
        return False, "デイリーはもう受け取ったわ。明日また来い"
    rec["daily"] = today
    rec["balance"] += CASINO_DAILY_AMOUNT
    _casino_save(guild_id)
    return True, f"デイリー +{CASINO_DAILY_AMOUNT} 2009コイン。残高 {rec['balance']:,}"


def casino_chat_reward(guild_id: int, user_id: int) -> bool:
    """Chatting earns a little: +2 coins, at most once a minute per user."""
    rec = casino_user(guild_id, user_id)
    now = time.time()
    if now - float(rec.get("last_chat_reward") or 0) < CASINO_CHAT_REWARD_COOLDOWN:
        return False
    rec["last_chat_reward"] = now
    rec["balance"] += CASINO_CHAT_REWARD
    _casino_save(guild_id)
    return True


def casino_settle(guild_id: int, user_id: int, bet: int, payout: int) -> dict:
    """Deduct bet and credit payout for one finished game."""
    rec = casino_user(guild_id, user_id)
    if bet <= 0:
        raise CasinoError("賭け金がおかしいわ")
    if rec["balance"] < bet:
        raise CasinoError(
            f"残高不足だわ（残高 {rec['balance']:,} / 賭け金 {bet:,}）。"
            "デイリー受け取るか発言して貯めろ")
    rec["balance"] += int(payout) - bet
    rec["total_wagered"] += bet
    rec["total_won"] += int(payout)
    rec["games"] += 1
    _casino_save(guild_id)
    return rec


CASINO_SESSION_TTL = 2 * 3600
CASINO_SETTLE_RE = re.compile(r"NCSETTLE:([A-Za-z0-9_\-]+)\.([0-9a-f]{16})")
CASINO_SETTLE_KEY = "nume-casino-settle-v1"


def _casino_sessions(guild_id: int) -> dict:
    return _casino_data(guild_id).setdefault("sessions", {})


def casino_sweep_sessions(guild_id: int, user_id: int) -> int:
    """Refund the locked coins of the user's expired web sessions."""
    now = time.time()
    refunded = 0
    for sess in _casino_sessions(guild_id).values():
        if (sess.get("user_id") == user_id and sess.get("state") == "active"
                and now > float(sess.get("expires") or 0)):
            sess["state"] = "expired"
            rec = casino_user(guild_id, user_id)
            rec["balance"] += int(sess.get("locked") or 0)
            refunded += int(sess.get("locked") or 0)
    if refunded:
        _casino_save(guild_id)
    return refunded


def casino_issue_session(guild_id: int, user_id: int, display: str = ""):
    """Move the user's whole balance into a web-casino session and
    return (session, reused). The coins stay locked until the user
    settles with the code the web page issues, or the session expires
    (2h) and the sweep refunds them."""
    casino_sweep_sessions(guild_id, user_id)
    sessions = _casino_sessions(guild_id)
    now = time.time()
    for sess in sessions.values():
        if sess.get("user_id") == user_id and sess.get("state") == "active":
            return sess, True
    rec = casino_user(guild_id, user_id, display)
    locked = int(rec["balance"])
    rec["balance"] = 0
    sess = {"sid": secrets.token_urlsafe(18), "user_id": user_id,
            "locked": locked, "issued": now,
            "expires": now + CASINO_SESSION_TTL, "state": "active"}
    sessions[sess["sid"]] = sess
    _casino_save(guild_id)
    return sess, False


def _settle_sig(sid, final, wagered, won, games, items) -> str:
    msg = (f"{sid}:{final}:{wagered}:{won}:{games}:"
           f"{','.join(items)}:{CASINO_SETTLE_KEY}")
    return hashlib.sha256(msg.encode("utf-8")).hexdigest()[:16]


def casino_settle_session(guild_id: int, user_id: int, b64: str, sig: str):
    """Verify and apply a settlement code from the web casino page.
    Returns (ok, message). The arithmetic is fully rechecked here:
    final must equal locked - wagered + won - shop cost."""
    try:
        raw = base64.urlsafe_b64decode(b64 + "=" * (-len(b64) % 4))
        payload = json.loads(raw.decode("utf-8"))
        sid = str(payload["sid"])
        final = int(payload["final"])
        wagered = int(payload["wagered"])
        won = int(payload["won"])
        games = int(payload["games"])
        entries = []
        canon = []
        for raw_item in payload.get("items") or []:
            if isinstance(raw_item, dict):
                iid = str(raw_item["id"])
                cost = int(raw_item["cost"])
                nm = str(raw_item.get("name") or "")[:60]
                entries.append({"id": iid, "cost": cost, "name": nm})
                canon.append(f"{iid}@{cost}")
            else:
                iid = str(raw_item)
                entries.append({"id": iid, "cost": None, "name": ""})
                canon.append(iid)
        items = [e["id"] for e in entries]
    except Exception:
        return False, "精算コードの形式がおかしいわ"
    if min(final, wagered, won, games) < 0 or len(set(items)) != len(items):
        return False, "精算コードの内容がおかしいわ"
    if not hmac.compare_digest(
            _settle_sig(sid, final, wagered, won, games, canon), sig):
        return False, "精算コードの署名が合わねえわ"
    casino_sweep_sessions(guild_id, user_id)
    sess = _casino_sessions(guild_id).get(sid)
    if sess is None or sess.get("user_id") != user_id:
        return False, "そのセッション見つからねえわ（自分のコードだけ使える）"
    if sess.get("state") == "settled":
        return False, "そのセッションはもう精算済みだわ"
    if sess.get("state") != "active":
        return False, "そのセッションは期限切れだわ。持ち込み分は残高に戻してある"
    rec = casino_user(guild_id, user_id)
    shop_cost = 0
    for e in entries:
        if e["id"] in rec.get("items", []):
            return False, "ショップの内容がおかしいわ"
        if e["cost"] is None:
            item = CASINO_SHOP_BY_ID.get(e["id"])
            if item is None:
                return False, "ショップの内容がおかしいわ"
            e["cost"] = item["price"]
            e["name"] = item["name"]
        elif e["cost"] <= 0:
            return False, "ショップの内容がおかしいわ"
        shop_cost += e["cost"]
    expected = int(sess["locked"]) - wagered + won - shop_cost
    if final != expected:
        return False, "精算コードの計算が合わねえわ"
    sess["state"] = "settled"
    rec["balance"] += final
    rec["total_wagered"] += wagered
    rec["total_won"] += won
    rec["games"] += games
    rec.setdefault("items", []).extend(items)
    names = rec.setdefault("item_names", {})
    for e in entries:
        if e["name"]:
            names[e["id"]] = e["name"]
    _casino_save(guild_id)
    gained = "".join(f"\n・{e['name'] or e['id']} を手に入れた！"
                     for e in entries)
    return True, (f"精算したわ。Webカジノ {games} ゲーム、"
                  f"{wagered:,} ベット → {final:,} 2009コイン持ち帰り。"
                  f"残高 {rec['balance']:,}{gained}")


async def try_casino_settle(channel_id: int, content: str, d: dict,
                            store: GuildStore) -> bool:
    """Mechanical path for web-casino settlement codes pasted in chat."""
    m = CASINO_SETTLE_RE.search(content or "")
    if not m:
        return False
    guild_id = channel_guild.get(channel_id) or MAIN_GUILD_ID
    author = d.get("author", {})
    ok, msg = casino_settle_session(guild_id, int(author["id"]),
                                    m.group(1), m.group(2))
    await rest.send_message(channel_id, msg,
                            reply_to=int(d["id"]) if d.get("id") else None)
    st = channel_state[channel_id]
    st["last_bot_msg"] = time.time()
    if store is not None:
        store.add({"id": 0, "author": rest.bot_name or "Bot",
                   "author_id": rest.bot_id or 0,
                   "content": "（Webカジノの精算をした）",
                   "channel_id": channel_id, "ts": time.time()})
    return True


def casino_buy(guild_id: int, user_id: int, item_id: str):
    item = CASINO_SHOP_BY_ID.get(item_id)
    if item is None:
        return False, "そんな商品ねえわ"
    rec = casino_user(guild_id, user_id)
    if item_id in rec.get("items", []):
        return False, f"{item['name']} はもう持ってるわ"
    if rec["balance"] < item["price"]:
        return False, (f"コイン足りねえ（残高 {rec['balance']:,} / "
                       f"価格 {item['price']:,}）")
    rec["balance"] -= item["price"]
    rec.setdefault("items", []).append(item_id)
    _casino_save(guild_id)
    return True, f"{item['name']} を手に入れた！ 残高 {rec['balance']:,}"


# ------------------------------------------------------------ game logic

SLOT_SYMBOLS = ["🍒", "🍋", "🔔", "⭐", "💎", "🎰"]
SLOT_WEIGHTS = [30, 26, 20, 12, 8, 4]
SLOT_TRIPLE = {"🎰": 20, "💎": 10, "⭐": 7, "🔔": 5, "🍋": 4, "🍒": 3}


def slot_spin(bet: int):
    reels = random.choices(SLOT_SYMBOLS, weights=SLOT_WEIGHTS, k=3)
    if reels[0] == reels[1] == reels[2]:
        payout = bet * SLOT_TRIPLE[reels[0]]
    elif reels[0] == reels[1] or reels[1] == reels[2] or reels[0] == reels[2]:
        payout = int(bet * 1.5)
    else:
        payout = 0
    return reels, payout


def coin_flip(bet: int, choice: str):
    flip = random.choice(["h", "t"])
    payout = int(bet * 1.9) if flip == choice else 0
    return flip, payout


def dice_duel(bet: int):
    me, bot = random.randint(1, 6), random.randint(1, 6)
    if me > bot:
        payout = int(bet * 1.9)
    elif me == bot:
        payout = bet  # tie: bet back
    else:
        payout = 0
    return me, bot, payout


BJ_RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
BJ_SUITS = ["♠", "♥", "♦", "♣"]


def bj_new_deck() -> list:
    deck = [f"{r}{s}" for s in BJ_SUITS for r in BJ_RANKS]
    random.shuffle(deck)
    return deck


def bj_hand_value(cards: list) -> int:
    total, aces = 0, 0
    for c in cards:
        r = c[:-1]
        if r == "A":
            total += 11
            aces += 1
        elif r in ("J", "Q", "K"):
            total += 10
        else:
            total += int(r)
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total


bj_sessions = {}  # panel message_id -> session dict (in-memory only)


def bj_resolve(session: dict):
    """Dealer plays out and the final payout is decided (bet already
    deducted at start). Returns (payout, result_text)."""
    bet = session["bet"]
    deck, dealer = session["deck"], session["dealer"]
    while bj_hand_value(dealer) < 17 and deck:
        dealer.append(deck.pop())
    pv, dv = bj_hand_value(session["player"]), bj_hand_value(dealer)
    if pv > 21:
        return 0, f"バースト… {pv} で負け。-{bet:,} 2009コイン"
    if dv > 21:
        return bet * 2, f"ディーラーがバースト（{dv}）！ 勝ち +{bet:,} 2009コイン"
    if pv > dv:
        return bet * 2, f"{pv} vs {dv} で勝ち！ +{bet:,} 2009コイン"
    if pv < dv:
        return 0, f"{pv} vs {dv} で負け。-{bet:,} 2009コイン"
    return bet, f"{pv} vs {dv} で引き分け。賭け金は戻したわ"


# ------------------------------------------------------------ panel UI

def _casino_cid(action: str, uid: int, *args) -> str:
    return ":".join(["casino", action, str(uid)] + [str(a) for a in args])


def _casino_btn(custom_id: str, label: str, style: int = 1, emoji: str = None,
                disabled: bool = False) -> dict:
    b = {"type": 2, "custom_id": custom_id, "label": label, "style": style,
         "disabled": disabled}
    if emoji:
        b["emoji"] = {"name": emoji}
    return b


def _casino_row(*buttons) -> dict:
    return {"type": 1, "components": list(buttons)}


def casino_components(view: str, uid: int, bet: int) -> list:
    if view == "coin":
        return [_casino_row(
            _casino_btn(_casino_cid("coin", uid, bet, "h"), "表に賭ける", 1, "🪙"),
            _casino_btn(_casino_cid("coin", uid, bet, "t"), "裏に賭ける", 1, "🪙"),
            _casino_btn(_casino_cid("main", uid, bet), "戻る", 2))]
    if view == "shop":
        rows, row = [], []
        for i, it in enumerate(CASINO_SHOP):
            row.append(_casino_btn(_casino_cid("buy", uid, bet, it["id"]),
                                   f"{i + 1}番を買う", 1, "🛒"))
            if len(row) == 5:
                rows.append(_casino_row(*row))
                row = []
        if row:
            rows.append(_casino_row(*row))
        rows.append(_casino_row(
            _casino_btn(_casino_cid("main", uid, bet), "戻る", 2)))
        return rows
    if view == "ranking":
        return [_casino_row(
            _casino_btn(_casino_cid("main", uid, bet), "戻る", 2))]
    if view == "bj":
        return [_casino_row(
            _casino_btn(_casino_cid("bj_hit", uid, bet), "ヒット", 1, "🃏"),
            _casino_btn(_casino_cid("bj_stand", uid, bet), "スタンド", 3, "✋"),
            _casino_btn(_casino_cid("close", uid, bet), "やめる（賭け金没収）", 4))]
    nxt = CASINO_BETS[(CASINO_BETS.index(bet) + 1) % len(CASINO_BETS)] \
        if bet in CASINO_BETS else CASINO_BETS[0]
    return [
        _casino_row(
            _casino_btn(_casino_cid("slots", uid, bet), f"スロット {bet}", 1, "🎰"),
            _casino_btn(_casino_cid("coin_view", uid, bet), f"コイントス {bet}", 1, "🪙"),
            _casino_btn(_casino_cid("dice", uid, bet), f"ダイス {bet}", 1, "🎲"),
            _casino_btn(_casino_cid("bj_start", uid, bet), f"ブラックジャック {bet}", 1, "🃏")),
        _casino_row(
            _casino_btn(_casino_cid("bet", uid, bet), f"賭け金切替 → {nxt}", 2, "💱"),
            _casino_btn(_casino_cid("daily", uid, bet), "デイリー受取", 2, "💰"),
            _casino_btn(_casino_cid("shop", uid, bet), "ショップ", 2, "🛒"),
            _casino_btn(_casino_cid("ranking", uid, bet), "ランキング", 2, "🏆")),
        _casino_row(
            _casino_btn(_casino_cid("close", uid, bet), "閉じる", 4)),
    ]


def _casino_item_name(rec: dict, iid: str) -> str:
    names = rec.get("item_names") or {}
    if iid in names:
        return str(names[iid])
    item = CASINO_SHOP_BY_ID.get(iid)
    return item["name"] if item else iid


def _casino_items_text(rec: dict) -> str:
    items = [_casino_item_name(rec, i) for i in rec.get("items", [])]
    return "、".join(items) if items else "なし"


def casino_embed(guild_id: int, uid: int, display: str, view: str = "main",
                 bet: int = 100, note: str = None) -> dict:
    rec = casino_user(guild_id, uid, display)
    if view == "shop":
        lines = ["コインと交換できるのはBot内のバッジ・称号だけ。"
                 "換金・譲渡・現実の商品との交換はできません。\n"]
        for i, it in enumerate(CASINO_SHOP):
            owned = "（所有済み）" if it["id"] in rec.get("items", []) else ""
            lines.append(f"**{i + 1}. {it['name']}** — {it['price']:,} 2009コイン{owned}\n"
                         f"　{it['desc']}")
        desc = "\n".join(lines)
        if note:
            desc = f"{note}\n\n" + desc
        return {"title": "🛒 カジノショップ",
                "description": desc[:4000],
                "footer": {"text": f"残高 {rec['balance']:,} 2009コイン"}}
    if view == "ranking":
        users = sorted(_casino_data(guild_id)["users"].items(),
                       key=lambda kv: kv[1].get("balance", 0), reverse=True)
        lines = [f"{i + 1}. {r.get('name', '?')} — {r.get('balance', 0):,} 2009コイン"
                 for i, (_, r) in enumerate(users[:10])]
        return {"title": "🏆 カジノランキング",
                "description": "\n".join(lines) or "まだ誰も遊んでないわ",
                "footer": {"text": "残高ランキング（上位10人）"}}
    if view == "coin":
        desc = f"表か裏か、当てれば賭け金の1.9倍。賭け金は {bet:,} 2009コイン。"
        if note:
            desc = f"{note}\n\n" + desc
        return {"title": "🪙 コイントス", "description": desc,
                "footer": {"text": f"残高 {rec['balance']:,} 2009コイン"}}
    lines = []
    if note:
        lines.append(note)
        lines.append("")
    lines += [
        f"**残高： {rec['balance']:,} 2009コイン**（賭け金 {bet:,}）",
        f"所持アイテム： {_casino_items_text(rec)}",
        f"戦績： {rec.get('games', 0)} ゲーム / 累計ベット {rec.get('total_wagered', 0):,} / "
        f"累計獲得 {rec.get('total_won', 0):,}",
        "",
        "遊び方：下のボタンでゲームを選ぶだけ。デイリー（毎日+500）と"
        "発言（1分ごと+2）でコインは貯まる。",
        "コインは無料で配るBot内専用ポイント。購入・換金・譲渡・"
        "現実の商品との交換はできません。",
    ]
    return {"title": "🎰 #2009 Casino", "description": "\n".join(lines)[:4000],
            "footer": {"text": "パネルは開いた本人のボタンだけ反応します"}}


def casino_bj_embed(session: dict, note: str = None) -> dict:
    pv = bj_hand_value(session["player"])
    dealer = session["dealer"]
    if session.get("finished"):
        dshow = " ".join(dealer) + f"（{bj_hand_value(dealer)}）"
    else:
        dshow = f"{dealer[0]} 🂠"
    lines = []
    if note:
        lines.append(note)
        lines.append("")
    lines += [f"あなたの手札： {' '.join(session['player'])}（{pv}）",
              f"ディーラー： {dshow}",
              f"賭け金： {session['bet']:,} 2009コイン"]
    if not session.get("finished"):
        lines.append("ヒットで1枚引く、スタンドで勝負。21を超えたらバーストで負け。")
    return {"title": "🃏 ブラックジャック", "description": "\n".join(lines)}


async def casino_open_panel(channel_id: int, guild_id: int, user_id: int,
                            display: str, store: GuildStore = None,
                            reply_to: int = None):
    casino_sweep_sessions(guild_id, user_id)
    casino_user(guild_id, user_id, display)
    msg = await rest.send_message(
        channel_id, None,
        embeds=[casino_embed(guild_id, user_id, display)],
        components=casino_components("main", user_id, CASINO_BETS[1]),
        reply_to=reply_to)
    st = channel_state[channel_id]
    st["last_bot_msg"] = time.time()
    st["last_ambient"] = time.time()
    if store is not None:
        store.add({"id": 0, "author": rest.bot_name or "Bot",
                   "author_id": rest.bot_id or 0,
                   "content": "（カジノのパネルを開いた）",
                   "channel_id": channel_id, "ts": time.time()})
    return msg


async def try_casino_trigger(channel_id: int, content: str, d: dict,
                             store: GuildStore) -> bool:
    """Mechanical casino path: 'カジノしたい' opens the button panel.
    Returns True when it handled the message."""
    if not casino_wants(content):
        return False
    guild_id = channel_guild.get(channel_id) or MAIN_GUILD_ID
    author = d.get("author", {})
    member = d.get("member") or {}
    display = (member.get("nick") or author.get("global_name")
               or author.get("username", "?"))
    web_url = (CFG.get("casino_web_url") or "").strip()
    if web_url:
        from urllib.parse import quote
        sess, reused = casino_issue_session(guild_id, int(author["id"]), display)
        link = (f"{web_url}#s={sess['sid']}&b={sess['locked']}"
                f"&u={quote(display)}")
        head = ("前のセッションまだ有効だから同じリンク使って" if reused
                else "Webカジノのセッション作ったわ")
        text = (f"🎰 {head}\n"
                f"残高 {sess['locked']:,} 2009コインを持ち込んで、2時間有効。\n"
                f"▶ {link}\n"
                "終わったらサイトの「精算コードを発行」を押して、"
                "出てきたコードをこのチャットに貼って。精算しないまま"
                "期限が切れたら持ち込み分は自動で残高に戻るわ")
        await rest.send_message(channel_id, text,
                                reply_to=int(d["id"]) if d.get("id") else None)
        st = channel_state[channel_id]
        st["last_bot_msg"] = time.time()
        st["last_ambient"] = time.time()
        if store is not None:
            store.add({"id": 0, "author": rest.bot_name or "Bot",
                       "author_id": rest.bot_id or 0,
                       "content": "（Webカジノのセッションリンクを送った）",
                       "channel_id": channel_id, "ts": time.time()})
        return True
    await casino_open_panel(channel_id, guild_id, int(author["id"]), display,
                            store=store,
                            reply_to=int(d["id"]) if d.get("id") else None)
    return True


async def handle_casino_component(d: dict):
    """Button presses on a casino panel (interaction type 3)."""
    data = d.get("data") or {}
    parts = (data.get("custom_id") or "").split(":")
    if len(parts) < 3 or parts[0] != "casino":
        return
    action = parts[1]
    try:
        owner_id = int(parts[2])
    except ValueError:
        return
    muser = ((d.get("member") or {}).get("user") or d.get("user") or {})
    presser = int(muser.get("id", 0) or 0)
    guild_id = int(d["guild_id"]) if d.get("guild_id") else MAIN_GUILD_ID
    message_id = int((d.get("message") or {}).get("id", 0) or 0)
    member = d.get("member") or {}
    display = (member.get("nick") or muser.get("global_name")
               or muser.get("username", "?"))
    if presser != owner_id:
        await interaction_callback(d, 4, {
            "content": "それは他人のパネルだわ。自分の分は『カジノしたい』って言って開け",
            "flags": 64})
        return
    uid = presser
    casino_user(guild_id, uid, display)
    bet = CASINO_BETS[1]
    if len(parts) > 3 and parts[3].isdigit():
        bet = int(parts[3])

    async def update(view, note=None, b=None):
        bb = bet if b is None else b
        await interaction_callback(d, 7, {
            "embeds": [casino_embed(guild_id, uid, display, view, bb, note)],
            "components": casino_components(view, uid, bb)})

    if action == "close":
        bj_sessions.pop(message_id, None)
        await interaction_callback(d, 7, {
            "content": "🎰 閉じたわ。また『カジノしたい』で開ける",
            "embeds": [], "components": []})
        return
    if action == "main":
        bj_sessions.pop(message_id, None)
        await update("main")
        return
    if action == "bet":
        nxt = CASINO_BETS[(CASINO_BETS.index(bet) + 1) % len(CASINO_BETS)] \
            if bet in CASINO_BETS else CASINO_BETS[0]
        await update("main", f"賭け金を {nxt:,} にしたわ", nxt)
        return
    if action == "daily":
        _, msg = casino_claim_daily(guild_id, uid)
        await update("main", msg)
        return
    if action == "shop":
        await update("shop")
        return
    if action == "buy":
        item_id = parts[4] if len(parts) > 4 else ""
        _, msg = casino_buy(guild_id, uid, item_id)
        await update("shop", msg)
        return
    if action == "ranking":
        await update("ranking")
        return
    if action == "coin_view":
        await update("coin")
        return
    if action == "slots":
        reels, payout = slot_spin(bet)
        try:
            rec = casino_settle(guild_id, uid, bet, payout)
        except CasinoError as e:
            await update("main", str(e))
            return
        shown = " ".join(reels)
        if payout > bet:
            note = f"{shown}　当たり！ +{payout - bet:,} 2009コイン（残高 {rec['balance']:,}）"
        elif payout == bet:
            note = f"{shown}　賭け金は戻ったわ（残高 {rec['balance']:,}）"
        elif payout > 0:
            note = f"{shown}　ペアで少し戻ったわ -{bet - payout:,}（残高 {rec['balance']:,}）"
        else:
            note = f"{shown}　ハズレ。-{bet:,} 2009コイン（残高 {rec['balance']:,}）"
        await update("main", note)
        return
    if action == "coin":
        choice = parts[4] if len(parts) > 4 else "h"
        flip, payout = coin_flip(bet, choice)
        try:
            rec = casino_settle(guild_id, uid, bet, payout)
        except CasinoError as e:
            await update("coin", str(e))
            return
        names = {"h": "表", "t": "裏"}
        if payout:
            note = (f"結果は {names[flip]}！ 当たり +{payout - bet:,} 2009コイン"
                    f"（残高 {rec['balance']:,}）")
        else:
            note = (f"結果は {names[flip]}… ハズレ。-{bet:,} 2009コイン"
                    f"（残高 {rec['balance']:,}）")
        await update("coin", note)
        return
    if action == "dice":
        me, bot, payout = dice_duel(bet)
        try:
            rec = casino_settle(guild_id, uid, bet, payout)
        except CasinoError as e:
            await update("main", str(e))
            return
        if me > bot:
            note = f"🎲 あなた {me} vs 僕 {bot}　勝ち！ +{payout - bet:,} 2009コイン（残高 {rec['balance']:,}）"
        elif me == bot:
            note = f"🎲 あなた {me} vs 僕 {bot}　引き分け、賭け金は戻したわ（残高 {rec['balance']:,}）"
        else:
            note = f"🎲 あなた {me} vs 僕 {bot}　負け。-{bet:,} 2009コイン（残高 {rec['balance']:,}）"
        await update("main", note)
        return
    if action == "bj_start":
        if message_id in bj_sessions:
            await update("main", "そのパネルではもうブラックジャック中だわ")
            return
        rec = casino_user(guild_id, uid, display)
        if rec["balance"] < bet:
            await update("main", f"残高不足だわ（残高 {rec['balance']:,} / 賭け金 {bet:,}）")
            return
        rec["balance"] -= bet
        rec["total_wagered"] += bet
        rec["games"] += 1
        deck = bj_new_deck()
        session = {"guild_id": guild_id, "user_id": uid, "bet": bet,
                   "deck": deck, "player": [deck.pop(), deck.pop()],
                   "dealer": [deck.pop(), deck.pop()], "finished": False}
        pv = bj_hand_value(session["player"])
        if pv == 21:
            if bj_hand_value(session["dealer"]) == 21:
                payout, note = bet, "両方ブラックジャック… 引き分け、賭け金は戻したわ"
            else:
                payout, note = int(bet * 2.5), f"ブラックジャック！ +{int(bet * 1.5):,} 2009コイン"
            rec["balance"] += payout
            rec["total_won"] += payout
            _casino_save(guild_id)
            session["finished"] = True
            await interaction_callback(d, 7, {
                "embeds": [casino_bj_embed(session, note)],
                "components": casino_components("main", uid, bet)})
            return
        _casino_save(guild_id)
        bj_sessions[message_id] = session
        await interaction_callback(d, 7, {
            "embeds": [casino_bj_embed(session)],
            "components": casino_components("bj", uid, bet)})
        return
    if action in ("bj_hit", "bj_stand"):
        session = bj_sessions.get(message_id)
        if session is None or session["user_id"] != uid:
            await interaction_callback(d, 4, {
                "content": "その勝負はもう終わってるか、再起動で消えたわ。パネル開き直して",
                "flags": 64})
            return
        if action == "bj_hit":
            session["player"].append(session["deck"].pop())
            if bj_hand_value(session["player"]) <= 21:
                await interaction_callback(d, 7, {
                    "embeds": [casino_bj_embed(session)],
                    "components": casino_components("bj", uid, bet)})
                return
        payout, note = bj_resolve(session)
        rec = casino_user(guild_id, uid, display)
        rec["balance"] += payout
        rec["total_won"] += payout
        _casino_save(guild_id)
        session["finished"] = True
        bj_sessions.pop(message_id, None)
        note = f"{note}（残高 {rec['balance']:,}）"
        await interaction_callback(d, 7, {
            "embeds": [casino_bj_embed(session, note)],
            "components": casino_components("main", uid, bet)})
        return


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
        started = match_started_template(store, text)
        if (started and "○○" not in started["text"]
                and "〇〇" not in started["text"]):
            st["last_ambient"] = time.time()
            await send_template_paragraphs(channel_id, started["text"], store,
                                           resume=not FRESH_START_RE.search(target_content))
            record_trigger(st, author_id, target_content)
            if st["pending_task"] and not st["pending_task"].done():
                st["pending_task"].cancel()
            await _ts.__aexit__()
            return
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
    dm_channels.add(channel_id)
    content = (d.get("content") or "").strip()
    image_note, has_image = await vision_note_for_message(d, channel_id, force=True)
    if not content and not has_image:
        return
    display = author.get("global_name") or author.get("username", "?")
    store = store_for(MAIN_GUILD_ID)
    stored = content
    for m in d.get("mentions", []):
        stored = stored.replace(f"<@{m['id']}>", f"@{m.get('username', 'user')}")
        stored = stored.replace(f"<@!{m['id']}>", f"@{m.get('username', 'user')}")
    stored = _with_image_note(stored, image_note, has_image)
    store.add({"id": int(d["id"]), "author": display, "author_id": int(author.get("id", 0)), "content": stored,
               "channel_id": channel_id, "ts": time.time()})
    st = channel_state[channel_id]
    if st.get("para_active") and STOP_RE.search(content):
        st["para_stop"] = True
        return
    if await try_casino_settle(channel_id, content, d, store):
        record_trigger(st, int(author.get("id", 0)), content)
        return
    if await try_casino_trigger(channel_id, content, d, store):
        record_trigger(st, int(author.get("id", 0)), content)
        return
    if await try_fetch_media(channel_id, content, d, store):
        record_trigger(st, int(author.get("id", 0)), content)
        return
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
            started = match_started_template(store, text)
            if (started and "○○" not in started["text"]
                    and "〇〇" not in started["text"]):
                await send_template_paragraphs(channel_id, started["text"], store,
                                               resume=not FRESH_START_RE.search(content))
                sent = True
                record_trigger(st, int(author.get("id", 0)), target_content)
            else:
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

    _mentioned_early = any(int(m["id"]) == rest.bot_id for m in d.get("mentions", []))
    image_note, has_image = await vision_note_for_message(
        d, channel_id, force=_mentioned_early)
    if content or has_image:
        # store a readable version: raw <@123...> mention markup teaches the
        # model to emit ID digits, so swap in the mentioned user's name
        name_map = {}
        for u in d.get("mentions") or []:
            name_map[str(u["id"])] = u.get("global_name") or u.get("username") or "メンション"
        stored_content = re.sub(
            r"<@[!&]?(\d+)>",
            lambda m: "@" + name_map.get(m.group(1), "メンション"), content)
        stored_content = _with_image_note(stored_content, image_note, has_image)
        store.add({"id": int(d["id"]), "author": display, "author_id": int(author["id"]), "content": stored_content,
                   "channel_id": channel_id, "ts": time.time()})
        if st.get("para_active") and STOP_RE.search(content):
            st["para_stop"] = True
        casino_chat_reward(guild_id, int(author["id"]))
        update_heat(st, int(author["id"]), content)
        if store.profile_due():
            asyncio.create_task(refresh_profile(store))
        if store.knowledge_due():
            asyncio.create_task(refresh_knowledge(store))

    st = channel_state[channel_id]
    if (content or has_image) and not channel_speaking(guild_id, channel_id):
        return
    if content and not st["paused"]:
        if await try_casino_settle(channel_id, content, d, store):
            record_trigger(st, int(author["id"]), content)
            return
        if await try_casino_trigger(channel_id, content, d, store):
            record_trigger(st, int(author["id"]), content)
            return
        if await try_fetch_media(channel_id, content, d, store):
            record_trigger(st, int(author["id"]), content)
            return
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
            if content or has_image:
                web = await web_lookup(content) if content and needs_web_lookup(content) else None
                text = await compose_reply(store, channel_id, display, web=web)
                if text:
                    started = match_started_template(store, text)
                    if (started and "○○" not in started["text"]
                            and "〇〇" not in started["text"]):
                        st["last_ambient"] = time.time()
                        await send_template_paragraphs(channel_id, started["text"], store,
                                                       resume=not FRESH_START_RE.search(content))
                        sent = True
                        record_trigger(st, int(author["id"]), content)
                    else:
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
    addr_content = content or _with_image_note("", image_note, has_image)
    if CFG.get("addressed_detection") and addr_content and not st["paused"] and not taunt_muted and not (
            laugh_only and random.random() < 0.8):
        if trigger_is_duplicate(st, int(author["id"]), addr_content):
            if random.random() < 0.3:
                await rest.add_reaction(channel_id, int(d["id"]),
                                        random.choice(["👀", "👍", "🫡"]))
        elif addressed_candidate(channel_id, addr_content, reply_to_other, store, display):
            await run_addressed(channel_id, guild_id, d, display, addr_content)
            return

    if not CFG["ambient_enabled"] or st["paused"]:
        return
    now = time.time()
    st["burst"] = [b for b in st["burst"] if now - b["ts"] < 120]
    if has_image:
        # an image post is a conversation beat on its own: count it twice
        # so a lone image can trigger an ambient reaction (still gated by
        # maybe_ambient's cooldown and probability)
        st["burst"].append({"user": int(author["id"]), "ts": now})
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
    {"name": "カジノ", "type": 1, "description": "仮想コインのカジノパネルを開く（スロット等・換金不可のBot内ポイント）"},
    {"name": "発言切替", "type": 1, "description": "Botの発言オン/オフを切り替える（管理者のみ）",
     "default_member_permissions": "8",
     "options": [{"name": "状態", "type": 3, "required": False,
                  "description": "オン=発言する / オフ=発言しない / 切替=今と逆 / 既定=チャンネル個別設定を解除",
                  "choices": [{"name": "オン（発言する）", "value": "on"},
                              {"name": "オフ（発言しない）", "value": "off"},
                              {"name": "切替", "value": "toggle"},
                              {"name": "既定に戻す", "value": "default"}]},
                 {"name": "チャンネル", "type": 7, "required": False,
                  "description": "指定すると、そのチャンネルだけの設定になる（未指定=サーバー全体）"}]},
]


def _canon_cmd(c: dict) -> str:
    """Canonical form of a slash-command definition, keeping only the
    fields we set, so a definition change (not just a new command)
    triggers re-registration."""
    opts = []
    for o in c.get("options") or []:
        oo = {k: o.get(k) for k in ("name", "description", "type", "required")}
        if o.get("choices"):
            oo["choices"] = [{k: ch.get(k) for k in ("name", "value")}
                             for ch in o["choices"]]
        opts.append(oo)
    base = {"name": c.get("name"), "type": c.get("type", 1),
            "description": c.get("description"),
            "default_member_permissions": c.get("default_member_permissions"),
            "options": opts}
    return json.dumps(base, ensure_ascii=False, sort_keys=True)


async def ensure_guild_commands(guild_id: int):
    """Slash commands are wiped when the bot is kicked from a guild, so
    re-register them automatically whenever the bot (re)joins one. Checks
    first and only writes when the registered set differs."""
    try:
        current = await rest.request(
            "GET", f"/applications/{APP_ID}/guilds/{guild_id}/commands")
        names = {c.get("name") for c in current} if isinstance(current, list) else set()
        have = {_canon_cmd(c) for c in current} if isinstance(current, list) else set()
        want = {_canon_cmd(c) for c in COMMAND_DEFS}
        if have == want:
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

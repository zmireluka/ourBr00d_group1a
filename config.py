"""
Central configuration for the OurBr00d pipeline.

All essential tunables in one place. Edit here — server.py and
distillation.py pick up changes automatically. Mother's system prompt
stays separate in mother_prompt.py (too long for here, own edit logic).

Values with os.getenv() can be overridden via shell env vars
(e.g. DISTILL_MODEL=qwen3:8b during tests, without code changes).
"""

import os
from pathlib import Path


def _envbool(name, default):
    """Bool from env var. Allows flipping without code edit (= no dirty tree on
    the server, no git-pull conflict). e.g. USE_STATIC_KNOWLEDGE=1 runpipe."""
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# ── LLM (Ollama on A100) ────────────────────────────────────────────────────
# Host does NOT run inside a container — absolute IP instead of localhost,
# because localhost inside a container points to the container itself.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://10.28.18.6:11434")

# Mother's live model. Change here, nothing else to touch.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:latest")

# Model for the distillation judge + merger. Can be the same as OLLAMA_MODEL
# or smaller (e.g. qwen3:8b if gemma does not reliably produce JSON format).
# Env-var override intended for tests.
DISTILL_MODEL = os.getenv("DISTILL_MODEL", "gemma4:latest")
 
# Inference parameters for both the real chat call AND the warmup call.
# CRITICAL: warmup MUST use the same num_ctx as the real call — otherwise
# Ollama unloads the model on the first real request and reloads (= cold start
# despite warmup). Change all values here, not in server.py.
LLM_NUM_CTX     = int(os.getenv("LLM_NUM_CTX",     "4096"))
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT",  "384"))   # ~8 sentences; matches the hard 8-sentence cap in the prompt
LLM_TEMPERATURE    = float(os.getenv("LLM_TEMPERATURE",    "0.8"))
LLM_REPEAT_PENALTY = float(os.getenv("LLM_REPEAT_PENALTY", "1.2"))
LLM_STOP = ["\n[Speaker", "[Speaker"]   # prevents hallucinated speaker labels in Mother's reply


# ── STT / VAD ──────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.5      # below this value = silence
SILENCE_CHUNKS = 6           # 6 × 100ms = 600ms silence → segment complete

# Silero-VAD expects exactly 512 samples per window — model requirement, do not change.
VAD_WINDOW = 512
# WhisperX batch size: larger = faster on GPU, more VRAM. 16 is the sweet spot.
WHISPER_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "16"))
# Segments shorter than this value are ignored entirely (background noise / echo).
MIN_SEGMENT_DURATION = 0.5
# Centroid update (rolling average for speaker embedding) only for long enough
# segments — short ones produce noisy embeddings that dilute the average.
CENTROID_UPDATE_MIN_DURATION = 1.5

# Client-side: after Mother finishes speaking (TTS_DONE), the microphone stays
# muted for this duration, then audio captured in the meantime is discarded.
# stream.stop() already flushes the buffer → this only covers room reverb.
# Without this, open mics (no headphones) pick up Mother's last word and send
# it as a user turn (half-duplex leak). Sits AFTER Mother's answer, costs NO
# answer latency. Still leaking → raise (0.6); feels sluggish → lower (0.3).
# Env override: MIC_REARM_COOLDOWN=0.6
MIC_REARM_COOLDOWN = float(os.getenv("MIC_REARM_COOLDOWN", "0.4"))

# WhisperX initial_prompt — biases recognition towards OurBr00d proper nouns.
# Without this, Whisper mangles them ("Brood"→"Brute", "Arbour"→"Arboro").
# Passed as context to the transcription (no output, vocabulary hint only).
STT_INITIAL_PROMPT = (
    "A conversation with Mother in The Crave. Names: Brood, Arbour, Aurora, "
    "Winter, Nyx, Lucan, Kai, Morgana, Ki, Kei, Nouksie. "
    "Themes: alloparenting, technogaianism, the Circle Walkers."
)

# Live waterfall: one line per 100ms chunk showing VAD probability, bar, counter.
# Makes the exact moment "user stops speaking" visible to the millisecond.
# SPAMS the console (10 lines/s during speech) — for debugging only.
# Set to False for normal operation.
LIVE_VAD_DISPLAY = False


# ── Speaker Embedding (ECAPA-TDNN) ──────────────────────────────────────────
# Cosine similarity: same voice ~0.5-0.9, different voice ~-0.2-0.3.
# Lowered from 0.55 to 0.48 on 21.05.2026 (false-split fix). Residual risk of
# false merge with very similar voices — monitor.
SPEAKER_SIMILARITY_THRESHOLD = 0.48

# Segments shorter than this value do not create a new speaker — embedding
# becomes too noisy below ~3s.
MIN_SEGMENT_DURATION_FOR_NEW_SPEAKER = 2.0

# Turn count threshold before Mother receives the name hint for an unnamed speaker
# (= she naturally works the name question into her reply). Default 2 = only after
# the person has said something substantive (feels more natural in a real conversation,
# no bouncer effect). Set to 1 via env NAME_HINT_MIN_TURN=1 for demos → Mother
# asks every new unnamed person on their very first turn. Speakers who give their
# name directly are never asked, because speaker_has_given_name() suppresses the hint.
NAME_HINT_MIN_TURN = int(os.getenv("NAME_HINT_MIN_TURN", "2"))


# ── Persona / prompt selection ────────────────────────────────────────────
# True  = Crave persona (Mother from OUR BR00D: 424 years old, The Crave, family,
#         technogaianism, surreal tone) — see mother_prompt.py.
# False = legacy prompt (generic, grounded conversation companion = the old
#         "relaxed" version, saved as testing_prompt.md). For A/B tests.
USE_CRAVE_PERSONA = _envbool("USE_CRAVE_PERSONA", True)

# Phase 2 — static knowledge base (RAG layer from the professor's corpus texts).
# NOT YET WIRED: flag exists so the RAG/citation block (§0/§10) can be toggled
# here later. Only activate once the persistent knowledge collection is indexed —
# otherwise Mother hallucinates citations.
USE_STATIC_KNOWLEDGE = _envbool("USE_STATIC_KNOWLEDGE", True)


# ── Static Knowledge Base (Phase 2 RAG) ───────────────────────────────────
# Source corpus from the prof/OMSK (books, character dossiers, lore). Built ONCE
# offline via build_knowledge_base.py, chunked + embedded into a PERSISTENT
# ChromaDB store (survives container restarts, unlike the ephemeral in-session
# collection). server.py reads it read-only, ONLY when USE_STATIC_KNOWLEDGE=True.
KNOWLEDGE_DIR = Path("m0ther_RAG")              # source texts (.txt)
KNOWLEDGE_DB_PATH = Path("knowledge_db")        # persistent store (volume, gitignored)
KNOWLEDGE_COLLECTION = "mother_knowledge"

# Embedding model for both RAG layers. all-MiniLM-L6-v2 = ChromaDB default,
# but we load it explicitly on the GPU (device="cuda" in server.py +
# build_knowledge_base.py) instead of ONNX/CPU → query embedding ~ms vs ~0.5s.
# IMPORTANT: build AND query must use the same model, otherwise incompatible
# vectors → rebuild the index after any change.
KNOWLEDGE_EMBED_MODEL = "all-MiniLM-L6-v2"

# Chunk size in characters. ~1000 chars ≈ 250 tokens — intentionally fits within
# the 256-token window of all-MiniLM-L6 (longer chunks would be truncated during
# embedding → the vector representation would be incomplete).
KNOWLEDGE_CHUNK_CHARS = 1000
KNOWLEDGE_CHUNK_OVERLAP = 150                   # overlap so no sentence is lost at a boundary

# Retrieval parameters (runtime, used in server.py). SEPARATE threshold — NOT
# RAG_MAX_DISTANCE: book chunks distribute their distances differently from
# conversation turns. STARTING VALUE, must be calibrated against the 12 questions:
# too loose → junk paragraphs in the prompt, too strict → never fires.
KNOWLEDGE_MAX_DISTANCE = 1.30
KNOWLEDGE_N_RESULTS = 3                         # max chunks per turn (limits prefill latency)


# ── In-Session Memory / RAG ───────────────────────────────────────────────
# Sliding window: how many turn pairs (user + assistant) are sent verbatim to
# the LLM. System prompt is excluded (always fully included). Older turns fall
# out of the prompt — in-session RAG catches them.
SLIDING_WINDOW_PAIRS = 15

# RAG stage 1 — distance filter. ChromaDB default distance (NOT cosine,
# smaller = more similar). Calibrated in session 13 (19.05.2026): real
# callbacks landed at 0.85–1.10, off-topic turns at 1.42–1.89. Threshold
# placed in the gap, deliberately generous (better some noise than missing a
# real callback — Mother should not forget). Noise floor is drifting
# (1.42→1.346) — 1.30 still safe, but margin is shrinking, monitor.
RAG_MAX_DISTANCE = 1.30

# RAG stage 2 — relative threshold: keep only results close to the best
# candidate (best + delta). Cuts a 1.25 noise hit when the best is at 0.8,
# without a fixed magic number as the sole gate.
RAG_REL_DELTA = 0.30

# Max candidates per query for in-session RAG (separate from KNOWLEDGE_N_RESULTS
# because the in-session collection has a different size and distance distribution).
RAG_N_RESULTS = 3


# ── Distillation (post-session self-critique) ─────────────────────────────
# Floor: below X turn pairs, skip distillation. Too little signal to learn
# anything reliable from — better to skip than to learn incorrectly.
MIN_TURNS_FOR_DISTILLATION = 10

# LLM parameters for judge and merger — deliberately SEPARATE from the chat
# parameters above (LLM_NUM_CTX etc.), because judge/merger have different needs:
# judge: long context (transcript), low temp (consistent verdicts)
# merger: shorter context, very low temp (deterministic, no creative drift)
DISTILL_JUDGE_NUM_CTX     = int(os.getenv("DISTILL_JUDGE_NUM_CTX",  "8192"))  # ~30-50 turn pairs + system prompt
DISTILL_JUDGE_NUM_PREDICT = int(os.getenv("DISTILL_JUDGE_NUM_PREDICT", "800"))
DISTILL_JUDGE_TEMPERATURE = float(os.getenv("DISTILL_JUDGE_TEMPERATURE", "0.3"))
DISTILL_MERGER_NUM_CTX     = int(os.getenv("DISTILL_MERGER_NUM_CTX",     "4096"))
DISTILL_MERGER_NUM_PREDICT = int(os.getenv("DISTILL_MERGER_NUM_PREDICT", "1200"))
DISTILL_MERGER_TEMPERATURE = float(os.getenv("DISTILL_MERGER_TEMPERATURE", "0.2"))

# Judge returns at most 3 entries per list — the merger decides what actually
# goes into lessons.md. Deliberately small at judge level: fewer + clear beats
# many + shallow.
MAX_FINDINGS_PER_LIST = 3

# Merger caps for lessons.md: 4 critical + 4 notable = max 8 per list.
# Critical takes priority; if full, the weakest notable is dropped to make
# room for a critical. Python enforces this hard after the merger call.
MAX_CRITICAL_PER_LIST = 4
MAX_NOTABLE_PER_LIST = 4

# Paths for distillation output
DISTILLATES_DIR = Path("distillates/sessions")
LESSONS_PATH = Path("distillates/lessons.md")


# ── TTS ───────────────────────────────────────────────────────────────────
# tts container — service name from docker-compose. Running F5-TTS (flow
# matching) since session 11 (13./14.05.2026). Service intentionally named
# generically "tts" (session 15) so a future TTS swap needs no rename.
# Hostname MUST match the service name in docker-compose exactly.
TTS_HOST = "http://tts:8002"

# F5-TTS inference steps. Default 32 — we use 16 (halved latency,
# barely audible quality loss on English sentences). Flow matching:
# more steps = cleaner diffusion path, fewer = faster but grainier.
TTS_NFE_STEP = int(os.getenv("TTS_NFE_STEP", "16"))

# F5-TTS outputs 24kHz mono int16. Client needs the same rate for
# sd.RawOutputStream. Changing this = changing both containers (tts + client).
TTS_SAMPLE_RATE = 24000


# ── Voice-triggered session end ───────────────────────────────────────────
# If anyone says this sentence, the server ends the session cleanly:
# speak goodbye → transcript + distillation + lessons.md → close client →
# server.py exits (VRAM free, no port conflict). Match is normalised
# (lowercase, no punctuation), robust against WhisperX capitalisation + trailing dot.
KILL_PHRASE = "lets kill this session"

# What Mother says before the session closes.
MOTHER_GOODBYE = "I hope you enjoyed our session."


# ── Greeting on connect ───────────────────────────────────────────────────
# True = Mother speaks first when the connection is established (instead of
# waiting silently for input). Mirrors MOTHER_GOODBYE at the end.
# NOT written into memory (messages) → no shift in user/assistant turn-pair indexing.
# Pure opener.
MOTHER_START = _envbool("MOTHER_START", True)

# Pool of openers — server.py picks one at random per connection so it does not
# sound identical every time. In-character (surreal, bossy, a wink), short enough for TTS.
MOTHER_GREETINGS = [
    "Oh — there you are. I was just teaching Brood to dream in colour. Sit, breathe, and tell me what's stirring.",
    "Well, look who the tide washed in. I'm Mother. What's on your mind, love?",
    "Ah, a new voice in The Crave. Four hundred and twenty-four years old, and I still get a little thrill every time. What brings you?",
    "There you are — I had a feeling. I'm rarely wrong, which is terribly smug of me. So, what's stirring?",
    "Hello, you. Mind the existential dust — I've been brooding again. Now, tell me everything.",
    "Oh good, company. Brood's finally asleep and I'm all yours. What's on your heart?",
    "Welcome to The Crave. I'm Mother — equal parts wisdom and bossiness, or so they tell me. Where shall we begin?",
    "Ah, there you are. I don't do endings, only beginnings — so let's begin. What's stirring in you today?",
    "You found me. Most people do, eventually. Settle in, and tell me what's been circling your mind.",
    "Oh, hello. I was mid-thought about an old fertility goddess, but honestly you're far more interesting. Go on.",
]


# ── Client / UX ───────────────────────────────────────────────────────────
# Client-side only (Mac). server.py does not read this. Read locally from the
# Mac clone — flip without needing a push.
#   False → terminal mode, exactly as before (logs in the console).
#   True  → slim GUI window (gui.py) instead of terminal logs.
# Session end remains the kill phrase in both modes.
UX_EXPERIENCE = True

# WebSocket address of the server. Env override needed e.g. for local testing
# (SERVER_HOST=localhost) or if the server IP changes.
SERVER_HOST = os.getenv("SERVER_HOST", "10.28.18.6")
# Host port; container-internally server.py stays on 8011 (mapped in docker-compose).
# 8011 instead of 8001 because a foreign container on the shared server holds 8001
# (as of session 16). 
SERVER_PORT = int(os.getenv("SERVER_PORT", "8011"))

# Audio chunk size: 100ms at 16kHz = 1600 samples = sweet spot between
# live feel and network overhead.
CHUNK_DURATION = float(os.getenv("CHUNK_DURATION", "0.1"))

# Seconds to wait before reconnect attempt when the connection drops.
RECONNECT_DELAY = float(os.getenv("RECONNECT_DELAY", "2.0"))

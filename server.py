import asyncio
import os
import random
import signal
import re
import datetime
import websockets
import numpy as np
import torch
import whisperx
import time
import ollama
import httpx
import chromadb
from dotenv import load_dotenv
from speechbrain.inference.speaker import SpeakerRecognition

from mother_prompt import MOTHER_SYSTEM_PROMPT
from distillation import (
    parse_lessons_md,
    render_lessons_for_prompt,
    run_distillation,
)
from config import (
    CENTROID_UPDATE_MIN_DURATION,
    DISTILLATES_DIR,
    KILL_PHRASE,
    KNOWLEDGE_COLLECTION,
    KNOWLEDGE_DB_PATH,
    KNOWLEDGE_EMBED_MODEL,
    KNOWLEDGE_MAX_DISTANCE,
    KNOWLEDGE_N_RESULTS,
    LESSONS_PATH,
    LIVE_VAD_DISPLAY,
    LLM_NUM_CTX,
    LLM_NUM_PREDICT,
    LLM_REPEAT_PENALTY,
    LLM_STOP,
    LLM_TEMPERATURE,
    MIN_SEGMENT_DURATION,
    MIN_SEGMENT_DURATION_FOR_NEW_SPEAKER,
    MOTHER_GOODBYE,
    MOTHER_GREETINGS,
    MOTHER_START,
    NAME_HINT_MIN_TURN,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    RAG_MAX_DISTANCE,
    RAG_N_RESULTS,
    RAG_REL_DELTA,
    SAMPLE_RATE,
    SILENCE_CHUNKS,
    SILENCE_THRESHOLD,
    SLIDING_WINDOW_PAIRS,
    SPEAKER_SIMILARITY_THRESHOLD,
    STT_INITIAL_PROMPT,
    TTS_HOST,
    USE_STATIC_KNOWLEDGE,
    VAD_WINDOW,
    WHISPER_BATCH_SIZE,
)

# Load .env — provides optional env vars (e.g. STT_VRAM_GB)
load_dotenv()

# Load VAD model — once at startup, stays in memory (CPU, no VRAM)
print("Loading VAD model...")
vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', verbose=False)
print("VAD ready.")

# Load WhisperX model — once at startup, stays in VRAM (~4.5GB)
print("Loading WhisperX model...")
whisper_model = whisperx.load_model("large-v3", "cuda", compute_type="int8", language="en",
                                    asr_options={"initial_prompt": STT_INITIAL_PROMPT})
print("WhisperX ready.")

# Load ECAPA-TDNN — 192-dim vector, more robust than pyannote/embedding on short segments (<3s)
# No HF token needed — model is public
print("Loading embedding model...")
embedding_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="/root/.cache/speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cuda"}
)
print("Embedding ready.")

# ChromaDB in-memory client — in-session RAG (phase 1).
# EphemeralClient = no volume, no persistence, dies with the process.
# Default embedding (all-MiniLM) is loaded on first add()/query().
print("Loading ChromaDB client...")
chroma_client = chromadb.EphemeralClient()
print("ChromaDB ready.")

# Embedding function for BOTH RAG layers — all-MiniLM on GPU instead of
# the ChromaDB default (ONNX/CPU). One query on CPU costs ~0.5s,
# on GPU ~milliseconds. IMPORTANT: build_knowledge_base.py MUST use the
# same function/model, otherwise incompatible vectors → rebuild the index.
from chromadb.utils import embedding_functions
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=KNOWLEDGE_EMBED_MODEL, device="cuda")
print(f"Embedding function ready (all-MiniLM on GPU).")

# Static knowledge base (phase 2) — PERSISTENT collection, read-only.
# Separate from the ephemeral client above: pre-indexed via build_knowledge_base.py,
# survives the process, fully available from turn 1. Only loaded when enabled.
# Handles the case where the index has not been built yet → pipeline continues without it.
knowledge_collection = None
if USE_STATIC_KNOWLEDGE:
    print("Loading knowledge base...")
    try:
        kb_client = chromadb.PersistentClient(path=str(KNOWLEDGE_DB_PATH))
        knowledge_collection = kb_client.get_collection(
            name=KNOWLEDGE_COLLECTION, embedding_function=embed_fn)
        print(f"Knowledge base ready ({knowledge_collection.count()} chunks).")
    except Exception as e:
        print(f"WARNING:Knowledge base not loadable ({e}) — running WITHOUT static knowledge. "
              f"Index built? → python3 build_knowledge_base.py")
        knowledge_collection = None

# All tunables (models, hosts, thresholds, sliding window, RAG limits)
# live in config.py. MOTHER_SYSTEM_PROMPT in mother_prompt.py.

ollama_client = ollama.AsyncClient(host=OLLAMA_HOST)


# Runtime state (not a tunable, does not belong in config.py)
session_active = False              # single client lock

# Set when the kill phrase is spoken — ends the serve loop in main()
# after the last session's distillation is complete. asyncio.Event can
# be created without a running loop from Python 3.10.
shutdown_event = asyncio.Event()


def is_kill_phrase(text):
    """True if the transcript text contains the end phrase.
    Normalised (lowercase, letters+spaces only) → robust against
    WhisperX capitalisation and punctuation ("Let's kill this session.")."""
    norm = re.sub(r"[^a-z ]", "", text.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    return KILL_PHRASE in norm

def cosine_similarity(a, b):
    """Similarity between two vectors: 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def speaker_has_given_name(messages, label):
    """True if this speaker has introduced themselves in an earlier utterance.
    Looks for clear introduction patterns (I'm X / I am X / my name is X / call me X)
    with a capitalised name following. Prevents Mother from redundantly asking for a
    name already given — the name hint note is then NOT injected."""
    prefix = f"[Speaker {label}]:"
    pattern = re.compile(r"(?:[Ii]'?m|[Ii] am|[Mm]y name is|[Cc]all me)\s+[A-Z][a-z]+")
    for m in messages:
        if m.get("role") == "user" and m["content"].startswith(prefix):
            if pattern.search(m["content"]):
                return True
    return False


def extract_sentence(buffer):
    """Finds a complete sentence in the buffer.
    Splits only at [.!?] + whitespace — guards against 'e.g.' / 'Dr.' false positives.
    Returns (sentence, remaining_buffer) or (None, buffer) if no sentence is complete.
    """
    match = re.search(r'[.!?]\s', buffer)
    if match:
        end = match.start() + 1   # inclusive of punctuation
        sentence = buffer[:end].strip()
        remaining = buffer[match.end():]
        return sentence, remaining
    return None, buffer


def index_turn_pair(collection, pair_number, user_content, mother_text):
    """Stores a turn pair that has fallen out of the sliding window into the
    in-session RAG collection. Synchronous (ChromaDB add() blocks) — called
    via run_in_executor outside the answer path. user_content already contains
    the [Speaker X] label. ChromaDB embeds automatically (all-MiniLM).
    """
    document = f"{user_content}\nMother: {mother_text}"
    collection.add(
        documents=[document],
        ids=[f"turn_{pair_number}"],
        metadatas=[{"pair": pair_number}],
    )


async def index_turn_pair_bg(collection, pair_number, user_content, mother_text):
    """Background wrapper: runs the blocking ChromaDB add() in the executor
    (so the event loop does not stall) and catches/logs errors — no silent fail.
    """
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, index_turn_pair, collection, pair_number, user_content, mother_text
        )
        print(f"→ RAG: pair {pair_number} indexed ({collection.count()} in store)")
    except Exception as e:
        print(f"WARNING:RAG index error at pair {pair_number}: {e}")


async def tts_speak(tts_client, sentence, websocket):
    """Sends a sentence to the tts service, streams PCM chunks directly via WebSocket to the client."""
    async with tts_client.stream("POST", f"{TTS_HOST}/synthesize", json={"text": sentence}) as tts_response:
        async for chunk in tts_response.aiter_bytes():
            if chunk:
                await websocket.send(chunk)

async def handle_client(websocket):
    global session_active

    # Reject if someone is already connected
    if session_active:
        print("Connection rejected — session already running")
        await websocket.close(1008, "Session already running")
        return

    session_active = True
    print("Client connected — session started")

    audio_buffer = []       # collects speech chunks
    silence_counter = 0     # counts consecutive silence chunks
    t_speech_end = time.time()   # timestamp of last speech end (for VAD latency)
    t_session_start = time.time()   # for waterfall timestamps

    # In-session memory — Mother's memory for this session.
    # System prompt = original from mother_prompt.py + any distilled lessons
    # from previous sessions (lessons.md). Empty on first run → Mother gets
    # only the original. Once sessions of ≥ 10 turns have run, the block fills up.
    lessons = parse_lessons_md(LESSONS_PATH)
    lessons_block = render_lessons_for_prompt(lessons)
    if lessons_block:
        system_content = (
            MOTHER_SYSTEM_PROMPT
            + "\n\nReflections from earlier sessions — keep doing the first kind, "
              "avoid the second:\n\n"
            + lessons_block
        )
        n_worked = len(lessons["what_worked"])
        n_avoid = len(lessons["what_to_avoid"])
        print(f"→ lessons.md loaded: {n_worked} worked + {n_avoid} avoid")
    else:
        system_content = MOTHER_SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_content}]

    # In-session RAG collection — empty, dies with the session.
    # Catches turns that fall out of the sliding window.
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rag_collection = chroma_client.create_collection(
        name=f"session_{session_id}", embedding_function=embed_fn)
    next_to_index = 1          # next turn pair to index into ChromaDB
    rag_tasks = set()          # running background index tasks (keep reference)

    # Speaker registry — dies with the session
    # speakers = {"A": centroid_vector, "B": centroid_vector, ...}
    # centroid = average embedding of all segments from this speaker
    speakers = {}
    next_speaker_id = "A"   # first speaker becomes "A", then "B", "C", ...
    speaker_turn_count = {}  # how many times each speaker has spoken
    speaker_name_asked = set()  # which speakers have already received the name hint

    terminate_after = False  # True when kill phrase is spoken → shut down server after

    # Greeting: Mother speaks first instead of waiting silently.
    # Random opener from the pool. "BUSY" first → client mutes the mic
    # immediately (covers the synthesis window). NOT added to messages → no
    # turn-pair offset. Own try/except: a TTS hang does not kill the session.
    if MOTHER_START and MOTHER_GREETINGS:
        greeting = random.choice(MOTHER_GREETINGS)
        try:
            await websocket.send("BUSY")
            async with httpx.AsyncClient(timeout=30) as tts_client:
                await tts_speak(tts_client, greeting, websocket)
            await websocket.send("TTS_DONE")
            print(f"Greeting: {greeting}")
        except Exception as e:
            print(f"WARNING:Greeting skipped: {e}")

    try:
        async for chunk in websocket:
            if len(chunk) == 0:
                continue

            if not isinstance(chunk, bytes):
                print("Warning: received non-audio bytes")
                continue

            # bytes → numpy array
            audio = np.frombuffer(chunk, dtype=np.float32).copy()

            # VAD expects exactly VAD_WINDOW=512 samples — split chunk into windows
            probability = 0.0
            for i in range(0, len(audio) - VAD_WINDOW + 1, VAD_WINDOW):
                window = torch.from_numpy(audio[i:i + VAD_WINDOW])
                prob = vad_model(window, SAMPLE_RATE).item()
                if prob > probability:
                    probability = prob  # take highest probability

            # Live waterfall (in-place): ONE line that overwrites itself per chunk
            # (\r, no newline). Saves console space when scrolling. On the next
            # regular print() (e.g. "segment complete") the last state is frozen
            # as a snapshot — so the moment of the segment trigger is always visible.
            if LIVE_VAD_DISPLAY and (len(audio_buffer) > 0
                                     or probability > SILENCE_THRESHOLD):
                t_rel = time.time() - t_session_start
                bars = int(probability * 10)
                bar = "█" * bars + "░" * (10 - bars)
                status = "REC" if probability > SILENCE_THRESHOLD else "..."
                line = (f"[{t_rel:7.3f}s] prob {probability:.2f} {bar} "
                        f" {status}  buf={len(audio_buffer):>2}  "
                        f"sil={silence_counter}")
                # ljust against ghost characters if new frame is shorter than previous
                print(line.ljust(70), end="\r", flush=True)

            if probability > SILENCE_THRESHOLD:
                audio_buffer.append(audio)
                silence_counter = 0

            else:
                silence_counter += 1
                # First silence chunk after speech = user just stopped talking
                if silence_counter == 1 and len(audio_buffer) > 0:
                    t_speech_end = time.time()

                if silence_counter >= SILENCE_CHUNKS and len(audio_buffer) > 0:
                    # 600ms silence after speech → segment complete
                    segment = np.concatenate(audio_buffer)
                    duration = len(segment) / SAMPLE_RATE

                    # Ignore segments that are too short (background noise, echo)
                    if duration < MIN_SEGMENT_DURATION:
                        audio_buffer = []
                        silence_counter = 0
                        continue

                    # Pipeline triggers NOW — mute client immediately. BUSY covers the
                    # entire processing window (STT→LLM→TTS), not just from first
                    # Mother audio. Only TTS_DONE lifts the mute again.
                    await websocket.send("BUSY")

                    # VAD wait time: from last speech chunk to segment trigger
                    latency_vad = time.time() - t_speech_end

                    print()
                    print("=" * 60)
                    print(f"→ Segment complete: {duration:.1f}s audio  |  VAD wait {latency_vad:.2f}s")

                    # Transcribe with WhisperX
                    t_start = time.time()
                    result = whisper_model.transcribe(segment, batch_size=WHISPER_BATCH_SIZE)
                    latency_stt = time.time() - t_start
                    text = " ".join(s["text"].strip() for s in result["segments"])
                    print(f"→ Text: {text}")
                    print(f"→ STT latency: {latency_stt:.2f}s")

                    # Ignore empty text
                    if not text.strip():
                        audio_buffer = []
                        silence_counter = 0
                        continue

                    # Kill phrase → end session cleanly. Speak goodbye,
                    # close client, exit loop. The finally block then runs
                    # transcript + distillation and sets the shutdown_event
                    # → server.py shuts itself down.
                    # Deliberately NOT added to messages (command, not a turn).
                    if is_kill_phrase(text):
                        print("→ Kill phrase detected — ending session")
                        async with httpx.AsyncClient(timeout=30) as tts_client:
                            await tts_speak(tts_client, MOTHER_GOODBYE, websocket)
                        await websocket.send("TTS_DONE")    # goodbye done → unmute client
                        await websocket.send("SESSION_END")  # client: end cleanly (no reconnect)
                        terminate_after = True
                        break

                    # Extract speaker embedding — 192-dim vector from audio segment.
                    # ECAPA-TDNN (SpeakerRecognition.encode_batch) handles windowing +
                    # pooling internally — device was set at init.
                    t_emb = time.time()

                    # ECAPA-TDNN: (1, samples) → (1, 1, 192) → (192,)
                    waveform = torch.from_numpy(segment).float().unsqueeze(0)
                    with torch.no_grad():
                        embedding = embedding_model.encode_batch(waveform).squeeze().cpu().numpy()

                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm

                    latency_emb = time.time() - t_emb

                    # Identify speaker: cosine similarity to known centroids
                    if not speakers:
                        label = next_speaker_id
                        speakers[label] = embedding
                        next_speaker_id = chr(ord(next_speaker_id) + 1)
                        speaker_info = f"Speaker {label} (new, first of session)"
                    else:
                        best_id, best_sim = max(
                            ((sid, cosine_similarity(embedding, c)) for sid, c in speakers.items()),
                            key=lambda x: x[1]
                        )
                        if best_sim > SPEAKER_SIMILARITY_THRESHOLD:
                            label = best_id
                            # Update centroid only with long enough segments — short ones are noisy
                            if duration >= CENTROID_UPDATE_MIN_DURATION:
                                speakers[label] = 0.9 * speakers[label] + 0.1 * embedding
                            speaker_info = f"Speaker {label} (matched, sim {best_sim:.2f})"
                        elif duration < MIN_SEGMENT_DURATION_FOR_NEW_SPEAKER:
                            # Short segment below threshold — no new speaker, force best match
                            label = best_id
                            speaker_info = f"Speaker {label} (short {duration:.1f}s, forced, sim {best_sim:.2f})"
                        else:
                            # Long segment, no match → new speaker
                            label = next_speaker_id
                            speakers[label] = embedding
                            next_speaker_id = chr(ord(next_speaker_id) + 1)
                            speaker_info = f"Speaker {label} (new, best sim {best_sim:.2f} < {SPEAKER_SIMILARITY_THRESHOLD})"

                    print(f"→ {speaker_info}")
                    print(f"→ Embedding latency: {latency_emb:.3f}s  |  speakers: {list(speakers.keys())}")

                    speaker_turn_count[label] = speaker_turn_count.get(label, 0) + 1

                    # Append user utterance to session memory (clean, without hint)
                    user_content = f"[Speaker {label}]: {text}"
                    messages.append({"role": "user", "content": user_content})

                    # Sliding window: system prompt (always full) + last N turn pairs.
                    # messages[0] = system, messages[1:] = turns. Slicing is safe
                    # when fewer turns are present (just takes all of them).
                    llm_messages = [messages[0]] + messages[1:][-(SLIDING_WINDOW_PAIRS * 2):]

                    # In-session RAG — retrieve relevant older turns. Lazy:
                    # only when ChromaDB is not empty (= once the first pair has
                    # fallen out of the sliding window). Queries: original + expanded
                    # query (stage 2, see below). query() runs in executor
                    # (event loop free for background index tasks). Hits appended to
                    # system prompt (context knowledge = system level, does not disturb
                    # user/assistant sequence; llm_messages[-1] stays the current
                    # turn → name hint ok).
                    latency_rag = 0.0   # stays 0 if RAG did not run (store empty)
                    if rag_collection.count() > 0:
                        loop = asyncio.get_running_loop()
                        t_rag = time.time()

                        # Stage 2 — query expansion: original query (raw turn) always
                        # stays in. Additionally an expanded query from the last 1-2
                        # user turns + current turn. Thin utterances ("yeah", "hmm")
                        # embed alone as noise → the expanded query rescues the hit.
                        # Deliberately NOT replacing, only adding (otherwise it drags
                        # the old topic into a clear topic change → contamination).
                        prev_user = [m["content"] for m in messages
                                     if m["role"] == "user"][-3:-1]
                        expanded = (" ".join(prev_user) + " " + text).strip()
                        queries = [text]
                        if expanded and expanded != text:
                            queries.append(expanded)
                        n = min(RAG_N_RESULTS, rag_collection.count())
                        res = await loop.run_in_executor(
                            None,
                            lambda: rag_collection.query(
                                query_texts=queries, n_results=n),
                        )

                        # Merge candidates from both queries, keep the smallest
                        # (= best) distance per document.
                        best_by_doc = {}
                        for qi in range(len(queries)):
                            for d, dist in zip(res["documents"][qi],
                                               res["distances"][qi]):
                                if d not in best_by_doc or dist < best_by_doc[d]:
                                    best_by_doc[d] = dist
                        cand = sorted(best_by_doc.items(), key=lambda x: x[1])

                        # Threshold: relative to best distance (best + δ),
                        # capped by the absolute RAG_MAX_DISTANCE. No real hit
                        # (best > cap) → nothing survives → NO RAG block.
                        kept, limit = [], RAG_MAX_DISTANCE
                        if cand:
                            best = cand[0][1]
                            limit = min(RAG_MAX_DISTANCE,
                                        best + RAG_REL_DELTA)
                            kept = [d for d, dist in cand if dist <= limit]
                        if kept:
                            rag_block = ("\n\nRelevant earlier moments from this "
                                         "conversation (use only if relevant):\n"
                                         + "\n".join(f"- {d}" for d in kept))
                            llm_messages[0] = {
                                "role": "system",
                                "content": MOTHER_SYSTEM_PROMPT + rag_block,
                            }
                        latency_rag = (time.time() - t_rag) * 1000
                        dist_str = "[" + ", ".join(
                            f"{dist:.3f}" for _, dist in cand) + "]"
                        print(f"→ RAG: {len(kept)}/{len(cand)} kept "
                              f"({latency_rag:.0f}ms, {rag_collection.count()} "
                              f"in store) q={len(queries)} dist={dist_str} "
                              f"thr={limit:.3f}")

                    # Static knowledge base (phase 2) — parallel second RAG layer.
                    # Own persistent collection, own threshold, naive query (raw turn).
                    # Fires from turn 1 (pre-indexed). Hits are APPENDED to the system
                    # prompt (additive on top of whatever in-session RAG may have set).
                    # Mother weaves the knowledge into her own voice — NO titles/authors/
                    # sources (canonical: mythic, not academic).
                    latency_kb = 0.0
                    if knowledge_collection is not None:
                        loop = asyncio.get_running_loop()
                        t_kb = time.time()
                        n_kb = min(KNOWLEDGE_N_RESULTS, knowledge_collection.count())
                        kb_res = await loop.run_in_executor(
                            None,
                            lambda: knowledge_collection.query(
                                query_texts=[text], n_results=n_kb,
                                include=["documents", "distances", "metadatas"]),
                        )
                        kb_docs = kb_res["documents"][0]
                        kb_dists = kb_res["distances"][0]
                        kb_metas = kb_res["metadatas"][0]
                        kb_kept = [(d, m) for d, dist, m in
                                   zip(kb_docs, kb_dists, kb_metas)
                                   if dist <= KNOWLEDGE_MAX_DISTANCE]
                        if kb_kept:
                            kb_block = (
                                "\n\nBackground knowledge you may quietly draw on — let it "
                                "color your reply in your own voice. Never name titles, "
                                "authors, or sources:\n"
                                + "\n".join(f"- {d}" for d, _ in kb_kept)
                            )
                            llm_messages[0] = {
                                "role": "system",
                                "content": llm_messages[0]["content"] + kb_block,
                            }
                        latency_kb = (time.time() - t_kb) * 1000
                        kb_dist_str = "[" + ", ".join(f"{d:.3f}" for d in kb_dists) + "]"
                        print(f"→ KB: {len(kb_kept)}/{len(kb_docs)} kept "
                              f"({latency_kb:.0f}ms, {knowledge_collection.count()} chunks) "
                              f"dist={kb_dist_str} thr={KNOWLEDGE_MAX_DISTANCE}")
                        # Source debug: which works were pulled → verifies KB relevance
                        if kb_kept:
                            kb_src = " | ".join(
                                f"{m.get('source', '?')[:32]}#{m.get('chunk', '?')}"
                                for _, m in kb_kept)
                            print(f"   KB sources: {kb_src}")

                    # Name hint injected ephemerally from NAME_HINT_MIN_TURN — NOT saved
                    # to messages. ONLY if the name has not been given yet: if the speaker
                    # already introduced themselves, the note would force Mother to ask
                    # redundantly → skip it and mark the topic as resolved.
                    # (Demo: NAME_HINT_MIN_TURN=1 → ask on the very first turn.)
                    name_hint_injected = False
                    if speaker_turn_count[label] >= NAME_HINT_MIN_TURN and label not in speaker_name_asked:
                        if speaker_has_given_name(messages, label):
                            speaker_name_asked.add(label)   # name known → no hint needed
                            print(f"→ Name hint skipped — Speaker {label} already introduced themselves")
                        else:
                            llm_messages[-1] = {"role": "user", "content": user_content + " (Note to Mother: you haven't asked this person their name yet — work it naturally into your reply)"}
                            speaker_name_asked.add(label)
                            name_hint_injected = True
                            print(f"→ Name hint injected for Speaker {label}")

                    # LLM → sentence split → TTS — all in parallel.
                    # Ollama streams tokens; as soon as a sentence is complete it is
                    # sent immediately to the tts service. While Mother speaks sentence 1,
                    # Ollama generates sentence 2/3 → perceived latency drops massively.
                    t_llm = time.time()
                    try:
                        stream = await ollama_client.chat(
                            model=OLLAMA_MODEL,
                            messages=llm_messages,
                            think=False,
                            stream=True,
                            options={
                                "num_ctx": LLM_NUM_CTX,
                                "num_predict": LLM_NUM_PREDICT,
                                "temperature": LLM_TEMPERATURE,
                                "repeat_penalty": LLM_REPEAT_PENALTY,
                                "stop": LLM_STOP,
                            }
                        )
                    except Exception as e:
                        print(f"WARNING:Ollama unreachable: {e}")
                        await websocket.send("TTS_DONE")   # str not bytes — otherwise client won't recognise it, mother_speaking would stay True
                        continue

                    mother_text = ""
                    buffer = ""
                    first_token_latency = None
                    t_first_sentence = None
                    sentences_sent = 0
                    prefix_buf = ""      # collects first tokens until prefix check is possible
                    prefix_done = False  # strip leading [Mother] label once

                    async with httpx.AsyncClient(timeout=30) as tts_client:
                        async for ollama_chunk in stream:
                            if first_token_latency is None:
                                first_token_latency = time.time() - t_llm
                            token = ollama_chunk["message"]["content"]

                            # Base models (not fine-tuned) mimic the [Speaker X]: pattern
                            # and prefix their reply with [Mother]. Collect first tokens,
                            # strip the label before anything goes to TTS/messages.
                            if not prefix_done:
                                prefix_buf += token
                                if len(prefix_buf) < 14:   # "[Speaker Z]: " = 14 chars
                                    continue
                                prefix_buf = re.sub(r'^\s*(\[Mother\]|\[Speaker [A-Z]\]):?\s*', '', prefix_buf)
                                prefix_done = True
                                token = prefix_buf
                                prefix_buf = ""

                            mother_text += token
                            buffer += token

                            # Pull all complete sentences from the buffer and send each to TTS
                            while True:
                                sentence, buffer = extract_sentence(buffer)
                                if sentence is None:
                                    break
                                sentences_sent += 1
                                if t_first_sentence is None:
                                    t_first_sentence = time.time() - t_llm
                                    print(f"→ Sentence 1 → TTS: {t_first_sentence:.2f}s after LLM start")
                                print(f"  [Sentence {sentences_sent}] {sentence}")
                                await tts_speak(tts_client, sentence, websocket)

                        # Very short reply (< 14 chars) → prefix_buf was never flushed
                        if not prefix_done and prefix_buf.strip():
                            rest = re.sub(r'^\s*(\[Mother\]|\[Speaker [A-Z]\]):?\s*', '', prefix_buf)
                            mother_text += rest
                            buffer += rest

                        # Stream ended — remainder in buffer is last sentence (possibly without punctuation)
                        if buffer.strip():
                            sentences_sent += 1
                            if t_first_sentence is None:
                                t_first_sentence = time.time() - t_llm
                                print(f"→ Sentence 1 → TTS: {t_first_sentence:.2f}s after LLM start")
                            print(f"  [Sentence {sentences_sent} final] {buffer.strip()}")
                            await tts_speak(tts_client, buffer.strip(), websocket)

                    await websocket.send("TTS_DONE")
                    latency_llm = time.time() - t_llm
                    messages.append({"role": "assistant", "content": mother_text})

                    # Fallback if LLM returned nothing (no sentence → None)
                    if t_first_sentence is None:
                        t_first_sentence = latency_llm
                    if first_token_latency is None:
                        first_token_latency = latency_llm

                    # Latency to first syllable = what the user experiences as "wait time".
                    # RAG runs on the critical path (after embedding, before LLM) →
                    # MUST be included, otherwise the display understates real wait time.
                    # latency_rag is in ms → /1000 for seconds sum.
                    latency_e2e = (latency_vad + latency_stt + latency_emb
                                        + latency_rag / 1000 + latency_kb / 1000
                                        + t_first_sentence)

                    print(f"→ Mother: {mother_text}")
                    print(f"→ LLM latency: first token {first_token_latency:.2f}s, total {latency_llm:.2f}s")
                    print(f"→ Sentences spoken: {sentences_sent}")
                    print(f"→ Memory: {(len(messages)-1)//2} turn pairs"
                          + ("  |  name hint injected" if name_hint_injected else ""))
                    print("-" * 60)
                    print(f"  LATENCY to first syllable  ≈ {latency_e2e:.2f}s")
                    print(f"    VAD silence     {latency_vad:.2f}s")
                    print(f"    WhisperX STT    {latency_stt:.2f}s")
                    print(f"    Embedding       {latency_emb:.2f}s")
                    print(f"    RAG retrieval   {latency_rag/1000:.2f}s")
                    print(f"    KB retrieval    {latency_kb/1000:.2f}s")
                    print(f"    LLM → sent. 1   {t_first_sentence:.2f}s  (1st token {first_token_latency:.2f}s)")
                    print("=" * 60)
                    print()

                    # Turn pairs that have fallen out of the sliding window
                    # → store in ChromaDB (background, outside the answer path).
                    # messages: [0]=system, pair p: user=[2p-1], assistant=[2p].
                    total_pairs = (len(messages) - 1) // 2
                    while next_to_index <= total_pairs - SLIDING_WINDOW_PAIRS:
                        p = next_to_index
                        u = messages[2 * p - 1]["content"]
                        m = messages[2 * p]["content"]
                        task = asyncio.create_task(
                            index_turn_pair_bg(rag_collection, p, u, m)
                        )
                        rag_tasks.add(task)
                        task.add_done_callback(rag_tasks.discard)
                        next_to_index += 1

                    # Clear buffer for next turn
                    audio_buffer = []
                    silence_counter = 0

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

    finally:
        session_active = False
        # Delete in-session RAG collection — dies with the session
        try:
            chroma_client.delete_collection(name=f"session_{session_id}")
        except Exception as e:
            print(f"WARNING:RAG collection deletion failed: {e}")
        # Save transcript — all turns except system prompt
        if len(messages) > 1:
            DISTILLATES_DIR.mkdir(parents=True, exist_ok=True)
            # Format: YY-MM-DD-HH-MM (e.g. session_26-06-22-18-52.txt).
            # Minutes included so two sessions in the same hour don't overwrite each other.
            # timestamp is also passed to run_distillation → transcript + findings +
            # marker are guaranteed to land in the same file.
            timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
            path = DISTILLATES_DIR / f"session_{timestamp}.txt"
            with open(path, "w") as f:
                for msg in messages[1:]:  # skip system prompt
                    role = msg["role"].upper()
                    f.write(f"[{role}]\n{msg['content']}\n\n")
            print(f"Transcript saved: {path}")

            # Await distillation deterministically here — judge + merger run
            # immediately after disconnect (not as a loose background task that
            # dies with the process). Single-client server: the brief block
            # (~5-15s) affects nothing, but guarantees findings + lessons.md
            # are written before the server is free again. run_distillation
            # checks the 10-turn floor itself and never raises outward.
            # MOTHER_SYSTEM_PROMPT pure (without lessons.md appendix) → judge
            # evaluates against the unchanged standard.
            await run_distillation(messages, MOTHER_SYSTEM_PROMPT, timestamp)

        # Kill phrase was spoken → now (after distillation is complete) shut down
        # the server. Marker is on disk → client sync finds it even after server.py exits.
        if terminate_after:
            print("Kill phrase: distillation complete — shutting down server.")
            shutdown_event.set()
        else:
            print("Session ended — ready for new connection")

async def warmup():
    """Avoid cold start: pay the expensive first-call costs at server startup
    instead of on Mother's first reply (~10s the very first time). Biggest item
    is loading the Ollama model into A100 VRAM (~8s on first chat call); plus
    a short WhisperX + F5 poke (CUDA kernel warmup). Each step in its own
    try/except — if one fails (e.g. tts not yet up), the server still starts.
    After warmup Ollama keeps the model in VRAM for ~5min by default."""
    t0 = time.time()
    print("Warmup: preheating pipeline (avoiding cold start)...")

    # 1) Ollama — load model into A100 VRAM (the ~8s item).
    #    CRITICAL: num_ctx MUST match the real call (handle_client) exactly!
    #    Ollama binds the loaded model instance to the context size — if the
    #    first real request arrives with a different num_ctx, Ollama unloads +
    #    reloads the model (= cold start again). So same num_ctx=4096 here.
    try:
        await ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            think=False,
            options={"num_ctx": LLM_NUM_CTX, "num_predict": 1},
        )
        print(f"  Ollama warm ({OLLAMA_MODEL}, num_ctx={LLM_NUM_CTX})")
    except Exception as e:
        print(f"  WARNING:Ollama warmup skipped: {e}")

    # 2) WhisperX — first transcribe compiles CUDA kernels (1s of silence is enough)
    try:
        whisper_model.transcribe(np.zeros(16000, dtype=np.float32), batch_size=WHISPER_BATCH_SIZE)
        print("  WhisperX warm")
    except Exception as e:
        print(f"  WARNING:WhisperX warmup skipped: {e}")

    # 3) F5-TTS — preheat first synthesis in the tts container
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            async with c.stream("POST", f"{TTS_HOST}/synthesize",
                                 json={"text": "Hello."}) as r:
                async for _ in r.aiter_bytes():
                    pass
        print("  F5-TTS warm")
    except Exception as e:
        print(f"  WARNING:F5-TTS warmup skipped: {e}")

    # 4) Knowledge base — first query loads the embedding model (all-MiniLM).
    #    Also covers in-session RAG (same model).
    if knowledge_collection is not None:
        try:
            knowledge_collection.query(query_texts=["hello"], n_results=1)
            print("  Knowledge base warm")
        except Exception as e:
            print(f"  WARNING:KB warmup skipped: {e}")

    print(f"Warmup complete ({time.time() - t0:.1f}s) — first reply will be warm.")


async def main():
    # Preheat before opening the port → the port only accepts connections
    # once the pipeline is warm. The client (reconnect loop) simply waits
    # those few seconds.
    await warmup()

    # ping_interval=None: no keepalive kill. Synchronous GPU calls (WhisperX,
    # ECAPA) briefly block the event loop — under VRAM pressure > 20s. The
    # default ping would then falsely assume a dead connection and disconnect.
    # Single client + client.py reconnect make active keepalive unnecessary.
    async with websockets.serve(handle_client, "0.0.0.0", 8001, ping_interval=None):
        print("Server running on port 8001 — waiting for connection...")
        # Runs until the kill phrase sets the shutdown_event (after distillation
        # is complete). Then the async with exits the serve loop →
        # process ends cleanly, VRAM freed.
        await shutdown_event.wait()
        print("Server shutting down.")

signal.signal(signal.SIGINT, signal.SIG_IGN)
asyncio.run(main())

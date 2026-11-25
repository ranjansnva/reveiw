# Local Loki (`Loki AI Agent`) — Technology Stack & Granular Technical Use-Case Reference

This document maps **every technology, model, library, protocol, and service** used in `Local_loki` to its exact **technical use case** (from core AI pipelines down to minor utility helpers).

---

## 1. Architecture Modes Overview

`Local_loki` is a real-time, self-hosted voice AI agent for airline flight search and booking (`Travomint`), built around four modular layers:

| Module | Directory | Role |
| :--- | :--- | :--- |
| **Voice Core** | `1-voice-core/` | Pipecat streaming pipeline, FastAPI/WebSocket server, STT/LLM/TTS, guardrails, flight search, SES mailer |
| **Short-Term Memory** | `2-memory-service/` | Standalone FastAPI + async Redis microservice for per-call turn history & metadata |
| **Long-Term Profile** | `3-profile-service/` | Caller phone/PNR CRM lookup, loyalty tier, and prompt context injection |
| **SIP Telephony** | `4-calling-agent/` | BigPBX/Asterisk SIP registration, RTP audio transcoding, VAD, and call transfer |

---

## 2. Large Language Models & Inference (`llama.cpp` / OpenAI-Compatible API)

All LLM inference runs against self-hosted `llama.cpp` servers exposing OpenAI-compatible `/v1/chat/completions` endpoints, switched dynamically via `MODEL_MODE` in `config.py`.

| Technology / Model | Endpoint / Config | Exact Technical Use Cases |
| :--- | :--- | :--- |
| **Qwen3-Omni-30B-A3B-Instruct** (`Q4_K_M.gguf` + `--mmproj`) | `OMNI_BASE_URL` (`http://10.20.1.75:8080/v1`) | **1. Direct Multimodal Audio Dialogue (`MODEL_MODE=omni_audio`)**: Accepts raw WAV audio encoded as base64 via `{"type": "input_audio", "input_audio": {"data": ..., "format": "wav"}}` directly in `/api/chat_and_speak` (`1-voice-core/pipeline.py`), bypassing a separate STT step.<br>**2. Standalone Sub-300ms ASR (`OmniSTTService`)**: Used in `1-voice-core/services/omni_stt.py` (`temperature=0.0`, `max_tokens=150`) with a strict telephony ASR system prompt (`OMNI_TELEPHONY_SYSTEM_PROMPT`) to transcribe 16kHz PCM/WAV audio turns. |
| **Qwen3.6-35B** (`Qwen3-6-35B.gguf`) | `QWEN_BASE_URL` (`http://10.20.1.75:8083/v1`) | **Text-Only Dialogue LLM (`MODEL_MODE=whisper_qwen`)**: Coupled with WhisperX or Omni STT for text-based reasoning and token streaming. |
| **Thinking-Mode Suppression (`/no_think` & `</think>` prefill)** | `1-voice-core/prompts.py`, `llm_extractor.py`, `date_resolver.py`, `detail_resolver.py`, `weekend_helper.py` | **Zero-Overhead Latency**: System prompts start with `/no_think` and micro-calls prefill `{"role": "assistant", "content": "</think>\n"}` (and strip `<think>`/`</think>` tags) so Qwen models stream or return answers immediately without reasoning token delay. |

### Micro-Level Auxiliary LLM Calls (Targeted `aiohttp` Calls)
Beyond main conversation generation, the active `LLM_BASE_URL` is called directly with `temperature=0.0`, `stream=False`, and short timeouts (4–5s) for narrow, code-validated tasks:

1. **Binary Affirmation Classifier (`1-voice-core/services/weekend_helper.py` -> `classify_affirmation_llm`)**: Classifies ambiguous caller replies as `YES` or `NO` (`max_tokens=5`) for date/detail/email confirmations.
2. **Language Switch Classifier (`1-voice-core/services/language_manager.py` -> `classify_confirmation`)**: Classifies whether a caller agreed to switch languages (`YES`/`NO`, `max_tokens=5`), with multilingual keyword fallback.
3. **Fallback Structured Slot Extractor (`1-voice-core/services/llm_extractor.py` -> `extract_llm_slots`)**: Extracts JSON for the single `awaiting_field` (`trip_details`, `cabin_pax`, `name`, `phone`, `email`) when regex extraction misses natural phrasing (e.g., *"myself and my wife"* -> `passengers: 2`), strictly validated in `_validate()`.
4. **Spoken Date Resolver (`1-voice-core/services/date_resolver.py` -> `resolve_date_llm`)**: Gated by `may_contain_date()`, converts spoken dates (*"twenty ninth of november"*) into `YYYY-MM-DD`, then validates range (`today <= date <= today + 366d`) and spoken month consistency.
5. **Partial Detail Correction Resolver (`1-voice-core/services/detail_resolver.py` -> `resolve_corrections`)**: Applies partial spoken edits (*"there is an M in the starting"*, *"it's Priyanshu, not Priyansh"*) against currently stored `name`/`phone`/`email`, validated via regex before updating state.

---

## 3. Speech-to-Text (STT) & Transcript Sanitization

| Technology | File | Exact Technical Use Cases |
| :--- | :--- | :--- |
| **Qwen-Omni Multimodal ASR** (`OmniSTTService`) | `1-voice-core/services/omni_stt.py` | **Primary High-Speed STT (`STT_ENGINE=omni`)**: Wraps raw 16kHz PCM-16 in a RIFF WAV header, base64-encodes it, and posts to `/v1/chat/completions`. Drops turns `< 3200` bytes (`< 0.1s`). |
| **Vesper CallSense / WhisperX** (`turbo` on GPU) | `1-voice-core/services/vesper_stt.py` | **Alternative GPU STT Pool (`STT_ENGINE=whisper`)**: Writes 16kHz audio to a temp `.wav` in `VESPER_TEMP_DIR`, sends `POST http://10.20.1.110:8002/callsense/transcribe` with `whisper_model="turbo"`, `device="gpu"`, `project="dowser"`, and cleans up the temp file. |
| **Telephony Transcript Cleaner** (Regex & Token Filter) | `1-voice-core/services/telephony_cleaner.py` | **Hallucination Suppression**: Strips Whisper/Omni silence artifacts (`"thank you"`, `"thanks for watching"`, `"amara.org"`, `"subtitles by"`), CJK/Hangul/Cyrillic script hallucinations (`[\u4e00-\u9fff...]`), isolated Hindi postpositions (`ने`, `का`, `की`, etc.) caused by breathing, and utterances `< 2` alphanumeric chars (while preserving single digits `0–9` for menu/option selection). |

---

## 4. Voice Activity Detection (VAD) & Barge-In

| Technology | File | Exact Technical Use Cases |
| :--- | :--- | :--- |
| **Silero VAD (ONNX Runtime)** | `1-voice-core/pipeline.py`, `4-calling-agent/silero_vad.py` | **Offline Neural Speech Detection**: Runs `SileroVADAnalyzer` locally on CPU (`< 1ms` per 512-sample / 1024-byte 32ms window at 16kHz) in both the Pipecat WebSocket pipeline and the SIP calling agent (`SILERO_VAD_THRESHOLD=0.50`). |
| **Picovoice Cobra VAD (`pvcobra`)** | `4-calling-agent/cobra_vad.py` | **Optional Sub-10ms Neural Telephony VAD**: Processes 512-sample 16kHz PCM16 frames (`COBRA_VAD_THRESHOLD=0.65`) when `PICOVOICE_ACCESS_KEY` is configured, resilient to PSTN hiss and acoustic echo. |
| **RMS Energy & Noise Gate (`audioop.rms`)** | `4-calling-agent/audio_bridge.py`, `config.py` | **Fallback / Pre-Filter Energy Check**: Measures RMS amplitude against `SIP_ENERGY_THRESHOLD` (`450`) and `SIP_NOISE_GATE_ENERGY` (`320`), ignoring zeroed or constant-byte RTP frames. |
| **BargeInController** (`FrameProcessor`) | `1-voice-core/services/barge_in.py` | **False-Trigger-Protected Interruption (`< 300ms`)**: Tracks `LISTENING -> THINKING -> SPEAKING` states. Requires sustained speech `>= BARGE_IN_MIN_SPEECH_MS` (`250ms`) while the bot is `THINKING` or `SPEAKING` before emitting `InterruptionFrame` + `CancelFrame` to halt TTS, flush audio, and trigger history `[interrupted]` marking. |

---

## 5. Text-to-Speech (TTS) & Adaptive Routing

| Technology | File | Exact Technical Use Cases |
| :--- | :--- | :--- |
| **SmartTTSRouter** | `1-voice-core/services/smart_tts.py` | **Adaptive Hybrid TTS Orchestrator (`TTS_ROUTER_MODE`)**:<br>- `hybrid` (default): Routes English turns to **Kokoro-80M** and non-English/multilingual turns to **Chatterbox**.<br>- `kokoro`: Forces Kokoro-80M for English/ASCII text, falling back to Chatterbox for non-Latin scripts.<br>- `chatterbox`: Uses Chatterbox exclusively.<br>- **Pre-TTS Phone Formatting**: Calls `format_phone_for_tts()` so phone numbers are spoken digit-by-digit instead of as billions. |
| **Kokoro-80M Neural TTS** | `1-voice-core/services/kokoro_tts.py` | **Low-Latency English Voice (`af_heart`, 24kHz)**:<br>- Connects to remote HTTP server (`KOKORO_BASE_URL=http://10.20.1.75:8880`) via `/tts/synthesize` (PCM/WAV) and `/tts/stream` (4KB chunked HTTP streaming).<br>- **In-Memory Cache**: Caches up to 200 synthesized phrases (`< 250` chars) keyed by text/voice/speed/lang.<br>- **Local ONNX Fallback**: Uses `onnxruntime` (`CUDAExecutionProvider` -> `CPUExecutionProvider`) + `kokoro_onnx` (`kokoro-v1.0.onnx`, `voices-v1.0.bin`) if the remote server is unreachable. |
| **Chatterbox / `edge-tts` (Azure Neural)** | `1-voice-core/services/chatterbox_tts.py`, `4-calling-agent/audio_bridge.py` | **Multilingual Streaming TTS**: Streams `audio/mp3` chunks via `edge_tts.Communicate` for `en` (`en-US-JennyNeural`), `hi`/`hinglish` (`hi-IN-SwaraNeural`), `es` (`es-ES-ElviraNeural`), `de` (`de-DE-KatjaNeural`), and `zh` (`zh-CN-XiaoxiaoNeural`). Supports `InterruptionFrame`/`CancelFrame` cancellation. |

---

## 6. Real-Time Pipeline, Web Server & Browser UI

| Technology | File(s) | Exact Technical Use Cases |
| :--- | :--- | :--- |
| **Pipecat (`pipecat-ai`)** | `1-voice-core/pipeline.py`, `services/*` | **Frame-Based Streaming Pipeline**: Connects `FastAPIWebsocketTransport` (with `ProtobufFrameSerializer`) -> `VADProcessor` -> `VesperSTTService`/`OmniSTTService` -> `BargeInController` -> `LanguageManager` -> `OpenAILLMService` -> `SentenceAggregator` (clause-level chunking) -> TTS -> Transport Output. Includes custom `LatencyTracker` measuring `T_eos -> T_stt -> T_first_tok -> T_first_audio`. |
| **FastAPI & Uvicorn** | `1-voice-core/pipeline.py`, `2-memory-service/app/main.py` | **API & WebSocket Serving**: Hosts the voice core on `:8765` (`/`, `/dialer`, `/phone`, `/health`, `/ws`, `/api/transcribe`, `/api/speak`, `/api/chat_and_speak`, `/api/config/model`, `/api/telephony/*`, `/api/session_memory`, `/api/clear_memory`) and the memory service on `:8001`. |
| **Browser Web Audio (`getUserMedia`, `MediaRecorder`, `AudioContext`)** | `1-voice-core/client/index.html`, `1-voice-core/client/dialer.html` | **Interactive Test Console & Softphone UI**: Captures microphone audio in the browser, streams/posts audio to the server, and plays back synthesized agent audio. |

---

## 7. Deterministic Guardrails & Booking State Machine

To prevent LLM hallucination and flow skipping during live calls, `1-voice-core` enforces strict deterministic gates around the LLM:

| Component | File | Exact Technical Use Cases |
| :--- | :--- | :--- |
| **InMemoryStateStore** | `1-voice-core/services/state_store.py` | **Single Source of Truth**: Tracks per-call `customer` (`name`, `email`, `phone`, `verified_email`), `trip` (`origin`, `destination`, `date`, `passengers`, `class`, confirmation flags), `search` (`done`, `results`, `selected_flight`, `options_offered`, `shown_flights`), `booking` (`status`, `booking_ref`, `pnr`), and `control.awaiting_field`. |
| **Write Gate** | `1-voice-core/services/write_gate.py` | **Slot-Write Filtering (`filter_extracted`, `option_selection_allowed`)**: Restricts which extracted keys may be written to state based on `control.awaiting_field` so stray utterances cannot overwrite `name`, `phone`, `email`, or lock a flight prematurely. |
| **Flow Enforcer** | `1-voice-core/services/flow_enforcer.py` | **Pre-TTS Reply Override (`enforce_flow`)**: Intercepts and replaces LLM replies if the LLM asks for personal details before `class_stated && pax_count_stated` (Phase A), asks for personal details before flight selection (Phase B), or fabricates flight options when `search.done` is `False` (Phase C). |
| **Claim Guard** | `1-voice-core/services/claim_guard.py` | **Action-Claim Verification (`reply_claims_mail_action`, `reply_claims_updated_email`)**: Regex-detects when the LLM falsely claims *"I have sent the email"*, *"Your booking is confirmed"*, or *"I have updated your email"* so the pipeline can verify against real state/SES status before speaking. |
| **Language Gate & Manager** | `1-voice-core/services/language_gate.py`, `language_manager.py` | **English-Only Enforcement & Bilingual Switch Infra**: `is_non_english_speech()` blocks Devanagari-dominant transcripts (`>= 2` Devanagari chars and `>=` Latin chars) from mutating booking state; `LanguageManager` provides atomic STT/LLM/TTS language switching hooks. |
| **Deterministic Matchers** | `1-voice-core/services/option_match.py`, `weekend_helper.py` | **Option Selection & Calendar Math**: Word-boundary regex (`match_option_selection`) maps *"option one"* / *"cheapest"* vs *"option two"* / *"fastest"* (English & Hindi). `compute_upcoming_saturday()` computes upcoming Saturday dates in Python rather than trusting the LLM. |

---

## 8. Telephony & SIP Integration (`4-calling-agent`)

| Technology | File | Exact Technical Use Cases |
| :--- | :--- | :--- |
| **pyVoIP (`VoIPPhone`, `VoIPCall`, `RTPClient`, `SIPClient`)** | `4-calling-agent/sip_handler.py` | **Enterprise SIP/RTP Call Control**: Registers extension `991` on BigPBX/Asterisk (`crm.bigpbx.com:50040`), auto-answers inbound calls, places outbound calls (caller ID `16672847136`), negotiates `PCMU` (G.711u, payload 0) and `PCMA` (G.711a, payload 8) + RFC 2833 events (101), patches `VoIPCall.renegotiate` to reply to Asterisk 90–120s session-timer `re-INVITE`s at the sender's `Via` IP/port, and supports `SIP REFER` human agent transfers. |
| **`audioop` & `soundfile` (`SIPAudioBridge`)** | `4-calling-agent/audio_bridge.py` | **Bi-Directional Telephony Transcoding**:<br>- **Inbound**: Detects `b"\x00"`/`b"\x80"` silence, removes `+128` bias from 8kHz unsigned 8-bit PCM, converts 8-bit -> 16-bit (`lin2lin`), strips DC offset (`audioop.avg` + `bias`), and resamples 8kHz -> 16kHz (`audioop.ratecv`).<br>- **Outbound**: Resamples 24kHz/16kHz TTS output (including decoding `edge-tts` MP3 via `soundfile`) down to 8kHz, applies peak headroom limiting (`max_amp` capped at `30000` with `gain=1.15`), and encodes back to 8-bit unsigned `+128` biased PCM for `pyVoIP`. |
| **CLI Outbound Dialer (`requests`)** | `dial.py` | **Command-Line Call Trigger**: Posts `{"phone_number": ..., "initial_message": ...}` to `http://127.0.0.1:8765/api/telephony/call` to initiate outbound AI calls. |

---

## 9. Memory & User Profiling Services

| Technology | File(s) | Exact Technical Use Cases |
| :--- | :--- | :--- |
| **Redis (`redis.asyncio`) & `fakeredis`** | `2-memory-service/app/redis_client.py`, `routes.py`, `tests/conftest.py` | **Short-Term Per-Call Memory**:<br>- `session:{call_id}:turns` (`LIST`): Stores turn JSON (`role`, `content`, `timestamp`, `interrupted`, `lang`) via atomic `RPUSH` + `LTRIM` (`MAX_TURNS=40`) + `EXPIRE` (`SESSION_TTL_SECONDS=7200`).<br>- `session:{call_id}:meta` (`HASH`): Tracks `created_at`, `last_active_at`, `profile_id`, and `extra` JSON.<br>- `PATCH /sessions/{call_id}/turns/last`: Marks the last assistant turn with `[interrupted]` on barge-in.<br>- `fakeredis` provides zero-dependency in-memory Redis for automated tests. |
| **API Key Auth & Audit Middleware** | `2-memory-service/app/auth.py`, `main.py` | **Security & Privacy**: Validates `X-API-Key` header and `call_id` format; HTTP audit middleware logs only method, path, status, and latency (never request bodies/transcripts). |
| **ProfileManager (Pydantic v2)** | `3-profile-service/profile_manager.py`, `models.py` | **Long-Term Caller CRM**: Resolves `CallerProfile` (`LoyaltyTier`, `UserPreferences`, `FlightRecord`) by phone number or PNR and formats `[CALLER CRM PROFILE]` context for system prompt injection. |

---

## 10. External APIs, Tools & Notifications

| Technology | File | Exact Technical Use Cases |
| :--- | :--- | :--- |
| **Travomint MetaSearch API (`requests`)** | `1-voice-core/tools/flight_search.py` | **Live Flight Search**: Queries `https://wegoroutes.travomint.com/v1/api/metaSearchResult`, resolves city names (English & Hindi) to 3-letter IATA codes, and selects exactly **two options** (cheapest fare and fastest duration) for voice presentation. |
| **Mock Travel Tools** | `1-voice-core/tools/travel_tools.py` | **Function-Calling Schema & Mock DB**: Provides `get_booking_by_pnr`, `get_flight_status`, `get_refund_policy`, and `cancel_booking` against `MOCK_BOOKINGS` / `MOCK_FLIGHTS` / `MOCK_REFUND_POLICIES`. |
| **AWS SES (`boto3`)** | `1-voice-core/services/mailer.py` | **Booking Confirmation Email**: Generates a responsive HTML flight & passenger summary card (`generate_booking_confirmation_html`) with a `Confirm Booking` link and dispatches it via `boto3.client("ses")` in `loop.run_in_executor()` so the async voice loop is never blocked. |

---

## 11. Configuration, Logging & Testing

| Technology | File(s) | Exact Technical Use Cases |
| :--- | :--- | :--- |
| **`pydantic-settings` & `python-dotenv`** | `config.py`, `1-voice-core/config.py`, `2-memory-service/app/config.py`, `.env` | **Centralized Typed Config**: Loads model endpoints, VAD/barge-in thresholds, TTS routing, SIP credentials, and Redis settings from `.env` with property-based switching (`active_llm_url`, `active_llm_model`). |
| **Python `logging` (`FileHandler` + `StreamHandler`)** | `1-voice-core/pipeline.py`, `4-calling-agent/sip_handler.py` | **Separated Diagnostics**: Writes voice pipeline logs to `logs/loki.log` and SIP/pyVoIP raw packet logs to `logs/telephony.log`. |
| **`pytest`, `pytest-asyncio`, `httpx`** | `1-voice-core/tests/`, `2-memory-service/tests/`, `1-voice-core/scripts/` | **Automated Verification**: Unit/integration tests for booking rules, email correction & SES dispatch, calling-mode memory, and async memory API endpoints, plus standalone scripts for latency (`test_latency.py`), barge-in (`test_barge_in.py`), transcription comparison (`compare_transcription.py`), and SES delivery (`ses_delivery_probe.py`). |

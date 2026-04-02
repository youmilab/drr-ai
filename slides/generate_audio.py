#!/usr/bin/env python3
"""
Generate narration audio for DRR-AI slide decks using ElevenLabs TTS.

Setup:
  pip install requests
  export ELEVENLABS_API_KEY=your_key_here

Usage (from Website/slides/):
  python generate_audio.py

Audio files are saved to Website/slides/audio/.
Skips files that already exist — safe to re-run after adding new slides.
"""

import os, sys, requests

API_KEY = os.environ.get("ELEVENLABS_API_KEY") or input("ElevenLabs API key: ").strip()
if not API_KEY:
    sys.exit("No API key provided.")

# ── Voice selection ────────────────────────────────────────────────────────────
# Uncomment your preferred voice:
# Rachel  (warm, professional female): 21m00Tcm4TlvDq8ikWAM
# Adam    (calm, authoritative male):  pNInz6obpgDQGcFmaJgB
# Aria    (natural female):            9BWtsMINqrJLrRacOk9x
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"   # Rachel
MODEL_ID  = "eleven_turbo_v2_5"

# ── Narration scripts ──────────────────────────────────────────────────────────
SCRIPTS = {

    # ── Project Motivation (7 slides) ──────────────────────────────────────────
    "motivation_s1": (
        "Welcome to DRR-AI. This video explains why automated disclosure risk review "
        "matters — and how DRR-AI supports education researchers navigating the IES "
        "and ICPSR review process."
    ),
    "motivation_s2": (
        "Restricted-use datasets from IES and ICPSR contain far richer information "
        "than public-use data — including detailed individual characteristics, "
        "transcripts, and geographic identifiers. This richness allows researchers to "
        "investigate more nuanced and substantive questions. But because of this "
        "sensitivity, all products must pass a Disclosure Risk Review before "
        "publication or dissemination."
    ),
    "motivation_s3": (
        "Historically, IES reviews took just five to ten business days. Today, due "
        "to significant budget cuts at IES and the Department of Education, "
        "researchers are waiting several months. These delays are seriously disrupting "
        "publication and dissemination timelines across the field."
    ),
    "motivation_s4": (
        "DRR-AI was created to address this bottleneck. It supports researchers and "
        "reviewers in the disclosure risk review process — reducing delays, errors, "
        "and re-submissions. In compliance with restricted-use data license policies, "
        "DRR-AI uses a zero-retention architecture: all reviews happen entirely in "
        "memory, and no user data is ever retained."
    ),
    "motivation_s5": (
        "DRR-AI supports you in three ways. First, as a self-check tool for "
        "researchers — catch common disclosure issues before submitting to official "
        "channels. Second, as support for human reviewers — streamlining internal "
        "DRR processes. And third, through tailored compliance models — supporting "
        "IES/NCES and ICPSR frameworks today, with custom models available on request."
    ),
    "motivation_s6": (
        "An important note: DRR-AI is a complement, not a replacement, for official "
        "DRR processes. It is not a substitute for the formal review required for "
        "publication approval. Instead, it helps researchers self-check their work "
        "and assists reviewers in working more efficiently — toward a stronger, more "
        "efficient education research environment."
    ),
    "motivation_s7": (
        "Try DRR-AI today at Yu-mi Lab dot ai slash drr-ai. Upload your manuscript, "
        "select your compliance framework, and receive a structured audit report — "
        "instantly and securely. Need a custom compliance model for your institution? "
        "Contact us — we would be happy to help."
    ),

    # ── The Privacy Model (6 slides) ───────────────────────────────────────────
    "privacy_s1": (
        "This video explains the DRR-AI privacy model — how your manuscript is "
        "handled securely, with zero data retention and fully in-memory processing."
    ),
    "privacy_s2": (
        "DRR-AI is designed to comply with restricted-use data license policies. "
        "Your manuscript data is never retained, logged, or shared. All reviews are "
        "conducted entirely in memory using an ephemeral, zero-retention architecture "
        "— nothing is written to disk or stored on servers."
    ),
    "privacy_s3": (
        "DRR-AI uses a zero-data-retention API built for commercial enterprise use. "
        "Unlike consumer AI products, these enterprise APIs are governed by legally "
        "binding agreements stating that prompt data is not stored and is not used "
        "to train AI models. Your manuscript content is processed and immediately "
        "discarded by the AI provider."
    ),
    "privacy_s4": (
        "On the backend, DRR-AI parses Word and PDF files directly from the incoming "
        "web request buffer. Files are never written to disk. Review results are "
        "shown only within the active session — once you leave, they are gone. "
        "We log only non-identifying metadata: timestamp, framework used, token "
        "counts, and processing time. No file content is ever logged."
    ),
    "privacy_s5": (
        "Here is the complete flow. Your file is uploaded over HTTPS. Text is "
        "extracted directly from the request buffer — no disk write. A zero-retention "
        "enterprise AI audits the text. The results are returned to your session and "
        "then discarded. At every step, no manuscript content touches a database or disk."
    ),
    "privacy_s6": (
        "If you have any questions about our privacy model, please contact us at "
        "Yu-mi Lab dot ai slash drr-ai. You can also watch our other videos — "
        "How to Use DRR-AI and Live Audit Demo — to see the system in practice."
    ),
}

# ── Generate ───────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
os.makedirs(OUT_DIR, exist_ok=True)

def generate(key, text):
    out_path = os.path.join(OUT_DIR, key + ".mp3")
    if os.path.exists(out_path):
        print(f"  skip  {key}.mp3  (already exists)")
        return
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": API_KEY,
    }
    payload = {
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    if r.status_code != 200:
        print(f"  ERROR {key}: HTTP {r.status_code} — {r.text[:300]}", file=sys.stderr)
        return
    with open(out_path, "wb") as f:
        f.write(r.content)
    print(f"  done  {key}.mp3  ({len(r.content) // 1024} KB)")

print(f"Generating {len(SCRIPTS)} audio clips → {OUT_DIR}\n")
for key, text in SCRIPTS.items():
    generate(key, text)
print("\nAll done.")

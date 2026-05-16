from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from collections import Counter
import re, math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Tone phrase dictionary ────────────────────────────────────────────────────
TONE_PHRASES = [
    "slow burn", "rewatch forever", "just finished", "devastating", "masterpiece",
    "emotionally devastating", "gut punch", "haunting", "dark and bleak", "bingeable",
    "lives rent free", "beautiful", "perfect", "underrated", "hidden gem",
    "ending", "finale", "dark", "funny", "emotional",
]

# Lifecycle bucket thresholds (months since release)
BUCKETS = {
    "release":     (0, 6),
    "settled":     (6, 24),
    "rediscovery": (24, 60),
    "legacy":      (60, 9999),
}

STOPWORDS = {
    "the","a","an","is","are","was","were","it","its","i","my","me","we","you",
    "he","she","they","this","that","and","but","or","so","if","in","on","at",
    "to","of","for","with","be","been","being","have","has","had","do","did",
    "not","no","just","very","really","like","one","more","some","all","https",
    "www","spoilers","spoiler","season","episode","show","series","watch","watched",
    "watching","netflix","hbo","amazon","apple","re","would","could","should","get",
}

SENTIMENT_WORDS = {
    "positive": {"masterpiece","perfect","brilliant","beautiful","amazing","incredible",
                 "stunning","love","loved","best","greatest","underrated","hidden","gem",
                 "emotional","powerful","haunting","devastating","slow burn","rewatch"},
    "negative": {"boring","slow","bad","worst","terrible","awful","disappointing",
                 "overrated","trash","garbage","waste","drop"},
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text.lower()

def get_sentiment(words: list[str]) -> str:
    pos = sum(1 for w in words if w in SENTIMENT_WORDS["positive"])
    neg = sum(1 for w in words if w in SENTIMENT_WORDS["negative"])
    if pos > neg: return "positive"
    if neg > pos: return "negative"
    return "neutral"

def top_word(texts: list[str]) -> str:
    all_words = []
    for t in texts:
        words = clean_text(t).split()
        all_words.extend(w for w in words if len(w) > 3 and w not in STOPWORDS)
    if not all_words:
        return ""
    most_common = Counter(all_words).most_common(1)
    return most_common[0][0].upper() if most_common else ""

def count_mentions(texts: list[str], patterns: list[str]) -> int:
    combined = " ".join(texts).lower()
    return sum(1 for p in patterns if p in combined)

# ── Request / Response models ─────────────────────────────────────────────────

class PostIn(BaseModel):
    text: str
    score: Optional[int] = 0
    created_utc: Optional[str] = None

class ProcessRequest(BaseModel):
    show_name: str
    posts: list[PostIn]
    release_year: Optional[int] = None

# ── Main endpoint ─────────────────────────────────────────────────────────────

@app.post("/process")
def process_show(req: ProcessRequest):
    posts = req.posts
    texts = [p.text for p in posts if p.text and len(p.text) > 10]

    if not texts:
        raise HTTPException(status_code=400, detail="No usable text")

    # ── Tone tags ──
    phrase_counts = {}
    combined = " ".join(texts).lower()
    for phrase in TONE_PHRASES:
        count = combined.count(phrase)
        if count > 0:
            phrase_counts[phrase] = count
    tone_tags = [p for p, _ in sorted(phrase_counts.items(), key=lambda x: -x[1])][:6]
    if not tone_tags:
        tone_tags = ["just finished", "rewatch forever", "emotional"]

    # ── Honest stats ──
    finish_patterns  = ["finished", "just finished", "completed", "watched all", "binged"]
    rewatch_patterns = ["rewatch", "rewatched", "rewatching", "third time", "fourth time"]

    finished_count  = count_mentions(texts, finish_patterns)
    rewatch_count   = count_mentions(texts, rewatch_patterns)
    total           = len(texts)

    finished_pct  = min(round((finished_count  / total) * 100 * 3), 95)
    rewatch_pct   = min(round((rewatch_count   / total) * 100 * 4), 80)

    all_words = []
    for t in texts:
        words = clean_text(t).split()
        all_words.extend(w for w in words if len(w) > 3 and w not in STOPWORDS)
    word_counts = Counter(all_words)

    # Sentiment
    pos_count = sum(c for w, c in word_counts.items() if w in SENTIMENT_WORDS["positive"])
    neg_count = sum(c for w, c in word_counts.items() if w in SENTIMENT_WORDS["negative"])
    total_sentiment = pos_count + neg_count or 1
    overall_positive_pct = round((pos_count / total_sentiment) * 100)

    top = word_counts.most_common(5)
    tw = next((w.upper() for w, _ in top if w not in STOPWORDS and len(w) > 3), "")

    # ── Lifecycle ──
    from datetime import datetime, timezone

    lifecycle = {"release": None, "settled": None, "rediscovery": None, "legacy": None}
    release_year = req.release_year

    # Group posts by bucket if we have timestamps
    bucket_texts = {b: [] for b in BUCKETS}
    for p in posts:
        if not p.created_utc or not release_year:
            continue
        try:
            created = datetime.fromisoformat(str(p.created_utc).replace("Z", "+00:00"))
            months_since = (created.year - release_year) * 12 + created.month
            for bucket, (lo, hi) in BUCKETS.items():
                if lo <= months_since < hi:
                    bucket_texts[bucket].append(p.text)
                    break
        except Exception:
            continue

    for bucket, btexts in bucket_texts.items():
        if len(btexts) < 3:
            continue
        words = []
        for t in btexts:
            words.extend(w for w in clean_text(t).split() if len(w) > 3 and w not in STOPWORDS)
        if not words:
            continue
        dominant = Counter(words).most_common(1)[0][0]
        sentiment = get_sentiment(words)
        lifecycle[bucket] = {"dominant_word": dominant, "sentiment": sentiment}

    # Fallback: if no timestamps, infer from tone tags
    if all(v is None for v in lifecycle.values()):
        if "masterpiece" in tone_tags or "greatest" in " ".join(tone_tags):
            lifecycle["legacy"] = {"dominant_word": "masterpiece", "sentiment": "positive"}
        if "underrated" in tone_tags or "hidden gem" in tone_tags:
            lifecycle["rediscovery"] = {"dominant_word": "underrated", "sentiment": "positive"}
        lifecycle["settled"] = {"dominant_word": "classic", "sentiment": "positive"}

    return {
        "tone_tags": tone_tags,
        "honest_stats": {
            "finished_pct":        finished_pct,
            "rewatch_pct":         rewatch_pct,
            "top_word":            tw,
            "overall_positive_pct": overall_positive_pct,
        },
        "lifecycle": lifecycle,
    }

@app.get("/health")
def health():
    return {"status": "ok"}

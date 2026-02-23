from __future__ import annotations

import re

# Sentence-level punctuation for all chunks.
_SENTENCE_END_RE = re.compile(r"([。！？!?；：]+|\.{3,}|[\r\n]+)")
# Clause-level punctuation for faster first-chunk emission in Chinese.
_FIRST_CHUNK_CLAUSE_RE = re.compile(r"([，、；]+)")


class SentenceChunker:
    """Accumulate token deltas and emit chunks.

    Primary split: sentence punctuation.
    Fallback split: when no punctuation appears for a while, emit a soft chunk
    near whitespace once `soft_limit_chars` is reached.
    """

    def __init__(self, soft_limit_chars: int = 80) -> None:
        self._buf = ""
        self._soft_limit_chars = max(20, soft_limit_chars)
        self._first_soft_limit_chars = 12
        self.first_chunk_emitted = False

    def push(self, delta: str) -> list[str]:
        if not delta:
            return []
        self._buf += delta
        out: list[str] = []

        # First chunk: aggressively emit on short clauses to reduce time-to-first-audio.
        split_re = _FIRST_CHUNK_CLAUSE_RE if not self.first_chunk_emitted else _SENTENCE_END_RE

        cursor = 0
        for m in split_re.finditer(self._buf):
            end_idx = m.end()
            segment = self._buf[cursor:end_idx].strip()
            if segment:
                out.append(segment)
                if not self.first_chunk_emitted:
                    self.first_chunk_emitted = True
                    cursor = end_idx
                    break
            cursor = end_idx

        if cursor > 0:
            self._buf = self._buf[cursor:]

        # Flush remaining sentence boundaries for normal flow after first chunk is emitted.
        if self.first_chunk_emitted and self._buf:
            cursor = 0
            for m in _SENTENCE_END_RE.finditer(self._buf):
                end_idx = m.end()
                segment = self._buf[cursor:end_idx].strip()
                if segment:
                    out.append(segment)
                cursor = end_idx
            if cursor > 0:
                self._buf = self._buf[cursor:]

        # Soft chunk fallback when long run has no sentence punctuation.
        active_soft_limit = self._first_soft_limit_chars if not self.first_chunk_emitted else self._soft_limit_chars
        if len(self._buf) >= active_soft_limit:
            split_at = -1

            # Prefer Chinese pause punctuation for long Chinese text.
            puncts = ("，", "、", "；") if not self.first_chunk_emitted else ("，", "、")
            for punct in puncts:
                idx = self._buf.rfind(punct, 0, active_soft_limit)
                if idx > split_at:
                    split_at = idx
            if split_at >= 0:
                split_at += 1  # keep punctuation in current chunk

            # Then try whitespace.
            if split_at < 0:
                split_at = self._buf.rfind(" ", 0, active_soft_limit)

            # Last resort hard cut.
            if split_at < 0:
                split_at = active_soft_limit

            segment = self._buf[:split_at].strip()
            if segment:
                out.append(segment)
                if not self.first_chunk_emitted:
                    self.first_chunk_emitted = True
            self._buf = self._buf[split_at:].lstrip()

        return out

    def flush(self) -> str:
        tail = self._buf.strip()
        self._buf = ""
        return tail

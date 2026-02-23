import os
import tempfile

# Keep tests deterministic and avoid background warmup threads during teardown.
os.environ.setdefault("STT_WARMUP_ENABLED", "false")

# Isolate durable memory file per test run.
_test_memory_dir = tempfile.mkdtemp(prefix="ani-test-memory-")
os.environ.setdefault("MEMORY_PERSIST_PATH", os.path.join(_test_memory_dir, "memory_store.json"))

from app.config import config  # noqa: E402

config.stt_warmup_enabled = False
config.stt_language_hint = "auto"

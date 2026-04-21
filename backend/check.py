"""
check.py — Preflight health checker for The Stylist's Brain
Run before starting the server to catch missing deps, models, or services.

Usage:
    python check.py           # verbose output
    python check.py --quiet   # suppress output, exit code only
"""

import sys
import os

QUIET = "--quiet" in sys.argv

_ok    = []
_warn  = []
_fail  = []


def check(label: str, fn):
    try:
        result = fn()
        msg = result if isinstance(result, str) else "OK"
        _ok.append((label, msg))
        if not QUIET:
            print(f"  \033[32m✓\033[0m {label:<40} {msg}")
    except Exception as exc:
        _fail.append((label, str(exc)))
        if not QUIET:
            print(f"  \033[31m✗\033[0m {label:<40} {exc}")


def warn(label: str, fn):
    try:
        result = fn()
        msg = result if isinstance(result, str) else "OK"
        _ok.append((label, msg))
        if not QUIET:
            print(f"  \033[32m✓\033[0m {label:<40} {msg}")
    except Exception as exc:
        _warn.append((label, str(exc)))
        if not QUIET:
            print(f"  \033[33m⚠\033[0m {label:<40} {exc} (optional)")


if not QUIET:
    print("\n\033[1mThe Stylist's Brain — Preflight Check\033[0m\n")

# ── Python ─────────────────────────────────────────────────────────────────────
check("Python ≥ 3.11", lambda: (
    f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 11)
    else (_ for _ in ()).throw(RuntimeError(f"Need 3.11+, got {sys.version_info.major}.{sys.version_info.minor}"))
))

# ── Core packages ──────────────────────────────────────────────────────────────
check("fastapi",            lambda: __import__("fastapi").__version__)
check("uvicorn",            lambda: __import__("uvicorn").__version__)
check("httpx",              lambda: __import__("httpx").__version__)
check("chromadb",           lambda: __import__("chromadb").__version__)
check("ultralytics (YOLO)", lambda: __import__("ultralytics").__version__)
check("open_clip",          lambda: __import__("open_clip").__version__)
check("torch",              lambda: __import__("torch").__version__)
check("PIL (Pillow)",       lambda: __import__("PIL").__version__)
check("cv2 (OpenCV)",       lambda: __import__("cv2").__version__)
check("numpy",              lambda: __import__("numpy").__version__)

warn("mobile_sam",          lambda: __import__("mobile_sam") and "installed")
warn("google-auth",         lambda: __import__("google.oauth2") and "installed")

# ── Model weights ──────────────────────────────────────────────────────────────
from pathlib import Path
MODELS = Path(__file__).parent.parent / "models"

check("models/ directory",  lambda: str(MODELS) if MODELS.exists() else (_ for _ in ()).throw(FileNotFoundError(f"Missing: {MODELS}")))
warn("yolov8n.pt",          lambda: f"{(MODELS/'yolov8n.pt').stat().st_size//1024} KB" if (MODELS/"yolov8n.pt").exists() else (_ for _ in ()).throw(FileNotFoundError("Will auto-download on first intake")))
warn("mobile_sam.pt",       lambda: f"{(MODELS/'mobile_sam.pt').stat().st_size//1024} KB" if (MODELS/"mobile_sam.pt").exists() else (_ for _ in ()).throw(FileNotFoundError("Run setup.sh to download")))

# ── ChromaDB write test ────────────────────────────────────────────────────────
def _chroma_test():
    import chromadb
    from chromadb.config import Settings
    store = Path(__file__).parent.parent / "chroma_store"
    store.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(store), settings=Settings(anonymized_telemetry=False))
    col = client.get_or_create_collection("preflighttest")   # no leading underscore — ChromaDB requirement
    col.upsert(ids=["test1"], embeddings=[[0.0] * 512], metadatas=[{"ok": "1"}])
    client.delete_collection("preflighttest")
    return "read/write OK"

check("ChromaDB (disk r/w)", _chroma_test)

# ── SQLite write test ──────────────────────────────────────────────────────────
def _sqlite_test():
    import sqlite3
    db = Path(__file__).parent.parent / "stylist.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS _preflight (id INTEGER PRIMARY KEY)")
    conn.execute("INSERT OR IGNORE INTO _preflight VALUES (1)")
    conn.execute("DROP TABLE _preflight")
    conn.commit(); conn.close()
    return "read/write OK"

check("SQLite (disk r/w)",   _sqlite_test)

# ── Ollama ─────────────────────────────────────────────────────────────────────
def _ollama_test():
    import httpx
    r = httpx.get("http://localhost:11434/api/tags", timeout=5)
    r.raise_for_status()
    models = [m["name"] for m in r.json().get("models", [])]
    llm_models = [m for m in models if any(n in m for n in ("llama3", "mistral"))]
    if not llm_models:
        raise RuntimeError("No llama3:8b or mistral:7b found — run: ollama pull llama3:8b")
    return f"running | models: {', '.join(llm_models)}"

warn("Ollama server",        _ollama_test)

# ── OWM API key ────────────────────────────────────────────────────────────────
warn("OWM_API_KEY env var",  lambda: "set" if os.getenv("OWM_API_KEY") else (_ for _ in ()).throw(RuntimeError("Not set — weather will use fallback. See .env.example")))

# ── Summary ────────────────────────────────────────────────────────────────────
if not QUIET:
    print(f"\n  {'─'*52}")
    print(f"  \033[32m✓ {len(_ok)} passed\033[0m  "
          f"\033[33m⚠ {len(_warn)} warnings\033[0m  "
          f"\033[31m✗ {len(_fail)} failed\033[0m")
    if _fail:
        print("\n  \033[31mFailed checks must be fixed before the server will start.\033[0m")
        print("  Run \033[1mbash setup.sh\033[0m to install missing dependencies.\n")
    elif _warn:
        print("\n  \033[33mWarnings are optional — the core app will still work.\033[0m\n")
    else:
        print("\n  \033[32mAll checks passed — ready to run!\033[0m\n")

sys.exit(1 if _fail else 0)

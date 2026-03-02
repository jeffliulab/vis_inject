"""
Robust LAION-Art image downloader -- no img2dataset dependency.

Downloads images from LAION-Art parquet metadata using only standard
Python libraries + pyarrow + Pillow. Designed for HPC environments
where installing img2dataset is problematic.

Features:
  - Resume support: tracks progress in a JSON state file
  - Retry with exponential backoff
  - Parallel downloads via ThreadPoolExecutor
  - Automatic image resizing and center-crop to target size
  - WebDataset .tar output format (compatible with pretrain.py)
  - Graceful handling of dead URLs, timeouts, corrupted images
  - Comprehensive logging (console + file + failed URLs + error stats)
  - --test-run mode for quick validation
  - Signal handling for graceful shutdown

Usage:
    python download_images.py
    python download_images.py --test-run                       # download 100 images to test
    python download_images.py --workers 16 --resume            # resume interrupted download
    python download_images.py --output-format folder           # save as individual files
    python download_images.py --start-shard 50 --end-shard 100 # download specific range
"""

import argparse
import datetime
import io
import json
import logging
import os
import platform
import shutil
import signal
import socket
import sys
import tarfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str, run_id: str) -> logging.Logger:
    """
    Create a logger that writes to:
      1. Console (INFO level, compact format)
      2. Main log file (DEBUG level, full timestamps + module)
      3. Failed-URL log file (one line per failure with reason)
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("laion_downloader")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)-5s  %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(console)

    main_log = os.path.join(log_dir, f"download_{run_id}.log")
    fh = logging.FileHandler(main_log, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(funcName)-22s | %(message)s"
    ))
    logger.addHandler(fh)

    logger.info(f"Main log file   : {main_log}")
    return logger


def setup_fail_log(log_dir: str, run_id: str) -> logging.Logger:
    """Separate logger that records every failed URL with its error."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    fail_logger = logging.getLogger("laion_fail")
    fail_logger.setLevel(logging.DEBUG)
    fail_logger.handlers.clear()
    fail_logger.propagate = False

    fail_log = os.path.join(log_dir, f"failed_urls_{run_id}.log")
    fh = logging.FileHandler(fail_log, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    fail_logger.addHandler(fh)

    logging.getLogger("laion_downloader").info(f"Failed-URL log  : {fail_log}")
    return fail_logger

# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

class ErrorStats:
    """Categorizes and counts download errors for diagnostics."""

    CATEGORIES = [
        "timeout", "dns_failure", "connection_refused", "connection_reset",
        "ssl_error", "http_404", "http_403", "http_other",
        "invalid_url", "empty_response", "image_corrupt", "image_too_small",
        "pillow_error", "unknown",
    ]

    def __init__(self):
        self.counts = {cat: 0 for cat in self.CATEGORIES}
        self.recent_errors: list = []   # last N errors for the diagnostic dump
        self._max_recent = 50

    def record(self, category: str, url: str = "", detail: str = ""):
        if category not in self.counts:
            category = "unknown"
        self.counts[category] += 1
        entry = {
            "time": datetime.datetime.now().isoformat(),
            "category": category,
            "url": url[:200],
            "detail": detail[:300],
        }
        self.recent_errors.append(entry)
        if len(self.recent_errors) > self._max_recent:
            self.recent_errors.pop(0)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def summary_lines(self) -> list:
        lines = ["Error breakdown:"]
        for cat in self.CATEGORIES:
            cnt = self.counts[cat]
            if cnt > 0:
                lines.append(f"  {cat:25s}: {cnt:>8,}")
        lines.append(f"  {'TOTAL':25s}: {self.total:>8,}")
        return lines

    def to_dict(self) -> dict:
        return {"counts": self.counts, "recent_errors": self.recent_errors}

# ---------------------------------------------------------------------------
# System diagnostics
# ---------------------------------------------------------------------------

def log_system_info(logger: logging.Logger, output_dir: str):
    """Log system/environment info for debugging HPC issues."""
    logger.info("=" * 60)
    logger.info("SYSTEM DIAGNOSTICS")
    logger.info("=" * 60)
    logger.info(f"Hostname        : {socket.gethostname()}")
    logger.info(f"Platform        : {platform.platform()}")
    logger.info(f"Python          : {sys.version}")
    logger.info(f"Python path     : {sys.executable}")
    logger.info(f"PID             : {os.getpid()}")
    logger.info(f"CWD             : {os.getcwd()}")
    logger.info(f"Timestamp       : {datetime.datetime.now().isoformat()}")

    # SLURM environment
    slurm_vars = {k: v for k, v in os.environ.items() if k.startswith("SLURM")}
    if slurm_vars:
        logger.info("SLURM environment:")
        for k, v in sorted(slurm_vars.items()):
            logger.info(f"  {k} = {v}")
    else:
        logger.info("SLURM           : not detected (running outside SLURM)")

    # Disk space
    try:
        usage = shutil.disk_usage(output_dir if os.path.exists(output_dir)
                                  else os.path.expanduser("~"))
        logger.info(f"Disk total      : {usage.total / (1024**3):.1f} GB")
        logger.info(f"Disk free       : {usage.free / (1024**3):.1f} GB")
        logger.info(f"Disk used       : {usage.used / (1024**3):.1f} GB "
                     f"({usage.used / usage.total * 100:.1f}%)")
    except Exception as e:
        logger.warning(f"Disk check failed: {e}")

    # Network connectivity
    logger.info("Network connectivity tests:")
    test_hosts = [
        ("huggingface.co", 443),
        ("cdn-lfs.huggingface.co", 443),
        ("8.8.8.8", 53),
    ]
    for host, port in test_hosts:
        try:
            t0 = time.time()
            sock = socket.create_connection((host, port), timeout=5)
            sock.close()
            ms = (time.time() - t0) * 1000
            logger.info(f"  {host}:{port}  OK  ({ms:.0f}ms)")
        except Exception as e:
            logger.warning(f"  {host}:{port}  FAILED  ({type(e).__name__}: {e})")

    # Key Python packages
    logger.info("Python packages:")
    for pkg in ["pyarrow", "PIL", "pandas", "numpy"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            logger.info(f"  {pkg:15s}: {ver}")
        except ImportError:
            logger.warning(f"  {pkg:15s}: NOT INSTALLED")

    logger.info("=" * 60)


def log_config(logger: logging.Logger, args):
    """Log all configuration parameters."""
    logger.info("CONFIGURATION")
    logger.info("-" * 40)
    for k, v in sorted(vars(args).items()):
        logger.info(f"  {k:20s}: {v}")
    logger.info("-" * 40)

# ---------------------------------------------------------------------------
# Image downloading
# ---------------------------------------------------------------------------

def classify_download_error(e: Exception) -> str:
    """Map an exception to an error category."""
    msg = str(e).lower()
    etype = type(e).__name__

    if isinstance(e, HTTPError):
        code = e.code
        if code == 404:
            return "http_404"
        elif code == 403:
            return "http_403"
        return "http_other"
    if isinstance(e, TimeoutError) or "timed out" in msg:
        return "timeout"
    if "ssl" in msg or "certificate" in msg:
        return "ssl_error"
    if "name or service not known" in msg or "getaddrinfo" in msg:
        return "dns_failure"
    if "connection refused" in msg or "errno 111" in msg:
        return "connection_refused"
    if "connection reset" in msg or "errno 104" in msg:
        return "connection_reset"
    return "unknown"


def download_single_image(url: str, timeout: int = 15,
                          max_retries: int = 2) -> Tuple[Optional[bytes], str]:
    """
    Download a single image with retry logic.

    Returns (image_bytes, error_category).
    image_bytes is None on failure; error_category is "" on success.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; LAION-Art-Downloader/1.0)"
    }

    if not url or not url.startswith("http"):
        return None, "invalid_url"

    last_error_cat = "unknown"

    for attempt in range(max_retries + 1):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    return None, f"http_other"
                data = resp.read()
                if len(data) < 100:
                    return None, "empty_response"
                return data, ""
        except (URLError, HTTPError, TimeoutError, OSError,
                ConnectionError, ValueError) as e:
            last_error_cat = classify_download_error(e)
            if attempt < max_retries:
                time.sleep(0.5 * (2 ** attempt))
            continue
        except Exception as e:
            return None, "unknown"

    return None, last_error_cat


def resize_and_crop(image_bytes: bytes, target_size: int,
                    encode_quality: int = 95) -> Tuple[Optional[bytes], str]:
    """
    Resize image to target_size x target_size with center crop.

    Returns (jpeg_bytes, error_category).
    """
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")

        w, h = img.size
        if w < 10 or h < 10:
            return None, "image_too_small"

        scale = target_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        left = (new_w - target_size) // 2
        top = (new_h - target_size) // 2
        img = img.crop((left, top, left + target_size, top + target_size))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=encode_quality)
        return buf.getvalue(), ""

    except Exception as e:
        err_msg = str(e).lower()
        if "cannot identify" in err_msg or "truncated" in err_msg:
            return None, "image_corrupt"
        return None, "pillow_error"


def process_row(row: dict, target_size: int, timeout: int) -> dict:
    """
    Download and process a single row.

    Always returns a result dict (with success=True/False) so errors
    are never silently swallowed.
    """
    url = row.get("URL", "")

    image_bytes, dl_err = download_single_image(url, timeout=timeout)
    if image_bytes is None:
        return {"success": False, "url": url, "error": dl_err}

    processed, proc_err = resize_and_crop(image_bytes, target_size)
    if processed is None:
        return {"success": False, "url": url, "error": proc_err,
                "raw_size": len(image_bytes)}

    return {
        "success": True,
        "url": url,
        "error": "",
        "image_bytes": processed,
        "caption": row.get("TEXT", ""),
        "metadata": {
            "similarity": row.get("similarity"),
            "aesthetic": row.get("aesthetic"),
            "punsafe": row.get("punsafe"),
            "pwatermark": row.get("pwatermark"),
            "hash": row.get("hash"),
        }
    }

# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

class ShardWriter:
    """Writes downloaded images into WebDataset .tar shards."""

    def __init__(self, output_dir: str, shard_size: int = 10000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.current_shard_id = -1
        self.current_count = 0
        self.current_tar = None
        self.total_written = 0

    def _open_shard(self, shard_id: int):
        if self.current_tar is not None:
            self.current_tar.close()
        path = self.output_dir / f"{shard_id:05d}.tar"
        self.current_tar = tarfile.open(str(path), "w")
        self.current_shard_id = shard_id
        self.current_count = 0

    def _add_bytes(self, name: str, data: bytes):
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        info.mtime = int(time.time())
        self.current_tar.addfile(info, io.BytesIO(data))

    def write(self, sample: dict, global_idx: int):
        target_shard = global_idx // self.shard_size
        if self.current_tar is None or target_shard != self.current_shard_id:
            self._open_shard(target_shard)

        prefix = f"{global_idx:09d}"
        self._add_bytes(f"{prefix}.jpg", sample["image_bytes"])
        self._add_bytes(f"{prefix}.txt", sample["caption"].encode("utf-8"))
        meta_json = json.dumps(sample["metadata"], ensure_ascii=False)
        self._add_bytes(f"{prefix}.json", meta_json.encode("utf-8"))

        self.current_count += 1
        self.total_written += 1

    def close(self):
        if self.current_tar is not None:
            self.current_tar.close()
            self.current_tar = None


class FolderWriter:
    """Writes downloaded images as individual files."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "captions").mkdir(parents=True, exist_ok=True)
        self.total_written = 0

    def write(self, sample: dict, global_idx: int):
        prefix = f"{global_idx:09d}"
        with open(self.output_dir / "images" / f"{prefix}.jpg", "wb") as f:
            f.write(sample["image_bytes"])
        with open(self.output_dir / "captions" / f"{prefix}.txt", "w",
                  encoding="utf-8") as f:
            f.write(sample["caption"])
        self.total_written += 1

    def close(self):
        pass

# ---------------------------------------------------------------------------
# Progress tracking with resume
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Tracks download progress with resume support and periodic snapshots."""

    def __init__(self, state_file: str):
        self.state_file = Path(state_file)
        self.state = {
            "completed_rows": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "total_bytes": 0,
            "start_time": time.time(),
            "last_update": time.time(),
            "session_history": [],
        }
        if self.state_file.exists():
            with open(self.state_file) as f:
                saved = json.load(f)
                self.state.update(saved)

    def update(self, success: bool, image_size: int = 0):
        self.state["completed_rows"] += 1
        if success:
            self.state["successful_downloads"] += 1
            self.state["total_bytes"] += image_size
        else:
            self.state["failed_downloads"] += 1
        self.state["last_update"] = time.time()

    def save(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def record_session_start(self, info: dict):
        self.state["session_history"].append({
            "started": datetime.datetime.now().isoformat(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            **info,
        })

    @property
    def completed(self) -> int:
        return self.state["completed_rows"]

    @property
    def successful(self) -> int:
        return self.state["successful_downloads"]

    @property
    def failed(self) -> int:
        return self.state["failed_downloads"]

    def format_progress(self, total_rows: int) -> str:
        elapsed = time.time() - self.state["start_time"]
        rate = self.completed / max(elapsed, 1)
        remaining = (total_rows - self.completed) / max(rate, 0.01)
        pct = self.completed / max(total_rows, 1) * 100
        success_rate = self.successful / max(self.completed, 1) * 100
        size_gb = self.state["total_bytes"] / (1024 ** 3)
        hours_remaining = remaining / 3600

        return (
            f"Progress: {self.completed:,}/{total_rows:,} ({pct:.1f}%) | "
            f"OK: {self.successful:,} | Fail: {self.failed:,} | "
            f"Success rate: {success_rate:.1f}% | "
            f"Size: {size_gb:.2f} GB | "
            f"Speed: {rate:.1f} img/s | "
            f"ETA: {hours_remaining:.1f}h"
        )

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _resolve_hf_token(token_arg: Optional[str],
                      logger: logging.Logger) -> Optional[str]:
    """
    Resolve HuggingFace token from (in priority order):
      1. --hf-token CLI argument
      2. HF_TOKEN environment variable
      3. huggingface-cli saved token (~/.cache/huggingface/token)
    Returns the token string or None.
    """
    if token_arg:
        logger.info("HF token: provided via --hf-token argument")
        return token_arg

    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        logger.info("HF token: found in environment variable")
        return env_token

    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        token = token_file.read_text().strip()
        if token:
            logger.info(f"HF token: loaded from {token_file}")
            return token

    logger.warning("HF token: NOT FOUND. Gated datasets will return 401.")
    logger.warning("  Fix: run 'huggingface-cli login' on the login node, or")
    logger.warning("  set HF_TOKEN env var, or pass --hf-token <token>")
    return None


def download_parquet(parquet_url: str, parquet_path: str,
                     logger: logging.Logger,
                     hf_token: Optional[str] = None) -> str:
    """Download parquet metadata if not present, with HF auth support."""
    path = Path(parquet_path)
    if path.exists() and path.stat().st_size > 100_000_000:
        logger.info(f"Parquet exists: {path} ({path.stat().st_size / 1e6:.0f} MB)")
        return str(path)

    logger.info(f"Downloading parquet metadata from {parquet_url} ...")
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; LAION-Art-Downloader/1.0)"
        }
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
            logger.debug("  Using HF token for authentication")

        req = Request(parquet_url, headers=headers)
        with urlopen(req, timeout=120) as resp:
            with open(str(path), "wb") as f:
                while True:
                    chunk = resp.read(8 * 1024 * 1024)  # 8MB chunks
                    if not chunk:
                        break
                    f.write(chunk)

        logger.info(f"Parquet saved: {path} ({path.stat().st_size / 1e6:.0f} MB)")
    except HTTPError as e:
        logger.error(f"Parquet download FAILED: HTTP {e.code}")
        logger.error(traceback.format_exc())
        if e.code == 401:
            logger.error("")
            logger.error("=" * 50)
            logger.error("  401 Unauthorized -- HuggingFace token required!")
            logger.error("")
            logger.error("  LAION-Art is a gated dataset. To fix this:")
            logger.error("  1. Go to https://huggingface.co/datasets/laion/laion-art")
            logger.error("     and accept the dataset terms")
            logger.error("  2. Create a token at https://huggingface.co/settings/tokens")
            logger.error("  3. Then do ONE of the following:")
            logger.error("     a) On HPC login node: huggingface-cli login")
            logger.error("     b) Set env var: export HF_TOKEN=hf_xxx...")
            logger.error("     c) Pass argument: --hf-token hf_xxx...")
            logger.error("=" * 50)
        if path.exists():
            path.unlink()
        raise
    except Exception as e:
        logger.error(f"Parquet download FAILED: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        if path.exists():
            path.unlink()
        raise

    return str(path)


def load_parquet_rows(parquet_path: str, start: int = 0,
                      end: Optional[int] = None,
                      logger: logging.Logger = None) -> list:
    """Load rows from parquet file as list of dicts."""
    import pyarrow.parquet as pq

    logger.info(f"Loading parquet: {parquet_path}")
    table = pq.read_table(parquet_path)
    logger.info(f"  Total rows in parquet : {table.num_rows:,}")
    logger.info(f"  Columns              : {table.column_names}")

    df = table.to_pandas()

    # Log sample URLs for debugging
    if "URL" in df.columns and len(df) > 0:
        sample_urls = df["URL"].head(3).tolist()
        logger.debug("  Sample URLs:")
        for u in sample_urls:
            logger.debug(f"    {u}")

    if end is not None:
        df = df.iloc[start:end]
    elif start > 0:
        df = df.iloc[start:]

    rows = df.to_dict("records")
    logger.info(f"  Selected range: [{start}:{end or 'end'}] = {len(rows):,} rows")
    return rows

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown_requested = False

def _signal_handler(signum, frame):
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    logger = logging.getLogger("laion_downloader")
    logger.warning(f"Received {sig_name} -- will finish current batch then exit")
    _shutdown_requested = True

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    global _shutdown_requested

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_dir or os.path.join(args.output_dir, "logs")

    logger = setup_logging(log_dir, run_id)
    fail_logger = setup_fail_log(log_dir, run_id)
    error_stats = ErrorStats()

    # Register signal handlers
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    logger.info("=" * 60)
    logger.info("LAION-Art Robust Image Downloader")
    logger.info("=" * 60)

    # System diagnostics
    log_system_info(logger, args.output_dir)
    log_config(logger, args)

    # Resolve HuggingFace token
    hf_token = _resolve_hf_token(args.hf_token, logger)

    # Download parquet metadata
    try:
        parquet_path = download_parquet(args.parquet_url, args.parquet_path,
                                        logger, hf_token=hf_token)
    except Exception:
        logger.critical("Cannot proceed without parquet metadata. Exiting.")
        sys.exit(1)

    # Determine row range
    start = args.start_shard * args.shard_size if args.start_shard else 0
    end_val = args.end_shard * args.shard_size if args.end_shard else None

    if args.test_run:
        end_val = start + args.test_count
        logger.info(f"[TEST RUN] Limiting to {args.test_count} images")

    rows = load_parquet_rows(parquet_path, start, end_val, logger)
    total = len(rows)

    if total == 0:
        logger.error("No rows to process. Check --start-shard / --end-shard.")
        return

    # Writer
    if args.output_format == "webdataset":
        writer = ShardWriter(args.output_dir, shard_size=args.shard_size)
    else:
        writer = FolderWriter(args.output_dir)

    # Progress tracker (for resume)
    state_file = os.path.join(args.output_dir, ".download_state.json")
    tracker = ProgressTracker(state_file)

    # Resume?
    skip = 0
    if args.resume and tracker.completed > 0:
        skip = tracker.completed
        logger.info(f"[RESUME] Skipping {skip:,} already-processed rows")
        rows = rows[skip:]
        total = len(rows)
        logger.info(f"  Remaining: {total:,} rows")

    tracker.record_session_start({
        "mode": "resume" if args.resume else ("test" if args.test_run else "full"),
        "total_rows": total,
        "skip": skip,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "N/A"),
    })
    tracker.save()

    logger.info("")
    logger.info("Starting download loop")
    logger.info(f"  Rows to process : {total:,}")
    logger.info(f"  Workers         : {args.workers}")
    logger.info(f"  Batch size      : {args.workers * 4}")
    logger.info("")

    t0 = time.time()
    batch_size = args.workers * 4
    global_idx = skip
    batches_done = 0
    last_snapshot_time = t0

    for batch_start in range(0, total, batch_size):
        if _shutdown_requested:
            logger.warning("Shutdown requested. Stopping after current state save.")
            break

        batch = rows[batch_start:batch_start + batch_size]

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for i, row in enumerate(batch):
                idx = global_idx + i
                future = executor.submit(
                    process_row, row, args.image_size, args.timeout
                )
                futures[future] = (idx, row)

            for future in as_completed(futures):
                idx, row = futures[future]
                try:
                    result = future.result()
                    if result["success"]:
                        writer.write(result, idx)
                        tracker.update(True, len(result["image_bytes"]))
                    else:
                        tracker.update(False)
                        error_stats.record(
                            result["error"],
                            url=result.get("url", ""),
                            detail=f"raw_size={result.get('raw_size', 'N/A')}"
                        )
                        fail_logger.info(
                            f"{result['error']:20s} | "
                            f"idx={idx} | "
                            f"{result.get('url', '')}"
                        )
                except Exception as e:
                    tracker.update(False)
                    error_stats.record(
                        "unknown",
                        url=row.get("URL", ""),
                        detail=f"{type(e).__name__}: {e}"
                    )
                    fail_logger.info(
                        f"{'exception':20s} | "
                        f"idx={idx} | "
                        f"{row.get('URL', '')} | "
                        f"{type(e).__name__}: {e}"
                    )
                    logger.debug(f"Unexpected error at idx={idx}: "
                                 f"{traceback.format_exc()}")

        global_idx += len(batch)
        batches_done += 1

        # Progress log every 10 batches or on last batch
        if batches_done % 10 == 0 or batch_start + batch_size >= total:
            logger.info(tracker.format_progress(total + skip))

        # Periodic snapshot: save state + error stats every 60 seconds
        now = time.time()
        if now - last_snapshot_time > 60:
            tracker.save()
            _save_error_snapshot(log_dir, run_id, error_stats, tracker)
            last_snapshot_time = now

    # Final cleanup
    writer.close()
    tracker.save()
    _save_error_snapshot(log_dir, run_id, error_stats, tracker)

    elapsed = time.time() - t0

    # ---- Final diagnostic report ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE - DIAGNOSTIC REPORT")
    logger.info("=" * 60)
    logger.info(f"  Finished at    : {datetime.datetime.now().isoformat()}")
    logger.info(f"  Elapsed        : {elapsed / 3600:.2f} hours ({elapsed:.0f}s)")
    logger.info(f"  Rows processed : {tracker.completed:,}")
    logger.info(f"  Images saved   : {tracker.successful:,}")
    logger.info(f"  Failed         : {tracker.failed:,}")
    succ_rate = tracker.successful / max(tracker.completed, 1) * 100
    logger.info(f"  Success rate   : {succ_rate:.1f}%")
    logger.info(f"  Total size     : "
                f"{tracker.state['total_bytes'] / (1024**3):.2f} GB")
    speed = tracker.completed / max(elapsed, 1)
    logger.info(f"  Avg speed      : {speed:.1f} images/sec")
    logger.info(f"  Output dir     : {args.output_dir}")
    logger.info(f"  State file     : {state_file}")
    if _shutdown_requested:
        logger.info(f"  ** Stopped early due to signal. Use --resume to continue. **")
    logger.info("")

    for line in error_stats.summary_lines():
        logger.info(line)

    # Disk space after download
    try:
        usage = shutil.disk_usage(args.output_dir)
        logger.info(f"\n  Disk free after : {usage.free / (1024**3):.1f} GB")
    except Exception:
        pass

    logger.info("")
    logger.info("Log files:")
    logger.info(f"  Main log       : {log_dir}/download_{run_id}.log")
    logger.info(f"  Failed URLs    : {log_dir}/failed_urls_{run_id}.log")
    logger.info(f"  Error snapshot : {log_dir}/error_snapshot_{run_id}.json")
    logger.info(f"  State file     : {state_file}")
    logger.info("")
    logger.info("If you need help debugging, share these files.")
    logger.info("=" * 60)


def _save_error_snapshot(log_dir: str, run_id: str,
                         error_stats: ErrorStats, tracker: ProgressTracker):
    """Save a JSON snapshot of error stats + progress for diagnostics."""
    snapshot = {
        "timestamp": datetime.datetime.now().isoformat(),
        "progress": {
            "completed": tracker.completed,
            "successful": tracker.successful,
            "failed": tracker.failed,
            "total_bytes": tracker.state["total_bytes"],
        },
        "errors": error_stats.to_dict(),
    }
    path = os.path.join(log_dir, f"error_snapshot_{run_id}.json")
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robust LAION-Art image downloader (no img2dataset needed)"
    )

    # Data source
    parser.add_argument("--parquet-url", type=str,
        default="https://huggingface.co/datasets/laion/laion-art/resolve/main/laion-art.parquet")
    parser.add_argument("--parquet-path", type=str,
        default="/cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/metadata/laion-art.parquet")

    # HuggingFace authentication (needed for gated datasets like LAION-Art)
    parser.add_argument("--hf-token", type=str, default=None,
        help="HuggingFace access token. Also reads from HF_TOKEN env var "
             "or ~/.cache/huggingface/token (via huggingface-cli login)")

    # Output
    parser.add_argument("--output-dir", type=str,
        default="/cluster/tufts/c26sp1ee0141/pliu07/LAION_ART/webdataset")
    parser.add_argument("--output-format", type=str,
        choices=["webdataset", "folder"], default="webdataset")

    # Logging
    parser.add_argument("--log-dir", type=str, default=None,
        help="Log directory (defaults to <output-dir>/logs)")

    # Image processing
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--encode-quality", type=int, default=95)

    # Download settings
    parser.add_argument("--workers", type=int, default=32,
        help="Number of parallel download threads")
    parser.add_argument("--timeout", type=int, default=15,
        help="Per-image download timeout in seconds")
    parser.add_argument("--shard-size", type=int, default=10000,
        help="Images per WebDataset shard")

    # Resume / range
    parser.add_argument("--resume", action="store_true",
        help="Resume from last checkpoint")
    parser.add_argument("--start-shard", type=int, default=None,
        help="Start from this shard number")
    parser.add_argument("--end-shard", type=int, default=None,
        help="End at this shard number")

    # Test mode
    parser.add_argument("--test-run", action="store_true",
        help="Download only a small number of images to test")
    parser.add_argument("--test-count", type=int, default=100,
        help="Number of images to download in test mode")

    main(parser.parse_args())

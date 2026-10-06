"""The indexing job: one at a time, recorded in DIRAG_HOME/job.json.

The indexer writes the record as it goes, whether it was started from the app or
from the command line, so the app shows progress for either and can stop either.

    state        running | done | stopped | failed
    action       update | reindex | rechunk
    library      the folder being indexed
    pid          the indexing process
    started      unix time
    total, done                 books to process, books processed
    bytes_total, bytes_done     the same in file bytes, for the time estimate
    current      the book being processed
    counts       {ok, rechunk, skip, no_text, error, removed}
    error        why a failed run failed

Stopping sends SIGINT. The indexer stops at its next check (before each book and
after each embedding batch), rolls back the book in progress and keeps every
book already committed. A second SIGINT, or Ctrl-C twice, stops at once.
SIGTERM is handled like SIGINT.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import config

ACTIONS = ("update", "reindex", "rechunk")
LOG = config.HOME / "job.log"
_proc = None
_stop_requested = False


def catch_stop():
    """In the indexer: turn SIGINT and SIGTERM into a request that check() acts on."""
    def request(signum, frame):
        global _stop_requested
        if _stop_requested:
            raise KeyboardInterrupt
        _stop_requested = True
    signal.signal(signal.SIGINT, request)
    signal.signal(signal.SIGTERM, request)


def check():
    """In the indexer: raise KeyboardInterrupt if a stop was requested."""
    if _stop_requested:
        raise KeyboardInterrupt


def _alive(pid):
    """Whether `pid` is a live indexer. Zombies and reused pids count as dead where /proc exists."""
    if _proc is not None and _proc.pid == pid:
        return _proc.poll() is None
    proc = Path("/proc") / str(pid)
    if Path("/proc/self").exists():
        try:
            state = (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
            cmdline = (proc / "cmdline").read_bytes().split(b"\0")
        except (OSError, IndexError):
            return False
        return state != "Z" and any(b"dirag" in part for part in cmdline) and b"index" in cmdline
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def status():
    """The job record, with a dead 'running' job reported as failed and an estimate while running."""
    job = config.read_json(config.JOB, {})
    if job.get("state") == "running":
        if not _alive(job.get("pid") or 0):
            job.update(state="failed", error=job.get("error") or "the indexing process ended; see job.log")
        elif job.get("bytes_done"):
            elapsed = time.time() - job["started"]
            job["eta"] = round(elapsed * (job["bytes_total"] - job["bytes_done"]) / job["bytes_done"])
    return job


def running():
    return status().get("state") == "running"


def start(action, root):
    """Start `dirag index` for `root` in the background. Raises RuntimeError when one is running."""
    global _proc
    if action not in ACTIONS:
        raise ValueError(action)
    if running():
        raise RuntimeError("an indexing job is running")
    config.HOME.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "dirag", "index", "--library", str(root)] + ([f"--{action}"] if action != "update" else [])
    with open(LOG, "w", encoding="utf-8") as log:
        _proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    # Written here as well as by the indexer, so a poll right after starting already sees the job.
    config.write_json(config.JOB, {"state": "running", "action": action, "library": str(root), "pid": _proc.pid,
                                   "started": time.time(), "total": 0, "done": 0, "bytes_total": 0, "bytes_done": 0,
                                   "current": "", "counts": {}})


def stop():
    job = status()
    if job.get("state") == "running":
        os.kill(job["pid"], signal.SIGINT)


class Progress:
    """The indexer's side: writes the job record after every book."""

    def __init__(self, action, root):
        self.job = {"state": "running", "action": action, "library": str(root), "pid": os.getpid(),
                    "started": time.time(), "total": 0, "done": 0, "bytes_total": 0, "bytes_done": 0,
                    "current": "", "counts": {}}
        self.write()

    def write(self, **fields):
        self.job.update(fields)
        config.write_json(config.JOB, self.job)

    def finish(self, state, error=None):
        self.write(state=state, current="", finished=time.time(), **({"error": error} if error else {}))

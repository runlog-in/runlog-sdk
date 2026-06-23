# Copyright (c) 2026 Runlog (runlog.in). All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file or https://runlog.in/auth/terms for full terms.
# Unauthorized use with non-runlog.in servers is prohibited.
# Reverse engineering, sublicensing, or use in competing services is prohibited.
# Use of this code to train or fine-tune ML models requires written consent.

"""
RunLogger — Universal Training Monitor
=======================================
from runlogger import RunLogger

logger = RunLogger(
    base_url="http://localhost:8000",
    project_name="my-project",
    api_token="rl-...",
    run_name="run-1",
)
logger.log(step=100, loss=0.5, lr=0.001)
logger.finish()
"""

import asyncio
import glob
import hashlib
import hmac
import json
import os
import queue
import sqlite3
import sys
import threading
import time
from typing import Dict, List, Optional
import random
import requests

try:
    import websockets
except ImportError:
    websockets = None

try:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import pynvml
    pynvml.nvmlInit()
    _nvml = True
except Exception:
    _nvml = False

try:
    import psutil
    _psutil = True
except Exception:
    _psutil = False


def _gpu_stats() -> dict:
    if not _nvml:
        return {}
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        u = pynvml.nvmlDeviceGetUtilizationRates(h)
        m = pynvml.nvmlDeviceGetMemoryInfo(h)
        return {
            "gpu_util":      u.gpu,
            "gpu_mem_used":  m.used  // 1024 // 1024,
            "gpu_mem_total": m.total // 1024 // 1024,
        }
    except Exception:
        return {}


def _sys_stats() -> dict:
    if not _psutil:
        return {}
    try:
        vm = psutil.virtual_memory()
        return {
            "cpu_util":  psutil.cpu_percent(interval=None),
            "ram_used":  vm.used  // 1024 // 1024,
            "ram_total": vm.total // 1024 // 1024,
        }
    except Exception:
        return {}

_ADJECTIVES = [
    "amber", "ancient", "arctic", "ashen", "astral", "atomic", "azure",
    "blazing", "bold", "brave", "bright", "brisk", "bronze", "burning",
    "calm", "cardinal", "celestial", "cerulean", "chrome", "cinder", "cobalt", "cosmic", "crimson", "crystal",
    "dark", "dawn", "deft", "dense", "drifting", "dusty", "dynamic",
    "eager", "early", "electric", "ember", "emerald", "endless",
    "fading", "fierce", "flint", "fluid", "flying", "frozen",
    "gentle", "gilded", "glacial", "glowing", "golden", "grand",
    "happy", "hidden", "hollow", "humble",
    "idle", "imperial", "infinite", "iron",
    "jolly", "jovial",
    "keen", "kinetic",
    "light", "lively", "lone", "lucid", "lunar",
    "majestic", "mellow", "mighty", "misty", "molten", "mystic",
    "narrow", "nebular", "nimble", "noble", "nomad",
    "obsidian", "olive", "onyx", "orbital",
    "phantom", "polar", "proud", "pulse",
    "quiet",
    "radiant", "rapid", "raven", "risen", "roaming", "rugged", "rustic",
    "sacred", "sandy", "sapphire", "scarlet", "serene", "shadow", "shining", "silent", "silver", "sleek", "solar", "sonic", "sparse", "stardust", "steady", "stellar", "still", "stormy", "swift",
    "tidal", "tidy", "titan", "twilight",
    "ultra", "urban",
    "vast", "velvet", "vivid",
    "wandering", "warm", "wild", "windy", "wise",
    "xenial",
    "young",
    "zealous", "zenith", "zesty",
]

_NOUNS = [
    "abyss", "apex", "arc", "archipelago", "atlas", "aurora",
    "beacon", "blizzard", "boulder", "breeze",
    "canyon", "cavern", "circuit", "cliff", "cloud", "cluster", "comet", "core", "crater", "current",
    "delta", "desert", "drift", "dune", "dusk",
    "eclipse", "epoch", "equinox",
    "falcon", "field", "fjord", "flame", "flare", "forest", "frontier",
    "galaxy", "geyser", "glacier", "glade", "gorge",
    "harbor", "haven", "horizon", "hurricane",
    "island",
    "jungle",
    "lagoon", "lattice", "lava", "lightning",
    "meadow", "meteor", "mist", "moon", "mountain",
    "nebula", "nexus", "node",
    "oasis", "ocean", "orbit",
    "peak", "pinnacle", "planet", "plateau", "pulse",
    "quasar",
    "ravine", "reef", "ridge", "rift", "river",
    "satellite", "savanna", "shore", "signal", "solstice", "spark", "spectrum", "star", "storm", "stream", "summit", "surge",
    "thunder", "tide", "torch", "trench", "tundra",
    "valley", "vantage", "veil", "vertex", "vortex",
    "wave", "wind",
    "zenith",
]

def _generate_run_name() -> str:
    adj  = random.choice(_ADJECTIVES)
    noun = random.choice(_NOUNS)
    num  = random.randint(1, 999)
    return f"{adj}-{noun}-{num}"

class _LogCapture:
    def __init__(self, original, on_line):
        self.original  = original
        self._on_line  = on_line
        self._line_buf = ""

    def write(self, text):
        self.original.write(text)
        self._line_buf += text
        if '\n' in self._line_buf:
            lines, self._line_buf = self._line_buf.rsplit('\n', 1)
            self._on_line(lines + '\n')

    def flush(self):
        self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)

class RunLogger:

    def __init__(
        self,
        api_token:         str,
        project_name:      str,
        run_name:          Optional[str]  = None,
        config:            Optional[Dict] = None,
        base_url:          Optional[str] = "https://runlog.in/",
        start_step:        int            = 0,
        metrics:           Optional[List] = None,
        tags:              Optional[List] = None,
        notes:             str            = "",
        log_system_stats:  bool           = True,
        offline_mode:      bool           = True,
        capture_terminal:  bool           = True,
        verbose:           bool           = False,
    ):
        self.base           = base_url.rstrip("/")
        self.headers        = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
        self.api_token      = api_token
        self._log_system    = log_system_stats
        self._offline_mode  = offline_mode
        self._verbose       = verbose

        self._lock          = threading.Lock()
        self._pending       = []
        self._log_count     = 0
        self._stop          = False
        self._ws            = None
        self._pause_flag    = False
        self._banned        = False
        self._pkt_seq       = 0
        self._last_sent_pkt = 0
        self._knobs      = {}
        self._knob_dirty = False
        self.capture_terminal = capture_terminal

        self._log_queue = queue.Queue()
        run_name = run_name or _generate_run_name()

        def _on_line(text):
            if text.strip('\r\n') == '':
                return
            self._log_queue.put(text)

        if self.capture_terminal:
            sys.stdout = _LogCapture(sys.stdout, _on_line)
            sys.stderr = _LogCapture(sys.stderr, _on_line)

        self._db_path  = None
        self._db_queue = queue.Queue()
        self._cache    = {}

        self._global_cache_path = ".runlog_global.json"
        self._global_cache      = {}
        try:
            with open(self._global_cache_path) as f:
                self._global_cache = json.load(f)
        except Exception:
            pass

        try:
            r    = requests.get(f"{self.base}/api/me/plan-config", headers=self.headers, timeout=5)
            plan = r.json() if r.ok else {}
        except Exception:
            plan = {}

        self._log_interval = float(plan.get("log_interval",      10))
        self._secret       = plan.get("client_secret",           "")
        self._daily_limit  = int(plan.get("daily_log_limit",     -1))
        self._max_metrics  = int(plan.get("max_metrics_tracked", -1))

        _project_cache_key       = f"project:{project_name}:{api_token}"
        _project_offline_created = False
        try:
            r = requests.get(
                f"{self.base}/api/projects/by-name/{requests.utils.quote(project_name)}",
                headers=self.headers,
                timeout=5,
            )
            if r.status_code in (401, 403):
                raise RuntimeError(f"Invalid API token. Server: {r.text}")
            r.raise_for_status()
            project_id = r.json()["id"]
            self._global_cache[_project_cache_key] = project_id
            try:
                with open(self._global_cache_path, "w") as f:
                    json.dump(self._global_cache, f)
            except Exception:
                pass
        except RuntimeError:
            raise
        except Exception:
            project_id = self._global_cache.get(_project_cache_key)
            if not project_id:
                import uuid
                project_id               = uuid.uuid4().hex[:24]
                _project_offline_created = True
                self._log("offline — project will be synced on reconnect", "warn")
            else:
                self._log("offline — using cached project", "debug")

        _run_payload = {
            "name":       run_name,
            "config":     config     or {},
            "start_step": start_step,
            "metrics":    metrics    or [],
            "tags":       tags       or [],
            "notes":      notes,
        }
        _offline_created = False
        try:
            r = requests.post(
                f"{self.base}/api/projects/{project_id}/runs",
                headers=self.headers,
                json=_run_payload,
                timeout=5,
            )
            if r.status_code in (401, 403):
                raise RuntimeError(f"Could not create run. Server: {r.text}")
            r.raise_for_status()
            self.run_id = r.json()["id"]
        except RuntimeError:
            raise
        except Exception:
            import uuid
            self.run_id      = uuid.uuid4().hex[:24]
            _offline_created = True
            self._cache["offline_created"] = True
            self._log("offline — run will be synced on reconnect", "warn")

        print(f"[Runlog] {project_name} / {run_name}")

        if self._offline_mode:
            os.makedirs("dumps", exist_ok=True)
            self._db_path = os.path.join("dumps", f".runlog_{self.run_id}.db")
            self._init_local_db()

            _token    = api_token
            _run_id   = self.run_id
            _base_url = self.base
            _pid      = project_id
            _oc       = str(_offline_created)
            _rp       = json.dumps(_run_payload)
            self._db_submit(lambda con, t=_token, r=_run_id, b=_base_url,
                                    pid=_pid, oc=_oc, rp=_rp,
                                    pn=project_name, poc=str(_project_offline_created): con.executemany(
                "INSERT OR REPLACE INTO run_meta (key, value) VALUES (?, ?)",
                [
                    ("api_token",                t),
                    ("run_id",                   r),
                    ("base_url",                 b),
                    ("project_id",               pid),
                    ("offline_created",          oc),
                    ("run_payload",              rp),
                    ("project_name",             pn),
                    ("project_offline_created",  poc),
                ]
            ))

            self._cache_put("plan", plan)
            if plan:
                self._cache_put("banned", False)

            if self._cache_get("banned", False):
                raise RuntimeError("[Runlog] account is banned.")

        if self._offline_mode:
            if plan:
                plan_allows = plan.get("offline_mode", False)
                self._cache_put("plan_offline_mode", plan_allows)
                if not plan_allows:
                    self._log("offline mode not available on your plan — disabled", "warn")
                    self._offline_mode = False
            else:
                plan_allows = self._cache_get("plan_offline_mode", True)
                if not plan_allows:
                    self._log("offline mode not available on your plan — disabled", "warn")
                    self._offline_mode = False

        if self._offline_mode:
            threading.Thread(target=self._db_worker, daemon=True).start()

        threading.Thread(target=self._ws_loop, daemon=True).start()

    def _log(self, msg: str, level: str = "info") -> None:
        if level == "debug" and not self._verbose:
            return
        prefix = {
            "info":  "[Runlog]",
            "warn":  "[Runlog] ⚠",
            "error": "[Runlog] ✖",
            "debug": "[Runlog] ~",
        }.get(level, "[Runlog]")
        print(f"{prefix} {msg}")

    def log(self, step: int, **kwargs) -> bool:
        if self._banned:
            return False
        self._buffer(step, is_eval=False, **kwargs)
        if self._log_count > 0 and self._log_count % 5000 == 0:
            threading.Thread(target=self._refresh_plan, daemon=True).start()
        return True

    def log_eval(self, step: int, **kwargs) -> bool:
        if self._banned:
            return False
        self._buffer(step, is_eval=True, **kwargs)
        return True

    def should_pause(self) -> bool:
        if self._pause_flag:
            self._clear_pause()
            self._log("paused ⏸", "info")
            if not self._ws:
                self._flush_logs(trigger="paused", final=False)
            return True
        return False

    def finish(self, status: str = "completed") -> None:
        if self._banned:
            return
        if not self._ws:
            self._drain_pending_to_db() 

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            with self._lock:
                has_pending = bool(self._pending)
            if not has_pending and self._log_queue.empty():
                break
            time.sleep(0.05)

        self._stop = True
        self._flush_logs(trigger=status, final=True)
        if self._offline_mode:
            self._db_queue.put(None)
        self._set_run_status(status)
        self._log(f"run {status}", "info")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        self.finish("crashed" if exc_type else "completed")
        return False

    def _buffer(self, step: int, is_eval: bool, **kwargs):
        if self._cache_get("banned", False):
            self._log("account banned — dropping log", "error")
            return

        if self._daily_limit != -1:
            today     = time.strftime("%Y-%m-%d")
            daily_key = f"daily_logs:{today}"
            count     = self._cache_get(daily_key, 0)
            if count >= self._daily_limit:
                if not self._stop:
                    self._log(f"daily limit reached ({self._daily_limit}) — dropping", "warn")
                return
            count += 1
            self._cache[daily_key] = count
            if count % 10 == 0:
                self._cache_put(daily_key, count)

        ts = time.time()
        payload = {
            "step":  int(step),
            "_ts":   ts,
            "_sig":  self._sign(ts),
            "_eval": is_eval,
            **(_gpu_stats() if self._log_system else {}),
            **(_sys_stats() if self._log_system else {}),
            **kwargs,
        }

        self._pkt_seq += 1
        payload["_pkt"] = self._pkt_seq

        if self._offline_mode:
            p = json.dumps(payload)
            s = payload.get("step", -1)
            pkt = self._pkt_seq
            self._db_submit(lambda con, p=p, s=s, pkt=pkt: con.execute(
                "INSERT INTO queue (step, pkt, payload, synced) VALUES (?, ?, ?, 0)", (s, pkt, p)
            ))

        with self._lock:
            self._pending.append(payload)

    def _drain_pending_to_db(self):
        if not self._offline_mode or not self._db_path:
            return
        with self._lock:
            items         = list(self._pending)
            self._pending = []
        if not items:
            return
        self._log(f"draining {len(items)} pending items to local DB", "debug")
        for payload in items:
            p   = json.dumps(payload)
            s   = payload.get("step", -1)
            pkt = payload.get("_pkt", -1)
            self._db_submit(lambda con, p=p, s=s, pkt=pkt: con.execute(
                "INSERT OR IGNORE INTO queue (step, pkt, payload, synced) VALUES (?, ?, ?, 0)",
                (s, pkt, p)
            ))
    def _sign(self, ts: float) -> str:
        if not self._secret:
            return ""
        if not (self._pkt_seq % 10 == 0):
            return ""
        msg = f"{self.run_id}:{ts:.3f}".encode()
        return hmac.new(self._secret.encode(), msg, hashlib.sha256).hexdigest()

    def _ws_loop(self):
        async def _run():
            proto  = "wss" if self.base.startswith("https") else "ws"
            ws_url = f"{proto}://{self.base.split('://')[1]}/ws/ingest/{self.run_id}?token={self.api_token}"

            self._min_interval = self._log_interval
            self._ws_delay     = 0.0
            _last_send_t       = 0.0
            _banned_perm       = False
            _offline_logged    = False
            _register_logged   = False

            while not self._stop and not _banned_perm:
                try:
                    offline_created = self._cache_get("offline_created", False)
                    if offline_created:
                        try:
                            con  = sqlite3.connect(self._db_path)
                            meta = dict(con.execute("SELECT key, value FROM run_meta").fetchall())
                            con.close()

                            run_payload = json.loads(meta.get("run_payload", "{}"))
                            project_id  = meta.get("project_id")

                            r = requests.post(
                                f"{self.base}/api/projects/{project_id}/runs",
                                headers=self.headers,
                                json=run_payload,
                                timeout=5,
                            )
                            if r.ok:
                                real_run_id = r.json()["id"]
                                self.run_id = real_run_id
                                ws_url = f"{proto}://{self.base.split('://')[1]}/ws/ingest/{self.run_id}?token={self.api_token}"
                                self._db_submit(lambda con, r=real_run_id: con.executemany(
                                    "INSERT OR REPLACE INTO run_meta (key, value) VALUES (?, ?)",
                                    [("run_id", r), ("offline_created", "False")]
                                ))
                                self._cache["offline_created"] = False
                                self._log(f"run registered on server: {self.run_id}", "debug")
                                _register_logged = False
                        except Exception:
                            if not _register_logged:
                                self._log("offline — will register run and sync when server is back", "warn")
                                _register_logged = True
                            await asyncio.sleep(10)
                            continue
                    self._load_knobs_from_db()
                    async with websockets.connect(ws_url, additional_headers=self.headers) as ws:
                        self._ws         = ws
                        _offline_logged  = False
                        _register_logged = False
                        self._log("connected", "info")

                        if self._knobs:
                            await ws.send(json.dumps({
                                "_type": "knob_definitions",
                                "knobs": self._knobs,
                            }))
                            self._knob_dirty = False

                        _ctrl_inbox: asyncio.Queue = asyncio.Queue()

                        async def _recv_forever():
                            try:
                                async for raw_msg in ws:
                                    await _ctrl_inbox.put(raw_msg)
                            except Exception:
                                pass

                        recv_task = asyncio.create_task(_recv_forever())

                        try:
                            if self._offline_mode:
                                await self._flush_offline_queue(ws)
                                await self._flush_offline_logs(ws)
                                async def _run_orphan_dump():
                                    await asyncio.get_event_loop().run_in_executor(None, self._dump_orphaned_runs)
                                asyncio.create_task(_run_orphan_dump())
                            while not self._stop:

                                while not _ctrl_inbox.empty():
                                    try:
                                        msg  = _ctrl_inbox.get_nowait()
                                        data = json.loads(msg)
                                        ctrl = data.get("_control")

                                        if ctrl == "config":
                                            self._min_interval = float(data.get("min_interval", self._min_interval))
                                            self._ws_delay     = float(data.get("ws_delay", 0))
                                            self._log(f"interval={self._min_interval:.1f}s", "debug")

                                        elif ctrl == "throttle_update":
                                            self._min_interval = float(data.get("min_interval", self._min_interval))
                                            self._ws_delay     = float(data.get("ws_delay", self._ws_delay))
                                            self._log(f"throttled to {self._min_interval:.1f}s", "debug")

                                        elif ctrl == "plan_update":
                                            new_plan    = data.get("plan", {})
                                            plan_allows = new_plan.get("offline_mode", False)
                                            self._cache_put("plan",              new_plan)
                                            self._cache_put("plan_offline_mode", plan_allows)
                                            self._log_interval = float(new_plan.get("log_interval",      self._log_interval))
                                            self._daily_limit  = int(new_plan.get("daily_log_limit",     self._daily_limit))
                                            self._max_metrics  = int(new_plan.get("max_metrics_tracked", self._max_metrics))
                                            self._secret       = new_plan.get("client_secret",           self._secret)
                                            if self._offline_mode and not plan_allows:
                                                self._log("plan downgraded — offline mode disabled", "warn")
                                                self._offline_mode = False
                                            elif not self._offline_mode and plan_allows:
                                                if self._db_path is not None:
                                                    self._log("plan upgraded — offline mode enabled", "info")
                                                    self._offline_mode = True
                                                else:
                                                    self._log("plan upgraded but offline mode was not initialized — restart to enable", "warn")

                                        elif ctrl == "limit_reached":
                                            today = time.strftime("%Y-%m-%d")
                                            self._cache_put(f"daily_logs:{today}", self._daily_limit)
                                            self._log("daily log limit reached — stopping", "warn")
                                            self._stop = True
                                            break

                                        elif ctrl == "pause":
                                            self._pause_flag = True
                                            drain_deadline = time.monotonic() + 3.0
                                            while time.monotonic() < drain_deadline:
                                                log_lines = []
                                                while not self._log_queue.empty():
                                                    try:
                                                        log_lines.append(self._log_queue.get_nowait())
                                                    except queue.Empty:
                                                        break
                                                if log_lines:
                                                    await ws.send(json.dumps({
                                                        "_type": "terminal_log",
                                                        "text": "".join(log_lines),
                                                    }))
                                                await asyncio.sleep(0.1)

                                            self._pause_flag = True

                                        elif ctrl == "resume":
                                            self._pause_flag = False

                                        elif ctrl == "banned":
                                            self._cache_put("banned", True)
                                            self._log(f"account banned — {data.get('reason', '')}", "error")
                                            self._banned  = True
                                            _banned_perm  = True
                                            self._stop    = True
                                            raise RuntimeError("RunLogger: account banned")

                                        elif ctrl == "metrics_capped":
                                            self._log(f"metrics cap reached: {data.get('reason', '')}", "warn")

                                        elif ctrl == "knob_update":
                                            key = data.get("key")
                                            val = data.get("value")
                                            if key and key in self._knobs:
                                                self._knobs[key]["value"] = float(val)
                                                self._knob_dirty = False
                                                if self._offline_mode:
                                                    self._persist_knobs()
                                                self._log(f"knob '{key}' → {val}", "debug")

                                        elif ctrl == "ack":
                                            if self._offline_mode:
                                                for pkt in data.get("pkts", []):
                                                    p = pkt
                                                    self._db_submit(lambda con, p=p: con.execute(
                                                        "UPDATE queue SET synced=1 WHERE pkt=?", (p,)
                                                    ))
                                    except Exception:
                                        pass

                                if self._stop:
                                    break

                                batch = []
                                with self._lock:
                                    if self._pending:
                                        batch         = self._pending
                                        self._pending = []

                                if batch:
                                    now = time.monotonic()
                                    wait_remaining = self._min_interval - (now - _last_send_t)
                                    
                                    if wait_remaining > 0:
                                        with self._lock:
                                            self._pending = batch + self._pending 
                                        await asyncio.sleep(0.005)
                                        continue

                                    _send_start = time.monotonic()
                                    

                                    if self._ws_delay > 0:
                                        await asyncio.sleep(self._ws_delay)

                                    await ws.send(json.dumps(batch))
                                    _last_send_t = time.monotonic()

                                    self._log_count += len(batch)
                                    self._log(f"batch of {len(batch)} sent (total={self._log_count})", "debug")

                                    log_lines = []
                                    while not self._log_queue.empty():
                                        try:
                                            log_lines.append(self._log_queue.get_nowait())
                                        except queue.Empty:
                                            break
                                    if log_lines:
                                        await ws.send(json.dumps({
                                            "_type": "terminal_log",
                                            "text":  "".join(log_lines),
                                        }))
                                    await asyncio.sleep(0.001)
                                else:
                                    log_lines = []
                                    while not self._log_queue.empty():
                                        try:
                                            log_lines.append(self._log_queue.get_nowait())
                                        except queue.Empty:
                                            break
                                    if log_lines:
                                        await ws.send(json.dumps({
                                            "_type": "terminal_log",
                                            "text":  "".join(log_lines),
                                        }))
                                    if self._knob_dirty and self._knobs:
                                        await ws.send(json.dumps({
                                            "_type": "knob_definitions",
                                            "knobs": self._knobs,
                                        }))
                                        self._knob_dirty = False

                                    await asyncio.sleep(0.005)

                        finally:
                            recv_task.cancel()
                            try:
                                await recv_task
                            except asyncio.CancelledError:
                                pass
                    _offline_logged = False

                except Exception as e:
                    self._ws = None
                    self._drain_pending_to_db()
                    log_lines = []
                    while not self._log_queue.empty():
                        try:
                            log_lines.append(self._log_queue.get_nowait())
                        except queue.Empty:
                            break
                    if log_lines and self._offline_mode:
                        text = "".join(log_lines)
                        t = "offline"
                        self._db_submit(lambda con, text=text, t=t: con.execute(
                            "INSERT INTO terminal_logs (trigger, logs, synced) VALUES (?, ?, 0)", (t, text)
                        ))
                    if not _banned_perm and not _offline_logged:
                        queued = self._db_pending_count()
                        self._log(f"offline — {queued} packets queued locally", "warn")
                        _offline_logged = True
                    await asyncio.sleep(2)

        asyncio.run(_run())

    async def _flush_offline_queue(self, ws):
        loop = asyncio.get_event_loop()
        flush_event = threading.Event()
        self._db_queue.put(lambda con, e=flush_event: e.set())
        await loop.run_in_executor(None, lambda: flush_event.wait(timeout=5.0))

        try:
            con  = sqlite3.connect(self._db_path)
            rows = con.execute(
                "SELECT id, pkt, payload FROM queue WHERE synced=0 ORDER BY pkt ASC"
            ).fetchall()
            con.close()
        except Exception:
            return

        if not rows:
            return

        self._log(f"syncing {len(rows)} offline packets...", "info")
        threading.Thread(target=self._set_run_status, args=("dumping",), daemon=True).start()

        BATCH          = 1000
        all_synced_ids = []

        for i in range(0, len(rows), BATCH):
            batch    = rows[i:i+BATCH]
            ids      = [r[0] for r in batch]
            payloads = [json.loads(r[2]) for r in batch]
            try:
                resp = await loop.run_in_executor(
                    None,
                    lambda p=payloads: requests.post(
                        f"{self.base}/api/runs/{self.run_id}/offline-sync",
                        headers=self.headers,
                        json=p,
                        timeout=30,
                    )
                )
                if resp.ok:
                    data      = resp.json()
                    inserted  = data.get("inserted", len(batch))
                    limit_hit = data.get("skipped_limit", 0) > 0 or (
                        not data.get("ok", True) and data.get("reason") == "daily_limit_reached"
                    )
                    dupe_skip = data.get("skipped_dupe", 0)
                    if dupe_skip > 0:
                        self._log(f"{dupe_skip} duplicate packets skipped", "debug")

                    self._db_submit(lambda con, ids=ids: (
                        con.execute(
                            f"UPDATE queue SET synced=1 WHERE id IN ({','.join(['?']*len(ids))})", ids
                        ),
                        con.execute("DELETE FROM queue WHERE synced=1"),
                    ))
                    all_synced_ids.extend(ids)

                    if limit_hit:
                        self._log(f"daily limit hit — {inserted} synced, remaining kept in DB", "warn")
                        break
                else:
                    self._log(f"offline sync error: {resp.status_code}", "error")
                    break
            except Exception as e:
                self._log(f"offline sync error: {e}", "error")
                break

        if all_synced_ids:
            ids_tuple = tuple(all_synced_ids)
            self._db_submit(lambda con, ids=ids_tuple: con.execute(
                f"DELETE FROM queue WHERE id IN ({','.join(['?']*len(ids))})", ids
            ))

        self._log("offline sync complete ✓", "info")
        threading.Thread(target=self._set_run_status, args=("running",), daemon=True).start()

    async def _flush_offline_logs(self, ws):
        loop = asyncio.get_event_loop()
        log_lines = []
        while not self._log_queue.empty():
            try:
                log_lines.append(self._log_queue.get_nowait())
            except queue.Empty:
                break
        if log_lines:
            text = "".join(log_lines)
            t    = "offline"
            flush_event = threading.Event()
            self._db_submit(lambda con, text=text, t=t: (
                con.execute("INSERT INTO terminal_logs (trigger, logs, synced) VALUES (?, ?, 0)", (t, text)),
                flush_event.set()
            ))
            await loop.run_in_executor(None, lambda: flush_event.wait(timeout=3.0))

        try:
            con  = sqlite3.connect(self._db_path)
            rows = con.execute(
                "SELECT id, logs FROM terminal_logs WHERE synced=0 ORDER BY id"
            ).fetchall()
            con.close()
        except Exception:
            return

        if not rows:
            return

        combined = "".join(r[1] for r in rows)
        ids      = [r[0] for r in rows]

        await ws.send(json.dumps({
            "_type": "terminal_log",
            "text":  combined,
        }))

        self._db_submit(lambda con, ids=ids: con.execute(
            f"UPDATE terminal_logs SET synced=1 WHERE id IN ({','.join(['?']*len(ids))})",
            ids
        ))
        self._log(f"flushed {len(rows)} offline log chunks", "debug")

    def _dump_orphaned_runs(self):
        os.makedirs("dumps", exist_ok=True)
        orphans = glob.glob(os.path.join("dumps", ".runlog_*.db"))
        for db_path in orphans:
            if db_path == self._db_path:
                continue
            try:
                con  = sqlite3.connect(db_path)
                meta = dict(con.execute("SELECT key, value FROM run_meta").fetchall())
                rows = con.execute(
                    "SELECT id, payload FROM queue WHERE synced=0 ORDER BY id ASC"
                ).fetchall()
                logs = con.execute(
                    "SELECT id, trigger, logs FROM terminal_logs WHERE synced=0"
                ).fetchall()
                con.close()

                all_ids      = [r[0] for r in rows]
                all_payloads = [json.loads(r[1]) for r in rows]

                orphan_token  = meta.get("api_token")
                orphan_base   = meta.get("base_url", self.base)
                orphan_run_id = meta.get("run_id") or db_path.replace(".runlog_", "").replace(".db", "")

                if not orphan_token:
                    self._log(f"orphan {orphan_run_id[:8]}... — no token, skipping", "debug")
                    continue

                orphan_headers = {
                    "Authorization": f"Bearer {orphan_token}",
                    "Content-Type":  "application/json",
                }
                if meta.get("project_offline_created") == "True":
                    orphan_project_name = meta.get("project_name")
                    try:
                        r = requests.get(
                            f"{orphan_base}/api/projects/by-name/{requests.utils.quote(orphan_project_name)}",
                            headers=orphan_headers,
                            timeout=10,
                        )
                        if r.ok:
                            orphan_project_id = r.json()["id"]
                        else:
                            r = requests.post(
                                f"{orphan_base}/api/projects",
                                headers=orphan_headers,
                                json={"name": orphan_project_name},
                                timeout=10,
                            )
                            if not r.ok:
                                self._log(f"orphan project creation failed: {r.status_code}", "error")
                                continue
                            orphan_project_id = r.json()["id"]
                        self._log(f"orphan project resolved: {orphan_project_id}", "debug")
                    except Exception as e:
                        self._log(f"orphan project resolution error: {e}", "error")
                        continue
                else:
                    orphan_project_id = meta.get("project_id")

                resume_label = "syncing" if meta.get("offline_created") == "True" else "resuming"
                self._log(f"recovering orphan run ({len(rows)} packets, {resume_label})...", "info")

                if meta.get("offline_created") == "True":
                    orphan_run_payload = json.loads(meta.get("run_payload", "{}"))
                    try:
                        r = requests.post(
                            f"{orphan_base}/api/projects/{orphan_project_id}/runs",
                            headers=orphan_headers,
                            json=orphan_run_payload,
                            timeout=10,
                        )
                        real_run_id   = r.json()["id"]
                        all_payloads  = [{**p, "run_id": real_run_id} for p in all_payloads]
                        orphan_run_id = real_run_id
                        self._log(f"orphan run created on server: {orphan_run_id}", "debug")
                        _fix = sqlite3.connect(db_path)
                        _fix.executemany(
                            "INSERT OR REPLACE INTO run_meta (key, value) VALUES (?, ?)",
                            [
                                ("run_id",          real_run_id),
                                ("offline_created", "False"),
                            ]
                        )
                        for _rid, _p in _fix.execute("SELECT id, payload FROM queue WHERE synced=0").fetchall():
                            _pd = json.loads(_p)
                            _pd["run_id"] = real_run_id
                            _fix.execute("UPDATE queue SET payload=? WHERE id=?", (json.dumps(_pd), _rid))
                        _fix.commit()
                        _fix.close()
                    except Exception as e:
                        self._log(f"orphan run creation failed: {e}", "error")
                        continue

                if not rows and not logs:
                    os.remove(db_path)
                    continue

                try:
                    r = requests.get(
                        f"{orphan_base}/api/runs/{orphan_run_id}",
                        headers=orphan_headers,
                        timeout=5,
                    )
                    if r.status_code == 404:
                        orphan_run_payload = json.loads(meta.get("run_payload", "{}"))
                        if orphan_run_payload and orphan_project_id:
                            r = requests.post(
                                f"{orphan_base}/api/projects/{orphan_project_id}/runs",
                                headers=orphan_headers,
                                json=orphan_run_payload,
                                timeout=10,
                            )
                            if r.ok:
                                real_run_id   = r.json()["id"]
                                all_payloads  = [{**p, "run_id": real_run_id} for p in all_payloads]
                                orphan_run_id = real_run_id
                                _fix = sqlite3.connect(db_path)
                                _fix.executemany(
                                    "INSERT OR REPLACE INTO run_meta (key, value) VALUES (?, ?)",
                                    [("run_id", real_run_id), ("offline_created", "False")]
                                )
                                for _rid, _p in _fix.execute("SELECT id, payload FROM queue WHERE synced=0").fetchall():
                                    _pd = json.loads(_p)
                                    _pd["run_id"] = real_run_id
                                    _fix.execute("UPDATE queue SET payload=? WHERE id=?", (json.dumps(_pd), _rid))
                                _fix.commit()
                                _fix.close()
                            else:
                                self._log(f"orphan run recovery failed: {r.status_code}", "error")
                                continue
                        else:
                            self._log("orphan run unrecoverable (404, no payload) — discarding", "debug")
                            try:
                                os.remove(db_path)
                            except Exception:
                                pass
                            continue
                except Exception:
                    continue

                success      = True
                BATCH        = 1000
                synced_count = 0

                for i in range(0, len(all_payloads), BATCH):
                    batch = all_payloads[i:i+BATCH]
                    try:
                        resp = requests.post(
                            f"{orphan_base}/api/runs/{orphan_run_id}/offline-sync",
                            headers=orphan_headers,
                            json=batch,
                            timeout=30,
                        )
                        if not resp.ok:
                            self._log(f"orphan sync failed: {resp.status_code}", "error")
                            success = False
                            break
                        data      = resp.json()
                        inserted  = data.get("inserted", len(batch))
                        limit_hit = data.get("skipped_limit", 0) > 0 or (
                            not data.get("ok", True) and data.get("reason") == "daily_limit_reached"
                        )
                        dupe_skip = data.get("skipped_dupe", 0)
                        synced_count += inserted
                        if dupe_skip > 0:
                            self._log(f"{dupe_skip} orphan duplicate packets skipped", "debug")
                        if limit_hit:
                            self._log(f"daily limit hit during orphan sync — {inserted} inserted, remaining kept", "warn")
                            _synced = all_ids[:synced_count]
                            if _synced:
                                _fix = sqlite3.connect(db_path)
                                _fix.execute(
                                    f"UPDATE queue SET synced=1 WHERE id IN ({','.join(['?']*len(_synced))})",
                                    _synced
                                )
                                _fix.execute("DELETE FROM queue WHERE synced=1")
                                _fix.commit()
                                _fix.close()
                            success = False
                            break
                    except Exception as e:
                        self._log(f"orphan sync error: {e}", "error")
                        success = False
                        break

                if not success:
                    continue

                all_ids = all_ids[:synced_count]

                for _, trigger, log_text in logs:
                    try:
                        requests.post(
                            f"{orphan_base}/api/runs/{orphan_run_id}/logs",
                            headers=orphan_headers,
                            json={"logs": log_text, "trigger": trigger},
                            timeout=10,
                        )
                    except Exception:
                        pass

                con = sqlite3.connect(db_path)
                if all_ids:
                    con.execute(
                        f"UPDATE queue SET synced=1 WHERE id IN ({','.join(['?']*len(all_ids))})",
                        all_ids
                    )
                con.execute("DELETE FROM queue WHERE synced=1")
                con.commit()
                con.close()
                os.remove(db_path)
                try:
                    requests.patch(
                        f"{orphan_base}/api/runs/{orphan_run_id}",
                        headers=orphan_headers,
                        json={"status": "dumped"},
                        timeout=5,
                    )
                except Exception:
                    pass
                self._log("orphan run recovered ✓", "info")

            except Exception as e:
                self._log(f"orphan scan error: {e}", "error")

    def _flush_logs(self, trigger: str = "finished", final: bool = False):
        if final:
            try:
                if isinstance(sys.stdout, _LogCapture):
                    sys.stdout = sys.stdout.original
                if isinstance(sys.stderr, _LogCapture):
                    sys.stderr = sys.stderr.original
            except Exception:
                pass
        if not self._offline_mode:
            return

        log_lines = []
        while not self._log_queue.empty():
            try:
                log_lines.append(self._log_queue.get_nowait())
            except queue.Empty:
                break

        if not log_lines:
            return

        text = "".join(log_lines).strip()
        if not text:
            return

        t = trigger
        self._db_submit(lambda con, text=text, t=t: con.execute(
            "INSERT INTO terminal_logs (trigger, logs, synced) VALUES (?, ?, 0)", (t, text)
        ))
        threading.Thread(target=self._upload_pending_logs, daemon=True).start()

    def _upload_pending_logs(self):
        try:
            con  = sqlite3.connect(self._db_path)
            rows = con.execute(
                "SELECT id, trigger, logs FROM terminal_logs WHERE synced=0"
            ).fetchall()
            con.close()
        except Exception:
            return

        for rid, trigger, logs in rows:
            try:
                r = requests.post(
                    f"{self.base}/api/runs/{self.run_id}/logs",
                    headers=self.headers,
                    json={"logs": logs, "trigger": trigger},
                    timeout=10,
                )
                if r.ok:
                    self._db_submit(lambda con, rid=rid: con.execute(
                        "UPDATE terminal_logs SET synced=1 WHERE id=?", (rid,)
                    ))
            except Exception:
                self._log("terminal logs saved locally — will upload on reconnect", "debug")


    def _init_local_db(self):
        con = sqlite3.connect(self._db_path)
        con.executescript("""
            CREATE TABLE IF NOT EXISTS queue (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                step    INTEGER,
                pkt     INTEGER,
                payload TEXT,
                synced  INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS plan_cache (
                key        TEXT PRIMARY KEY,
                value      TEXT,
                updated_at REAL
            );
            CREATE TABLE IF NOT EXISTS terminal_logs (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger TEXT,
                logs    TEXT,
                synced  INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS run_meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        con.commit()
        con.close()

    def _db_worker(self):
        """Single thread owns all SQLite writes — zero contention with training loop."""
        con = sqlite3.connect(self._db_path, check_same_thread=False)
        while True:
            fn = self._db_queue.get()
            if fn is None:
                break
            try:
                fn(con)
                con.commit()
            except Exception as e:
                self._log(f"db error: {e}", "error")
        con.close()

    def _db_submit(self, fn):
        if not self._offline_mode:
            return
        self._db_queue.put(fn)

    def _db_pending_rows(self) -> list:
        try:
            con  = sqlite3.connect(self._db_path)
            rows = con.execute(
                "SELECT id, payload FROM queue WHERE synced=0 ORDER BY id ASC"
            ).fetchall()
            con.close()
            return [(r[0], json.loads(r[1])) for r in rows]
        except Exception:
            return []

    def _db_pending_count(self) -> int:
        try:
            con = sqlite3.connect(self._db_path)
            n   = con.execute("SELECT COUNT(*) FROM queue WHERE synced=0").fetchone()[0]
            con.close()
            return n
        except Exception:
            return 0


    def _cache_put(self, key: str, value):
        self._cache[key] = value
        v = json.dumps(value)
        t = time.time()
        self._db_submit(lambda con, v=v, t=t, key=key: con.execute(
            "INSERT OR REPLACE INTO plan_cache (key, value, updated_at) VALUES (?, ?, ?)",
            (key, v, t)
        ))

    def _cache_get(self, key: str, default=None):
        if key in self._cache:
            return self._cache[key]
        if not self._db_path:
            return default
        try:
            con = sqlite3.connect(self._db_path)
            row = con.execute(
                "SELECT value FROM plan_cache WHERE key=?", (key,)
            ).fetchone()
            con.close()
            if row:
                val = json.loads(row[0])
                self._cache[key] = val
                return val
        except Exception:
            pass
        return default


    def _set_run_status(self, status: str):
        try:
            requests.patch(
                f"{self.base}/api/runs/{self.run_id}",
                headers=self.headers,
                json={"status": status},
                timeout=5,
            )
        except Exception:
            pass

    def _refresh_plan(self):
        try:
            r = requests.get(f"{self.base}/api/me/plan-config", headers=self.headers, timeout=5)
            if r.ok:
                plan = r.json()
                if float(plan.get("log_interval", self._log_interval)) != self._log_interval:
                    self._log(f"plan interval updated: {self._log_interval}s → {plan['log_interval']}s", "debug")
                self._log_interval = float(plan.get("log_interval",      self._log_interval))
                self._daily_limit  = int(plan.get("daily_log_limit",     self._daily_limit))
                self._max_metrics  = int(plan.get("max_metrics_tracked", self._max_metrics))
                self._secret       = plan.get("client_secret",           self._secret)
                self._cache_put("plan", plan)
        except Exception:
            pass

    def _clear_pause(self):
        try:
            requests.delete(
                f"{self.base}/api/runs/pause/{self.run_id}",
                headers=self.headers,
                timeout=3,
            )
            self._pause_flag = False
        except Exception:
            pass
    def register_knob(self, key: str, value: float, *, min: float = 0.0, max: float = 1.0, label: str = None) -> None:
        self._knobs[key] = {
            "value": float(value),
            "min":   float(min),
            "max":   float(max),
            "label": label or key,
        }
        self._knob_dirty = True
        if self._offline_mode:
            self._persist_knobs()

    @property
    def knobs(self) -> dict:
        return {k: v["value"] for k, v in self._knobs.items()}
    
    def _persist_knobs(self):
        data = json.dumps(self._knobs)
        self._db_submit(lambda con, d=data: con.execute(
            "INSERT OR REPLACE INTO run_meta (key, value) VALUES (?, ?)",
            ("knobs", d)
        ))

    def _load_knobs_from_db(self):
        try:
            con = sqlite3.connect(self._db_path)
            row = con.execute("SELECT value FROM run_meta WHERE key='knobs'").fetchone()
            con.close()
            if row:
                db_knobs = json.loads(row[0])
                for k, v in db_knobs.items():
                    if k in self._knobs:
                        self._knobs[k]["value"] = v["value"]
        except Exception:
            pass
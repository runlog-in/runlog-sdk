# Copyright (c) 2026 Runlog (runlog.in). All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file or https://runlog.in/auth/terms for full terms.
# Unauthorized use with non-runlog.in servers is prohibited.
# Reverse engineering, sublicensing, or use in competing services is prohibited.
# Use of this code to train or fine-tune ML models requires written consent.

"""
RunLogger — Manual Sync
=======================
Scan for leftover .runlog_*.db files and upload them to the server.
Can be run without any training script.

Usage:
    python -m runlogger.sync                         # scan current dir
    python -m runlogger.sync --dir /path/to/runs     # scan specific dir
    python -m runlogger.sync --file .runlog_abc.db   # sync one specific file
    runlogger-sync                                   # if installed via pip

Environment variable fallbacks:
    RUNLOGGER_URL    — base URL  (used only if a DB has no run_meta token)
    RUNLOGGER_TOKEN  — API token (used only if a DB has no run_meta token)
"""

import argparse
import glob
import json
import os
import sqlite3
import sys

import requests


def _log(msg: str, level: str = "info", verbose: bool = False, indent: bool = True) -> None:
    """
    level:
      info  — always shown
      warn  — always shown
      error — always shown
      debug — only if verbose=True
    """
    if level == "debug" and not verbose:
        return
    prefix = {
        "info":  "  ✔",
        "warn":  "  ⚠",
        "error": "  ✖",
        "debug": "  ~",
    }.get(level, "  ✔")
    pad = prefix if indent else prefix.lstrip()
    print(f"{pad} {msg}")


def _patch_status(base: str, run_id: str, headers: dict, status: str) -> None:
    try:
        requests.patch(
            f"{base}/api/runs/{run_id}",
            headers=headers,
            json={"status": status},
            timeout=5,
        )
    except Exception:
        pass


def _sync_one(db_path: str, fallback_token: str, fallback_base: str, verbose: bool = False) -> bool:
    """
    Sync a single .runlog_*.db file.
    Returns True if fully synced and safe to delete, False otherwise.
    """
    def log(msg, level="info"):
        _log(msg, level=level, verbose=verbose)

    print(f"\n── {os.path.basename(db_path)}")

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
    except Exception as e:
        log(f"could not read DB: {e}", "error")
        return False

    token  = meta.get("api_token") or fallback_token
    base   = (meta.get("base_url") or fallback_base).rstrip("/")
    run_id = meta.get("run_id") or db_path.replace(".runlog_", "").replace(".db", "")

    if not token:
        log("no token found — pass --token or set RUNLOGGER_TOKEN", "error")
        return False

    if not base:
        log("no base URL found — pass --base-url or set RUNLOGGER_URL", "error")
        return False

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }

    print(f"     run    : {run_id}")
    print(f"     server : {base}")
    print(f"     packets: {len(rows)} metrics, {len(logs)} log entries")

    if not rows and not logs:
        log("nothing to sync — cleaned up")
        try:
            os.remove(db_path)
        except Exception:
            pass
        return True

    if meta.get("offline_created") == "True":
        project_id = meta.get("project_id")

        if meta.get("project_offline_created") == "True":
            project_name = meta.get("project_name")
            try:
                r = requests.get(
                    f"{base}/api/projects/by-name/{requests.utils.quote(project_name)}",
                    headers=headers,
                    timeout=10,
                )
                if r.ok:
                    project_id = r.json()["id"]
                else:
                    r = requests.post(
                        f"{base}/api/projects",
                        headers=headers,
                        json={"name": project_name},
                        timeout=10,
                    )
                    if not r.ok:
                        log(f"project creation failed ({r.status_code}): {r.text}", "error")
                        return False
                    project_id = r.json()["id"]
                log(f"project resolved: {project_id}", "debug")
            except Exception as e:
                log(f"project resolution error: {e}", "error")
                return False

        orphan_run_payload = json.loads(meta.get("run_payload", "{}"))
        if not orphan_run_payload or not project_id:
            log("offline run has no payload/project — unrecoverable, discarding", "error")
            try:
                os.remove(db_path)
            except Exception:
                pass
            return True
        try:
            r = requests.post(
                f"{base}/api/projects/{project_id}/runs",
                headers=headers,
                json=orphan_run_payload,
                timeout=10,
            )
            if not r.ok:
                log(f"run creation failed ({r.status_code}): {r.text}", "error")
                return False
            real_run_id = r.json()["id"]
            log(f"run created on server: {real_run_id}", "debug")

            con = sqlite3.connect(db_path)
            con.executemany(
                "INSERT OR REPLACE INTO run_meta (key, value) VALUES (?, ?)",
                [
                    ("run_id",          real_run_id),
                    ("offline_created", "False"),
                ]
            )
            for _rid, _p in con.execute("SELECT id, payload FROM queue WHERE synced=0").fetchall():
                _pd = json.loads(_p)
                _pd["run_id"] = real_run_id
                con.execute("UPDATE queue SET payload=? WHERE id=?", (json.dumps(_pd), _rid))
            con.commit()
            con.close()
            run_id = real_run_id
        except Exception as e:
            log(f"run creation error: {e}", "error")
            return False

    try:
        r = requests.get(f"{base}/api/runs/{run_id}", headers=headers, timeout=5)
        if r.status_code == 404:
            orphan_run_payload = json.loads(meta.get("run_payload", "{}"))
            project_id         = meta.get("project_id")
            if orphan_run_payload and project_id:
                r = requests.post(
                    f"{base}/api/projects/{project_id}/runs",
                    headers=headers,
                    json=orphan_run_payload,
                    timeout=10,
                )
                if r.ok:
                    run_id = r.json()["id"]
                    log(f"run recovered on server: {run_id}", "debug")
                    con = sqlite3.connect(db_path)
                    con.executemany(
                        "INSERT OR REPLACE INTO run_meta (key, value) VALUES (?, ?)",
                        [("run_id", run_id), ("offline_created", "False")]
                    )
                    for _rid, _p in con.execute("SELECT id, payload FROM queue WHERE synced=0").fetchall():
                        _pd = json.loads(_p)
                        _pd["run_id"] = run_id
                        con.execute("UPDATE queue SET payload=? WHERE id=?", (json.dumps(_pd), _rid))
                    con.commit()
                    con.close()
                else:
                    log(f"run recovery failed ({r.status_code}): {r.text}", "error")
                    return False
            else:
                log("run not found and no payload to recover — discarding", "warn")
                try:
                    os.remove(db_path)
                except Exception:
                    pass
                return True
    except Exception:
        pass

    _patch_status(base, run_id, headers, "dumping")

    BATCH        = 200
    all_ids      = [r[0] for r in rows]
    success      = True
    total_inserted = 0
    total_skipped  = 0

    for i in range(0, len(rows), BATCH):
        batch    = rows[i : i + BATCH]
        payloads = [json.loads(r[1]) for r in batch]
        try:
            resp = requests.post(
                f"{base}/api/runs/{run_id}/offline-sync",
                headers=headers,
                json=payloads,
                timeout=30,
            )
            if resp.ok:
                data       = resp.json()
                inserted   = data.get("inserted", len(payloads))
                skipped    = data.get("skipped", 0)
                dupe_skip  = data.get("skipped_dupe", 0)
                limit_hit  = data.get("skipped_limit", 0) > 0 or (
                    not data.get("ok", True) and data.get("reason") == "daily_limit_reached"
                )

                total_inserted += inserted
                total_skipped  += skipped

                if dupe_skip > 0:
                    log(f"{dupe_skip} duplicate packets skipped", "debug")

                if limit_hit:
                    log(f"daily limit reached — {total_inserted} uploaded, remaining kept in DB", "warn")
                    synced_ids = all_ids[:i + inserted]
                    if synced_ids:
                        con = sqlite3.connect(db_path)
                        con.execute(
                            f"UPDATE queue SET synced=1 WHERE id IN ({','.join(['?']*len(synced_ids))})",
                            synced_ids
                        )
                        con.execute("DELETE FROM queue WHERE synced=1")
                        con.commit()
                        con.close()
                    success = False
                    break
            else:
                log(f"upload failed ({resp.status_code}): {resp.text}", "error")
                success = False
                break
        except requests.exceptions.ConnectionError:
            log(f"could not reach {base} — is the server running?", "error")
            success = False
            break
        except Exception as e:
            log(f"upload error: {e}", "error")
            success = False
            break

    if not success:
        log("sync incomplete — retry when server is reachable", "warn")
        return False

    log(f"{total_inserted} packets uploaded ✓")

    try:
        con = sqlite3.connect(db_path)
        con.execute(
            f"UPDATE queue SET synced=1 WHERE id IN ({','.join(['?'] * len(all_ids))})",
            all_ids,
        )
        con.execute("DELETE FROM queue WHERE synced=1")
        con.commit()
        con.close()
    except Exception as e:
        log(f"could not mark metrics synced locally: {e}", "warn")

    for log_id, trigger, log_text in logs:
        try:
            resp = requests.post(
                f"{base}/api/runs/{run_id}/logs",
                headers=headers,
                json={"logs": log_text, "trigger": trigger},
                timeout=10,
            )
            if resp.ok:
                log(f"terminal log '{trigger}' uploaded", "debug")
            else:
                log(f"terminal log upload failed ({resp.status_code})", "warn")
        except Exception as e:
            log(f"terminal log upload error: {e}", "warn")

        try:
            con = sqlite3.connect(db_path)
            con.execute("UPDATE terminal_logs SET synced=1 WHERE id=?", (log_id,))
            con.commit()
            con.close()
        except Exception:
            pass

    _patch_status(base, run_id, headers, "dumped")

    try:
        os.remove(db_path)
    except Exception as e:
        log(f"could not delete DB file: {e}", "warn")

    return True

def main():
    parser = argparse.ArgumentParser(
        prog="runlogger-sync",
        description="Manually sync offline RunLogger DB files to the server.",
    )
    parser.add_argument(
        "--dir",
        default="dumps",
        help="Directory to scan for .runlog_*.db files (default: dumps/)",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Sync a single specific .runlog_*.db file instead of scanning",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("RUNLOGGER_URL", ""),
        help="Server base URL — fallback if not stored in DB (env: RUNLOGGER_URL)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("RUNLOGGER_TOKEN", ""),
        help="API token — fallback if not stored in DB (env: RUNLOGGER_TOKEN)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show debug-level output (project/run IDs, per-batch detail, log uploads)",
    )
    args = parser.parse_args()

    fallback_token = args.token
    fallback_base  = args.base_url.rstrip("/") if args.base_url else ""

    if args.file:
        if not os.path.isfile(args.file):
            print(f"✖ file not found: {args.file}")
            sys.exit(1)
        db_files = [args.file]
    else:
        db_files = sorted(glob.glob(os.path.join(args.dir, ".runlog_*.db")))

    if not os.path.isdir(args.dir) and not args.file:
        print(f"✔ no dumps directory found at '{args.dir}' — nothing to sync")
        sys.exit(0)

    if not db_files:
        print("✔ no offline RunLogger DB files found — nothing to sync")
        sys.exit(0)

    print(f"Found {len(db_files)} offline DB file(s)")

    ok  = 0
    err = 0
    for db_path in db_files:
        if _sync_one(db_path, fallback_token, fallback_base, verbose=args.verbose):
            ok += 1
        else:
            err += 1

    print(f"\n{'─'*40}")
    print(f"✔ synced: {ok}   ✖ failed: {err}")

    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()
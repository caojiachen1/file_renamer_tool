import os
import sys
import threading
import subprocess
import shlex
import webbrowser
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

from flask import Flask, jsonify, request, Response, send_from_directory


app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"), static_url_path="/static")


# --- Locate executable -------------------------------------------------------
def find_executable() -> Optional[Path]:
    here = Path(__file__).resolve().parent.parent  # repo root
    candidates = [
        here / "file_renamer.exe",
        here / "file_renamer_cli.exe",
        here / "build" / "Release" / "file_renamer_cli.exe",
        here / "build" / "Debug" / "file_renamer_cli_d.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


EXECUTABLE = find_executable()
_proc_lock = threading.Lock()
_current_proc: Optional[subprocess.Popen] = None


def build_args(payload: dict) -> List[str]:
    if EXECUTABLE is None:
        raise FileNotFoundError("file_renamer executable not found")

    directory = payload.get("directory", "").strip()
    if not directory:
        raise ValueError("目录路径不能为空")

    p = Path(directory)
    if not p.exists() or not p.is_dir():
        raise ValueError("目录不存在或不是文件夹: %s" % directory)

    args: List[str] = [str(EXECUTABLE), str(p)]

    # Algorithm
    algorithm = (payload.get("algorithm") or "MD5").upper()
    if algorithm not in {"MD5", "SHA1", "SHA256", "SHA512", "CRC32", "BLAKE2B"}:
        raise ValueError("不支持的算法: %s" % algorithm)
    if algorithm != "MD5":
        args += ["-a", algorithm]

    # Mode
    mode = (payload.get("mode") or "ultra-fast").lower()
    if mode == "single-thread":
        args += ["--single-thread"]
    elif mode == "multi-thread":
        args += ["--multi-thread"]
    elif mode == "batch-mode":
        args += ["--batch-mode"]
    else:
        args += ["--ultra-fast"]

    # Recursive
    if bool(payload.get("recursive")):
        args += ["-r"]

    # Execute vs preview
    execute = bool(payload.get("execute"))
    if execute:
        args += ["-e"]
        if bool(payload.get("yes")):
            args += ["-y"]

    # Quick check
    quick = payload.get("quick")
    if quick is not None:
        if bool(quick):
            args += ["-q"]
        else:
            args += ["--no-quick"]

    # Threads & batch
    threads = payload.get("threads")
    if threads:
        args += ["-t", str(int(threads))]
    else:
        # Ultra-fast mode by default uses all available logical CPUs if not specified
        if (payload.get("mode") or "ultra-fast").lower() == "ultra-fast":
            try:
                cpu_threads = os.cpu_count() or 0
            except Exception:
                cpu_threads = 0
            if cpu_threads > 0:
                args += ["-t", str(cpu_threads)]
    batch = payload.get("batch")
    if batch:
        args += ["-b", str(int(batch))]

    # Extensions filter (comma separated, keep raw)
    exts = (payload.get("extensions") or "").strip()
    if exts:
        args += ["-x", exts]

    # Device selection
    device = (payload.get("device") or "auto").strip()
    if device:
        # normalize values
        dval = device.lower()
        if dval in {"auto", "cpu", "list"}:
            args += ["-d", dval]
        else:
            # numeric or -1
            try:
                int(device)
            except Exception:
                raise ValueError("设备选择无效: %s" % device)
            args += ["-d", device]

    # Extreme tuning
    if bool(payload.get("extreme")):
        args += ["--extreme"]

    # Optional tunables
    def add_opt(key: str, flag: str):
        v = payload.get(key)
        if v is None or v == "":
            return
        args.extend([flag, str(int(v))])

    add_opt("gpu_min_kb", "--gpu-min-kb")
    add_opt("buffer_kb", "--buffer-kb")
    add_opt("mmap_chunk_mb", "--mmap-chunk-mb")
    add_opt("gpu_file_cap_mb", "--gpu-file-cap-mb")
    add_opt("gpu_batch_bytes_mb", "--gpu-batch-bytes-mb")
    add_opt("gpu_chunk_mb", "--gpu-chunk-mb")

    return args


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/which-exe")
def which_exe():
    global EXECUTABLE
    if EXECUTABLE is None:
        EXECUTABLE = find_executable()
    return jsonify({
        "path": str(EXECUTABLE) if EXECUTABLE else None,
        "exists": EXECUTABLE.exists() if EXECUTABLE else False,
    })


@app.get("/api/devices")
def list_devices():
    exe = EXECUTABLE or find_executable()
    if not exe:
        return jsonify({"ok": False, "error": "找不到可执行文件 file_renamer.exe"}), 500
    try:
        # CLI supports: file_renamer -d list
        proc = subprocess.run([str(exe), "-d", "list"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        out = proc.stdout.decode(errors="ignore")
        # Extract device ids
        import re
        ids = []
        for m in re.finditer(r"Device\s+(-?\d+):", out):
            try:
                ids.append(int(m.group(1)))
            except Exception:
                pass
        return jsonify({"ok": True, "output": out, "deviceIds": ids})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/browse-folder")
def browse_folder():
    """Open a native folder selection dialog on the server machine and return the selected path.

    Note: This runs on the server (local machine). It uses tkinter filedialog and will show
    a dialog on the machine where the Flask process runs.
    """
    try:
        # import locally to avoid tkinter dependency when not used
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        # ensure dialog is on top
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        path = filedialog.askdirectory()
        root.destroy()
        if not path:
            return jsonify({"ok": False, "error": "cancelled"}), 400
        return jsonify({"ok": True, "path": path})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def stream_process(proc: subprocess.Popen):
    try:
        for line in iter(proc.stdout.readline, b""):
            if not line:
                break
            try:
                chunk = line.decode("utf-8", errors="ignore")
            except Exception:
                chunk = line.decode(errors="ignore")
            yield chunk
        proc.wait()
        yield f"\n[exit-code] {proc.returncode}\n"
    finally:
        proc.stdout and proc.stdout.close()


@app.post("/api/run")
def run_cli():
    global _current_proc
    if _proc_lock.locked():
        return jsonify({"ok": False, "error": "已有任务在运行中，请先停止或等待完成。"}), 429

    payload = request.get_json(silent=True) or {}
    try:
        args = build_args(payload)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    def generate():
        global _current_proc
        with _proc_lock:
            try:
                _current_proc = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            except FileNotFoundError:
                yield "启动失败：找不到可执行文件。\n"
                return
            except Exception as e:
                yield f"启动失败：{e}\n"
                return

            yield f"运行命令: {' '.join(shlex.quote(a) for a in args)}\n\n"
            try:
                for chunk in stream_process(_current_proc):
                    yield chunk
            finally:
                _current_proc = None

    return Response(generate(), mimetype="text/plain; charset=utf-8")


@app.post("/api/stop")
def stop_cli():
    global _current_proc
    if not _current_proc:
        return jsonify({"ok": False, "error": "没有正在运行的任务。"}), 400
    try:
        _current_proc.terminate()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def main():
    # Allow port override
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "127.0.0.1")

    def run_server():
        # Use use_reloader=False to avoid spawning extra processes
        app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)

    def open_browser_when_ready(host: str, port: int, timeout: int = 10):
        url = f"http://{host}:{port}/"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1) as resp:
                    if resp.status == 200:
                        webbrowser.open(url)
                        return True
            except Exception:
                time.sleep(0.2)
        # fallback: still try to open (may show error page)
        try:
            webbrowser.open(url)
            return True
        except Exception:
            return False

    print(f"Serving UI on http://{host}:{port}")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Open browser from Python process unless explicitly disabled
    if os.environ.get("OPEN_BROWSER", "1") != "0":
        opened = open_browser_when_ready(host, port, timeout=10)
        if not opened:
            print("警告：无法自动打开浏览器，或在超时后尝试打开失败。请手动访问:", f"http://{host}:{port}/")

    try:
        # Wait for server thread to finish (runs until process killed)
        server_thread.join()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

"""ROS entry point that records and reports each MPPI controller run."""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time

from controls.random_seed import sampling_seed


def default_runs_root():
    for directory in Path(__file__).resolve().parents:
        if (directory / "src" / "controls" / "package.xml").is_file():
            return directory / "runs"
    return Path.home() / "rams_runs"


def stop(process, interrupt=True):
    if process is None or process.poll() is not None:
        return
    signals = [(signal.SIGINT, 20)] if interrupt else []
    signals += [(signal.SIGTERM, 5), (signal.SIGKILL, 3)]
    for sig, timeout in signals:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue


def mirror_output(process, log):
    for line in process.stdout:
        log.write(line)
        log.flush()
        try:
            print(line, end="", flush=True)
        except BrokenPipeError:
            # The file remains available if a terminal pipeline closes.
            pass


def recorder_ready(process, recorder_log, stopping):
    deadline = time.monotonic() + 25
    while not stopping.is_set() and process.poll() is None and time.monotonic() < deadline:
        if "Ctrl+C to finish" in recorder_log.read_text(encoding="utf-8", errors="replace"):
            return True
        stopping.wait(.1)
    return False


def write_status(target, status):
    spare = target.with_suffix(".tmp")
    spare.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    spare.replace(target)


def main():
    env = os.environ.copy()
    model = env.get("MPPI_MODEL", "exp13")
    try:
        seed = sampling_seed()
    except ValueError as exc:
        print(f"[run] {exc}", file=sys.stderr)
        return 2
    env["MPPI_SEED"] = str(seed)
    stamp = datetime.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    runs = Path(env.get("MPPI_RUNS_DIR", str(default_runs_root()))).expanduser().resolve()
    runs.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=stamp + "-", dir=runs))
    audit = Path(__file__).parent / "run_audit"
    env.update(MPPI_AUDIT="1", MPPI_CSV=str(run / "mppi.csv"),
               MPPI_TRAJECTORY_PNG=str(run / "trajectory.png"), PYTHONUNBUFFERED="1")
    # Keep ROS's own process logs with the captured topics and stdout.
    env["ROS_LOG_DIR"] = str(run / "ros")
    status = {
        "model": model,
        "mppi_seed": seed,
        "started": datetime.datetime.now().astimezone().isoformat(),
        "controller_args": sys.argv[1:],
        "ros_domain_id": env.get("ROS_DOMAIN_ID", "0"),
        "recorder_ready": False,
    }
    meta_file = run / "run.json"
    write_status(meta_file, status)
    print(f"[run] {run}", flush=True)
    stopping = threading.Event()
    previous = {}

    def request_stop(_signum, _frame):
        stopping.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, request_stop)

    recorder = controller = output = None
    try:
        with (run / "recorder.log").open("w", encoding="utf-8") as capture_log, \
             (run / "controller.log").open("w", encoding="utf-8") as controller_log:
            recorder = subprocess.Popen(
                [sys.executable, "-u", str(audit / "capture.py"), str(run)],
                stdout=capture_log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            try:
                status["recorder_ready"] = recorder_ready(recorder, run / "recorder.log", stopping)
                write_status(meta_file, status)
                if status["recorder_ready"] and not stopping.is_set():
                    controller = subprocess.Popen(
                        [sys.executable, "-u", "-m", "controls.control_node", *sys.argv[1:]],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                        encoding="utf-8", errors="replace", env=env, start_new_session=True)
                    output = threading.Thread(target=mirror_output,
                                              args=(controller, controller_log), daemon=True)
                    output.start()
                    while controller.poll() is None and recorder.poll() is None and not stopping.wait(.2):
                        pass
                    if recorder.poll() is not None and controller.poll() is None:
                        print("[run] Recorder stopped. Ending the controller run.", flush=True)
                        status["recorder_failed"] = True
                elif not stopping.is_set():
                    print(f"[run] Recorder failed to start. See {run / 'recorder.log'}", flush=True)
            finally:
                stop(controller)
                if output is not None:
                    output.join(timeout=5)
                stop(recorder)

        status["controller_exit"] = controller.returncode if controller is not None else None
        status["recorder_exit"] = recorder.returncode if recorder is not None else None
        print("[run] Generating the report...", flush=True)
        with (run / "report.log").open("w", encoding="utf-8") as report_log:
            if (run / "events.jsonl").is_file():
                result = subprocess.run(
                    [sys.executable, str(audit / "report.py"), str(run)], env=env,
                    stdout=report_log, stderr=subprocess.STDOUT, start_new_session=True)
                status["report_exit"] = result.returncode
            else:
                report_log.write("No event archive was created. See recorder.log.\n")
                status["report_exit"] = None
        if (run / "report.html").is_file():
            print(f"[run] Report: {run / 'report.html'}", flush=True)
        else:
            print(f"[run] No report generated. See {run / 'report.log'}", flush=True)
        if status.get("recorder_failed") or (not status["recorder_ready"] and not stopping.is_set()):
            return 1
        return status["controller_exit"] or 0
    finally:
        status["finished"] = datetime.datetime.now().astimezone().isoformat()
        write_status(meta_file, status)
        for signum, handler in previous.items():
            signal.signal(signum, handler)

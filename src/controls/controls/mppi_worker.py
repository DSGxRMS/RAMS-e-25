"""Run the existing MPPI solver under a separate CUDA-capable interpreter."""

import json
from pathlib import Path
import socket
import subprocess
import sys


def _send(sock, payload):
    sock.sendall(json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n")


def _receive(stream):
    line = stream.readline()
    if not line:
        raise RuntimeError("MPPI GPU worker exited. See its startup or traceback output above.")
    return json.loads(line)


class MPPIWorker:
    """A synchronous, local connection to a persistent MPPI solver."""

    def __init__(self, python):
        self.python = python
        self._path = None

    def __enter__(self):
        self._socket, child = socket.socketpair()
        self._stream = self._socket.makefile("rb")
        self._process = None
        try:
            with child:
                self._process = subprocess.Popen(
                    [self.python, "-I", "-u", str(Path(__file__).resolve()), str(child.fileno())],
                    pass_fds=(child.fileno(),))
            self._socket.settimeout(180)
            config = _receive(self._stream)
            self.ACCEL_MAX = config["ACCEL_MAX"]
            self.V_MAX = config["V_MAX"]
            self.DT_EFF = config["DT_EFF"]
            self.T_MAX = config["T_MAX"]
            self.K = config["K"]
            self.audit_enabled = config["audit_enabled"]
            self._socket.settimeout(5)
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *_exc):
        self.close()

    def close(self):
        self._stream.close()
        self._socket.close()
        if self._process is not None:
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()

    def _call(self, action, args=(), kwargs=None):
        try:
            _send(self._socket, [action, args, kwargs or {}])
            return _receive(self._stream)
        except socket.timeout as exc:
            raise RuntimeError("MPPI GPU worker timed out during " + action) from exc

    def update_path(self, path):
        if path != self._path:
            self._call("update_path", (path,))
            self._path = path

    def reset_steering_plan(self):
        self._call("reset_steering_plan")

    def compute(self, *args, steering_prediction=None, **kwargs):
        if steering_prediction is not None:
            angle, queued, solve_s, measured = steering_prediction
            steering_prediction = (angle, queued.tolist(), solve_s, measured)
        return self._call("compute", args, {**kwargs, "steering_prediction": steering_prediction})


def main(fd):
    # Isolated Python deliberately ignores the Galactic PYTHONPATH.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np

    from controls.mppi_controller import MPPIController

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"No CUDA in {sys.executable} (PyTorch {torch.__version__}).")
    # Verify that this PyTorch build can execute kernels on the selected GPU.
    torch.ones(8, device="cuda").sum().item()
    solver = MPPIController()
    actions = {"update_path": solver.update_path, "compute": solver.compute,
               "reset_steering_plan": solver.reset_steering_plan}
    with socket.socket(fileno=fd) as connection, connection.makefile("rb") as requests:
        _send(connection, {name: getattr(solver, name) for name in
                           ("ACCEL_MAX", "V_MAX", "DT_EFF", "T_MAX", "K", "audit_enabled")})
        for line in requests:
            action, args, kwargs = json.loads(line)
            prediction = kwargs.get("steering_prediction")
            if prediction is not None:
                prediction[1] = np.asarray(prediction[1], dtype=np.float32)
            _send(connection, actions[action](*args, **kwargs))


if __name__ == "__main__":
    try:
        main(int(sys.argv[1]))
    except (KeyboardInterrupt, BrokenPipeError, ConnectionResetError):
        pass

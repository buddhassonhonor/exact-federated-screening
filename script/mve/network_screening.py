"""Send verified causal waves over independent framed TCP loopback links."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import ctypes
from datetime import datetime, timezone
import gzip
import json
import os
import socket
import time

import numpy as np

from exact_tracking import independent_rank
from run_exact_study import RUNS, SCALE, make_method, measure, setup
from stream_data import digest

METHODS = ("dense-best", "tput", "nra-8", "absolute", "hybrid-25", "safe-zone", "anchor")
PROFILES = {
    "loopback": (0, None),
    "2ms-1Mbps": (2, 1),
    "2ms-100Mbps": (2, 100),
    "20ms-1Mbps": (20, 1),
    "20ms-100Mbps": (20, 100),
}
LOW_RATE = {
    "2ms-0.05Mbps": (2, 0.05),
    "2ms-0.1Mbps": (2, 0.1),
    "2ms-0.25Mbps": (2, 0.25),
    "2ms-0.5Mbps": (2, 0.5),
}
TIMED_ENDPOINTS = {"loopback": PROFILES["loopback"], "2ms-1Mbps": PROFILES["2ms-1Mbps"]}
ROUNDS = {
    "beijing": (0, 2, 6, 10, 14, 18, 21),
    "load": (0, 2, 6, 10, 14, 18, 21),
    "wide-gradual": (0, 3, 7, 13, 19, 24, 31),
}

if os.name == "nt":
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    current_thread = kernel.GetCurrentThread
    current_thread.restype = ctypes.c_void_p
    query_cycles = kernel.QueryThreadCycleTime
    query_cycles.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)]
    query_cycles.restype = ctypes.c_int


def cycles():
    if os.name != "nt":
        return 0
    count = ctypes.c_ulonglong()
    if not query_cycles(current_thread(), ctypes.byref(count)):
        raise ctypes.WinError(ctypes.get_last_error())
    return count.value


def read_exact(sock, size):
    tick = cycles()
    wall = time.perf_counter_ns()
    started = time.thread_time_ns()
    buf = bytearray(size)
    view = memoryview(buf)
    while len(view):
        n = sock.recv_into(view)
        if not n:
            raise EOFError("Truncated benchmark payload")
        view = view[n:]
    return buf, (time.thread_time_ns() - started, time.perf_counter_ns() - wall, cycles() - tick)


def write_exact(sock, payload, one_way_delay):
    if one_way_delay:
        time.sleep(one_way_delay)
    tick = cycles()
    wall = time.perf_counter_ns()
    started = time.thread_time_ns()
    sock.sendall(payload)
    return time.thread_time_ns() - started, time.perf_counter_ns() - wall, cycles() - tick


class Loopback:
    def __init__(self, clients):
        self.clients = clients
        self.links = []
        self.pool = ThreadPoolExecutor(max_workers=2 * clients)

    def __enter__(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.bind(("127.0.0.1", 0))
                listener.listen()
                for _ in range(self.clients):
                    client = socket.create_connection(listener.getsockname(), timeout=5)
                    server, _ = listener.accept()
                    self.links.append((client, server))
                    for sock in (client, server):
                        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        sock.settimeout(5)
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_):
        for link in self.links:
            for sock in link:
                sock.close()
        self.pool.shutdown()

    def transfer(self, frames, rtt, mbps):
        waves = defaultdict(lambda: defaultdict(bytearray))
        for item in frames:
            waves[item.phase, item.direction][item.client].extend(item.frame)
        client_cost = np.zeros((self.clients, 3), dtype=np.int64)
        server_cost = np.zeros(3, dtype=np.int64)
        n_bytes = 0
        theoretical = 0.0
        elapsed = time.perf_counter_ns()
        for phase, direction in sorted(waves, key=lambda key: (key[0], key[1] == "up")):
            requests = []
            delay_max = 0.0
            for i, payload in waves[phase, direction].items():
                size = len(payload)
                if size == 0:
                    continue
                client, server = self.links[i]
                sending, receiving = (client, server) if direction == "up" else (server, client)
                transfer_delay = rtt / 2000 + (size * 8 / (mbps * 1e6) if mbps else 0)
                delay_max = max(delay_max, transfer_delay)
                writer = self.pool.submit(write_exact, sending, payload, transfer_delay)
                reader = self.pool.submit(read_exact, receiving, size)
                requests.append((i, payload, reader, writer))
            for i, payload, reader, writer in requests:
                data, read_cost = reader.result()
                write_cost = writer.result()
                assert data == payload
                n_bytes += len(data)
                client_cost[i] += write_cost if direction == "up" else read_cost
                server_cost += read_cost if direction == "up" else write_cost
            theoretical += delay_max
        elapsed = time.perf_counter_ns() - elapsed
        return dict(elapsed_ms=elapsed / 1e6, debited_ms=1000 * theoretical, app_bytes=n_bytes,
                    goodput_kbps=8e6 * n_bytes / elapsed, server_socket_cpu_us=server_cost[0] / 1e3,
                    per_client_socket_cpu_us=(client_cost[:, 0] / 1e3).tolist(),
                    server_socket_call_ms=server_cost[1] / 1e6,
                    per_client_socket_call_ms=(client_cost[:, 1] / 1e6).tolist(),
                    server_socket_cycles=int(server_cost[2]), per_client_socket_cycles=client_cost[:, 2].tolist(),
                    waves=len(waves))


def capture(name, path, methods):
    with np.load(path, allow_pickle=False) as opened:
        scores, ranks = opened["scores"], opened["exact_rank"]
    c, d = scores.shape[1:]
    traces = []
    for kind in methods:
        model = make_method(kind, c, d, ranks.shape[1])
        previous = None
        for t, values in enumerate(scores):
            keep = t in ROUNDS[name]
            result, current, frames = measure(model, values, previous, capture=keep)
            assert np.array_equal(current, ranks[t]) and np.array_equal(current, independent_rank(values, ranks.shape[1]))
            if keep:
                assert sum(len(item.frame) for item in frames) == result["up"] + result["down"] + result["selection_bytes"]
                traces.append((kind, t, frames, result))
            previous = current
    return c, d, traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-run", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--extra-bandwidth", action="store_true")
    modes.add_argument("--endpoint-timing", action="store_true")
    args = parser.parse_args()
    run = RUNS / args.slug
    setup(run, args)
    cases = {
        name: RUNS / args.native_run / "artifacts" / f"{name}__block0.npz" for name in ("beijing", "load")
    }
    if args.extra_bandwidth:
        profiles = LOW_RATE
        methods = ("dense-best", "hybrid-25", "nra-8")
    elif args.endpoint_timing:
        profiles = TIMED_ENDPOINTS
        methods = ("dense-best", "hybrid-25", "nra-8")
    else:
        cases["wide-gradual"] = RUNS / SCALE / "artifacts" / "gradual__synthetic__width-2048__s7.npz"
        profiles, methods = PROFILES, METHODS
    seeds = np.random.default_rng(771)
    inputs = {name: {"path": str(path), "sha256": digest(path)} for name, path in cases.items()}
    detail = dict(profiles=profiles, cases=inputs, rounds=ROUNDS, methods=methods,
                  order_seed=771, scope="Application frames over real AF_INET loopback TCP")
    (run / "artifacts" / "network_design.json").write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
    total = 0
    with (run / "stdout_stderr.log").open("w", encoding="utf-8") as log, (
        run / "metrics.jsonl"
    ).open("w", encoding="utf-8") as out, gzip.open(run / "artifacts" / "frame_traces.jsonl.gz", "wt", encoding="utf-8") as archive:
        print(f"{datetime.now(timezone.utc).isoformat()} start", file=log, flush=True)
        for name, path in cases.items():
            c, d, traces = capture(name, path, methods)
            for method, t, frames, reference in traces:
                record = {"case": name, "method": method, "round": t, "reference": reference,
                          "frames": [{"client": r.client, "phase": r.phase, "direction": r.direction,
                                      "bytes": r.frame.hex()} for r in frames]}
                print(json.dumps(record, separators=(",", ":")), file=archive)
            jobs = [(p, repeat, i) for p in profiles for repeat in range(args.repeats) for i in range(len(traces))]
            seeds.shuffle(jobs)
            with Loopback(c) as network:
                for profile, repeat, index in jobs:
                    method, t, frames, reference = traces[index]
                    measured = network.transfer(frames, *profiles[profile])
                    assert measured["app_bytes"] == reference["up"] + reference["down"] + reference["selection_bytes"]
                    assert measured["waves"] == reference["phases_with_selection"]
                    row = dict(case=name, method=method, round=t, profile=profile, repeat=repeat, clients=c, d=d,
                               screen_bytes=reference["up"] + reference["down"], selection_bytes=reference["selection_bytes"],
                               **measured)
                    print(json.dumps(row, separators=(",", ":")), file=out, flush=True)
                    total += 1
            message = f"{datetime.now(timezone.utc).isoformat()} {name}: {len(jobs)} TCP measurements"
            print(message, file=log, flush=True)
            print(message, flush=True)
        print(f"{total} measurements and frame payload matches", file=log, flush=True)


if __name__ == "__main__":
    main()

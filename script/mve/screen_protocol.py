"""Serializable transports for additive top-k screening."""

import struct
import zlib
from dataclasses import dataclass

import numpy as np

I16 = np.dtype("<i2")
U16 = np.dtype("<u2")
PAIR = np.dtype([("index", U16), ("value", I16)])
HEADER = struct.Struct("<BBI")


class Wire:
    """In-memory application layer. Each send returns only decoded payload."""

    def __init__(self):
        self.up = self.down = 0
        self.client_id = self.phase = 0

    def route(self, client, phase=0):
        self.client_id, self.phase = client, phase

    @staticmethod
    def encode(kind, payload, compress=False):
        coded = zlib.compress(payload, level=1) if compress else payload
        use_compressed = len(coded) < len(payload)
        body = coded if use_compressed else payload
        return HEADER.pack(kind, use_compressed, len(body)) + body

    def deliver(self, direction, frame):
        if direction == "up":
            self.up += len(frame)
        else:
            self.down += len(frame)
        tag, flag, size = HEADER.unpack_from(frame)
        assert size == len(frame) - HEADER.size
        body = frame[HEADER.size:]
        return tag, zlib.decompress(body) if flag else body

    def send(self, direction, kind, payload, compress=False):
        return self.deliver(direction, self.encode(kind, payload, compress))[1]


def top(values, k):
    # The feature ID is the secondary key, including when all scores tie.
    return np.argsort(-values, kind="stable")[:k]


def integers(values, dtype=I16):
    return np.asarray(values, dtype=dtype).tobytes()


def packed_pairs(index, values):
    data = np.empty(len(index), dtype=PAIR)
    data["index"] = index
    data["value"] = values
    return data.tobytes()


def pairs(payload):
    data = np.frombuffer(payload, dtype=PAIR)
    return data["index"], data["value"]


def request(wire, index, dimension, compress=True):
    mask = np.zeros(dimension, dtype=np.uint8)
    mask[index] = 1
    listed = integers(index, U16)
    packed = np.packbits(mask).tobytes()
    mode, payload = (1, packed) if len(packed) < len(listed) else (0, listed)
    received = wire.send("down", 1, bytes([mode]) + payload, compress)
    if received[0] == 1:
        return np.flatnonzero(np.unpackbits(np.frombuffer(received[1:], dtype=np.uint8))[:dimension])
    return np.frombuffer(received[1:], dtype=U16)


def refresh(wire, scores, index, compress=True):
    d = len(scores)
    ids = request(wire, index, d, compress) if len(index) != d else np.arange(d)
    payload = wire.send("up", 2, integers(scores[ids]), compress)
    return ids, np.frombuffer(payload, dtype=I16)


@dataclass
class Outcome:
    selected: np.ndarray
    up: int
    down: int
    extra: int = 0
    candidates: int = 0
    fallback: bool = False
    gap: int = 0
    dense_clients: int = 0


class Protocol:
    def __init__(self, kind, clients, dimension, k, setting=0, groups=4, seed=0, dense_percent=75):
        self.kind = kind
        self.k = k
        self.setting = setting
        self.groups = groups
        self.dense_percent = dense_percent
        self.server = np.zeros((clients, dimension), dtype=I16)
        self.client = self.server.copy()
        self.previous = np.arange(k)
        self.group_id = np.arange(dimension) % max(1, groups - 1) + 1 if groups > 1 else np.zeros(dimension, dtype=int)
        self.rng = np.random.default_rng(seed)
        self.round = 0

    def step(self, scores, wire=None):
        self.round += 1
        wire = Wire() if wire is None else wire
        name = self.kind
        c, d = scores.shape
        k = self.k
        extra, candidates, fallback, gap, dense_clients = 0, 0, False, 0, 0
        if name == "random":
            selected = np.sort(self.rng.choice(d, size=k, replace=False))
        elif name == "periodic" and (self.round - 1) % self.setting:
            selected = self.previous
        elif name in {"dense", "lossless", "delta", "dense-best", "periodic"}:
            for i, local in enumerate(scores):
                wire.route(i)
                if name == "dense-best":
                    current = Wire.encode(3, integers(local), True)
                    differenced = Wire.encode(15, integers(local - self.client[i]), True)
                    kind, packet = wire.deliver("up", min((current, differenced), key=len))
                elif name == "delta":
                    kind, packet = 15, wire.send("up", 15, integers(local - self.client[i]), True)
                else:
                    kind, packet = 3, wire.send("up", 3, integers(local), name != "dense")
                received = np.frombuffer(packet, dtype=I16)
                self.server[i] = self.server[i] + received if kind == 15 else received
                self.client[i] = local
            selected = top(self.server.sum(axis=0), k)
        elif name == "sparse":
            aggregate = np.zeros(d, dtype=np.int64)
            for i, local in enumerate(scores):
                wire.route(i)
                ids = np.sort(top(local.astype(np.int32), min(d, self.setting)))
                ids, values = pairs(wire.send("up", 4, packed_pairs(ids, local[ids]), True))
                aggregate[ids] += values
            selected = top(aggregate, k)
        elif name == "tput":
            selected, candidates = self._tput(wire, scores)
            extra = 2
        elif name in {"hybrid", "absolute"}:
            selected, extra, candidates, fallback, gap, dense_clients = self._adaptive(wire, scores)
        else:
            selected, extra, candidates, fallback, gap = self._cached(wire, scores)
        self.previous = np.sort(selected)
        return Outcome(self.previous.copy(), wire.up, wire.down, extra, candidates, fallback, gap, dense_clients)

    def _adaptive(self, wire, scores):
        """Choose a dense or interval frame before transport, then refine as needed."""
        c, d = scores.shape
        hybrid = self.kind == "hybrid"
        group = self.group_id.copy()
        if self.groups > 1:
            group[self.previous] = 0
        group_masks = [group == g for g in range(self.groups)]
        lower = np.zeros((c, d), dtype=np.int64)
        upper = lower.copy()
        dense_clients = 0
        for i, local in enumerate(scores):
            wire.route(i)
            dense = Wire.encode(8, integers(local), True)
            residual = local - self.client[i] if hybrid else local.copy()
            ids = np.sort(top(np.abs(residual.astype(np.int32)), min(d, self.setting)))
            limits = np.zeros((self.groups, 2), dtype=I16)
            omitted = np.ones(d, dtype=bool)
            omitted[ids] = False
            for g, mask in enumerate(group_masks):
                values = residual[mask & omitted]
                if len(values):
                    limits[g] = values.min(), values.max()
            envelope = Wire.encode(9, limits.tobytes() + packed_pairs(ids, local[ids]), True)
            # Leave part of the dense cost available for possible refinement.
            use_dense = hybrid and (self.round == 1 or 100 * len(envelope) > self.dense_percent * len(dense))
            kind, received = wire.deliver("up", dense if use_dense else envelope)
            if kind == 8:
                self.server[i] = np.frombuffer(received, dtype=I16)
                self.client[i] = local
                lower[i] = upper[i] = self.server[i]
                dense_clients += 1
                continue
            bounds = np.frombuffer(received[:limits.nbytes], dtype=I16).reshape(-1, 2)
            lo, hi = bounds[group].T.copy()
            chosen, values = pairs(received[limits.nbytes:])
            if hybrid:
                self.server[i, chosen] = values
                self.client[i, ids] = local[ids]
                lo[chosen] = hi[chosen] = 0
                lower[i] = self.server[i] + lo
                upper[i] = self.server[i] + hi
            else:
                lo[chosen] = hi[chosen] = values
                lower[i], upper[i] = lo, hi

        low, high = lower.sum(axis=0), upper.sum(axis=0)
        initial = top(low, self.k)
        boundary = initial[-1]
        cutoff = low[boundary]
        ids = np.arange(d)
        # Include a tied upper bound exactly when its tie key can pass the cutoff.
        candidates = np.flatnonzero((high > cutoff) | ((high == cutoff) & (ids <= boundary)))
        outsiders = np.ones(d, dtype=bool)
        outsiders[initial] = False
        gap = int(cutoff - high[outsiders].max(initial=-1))
        if np.array_equal(np.sort(initial), candidates):
            return initial, 0, len(candidates), False, gap, dense_clients

        extra = False
        fallback = False
        for i, local in enumerate(scores):
            unknown = candidates[lower[i, candidates] != upper[i, candidates]]
            if not len(unknown):
                continue
            wire.route(i, 1)
            extra = True
            if len(unknown) == d:
                wire.send("down", 10, b"")
                query = ids
            else:
                query = request(wire, unknown, d)
            partial = Wire.encode(2, integers(local[query]), True)
            dense = Wire.encode(8, integers(local), True)
            kind, response = wire.deliver("up", min((partial, dense), key=len))
            chosen = ids if kind == 8 else query
            lower[i, chosen] = np.frombuffer(response, dtype=I16)
            if hybrid:
                self.server[i, chosen] = lower[i, chosen]
                self.client[i, chosen] = local[chosen]
            fallback |= kind == 8 or len(query) == d
        exact = lower[:, candidates].sum(axis=0)
        return candidates[top(exact, self.k)], int(extra), len(candidates), fallback, gap, dense_clients

    def _tput(self, wire, scores):
        """Three-phase uniform threshold for nonnegative additive scores."""
        c, d = scores.shape
        partial = np.zeros(d, dtype=np.int64)
        sent = np.zeros((c, d), dtype=bool)
        for i, local in enumerate(scores):
            wire.route(i)
            ids = np.sort(top(local.astype(np.int32), self.k))
            ids, values = pairs(wire.send("up", 5, packed_pairs(ids, local[ids]), True))
            partial[ids] += values
            sent[i, ids] = True
        threshold = int(np.partition(partial, -self.k)[-self.k])
        tau = (threshold + c - 1) // c
        for i, local in enumerate(scores):
            wire.route(i, 1)
            message = wire.send("down", 6, integers([tau]))
            local_tau = np.frombuffer(message, dtype=I16)[0]
            ids = np.flatnonzero((local >= local_tau) & ~sent[i])
            ids, values = pairs(wire.send("up", 7, packed_pairs(ids, local[ids]), True))
            partial[ids] += values
            sent[i, ids] = True
        upper = partial + np.sum(~sent, axis=0) * max(0, tau - 1)
        candidates = np.flatnonzero(upper >= threshold)
        for i, (local, sent_row) in enumerate(zip(scores, sent)):
            ids = candidates[~sent_row[candidates]]
            if len(ids):
                wire.route(i, 2)
                ids, values = refresh(wire, local, ids)
                partial[ids] += values
        return candidates[top(partial[candidates], self.k)], len(candidates)

    def _cached(self, wire, scores):
        c, d = scores.shape
        group = self.group_id.copy()
        if self.groups > 1:
            group[self.previous] = 0
        group_masks = tuple(group == j for j in range(self.groups))
        lower, upper = np.zeros(d, dtype=np.int64), np.zeros(d, dtype=np.int64)
        # A full first round initializes identical caches at both ends.
        if self.round == 1:
            for i, local in enumerate(scores):
                wire.route(i)
                packet = wire.send("up", 8, integers(local), True)
                self.server[i] = np.frombuffer(packet, dtype=I16)
                self.client[i] = local
            return top(self.server.sum(axis=0), self.k), 0, 0, False, 0

        for i, local in enumerate(scores):
            wire.route(i)
            residual = local - self.client[i]
            n = min(d, self.setting)
            if self.kind == "shuffled":
                ids = np.sort(self.rng.choice(d, n, replace=False))
            else:
                ids = np.sort(top(np.abs(residual.astype(np.int32)), n))
            residual[ids] = 0
            limits = np.zeros((self.groups, 2), dtype=I16)
            for g, mask in enumerate(group_masks):
                values = residual[mask]
                if len(values):
                    limits[g] = values.min(), values.max()
            message = wire.send("up", 9, limits.tobytes() + packed_pairs(ids, local[ids]), True)
            decoded = np.frombuffer(message[:limits.nbytes], dtype=I16).reshape(-1, 2)
            received_ids, values = pairs(message[limits.nbytes:])
            self.server[i, received_ids] = values
            self.client[i, ids] = local[ids]
            lo, hi = decoded[group].T.copy()
            lo[received_ids] = hi[received_ids] = 0
            lower += self.server[i] + lo
            upper += self.server[i] + hi

        selected = top(lower, self.k)
        cutoff = lower[selected[-1]]
        other = np.ones(d, dtype=bool)
        other[selected] = False
        gap = int(cutoff - upper[other].max(initial=-1))
        if gap > 0:
            return selected, 0, 0, False, gap
        candidates = np.flatnonzero(upper >= cutoff)
        uncertain = candidates[lower[candidates] != upper[candidates]]
        if len(uncertain):
            # The size comparison assumes uncompressed score words for both options.
            request_bytes = 1 + min(2 * len(uncertain), (d + 7) // 8)
            fallback = HEADER.size + request_bytes + 2 * len(uncertain) >= 2 * d
            index = np.arange(d) if fallback else uncertain
            for i, local in enumerate(scores):
                wire.route(i, 1)
                if fallback:
                    # A framed broadcast distinguishes fallback from no request.
                    wire.send("down", 10, b"")
                ids, values = refresh(wire, local, index)
                self.server[i, ids] = values
                self.client[i, ids] = local[ids]
            if fallback:
                return top(self.server.sum(axis=0), self.k), 1, len(candidates), True, gap
            lower[uncertain] = self.server[:, uncertain].sum(axis=0)
        return candidates[top(lower[candidates], self.k)], int(bool(len(uncertain))), len(candidates), False, gap

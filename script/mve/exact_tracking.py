"""Exact temporal and snapshot baselines for federated screening."""

from dataclasses import dataclass

import numpy as np

from screen_protocol import I16, U16, Wire, integers, packed_pairs, pairs, request, top


@dataclass
class Record:
    phase: int
    client: int
    direction: str
    frame: bytes


class TraceWire(Wire):
    def __init__(self, capture=False):
        super().__init__()
        self.capture = capture
        self.waves = set()
        self.messages = 0
        self.records = []

    def deliver(self, direction, frame):
        self.waves.add((self.phase, direction))
        self.messages += 1
        if self.capture:
            self.records.append(Record(self.phase, self.client_id, direction, frame))
        return super().deliver(direction, frame)


def independent_rank(scores, k):
    sums = scores.sum(axis=0, dtype=np.int64)
    return np.sort(np.lexsort((np.arange(len(sums)), -sums))[:k])


def encode_dense(current, anchor):
    """Select the shorter framed absolute or anchored delta payload."""
    a = Wire.encode(8, integers(current), True)
    b = Wire.encode(15, integers(current - anchor), True)
    return a if len(a) <= len(b) else b


def decode_dense(tag, body, anchor):
    values = np.frombuffer(body, dtype=I16)
    return values if tag == 8 else values + anchor


class AnchoredMonitor:
    """A scalar displacement certificate or an exact safe-zone comparator."""

    def __init__(self, clients, dimension, k, safe_zone=False):
        self.local = np.zeros((clients, dimension), dtype=I16)
        self.server = self.local.copy()
        self.global_at_client = np.zeros((clients, dimension), dtype=np.int64)
        self.k = k
        self.client_count, self.dimension = clients, dimension
        self.ids = np.arange(dimension)
        self.tie_offsets = dimension - 1 - self.ids
        self.safe_zone = safe_zone
        self.selected = None
        self.outside = None
        self.gap = 0

    def step(self, scores, wire):
        if self.selected is None:
            return self._refresh(scores, range(len(scores)), wire, 0), False, 0
        inside = self.selected
        outside = self.outside
        total_gap = 0
        unresolved = []
        violations = 0
        for i, values in enumerate(scores):
            wire.route(i)
            displacement = values - self.local[i]
            if self.safe_zone:
                # Each adjusted local order sums to clients times the global order.
                adjusted = (self.global_at_client[i] + self.client_count * displacement.astype(np.int64))
                keys = self.dimension * adjusted + self.tie_offsets
                if keys[inside].min() > keys[outside].max():
                    wire.send("up", 20, b"")
                    unresolved.append(i)
                    continue
                violations += 1
                tag, body = wire.deliver("up", encode_dense(values, self.local[i]))
                self.server[i] = decode_dense(tag, body, self.server[i])
                self.local[i] = values
            else:
                local_gap = int(displacement[inside].min()) - int(displacement[outside].max())
                returned = wire.send("up", 21, integers([local_gap]))
                total_gap += int(np.frombuffer(returned, dtype=I16)[0])
        if self.safe_zone:
            if violations == 0:
                return inside.copy(), True, 0
            return self._refresh(scores, unresolved, wire, 1), False, violations
        if self.gap + self.dimension * total_gap > 0:
            return inside.copy(), True, 0
        return self._refresh(scores, range(len(scores)), wire, 1), False, 0

    def _refresh(self, scores, sites, wire, phase):
        for i in sites:
            wire.route(i, phase)
            if phase:
                wire.send("down", 22, b"")
            tag, body = wire.deliver("up", encode_dense(scores[i], self.local[i]))
            self.server[i] = decode_dense(tag, body, self.server[i])
            self.local[i] = scores[i]
        aggregate = self.server.sum(axis=0, dtype=np.int64)
        selected = np.sort(top(aggregate, self.k))
        outside = np.setdiff1d(self.ids, selected, assume_unique=True)
        ranked = self.dimension * aggregate + self.tie_offsets
        self.gap = int(ranked[selected].min() - ranked[outside].max())
        self.selected, self.outside = selected, outside
        if self.safe_zone:
            dtype = U16 if 2048 * self.client_count < 2**16 else np.dtype("<i4")
            encoded = integers(aggregate, dtype)
            for i in range(len(scores)):
                wire.route(i, phase + 1)
                body = wire.send("down", 23, encoded, True)
                self.global_at_client[i] = np.frombuffer(body, dtype=dtype)
        return selected.copy()


class BatchedNRA:
    """Exact no-random-access algorithm of Fagin et al. with batched sorted access."""

    def __init__(self, k, block=8):
        self.k, self.block = k, block

    def step(self, scores, wire):
        c, d = scores.shape
        order = np.argsort(-scores.astype(np.int32), axis=1, kind="stable")
        known = np.zeros((c, d), dtype=bool)
        lower = np.zeros(d, dtype=np.int64)
        depth = 0
        while depth * self.block < d:
            end = min((depth + 1) * self.block, d)
            tail = np.zeros(c, dtype=np.int64)
            for i, local in enumerate(scores):
                wire.route(i, depth)
                if depth:
                    wire.send("down", 25, b"")
                indices = order[i, depth * self.block:end]
                received = wire.send("up", 26, packed_pairs(indices, local[indices]), True)
                ids, values = pairs(received)
                lower[ids] += values
                known[i, ids] = True
                tail[i] = values[-1]
            upper = lower + (tail[:, None] * ~known).sum(axis=0)
            initial = top(lower, self.k)
            cutoff = initial[-1]
            outsiders = np.ones(d, dtype=bool)
            outsiders[initial] = False
            upper_tie = (upper == lower[cutoff]) & (np.arange(d) < cutoff)
            if not np.any(outsiders & ((upper > lower[cutoff]) | upper_tie)):
                return np.sort(initial), depth + 1
            depth += 1
        raise AssertionError("Exhausted ranked lists without resolving an exact result")


def announce(wire, chosen, previous, c, d):
    if previous is None or not np.array_equal(chosen, previous):
        for i in range(c):
            wire.route(i, phase=10000)
            assert np.array_equal(request(wire, chosen, d), chosen)

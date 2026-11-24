"""Small framed site worker shared by local tests and the WSL2 service."""
from array import array
from pathlib import Path
import json
import socketserver
import struct
import sys
import threading
import zlib

HEADER = struct.Struct("<BBI")
PAIR = struct.Struct("<Hh")
ADJUST = struct.Struct("<Hi")
PIVOT = struct.Struct("<q")
OPEN, STOP, STEP, FETCH, FULL, MORE, COMMIT = range(30, 37)
OK, DENSE, DELTA, LIST, ENVELOPE, EXCEPTIONS, VALUES = range(40, 47)
NRABLOCK = 8


def words(values, fmt="h"):
    output = array(fmt, (int(x) for x in values))
    if sys.byteorder != "little":
        output.byteswap()
    return output.tobytes()


def unpack(data, fmt="h"):
    output = array(fmt)
    output.frombytes(data)
    if sys.byteorder != "little":
        output.byteswap()
    return output.tolist()


def pack(tag, body=b"", compress=True):
    zipped = zlib.compress(body, level=1) if compress else body
    use = len(zipped) < len(body)
    payload = zipped if use else body
    return HEADER.pack(tag, use, len(payload)) + payload


def decode(frame):
    tag, zipped, size = HEADER.unpack_from(frame)
    assert len(frame) == HEADER.size + size
    body = frame[HEADER.size:]
    return tag, zlib.decompress(body) if zipped else body


def receive(sock, size):
    body = bytearray(size)
    view = memoryview(body)
    while view:
        read = sock.recv_into(view)
        if not read:
            raise EOFError("TCP peer closed")
        view = view[read:]
    return bytes(body)


def receive_frame(sock):
    header = receive(sock, HEADER.size)
    _, _, size = HEADER.unpack(header)
    if size > 2**22:
        raise ValueError("Benchmark frame exceeded configured limit")
    return header + receive(sock, size)


def indices_body(ids, d):
    as_list = words(ids, "H")
    if len(as_list) <= (d + 7) // 8:
        return b"\x00" + as_list
    mask = bytearray((d + 7) // 8)
    for j in ids:
        mask[j // 8] |= 1 << (j % 8)
    return b"\x01" + mask


def indices_from(content, d):
    if content[0] == 0:
        return unpack(content[1:], "H")
    return [i for i in range(d) if content[1 + i // 8] & (1 << (i % 8))]


def pair_bytes(ids, scores):
    return b"".join(PAIR.pack(int(j), int(scores[j])) for j in ids)


class Site:
    def __init__(self, rows, method, clients, d, k):
        self.rows, self.method, self.clients, self.d, self.k = rows, method, clients, d, k
        self.cache, self.adjustment = [0] * d, [0] * d
        self.selected = set()
        self.pivot = 0
        self.row = None
        self.cursor = 0

    def dense(self, frame=None):
        if frame is None:
            a = pack(DENSE, words(self.row))
            b = pack(DELTA, words(x-y for x,y in zip(self.row,self.cache)))
            frame = min((a,b),key=len)
        self.cache = list(self.row)
        return frame

    def nra(self):
        ids = self.order[self.cursor:self.cursor+NRABLOCK]
        self.cursor += len(ids)
        return pack(LIST,pair_bytes(ids,self.row))

    def residual(self):
        residual = [x-a for x,a in zip(self.row,self.cache)]
        exact = set(sorted(range(self.d),key=lambda j:(-abs(residual[j]),j))[:2])
        groups = [[] for _ in range(4)]
        for i,val in enumerate(residual):
            if i not in exact:
                group = 0 if i in self.selected else 1+i%3
                groups[group].append(val)
        bounds = [coordinate for group in groups for coordinate in (min(group,default=0),max(group,default=0))]
        compressed = pack(ENVELOPE,words(bounds)+pair_bytes(sorted(exact),self.row))
        a=pack(DENSE,words(self.row))
        # Cycle-2 fixed 25% rule. Cached residuals are relative to exact values previously sent.
        if 4*len(compressed)>len(a):
            return self.dense(a)
        for i in exact:
            self.cache[i]=self.row[i]
        return compressed

    def violation(self):
        divisor = self.d*self.clients
        ids=[j for j,score in enumerate(self.row)
             if ((divisor*score+self.d*self.adjustment[j]+self.d-1-j)>self.pivot)!=(j in self.selected)]
        sparse = pack(EXCEPTIONS,pair_bytes(ids,self.row))
        if not ids:
            return sparse
        a=pack(DENSE,words(self.row))
        b=pack(DELTA,words(x-y for x,y in zip(self.row,self.cache)))
        full=min((a,b),key=len)
        if len(sparse)<=len(full):
            for j in ids:
                self.cache[j]=self.row[j]
            return sparse
        return self.dense(full)

    def handle(self, frame):
        tag, body = decode(frame)
        if tag == STEP:
            self.row = self.rows[unpack(body,"H")[0]]
            if self.method == "nra-8":
                self.order=sorted(range(self.d),key=lambda j:(-self.row[j],j))
                self.cursor=0
                return self.nra()
            if self.method == "dense-best" or not self.selected:
                return self.dense()
            return self.residual() if self.method == "hybrid-25" else self.violation()
        if tag == MORE:
            return self.nra()
        if tag == FULL:
            return self.dense()
        if tag == FETCH:
            ids=indices_from(body,self.d)
            for j in ids:
                self.cache[j]=self.row[j]
            return pack(VALUES,words(self.row[j] for j in ids))
        if tag == COMMIT:
            changed, install = body[:2]
            cursor=2
            if changed:
                self.selected=set(unpack(body[cursor:cursor+2*self.k],"H"))
                cursor+=2*self.k
            if install:
                self.pivot=PIVOT.unpack_from(body,cursor)[0]
                repairs=body[cursor+PIVOT.size:]
                if install==1:
                    for j,val in ADJUST.iter_unpack(repairs):
                        self.adjustment[j]=val
                elif install==2:
                    self.adjustment=unpack(repairs,"i")
            return pack(OK)
        raise ValueError(f"Unsupported site tag {tag}")


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            tag,payload=decode(receive_frame(self.request))
            if tag==STOP and payload.decode()==self.server.token:
                self.request.sendall(pack(OK))
                threading.Thread(target=self.server.shutdown,daemon=True).start()
                return
            if tag!=OPEN:
                raise ValueError("Expected new site initialization")
            req=json.loads(payload)
            source=Path(self.server.inputs) / req["case"] / f'{int(req["id"])}.json'
            if source.parent.parent.resolve()!=Path(self.server.inputs).resolve():
                raise ValueError("Case path must be in local input directory")
            site=Site(json.loads(source.read_text(encoding="ascii")),req["method"],req["c"],req["d"],req["k"])
            self.request.sendall(pack(OK,b"wsl-linux" if sys.platform.startswith("linux") else b"windows"))
            while True:
                self.request.sendall(site.handle(receive_frame(self.request)))
        except (EOFError,ConnectionResetError,BrokenPipeError):
            return


class Daemon(socketserver.ThreadingTCPServer):
    daemon_threads=True
    allow_reuse_address=True


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--port",type=int,required=True)
    p.add_argument("--inputs",required=True)
    p.add_argument("--token",required=True)
    args=p.parse_args()
    with Daemon(("127.0.0.1",args.port),Handler) as server:
        server.inputs=args.inputs
        server.token=args.token
        print("initialized",flush=True)
        server.serve_forever(poll_interval=.1)

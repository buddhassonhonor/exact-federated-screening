"""Shared-link online experiment across Windows and WSL2 TCP endpoints."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import random
import secrets
import socket
import subprocess
from time import perf_counter,sleep

from tracking_site import OPEN, OK, STOP, decode, pack, receive_frame

PROFILES={
    "narrow":(.25,2.0,1.0),
    "moderate":(2.0,5.0,3.0),
    "interactive":(20.0,14.0,7.0),
}


def linux_path(value):
    p=Path(value).resolve()
    return f"/mnt/{p.drive[0].lower()}/"+p.relative_to(p.anchor).as_posix()


def exchange(peer,frame):
    peer.sendall(frame)
    return receive_frame(peer)


class WslWorker:
    def __init__(self,inputs,log):
        self.inputs,self.log=inputs,log
        self.token=secrets.token_hex(16)
        self.proc=None

    def __enter__(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1",0))
            self.port=probe.getsockname()[1]
        worker=Path(__file__).with_name("tracking_site.py")
        args=["wsl.exe","-d","Ubuntu","--","python3",linux_path(worker),
              "--port",str(self.port),"--inputs",linux_path(self.inputs),"--token",self.token]
        self.file=self.log.open("wb")
        self.proc=subprocess.Popen(args,stdout=self.file,stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        for _ in range(60):
            try:
                with socket.create_connection(self.address,timeout=.2):
                    return self
            except OSError:
                if self.proc.poll() is not None:
                    break
                sleep(.12)
        self.__exit__()
        raise RuntimeError(f"WSL server did not initialize; inspect {self.log}")

    @property
    def address(self):
        return "127.0.0.1",self.port

    def __exit__(self,*unused):
        if self.proc.poll() is None:
            try:
                with socket.create_connection(self.address,timeout=2) as peer:
                    assert decode(exchange(peer,pack(STOP,self.token.encode())))[0]==OK
                self.proc.wait(timeout=6)
            except (OSError,EOFError,subprocess.TimeoutExpired):
                self.proc.terminate()
                self.proc.wait(timeout=10)
        self.file.close()


class SocketLinks:
    """Logical clients compute replies online. Timing shapes one shared byte budget per direction."""
    def __init__(self,address,case,method,c,d,k,profile):
        self.address,self.profile=address,profile
        self.session={i:pack(OPEN,json.dumps(dict(case=case,method=method,id=i,c=c,d=d,k=k)).encode()) for i in range(c)}
        self.pool=ThreadPoolExecutor(max_workers=c)
        self.peers={}
        self.fault=None
        self.seed=0
        self.reset_counts()

    def enter(self):
        try:
            self.connect()
        except BaseException:
            self.close()
            raise
        return self

    def connect(self):
        for i,frame in self.session.items():
            peer=socket.create_connection(self.address,timeout=10)
            peer.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            self.peers[i]=peer
            tag,body=decode(exchange(peer,frame))
            assert tag==OK and body==b"wsl-linux"

    def disconnect(self):
        for peer in self.peers.values():
            peer.close()
        self.peers.clear()

    def close(self):
        self.disconnect()
        self.pool.shutdown()

    def reset_counts(self):
        self.up=self.down=0
        self.waves=[]
        self.rtts=[]
        self.frames=[]
        self.rng=random.Random(self.seed)

    def wave(self,requests):
        mbps,center,spread=PROFILES[self.profile]
        delay=max(.2,center+self.rng.uniform(-spread,spread))
        if self.rng.random()<.07:
            delay+=center
        down=sum(map(len,requests.values()))
        sleep(delay/2000+8*down/(mbps*1e6))
        if self.fault in requests:
            self.peers[self.fault].close()
            self.fault=None
        tasks={i:self.pool.submit(exchange,self.peers[i],frame) for i,frame in requests.items()}
        replies={}
        failure=None
        for i,job in tasks.items():
            try:
                replies[i]=job.result()
            except (OSError,EOFError) as err:
                failure=err
        if failure is not None:
            raise failure
        up=sum(map(len,replies.values()))
        sleep(delay/2000+8*up/(mbps*1e6))
        self.up+=up
        self.down+=down
        self.waves.append((down,up))
        self.rtts.append(delay)
        self.frames.append((requests.copy(),replies))
        return replies

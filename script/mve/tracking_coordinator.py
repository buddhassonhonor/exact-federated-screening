"""Coordinator for exact online screening with independent site states."""
import numpy as np

from tracking_site import (
    ADJUST, COMMIT, DELTA, DENSE, ENVELOPE, EXCEPTIONS, FETCH, FULL, LIST, MORE, NRABLOCK,
    OK, PAIR, PIVOT, STEP, VALUES, Site, decode, indices_body, pack, unpack, words,
)

METHODS=("dense-best","hybrid-25","nra-8","partial-zone")
ZONE_METHODS={"partial-zone","full-zone","frozen-zone"}
MAX_SCORE=2048


def top(values,k):
    return np.argsort(-values,kind="stable")[:k]


def independent_rank(scores,k):
    summed=np.asarray(scores,dtype=np.int64).sum(axis=0)
    return np.sort(np.lexsort((np.arange(len(summed)),-summed))[:k])


def ranking_frontier(lo,hi,k):
    order=top(lo,k)
    boundary=order[-1]
    keep=(hi>lo[boundary])|((hi==lo[boundary])&(np.arange(len(lo))<=boundary))
    return np.flatnonzero(keep)


class LocalLinks:
    """In-process byte-accurate execution with identical request and response steps."""

    def __init__(self, scores, method, k):
        self.sites=[Site(panel.tolist(),method,scores.shape[1],scores.shape[2],k)
                    for panel in scores.transpose(1,0,2)]
        self.reset_counts()

    def reset_counts(self):
        self.up=self.down=0
        self.waves=[]

    def wave(self, requests):
        answers={i:self.sites[i].handle(request) for i,request in requests.items()}
        up=sum(map(len,answers.values()))
        down=sum(map(len,requests.values()))
        self.up+=up
        self.down+=down
        self.waves.append((down,up))
        return answers


class Coordinator:
    def __init__(self,c,d,k,method,links):
        self.c,self.d,self.k,self.method,self.links=c,d,k,method,links
        self.offsets=np.zeros((c,d),dtype=np.int64)
        self.cache=np.zeros((c,d),dtype=np.int64)
        self.ids=np.arange(d)
        self.tie=d-1-self.ids
        self.selected=np.array([],dtype=np.int64)
        self.pivot=0

    def broadcast(self,packet):
        return self.links.wave({i:packet for i in range(self.c)})

    def dense(self,i,tag,body):
        parsed=np.asarray(unpack(body),dtype=np.int64)
        if tag==DELTA:
            parsed+=self.cache[i]
        self.cache[i]=parsed
        return parsed

    def commit(self,selected,repairs=None,pivot=None):
        changed=not np.array_equal(self.selected,selected)
        if not changed and repairs is None:
            return
        head=words(selected,"H") if changed else b""
        requests={}
        for i in range(self.c):
            if repairs is None:
                requests[i]=pack(COMMIT,bytes([int(changed),0])+head)
            else:
                ids=np.flatnonzero(repairs[i]!=self.offsets[i])
                payload=bytes([int(changed),1])+head+PIVOT.pack(int(pivot))
                delta=pack(COMMIT,payload+b"".join(ADJUST.pack(j,repairs[i,j]) for j in ids))
                full=pack(COMMIT,bytes([int(changed),2])+head+PIVOT.pack(int(pivot))+words(repairs[i],"i"))
                requests[i]=min((delta,full),key=len)
        assert all(decode(x)[0]==OK for x in self.links.wave(requests).values())
        self.selected=selected
        if repairs is not None:
            self.offsets,self.pivot=repairs,int(pivot)

    def rank_nra(self,first):
        known=np.zeros((self.c,self.d),dtype=bool)
        lower=np.zeros(self.d,dtype=np.int64)
        parts=first
        for depth in range(0,self.d,NRABLOCK):
            tails=np.zeros(self.c,dtype=np.int64)
            for i,content in parts.items():
                tag,body=decode(content)
                assert tag==LIST
                batch=list(PAIR.iter_unpack(body))
                ids,values=map(np.asarray,zip(*batch))
                known[i,ids]=True
                lower[ids]+=values
                tails[i]=values[-1]
            hi=lower+((~known)*tails[:,None]).sum(axis=0)
            if len(ranking_frontier(lower,hi,self.k))==self.k:
                return np.sort(top(lower,self.k))
            if depth+NRABLOCK<self.d:
                parts=self.broadcast(pack(MORE))
        raise AssertionError("NRA exhausted lists")

    def refine(self,lo,hi):
        low,high=lo.sum(axis=0),hi.sum(axis=0)
        candidates=ranking_frontier(low,high,self.k)
        if len(candidates)==self.k:
            return np.sort(candidates),0
        fetch={}
        requests={}
        for i in range(self.c):
            ids=candidates[lo[i,candidates]!=hi[i,candidates]]
            if len(ids):
                fetch[i]=ids
                requests[i]=pack(FULL) if len(ids)==self.d else pack(FETCH,indices_body(ids,self.d))
        replies=self.links.wave(requests) if requests else {}
        for i,content in replies.items():
            tag,body=decode(content)
            if tag==VALUES:
                ids=fetch[i]
                self.cache[i,ids]=unpack(body)
                lo[i,ids]=hi[i,ids]=self.cache[i,ids]
            else:
                assert tag in (DENSE,DELTA)
                lo[i]=hi[i]=self.dense(i,tag,body)
        chosen=candidates[top(lo[:,candidates].sum(axis=0),self.k)]
        return np.sort(chosen),sum(map(len,fetch.values()))

    def full_resolution(self,lo,hi):
        incomplete=np.flatnonzero(np.any(lo!=hi,axis=1))
        if len(incomplete):
            requests={int(i):pack(FULL) for i in incomplete}
            for i,content in self.links.wave(requests).items():
                tag,body=decode(content)
                assert tag in (DENSE,DELTA)
                lo[i]=hi[i]=self.dense(i,tag,body)
        return np.sort(top(lo.sum(axis=0),self.k)),len(incomplete)*self.d

    def allocate(self,lo,hi,chosen):
        inside=np.zeros(self.d,dtype=bool)
        inside[chosen]=True
        low=self.d*lo.sum(axis=0)+self.tie
        high=self.d*hi.sum(axis=0)+self.tie
        pivot_min=high[~inside].max()
        pivot_max=low[inside].min()-1
        assert pivot_min<=pivot_max
        pivot=np.clip(self.pivot,pivot_min,pivot_max) if len(self.selected) else (pivot_min+pivot_max)//2
        required_lower=(pivot+1-self.tie+self.d-1)//self.d-self.c*lo
        required_upper=(pivot-self.tie)//self.d-self.c*hi
        wrong_low=np.any(self.offsets<required_lower,axis=0)&inside
        wrong_high=np.any(self.offsets>required_upper,axis=0)&~inside
        repairs=self.offsets.copy()
        for wrong,limit,sign in ((wrong_low,required_lower,1),(wrong_high,required_upper,-1)):
            indexes=np.flatnonzero(wrong)
            total=-sign*limit[:,indexes].sum(axis=0)
            assert np.all(total>=0)
            share,remainder=np.divmod(total,self.c)
            repairs[:,indexes]=limit[:,indexes]+sign*(share+(np.arange(self.c)[:,None]<remainder))
        assert np.all(repairs.sum(axis=0)==0)
        changed=np.count_nonzero(repairs!=self.offsets)
        send=pivot!=self.pivot or changed or not len(self.selected)
        return (repairs if send else None),int(pivot),int(changed)

    def advance(self,t):
        self.links.reset_counts()
        first=self.broadcast(pack(STEP,words([t],"H")))
        fetched,repairs,violators=0,0,0
        if self.method=="nra-8":
            chosen=self.rank_nra(first)
            self.commit(chosen)
        else:
            lo=np.zeros((self.c,self.d),dtype=np.int64)
            hi=np.full_like(lo,MAX_SCORE)
            inside=np.zeros(self.d,dtype=bool)
            inside[self.selected]=True
            groups=np.where(inside,0,1+self.ids%3)
            keep=True
            for i,answer in first.items():
                tag,body=decode(answer)
                if tag in (DENSE,DELTA):
                    lo[i]=hi[i]=self.dense(i,tag,body)
                    keep=False
                elif tag==ENVELOPE:
                    extrema=np.asarray(unpack(body[:16])).reshape(4,2)
                    lo[i]=self.cache[i]+extrema[groups,0]
                    hi[i]=self.cache[i]+extrema[groups,1]
                    body=body[16:]
                    keep=False
                elif tag==EXCEPTIONS:
                    numerator=self.pivot-self.tie-self.d*self.offsets[i]
                    divisor=self.c*self.d
                    lo[i,inside]=(numerator[inside]+divisor)//divisor
                    hi[i,~inside]=numerator[~inside]//divisor
                    if body:
                        violators+=1
                        keep=False
                else:
                    raise AssertionError(tag)
                for j,x in PAIR.iter_unpack(body) if tag in (EXCEPTIONS,ENVELOPE) else ():
                    self.cache[i,j]=lo[i,j]=hi[i,j]=x
            lo=np.clip(lo,0,MAX_SCORE)
            hi=np.clip(hi,0,MAX_SCORE)
            assert np.all(lo<=hi)
            if keep:
                assert self.method in ZONE_METHODS and len(self.selected)==self.k
                chosen=self.selected
            else:
                if self.method=="full-zone":
                    chosen,fetched=self.full_resolution(lo,hi)
                else:
                    chosen,fetched=self.refine(lo,hi)
                if self.method in ZONE_METHODS and (self.method!="frozen-zone" or not len(self.selected)):
                    changes,pivot,repairs=self.allocate(lo,hi,chosen)
                    self.commit(chosen,changes,pivot)
                else:
                    self.commit(chosen)
        stats=dict(up=int(self.links.up),down=int(self.links.down),wave_sizes=self.links.waves.copy(),
                   rounds_of_messages=len(self.links.waves),fetched=fetched,repairs=repairs,reported_violators=violators)
        return chosen.copy(),stats

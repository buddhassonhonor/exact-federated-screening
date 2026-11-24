"""Run the third research cycle with saved inputs, decisions, frames and checks."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path
import random
import shutil
from time import perf_counter
import traceback

from run_exact_study import ROOT,RUNS,archived_files,setup,choose_bin
from run_screening import local_histogram,score,evaluate,fit
import numpy as np
from native_bike import prepare
from native_streams import get_block
from stream_data import digest
from tracking_coordinator import Coordinator,LocalLinks,METHODS,independent_rank
from tracking_transport import PROFILES,SocketLinks,WslWorker

NEW_SOURCE=("tracking_site.py","tracking_coordinator.py","tracking_transport.py","native_bike.py","run_cycle3.py")


def snapshot(run,args):
    setup(run,args)
    import difflib
    with (run/"diff.patch").open("a",encoding="utf-8") as patch:
        for file in NEW_SOURCE:
            src=Path(__file__).with_name(file)
            shutil.copy2(src,run/"source"/file)
            patch.writelines(difflib.unified_diff([],src.read_text(encoding="utf-8-sig").splitlines(keepends=True),
                                                fromfile="/dev/null",tofile="b/script/mve/"+file))


def fixture(seed):
    rng=np.random.default_rng(seed)
    all_rows=[]
    for c in (1,3,12):
        for d in (2,19,70):
            for limit in (2,12,2049):
                for k in {1,min(4,d-1),d-1}:
                    stream=rng.integers(0,limit,(17,c,d),dtype=np.int16)
                    stream[1]=stream[0]
                    stream[2:,0,:k]=stream[1,0,:k]
                    # Small changes outside the selected sets and ties are evaluated too.
                    archive=(f"fixture-{c}-{d}-{limit}-{k}",seed)
                    all_rows.extend(offline(stream,k,*archive))
    return all_rows


def offline(scores,k,name,unit):
    c,d=scores.shape[1:]
    results=[]
    engines={m:Coordinator(c,d,k,m,LocalLinks(scores,m,k)) for m in METHODS}
    for t,s in enumerate(scores):
        expected=independent_rank(s,k)
        for m,engine in engines.items():
            start=perf_counter()
            ranked,stats=engine.advance(t)
            elapsed=(perf_counter()-start)*1000
            assert np.array_equal(ranked,expected),(name,m,unit,t,ranked,expected)
            results.append(dict(case=name,unit=unit,d=d,c=c,method=m,round=t,local_ms=elapsed,**stats))
    return results


def ablations(inputs,records):
    methods=(*METHODS,"full-zone","frozen-zone")
    for unit,(name,path) in enumerate(inputs.items()):
        with np.load(path,allow_pickle=False) as data:
            scores,ranks=data["scores"],data["exact_rank"]
        c,d=scores.shape[1:]
        k=ranks.shape[1]
        for method in methods:
            control=Coordinator(c,d,k,method,LocalLinks(scores,method,k))
            for t,expected in enumerate(ranks):
                ranked,stats=control.advance(t)
                assert np.array_equal(ranked,expected)
                print_rows(records,[dict(case=name,d=d,c=c,unit=unit,round=t,method=method,**stats)])


def replay(path):
    p=Path(path)
    with np.load(p,allow_pickle=False) as data:
        scores,ranks=data["scores"],data["exact_rank"]
    assert all(np.array_equal(independent_rank(s,ranks.shape[1]),r) for s,r in zip(scores,ranks,strict=True))
    case,unit=p.stem.rsplit("__s",1)
    return offline(scores,ranks.shape[1],case,int(unit))


def development_policy(path,output):
    import pandas as pd
    data=pd.read_json(path,lines=True)
    data["bytes"]=data.up+data.down
    data["width"]=data.d.map(choose_bin)
    table={}
    selected={}
    for profile,(mbps,rtt,_spread) in PROFILES.items():
        estimated=rtt*data.rounds_of_messages+.008*data.bytes/mbps
        data["model_ms"]=estimated
        pivot=data.pivot(index=["case","unit","width","round"],columns="method",values="model_ms")
        ratios=pivot.div(pivot["dense-best"],axis=0)
        cases=ratios.groupby(level=["case","unit","width"])
        score_by_case=(cases.quantile(.5)+cases.quantile(.9))*.5
        table[profile]=score_by_case.groupby(level=["case","width"]).mean().groupby(level="width").mean()[list(METHODS)].to_dict("index")
        selected[profile]={width:min(list(METHODS),key=cost.get) for width,cost in table[profile].items()}
    policy={"selection":selected,"development_objectives":table,"profiles":PROFILES,
            "objective":"Mean across cases of paired median and p90 normalized modeled time, equal weight",
            "development_trace_sha256":digest(path),"alternatives":METHODS}
    output.write_text(json.dumps(policy,indent=2)+"\n",encoding="utf-8")
    return policy


def scores_for_bike(data_root,artifact,wide):
    path,meta=prepare(data_root,ROOT/"data"/"derived"/"native_v2",wide)
    name="bike-wide" if wide else "bike"
    (artifact/f"{name}_source.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    with np.load(path,allow_pickle=False) as source:
        x,y=source["features"],source["labels"]
    output={}
    for i,window in enumerate(meta["periods"]):
        local=[]
        predictions=[]
        histograms=deque(maxlen=3)
        model=before=None
        for data in get_block(x,y,window):
            predictions.append(evaluate(data,model).tolist())
            histograms.append(tuple(np.array(v) for v in zip(*(local_histogram(f,label) for f,label in data))))
            h,n=(np.stack(items).sum(axis=0) for items in zip(*histograms))
            scores=score(h,n)
            local.append(scores)
            features=independent_rank(scores,4)
            model,up,down=fit(h,n,features,before)
            before=features
        dest=artifact/f"{name}-block{i}.npz"
        np.savez_compressed(dest,scores=local,exact_rank=[independent_rank(s,4) for s in local],
                            predictions=predictions)
        output[f"{name}-block{i}"]=dest
    return output


def print_rows(out,rows):
    for row in rows:
        print(json.dumps(row,separators=(",",":")),file=out)
    out.flush()


def online(output,inputs,policy,log):
    tapes={}
    directory=output/"artifacts"/"site_inputs"
    for name,npz in inputs.items():
        with np.load(npz,allow_pickle=False) as data:
            scores,ranks=data["scores"],data["exact_rank"]
        tapes[name]=(scores,ranks)
        folder=directory/name
        folder.mkdir(parents=True)
        for i,s in enumerate(scores.transpose(1,0,2)):
            (folder/f"{i}.json").write_text(json.dumps(s.tolist(),separators=(",",":")),encoding="ascii")
    paths={name:{"path":str(p),"sha256":digest(p)} for name,p in inputs.items()}
    (output/"artifacts"/"inputs.json").write_text(json.dumps(paths,indent=2),encoding="utf-8")
    order=random.Random(20261124)
    jobs=[(name,profile,repetition,method) for name in tapes for profile in PROFILES for repetition in (0,1) for method in METHODS]
    order.shuffle(jobs)
    log("Online benchmark uses Windows coordinator and Linux WSL2 site workers.")
    with WslWorker(directory,output/"artifacts"/"wsl.log") as linux, (
        output/"metrics.jsonl"
    ).open("w",encoding="utf-8") as stream, gzip.open(
        output/"artifacts"/"wire_frames.jsonl.gz","wt",encoding="utf-8"
    ) as frames:
        for position,(name,profile,repetition,method) in enumerate(jobs,1):
            scores,ranks=tapes[name]
            c,d=scores.shape[1:]
            k=ranks.shape[1]
            backend=SocketLinks(linux.address,name,method,c,d,k,profile).enter()
            engine=Coordinator(c,d,k,method,backend)
            rows=[]
            try:
                for t,expected in enumerate(ranks):
                    backend.seed=(1+list(tapes).index(name))*1000000+repetition*10000+t
                    start=perf_counter()
                    chosen,stat=engine.advance(t)
                    elapsed=1000*(perf_counter()-start)
                    assert np.array_equal(chosen,expected)
                    rows.append(dict(case=name,profile=profile,repetition=repetition,method=method,round=t,
                                     elapsed_ms=elapsed,chosen=chosen.tolist(),rtts_ms=backend.rtts.copy(),
                                     chosen_by_policy=int(method==policy["selection"][profile][choose_bin(d)]),**stat))
                    item = dict(case=name, profile=profile, repetition=repetition, method=method, round=t,
                                waves=[[[i,a[i].hex(),b[i].hex()] for i in sorted(a)] for a,b in backend.frames])
                    print(json.dumps(item,separators=(",",":")),file=frames)
            finally:
                backend.close()
            print_rows(stream,rows)
            if position%15==0 or position==len(jobs):
                log(f"{position} of {len(jobs)} online sequences")
        recoveries=[]
        name=next(iter(tapes))
        scores,ranks=tapes[name]
        c,d=scores.shape[1:]
        k=ranks.shape[1]
        for method in METHODS:
            switch=SocketLinks(linux.address,name,method,c,d,k,"moderate").enter()
            core=Coordinator(c,d,k,method,switch)
            try:
                for t in range(3):
                    assert np.array_equal(core.advance(t)[0],ranks[t])
                switch.fault=c//2
                switch.seed=749
                start=perf_counter()
                try:
                    core.advance(3)
                except (OSError,EOFError) as failure:
                    detected=type(failure).__name__
                    switch.disconnect()
                    rebound=perf_counter()
                    switch.connect()
                    boundary=perf_counter()
                    reset=Coordinator(c,d,k,method,switch)
                    recovered,stats=reset.advance(3)
                    finish=perf_counter()
                    assert np.array_equal(recovered,ranks[3])
                    assert np.array_equal(reset.advance(4)[0],ranks[4])
                else:
                    raise AssertionError("Injected socket closure did not interrupt the round")
                recoveries.append(dict(case=name,method=method,round=3,detected=detected,
                                       interruption_ms=(rebound-start)*1000,reconnect_ms=(boundary-rebound)*1000,
                                       retry_ms=(finish-boundary)*1000,recovery_ms=(finish-start)*1000,**stats))
            finally:
                switch.close()
        (output/"artifacts"/"recoveries.json").write_text(json.dumps(recoveries,indent=2)+"\n",encoding="utf-8")
        log(f"Verified {len(recoveries)} interrupted TCP rounds and subsequent rounds")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("pilot","development","heldout","online","ablation"))
    parser.add_argument("--slug",required=True)
    parser.add_argument("--policy-run")
    parser.add_argument("--heldout-run")
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--data-root",type=Path,default=Path("D:/data"))
    args=parser.parse_args()
    run=RUNS/args.slug
    snapshot(run,args)
    artifact=run/"artifacts"
    with (run/"stdout_stderr.log").open("w",encoding="utf-8") as messages:
        def log(text):
            print(text,flush=True)
            print(text,file=messages,flush=True)
        log(f"{datetime.now(timezone.utc).isoformat()} {args.stage}")
        try:
            if args.stage in ("pilot","development"):
                if args.stage=="development":
                    archive=archived_files()
                    (artifact/"input_hashes.json").write_text(json.dumps(archive,indent=2),encoding="utf-8")
                    data,fn=tuple(archive),replay
                else:
                    data,fn=(7,13,19,29,41),fixture
                n=0
                with ProcessPoolExecutor(max_workers=args.workers) as pool,(run/"metrics.jsonl").open("w",encoding="utf-8") as records:
                    for i,rows in enumerate(pool.map(fn,data),1):
                        print_rows(records,rows)
                        n+=len(rows)
                        if i%50==0 or i==len(data):
                            log(f"{i}/{len(data)} sequences, {n} comparisons")
                if args.stage=="development":
                    log(development_policy(run/"metrics.jsonl",artifact/"policy.json")["selection"])
            else:
                if not args.policy_run:
                    raise ValueError("A frozen policy run is required")
                policy_file=RUNS/args.policy_run/"artifacts"/"policy.json"
                policy=json.loads(policy_file.read_text(encoding="utf-8"))
                shutil.copy2(policy_file,artifact/"frozen_policy.json")
                (artifact/"policy_sha256.txt").write_text(digest(policy_file)+"\n",encoding="utf-8")
                if args.stage=="heldout":
                    inputs={}
                    for wide in (False,True):
                        inputs.update(scores_for_bike(args.data_root,artifact,wide))
                    with (run/"metrics.jsonl").open("w",encoding="utf-8") as records:
                        for i,(case,path) in enumerate(inputs.items()):
                            with np.load(path,allow_pickle=False) as data:
                                scores,k=data["scores"],data["exact_rank"].shape[1]
                            print_rows(records,offline(scores,k,case,i))
                    hashes={name:digest(path) for name,path in inputs.items()}
                    (artifact/"score_hashes.json").write_text(json.dumps(hashes,indent=2),encoding="utf-8")
                    log(f"{len(hashes)} held-out chronological blocks")
                else:
                    previous=RUNS/args.heldout_run/"artifacts"
                    hashes=json.loads((previous/"score_hashes.json").read_text(encoding="utf-8"))
                    inputs={name:previous/f"{name}.npz" for name in hashes}
                    assert all(digest(file)==hashes[name] for name,file in inputs.items())
                    assert digest(previous/"frozen_policy.json")==digest(policy_file)
                    if args.stage=="online":
                        online(run,inputs,policy,log)
                    else:
                        with (run/"metrics.jsonl").open("w",encoding="utf-8") as records:
                            ablations(inputs,records)
                        log(f"{len(inputs)} paired blocks with six transport settings")
        except BaseException:
            log(traceback.format_exc())
            raise


if __name__=="__main__":
    main()

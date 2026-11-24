"""Verify third-cycle artifacts and generate paper results from measured frames."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_exact_study import ROOT,RUNS
from stream_data import digest
from tracking_coordinator import METHODS,independent_rank
from tracking_site import decode

DIMS={"bike": "Base dictionary", "bike-wide": "Expanded dictionary"}
NAMES={"dense-best":"Best dense","hybrid-25":"Grouped hybrid","nra-8":"Batched NRA",
       "partial-zone":"Partial repair","full-zone":"Full refresh","frozen-zone":"Frozen slack"}
PROFILES=("narrow","moderate","interactive")
FIG=ROOT/"figures"


def table(outfile,columns,rows,design):
    def line(values):
        return " & ".join(values)+r" \\"
    lines=[r"\begin{tabular}{"+design+"}",r"\toprule",line(columns),r"\midrule"]
    lines.extend(line(v) for v in rows)
    lines.extend([r"\bottomrule",r"\end{tabular}"])
    (FIG/outfile).write_text("\n".join(lines)+"\n",encoding="utf-8")


def verify(args,network):
    held=RUNS/args.heldout/"artifacts"
    staged=RUNS/args.online/"artifacts"
    dev=RUNS/args.dev
    policy_file=dev/"artifacts"/"policy.json"
    policy=json.loads(policy_file.read_text(encoding="utf-8"))
    assert digest(dev/"metrics.jsonl")==policy["development_trace_sha256"]
    assert digest(policy_file)==digest(held/"frozen_policy.json")==digest(staged/"frozen_policy.json")
    datas=json.loads((held/"score_hashes.json").read_text(encoding="utf-8"))
    ranks={}
    for name,sha in datas.items():
        path=held/f"{name}.npz"
        assert digest(path)==sha
        with np.load(path,allow_pickle=False) as src:
            samples=src["scores"]
            ranks[name]=src["exact_rank"]
        assert all(np.array_equal(independent_rank(x,ranks[name].shape[1]),y)
                   for x,y in zip(samples,ranks[name],strict=True))
    base=ROOT/"data"/"derived"/"native_v2"
    for name,local in (("bike","bike"),("bike-wide","bike_wide")):
        info=json.loads((held/f"{name}_source.json").read_text(encoding="utf-8"))
        assert info["derived_sha256"]==digest(base/f"{local}.npz")
        assert all(asset["sha256"]==digest(args.data_root/"capital_bikeshare_2022"/asset["name"])
                   for asset in info["source"]["assets"])
    checks={}
    fields=["case","profile","repetition","method","round"]
    verified_frames=0
    with gzip.open(staged/"wire_frames.jsonl.gz","rt",encoding="utf-8") as stream:
        for line in stream:
            row=json.loads(line)
            lengths=[]
            for wave in row["waves"]:
                up=down=0
                for _,s,r in wave:
                    request,reply=bytes.fromhex(s),bytes.fromhex(r)
                    decode(request)
                    decode(reply)
                    down+=len(request)
                    up+=len(reply)
                    verified_frames+=2
                lengths.append([down,up])
            checks[tuple(row[k] for k in fields)]=lengths
    for row in network.to_dict("records"):
        key=tuple(row[k] for k in fields)
        assert checks[key]==row["wave_sizes"]
        assert sum(b[0] for b in checks[key])==row["down"]
        assert sum(b[1] for b in checks[key])==row["up"]
        assert len(checks[key])==row["rounds_of_messages"]==len(row["rtts_ms"])
        assert row["chosen"]==ranks[row["case"]][row["round"]].tolist()
    assert len(checks)==len(network)==len(datas)*len(PROFILES)*4*2*7
    return {"policy":policy,"policy_sha256":digest(policy_file),"rank_rounds":sum(map(len,ranks.values())),
            "verified_frames":verified_frames,"verified_network_rounds":len(checks)}


def native(held):
    base=held/"artifacts"
    output={}
    for name in DIMS:
        paths=sorted(base.glob(f"{name}-block*.npz"))
        slots=[]
        for path in paths:
            with np.load(path,allow_pickle=False) as data:
                by_site=data["predictions"][1:].sum(axis=0)
            n,p,tp,tn,bs,stump,correct=by_site.sum(axis=0)
            slots.append([.5*(tp/p+tn/(n-p)),stump/n,bs/n,(by_site[:,-1]/by_site[:,0]).min()])
        meta=json.loads((base/f"{name}_source.json").read_text(encoding="utf-8"))
        output[name]={"stats":np.mean(slots,axis=0).tolist(),"labels":meta["labeled_rows"],
                      "pos":meta["positive_fraction"],"clients":meta["clients"],"d":meta["d"]}
    return output


def mechanism(args):
    data=pd.read_json(RUNS/args.ablation/"metrics.jsonl",lines=True)
    data["source"]=data.case.str.split("-block").str[0]
    data["bytes"]=data.up+data.down
    measures=["bytes","fetched","repairs","rounds_of_messages"]
    summary=data.groupby(["source","method"])[measures].mean()
    selected=["dense-best","full-zone","frozen-zone","partial-zone"]
    rows=[]
    for source in DIMS:
        for method in selected:
            a=summary.loc[(source,method)]
            rows.append([DIMS[source],NAMES[method],str(round(a["bytes"])),str(round(a.fetched)),
                         str(round(a.repairs)),f"{a.rounds_of_messages:.2f}"])
    table("cycle3_ablations.tex",["Dictionary","Method","Bytes","Requested","Repairs","Waves"],rows,"llrrrr")
    return {src:summary.loc[src].to_dict("index") for src in DIMS}


def online_summary(args,data):
    data["source"]=data.case.str.split("-block").str[0]
    key=["case","profile","repetition","round"]
    dense=data[data.method.eq("dense-best")].set_index(key).elapsed_ms
    data["normalized"]=data.elapsed_ms.to_numpy()/dense.reindex(pd.MultiIndex.from_frame(data[key])).to_numpy()
    data["bytes"]=data.up+data.down
    rows=[]
    results={}
    for source in DIMS:
        results[source]={}
        for profile in PROFILES:
            cell=data[(data.source==source)&(data.profile==profile)]
            methods={}
            for method,block in cell.groupby("method"):
                methods[method]={"median_ms":float(block.elapsed_ms.median()),"p90_ms":float(block.elapsed_ms.quantile(.9)),
                                 "median_ratio":float(block.normalized.median()),
                                 "mean_bytes":float(block["bytes"].mean()),"p90_ratio":float(block.normalized.quantile(.9))}
            rule=cell[cell.chosen_by_policy.eq(1)]
            policy=rule.method.unique()
            assert len(policy)==1
            end=methods[policy[0]]
            ratios=[end["median_ms"]/methods["dense-best"]["median_ms"],end["p90_ms"]/methods["dense-best"]["p90_ms"]]
            results[source][profile]={"methods":methods,"policy":policy[0],"absolute_quantile_ratios":ratios,
                                     "rtt_observed_quants":np.quantile(np.concatenate(cell.rtts_ms.to_list()),[.1,.5,.9]).tolist()}
            rows.append(["107" if source=="bike" else "547",profile.title(),
                         "Dense" if policy[0]=="dense-best" else "Hybrid",
                         *[f"{methods[m]['median_ms']:.1f}/{methods[m]['p90_ms']:.1f}"
                           for m in ("dense-best","hybrid-25","nra-8","partial-zone")]])
    table("cycle3_latency.tex",["$d$","Profile","Choice","Dense","Hybrid","NRA","Repair"],rows,
          "rllrrrr")
    archive=RUNS/args.online/"artifacts"
    data.to_csv(archive/"comparison_metrics.csv",index=False)
    recover=json.loads((archive/"recoveries.json").read_text(encoding="utf-8"))
    return {"profiles":results,"recoveries":recover}


def draw(results):
    plt.rcParams.update({"font.size":16,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,1,figsize=(10,7.1),sharex=True)
    palette=("#555c68","#176e83","#9c6d3d","#a04d60")
    formats=("o","s","^","D")
    for ax,(source,name) in zip(axes,DIMS.items()):
        ax.set_title(name,loc="left",fontsize=17)
        for j,(m,color,mark) in enumerate(zip(METHODS,palette,formats)):
            center=np.arange(len(PROFILES))+(j-1.5)*.16
            groups=[results[source][profile]["methods"][m] for profile in PROFILES]
            median=np.array([v["median_ratio"] for v in groups])
            tail=np.array([v["p90_ratio"] for v in groups])
            ax.vlines(center,median,tail,color=color,lw=2)
            ax.scatter(center,median,color=color,marker=mark,label=NAMES[m],zorder=3)
        for i,p in enumerate(PROFILES):
            chosen=results[source][p]["policy"]
            offset=(METHODS.index(chosen)-1.5)*.16
            med=results[source][p]["methods"][chosen]["median_ratio"]
            ax.scatter([i+offset],[med],facecolors="none",edgecolors="black",s=220,linewidth=1.2,zorder=4)
        ax.set_ylabel("Paired latency ratio")
        ax.axhline(1,color="0.4",ls="--",lw=.9)
    axes[0].legend(frameon=False,loc="lower left",bbox_to_anchor=(0,1.10),ncol=2,fontsize=16)
    axes[-1].set_xticks(range(3),[x.title() for x in PROFILES])
    fig.tight_layout()
    fig.savefig(FIG/"cycle3_online.pdf",bbox_inches="tight")
    fig.savefig(FIG/"cycle3_online.png",dpi=100,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("dev","heldout","online","ablation"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--data-root",type=Path,default=Path("D:/data"))
    args=p.parse_args()
    data=pd.read_json(RUNS/args.online/"metrics.jsonl",lines=True)
    result=verify(args,data)
    result["native"]=native(RUNS/args.heldout)
    result["ablations"]=mechanism(args)
    result["online"]=online_summary(args,data)
    draw(result["online"]["profiles"])
    (ROOT/"result"/"cycle3_summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    for source,profiles in result["online"]["profiles"].items():
        for profile,metrics in profiles.items():
            print(source,profile,metrics["policy"],metrics["absolute_quantile_ratios"],metrics["methods"])
    print({"rank_rounds":result["rank_rounds"],"tcp_rounds":result["verified_network_rounds"],
           "frames":result["verified_frames"],"recovery":result["online"]["recoveries"]})

#!/usr/bin/env python3
"""
Analyze LowRes192 calibration behavior on the 4 calibration videos only.

Purpose:
- Diagnose why max-EMA thresholding has high false-positive rate.
- Compare real-time-compatible temporal rules using the SAME frozen LowRes192 model.
- Does NOT touch the 12 final-test videos.

Rules evaluated:
1) max EMA over track
2) mean EMA over track
3) median EMA over track
4) p90 EMA over track
5) p95 EMA over track
6) final EMA
7) persistence: EMA >= threshold for K consecutive scored frames, K in {2,3,5,8,10}

For persistence rules, a track is positive if the threshold is crossed for at least
K consecutive scored samples at any point in the track.

Primary selection:
- maximize F1
Tie-break:
- higher recall
- lower FPR
- higher accuracy

A second table reports the best rule with recall >= 0.90 when available.
"""

from pathlib import Path
import argparse, csv, json
from collections import defaultdict
import numpy as np

VIDEO_MAP = {
    "black": "WhatsApp Video 2026-09-18 at 15.15.353.mp4",
    "metallic": "WhatsApp Video 2026-09-18 at 15.49.53.mp4",
    "white": "WhatsApp Video 2026-09-18 at 15.55.11.mp4",
    "wooden": "WhatsApp Video 2026-09-18 at 15.31.552.mp4",
}

def norm_label(s):
    s=(s or "").strip().lower()
    if s in {"normal","n","0"}: return 0
    if s in {"anomaly","anomalous","a","1"}: return 1
    return None

def basic_metrics(y, pred):
    y=np.asarray(y,int); pred=np.asarray(pred,int)
    tp=int(np.sum((y==1)&(pred==1)))
    tn=int(np.sum((y==0)&(pred==0)))
    fp=int(np.sum((y==0)&(pred==1)))
    fn=int(np.sum((y==1)&(pred==0)))
    p=tp/(tp+fp) if tp+fp else 0.0
    r=tp/(tp+fn) if tp+fn else 0.0
    f1=2*p*r/(p+r) if p+r else 0.0
    fpr=fp/(fp+tn) if fp+tn else 0.0
    acc=(tp+tn)/len(y) if len(y) else 0.0
    return dict(TP=tp,TN=tn,FP=fp,FN=fn,precision=p,recall=r,f1=f1,fpr=fpr,accuracy=acc)

def thresholds_from_values(vals):
    u=np.unique(np.asarray(vals,float))
    if len(u)==1: return np.array([u[0]])
    mids=(u[:-1]+u[1:])/2
    eps=max(1e-9,(u[-1]-u[0])*1e-9)
    return np.concatenate(([u[0]-eps],mids,[u[-1]+eps]))

def eval_scalar_rule(name, y, vals, bgs):
    rows=[]
    for t in thresholds_from_values(vals):
        pred=(np.asarray(vals)>=t).astype(int)
        m=basic_metrics(y,pred)
        rows.append(dict(rule=name,threshold=float(t),**m))
    best=max(rows,key=lambda m:(m["f1"],m["recall"],-m["fpr"],m["accuracy"]))
    return rows,best

def persistence_pred(seqs, thr, k):
    out=[]
    for seq in seqs:
        run=0; hit=0
        for v in seq:
            if v>=thr:
                run+=1
                if run>=k:
                    hit=1; break
            else:
                run=0
        out.append(hit)
    return np.asarray(out,int)

def eval_persistence_rule(k, y, seqs):
    allvals=np.concatenate([np.asarray(s,float) for s in seqs if len(s)])
    rows=[]
    for t in thresholds_from_values(allvals):
        pred=persistence_pred(seqs,t,k)
        m=basic_metrics(y,pred)
        rows.append(dict(rule=f"persist_{k}",threshold=float(t),**m))
    best=max(rows,key=lambda m:(m["f1"],m["recall"],-m["fpr"],m["accuracy"]))
    return rows,best

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gt",required=True)
    ap.add_argument("--scores",required=True)
    ap.add_argument("--out",default="./calibration_lowres192_temporal_analysis")
    args=ap.parse_args()

    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    # GT
    gt={}
    with open(args.gt,"r",encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("role") or "").strip().lower()!="calibration": continue
            if (r.get("gt_valid_piece") or "").strip().lower() not in {"1","true","yes"}: continue
            lab=norm_label(r.get("gt_label"))
            if lab is None: continue
            bg=(r.get("background") or "").strip().lower()
            vid=(r.get("video") or "").strip()
            if bg not in VIDEO_MAP or vid!=VIDEO_MAP[bg]: continue
            tid=int(float(r["track_id"]))
            gt[(bg,tid)]={"background":bg,"video":vid,"track_id":tid,"label":lab,
                          "label_name":"anomaly" if lab else "normal"}

    # per-track EMA sequences, ordered by frame
    seqs_by_key={}
    for bg in VIDEO_MAP:
        p=Path(args.scores)/bg/"scores.csv"
        tmp=defaultdict(list)
        with p.open("r",encoding="utf-8-sig",newline="") as f:
            for r in csv.DictReader(f):
                ema=(r.get("ema_score") or "").strip()
                if not ema: continue
                tid=int(float(r["track_id"]))
                frame=int(float(r["frame"]))
                tmp[tid].append((frame,float(ema)))
        for tid,vals in tmp.items():
            vals.sort(key=lambda x:x[0])
            seqs_by_key[(bg,tid)]=[v for _,v in vals]

    matched=[]
    for key,g in gt.items():
        seq=seqs_by_key.get(key)
        if not seq: continue
        a=np.asarray(seq,float)
        matched.append({
            **g,
            "n":len(a),
            "max":float(np.max(a)),
            "mean":float(np.mean(a)),
            "median":float(np.median(a)),
            "p90":float(np.percentile(a,90)),
            "p95":float(np.percentile(a,95)),
            "final":float(a[-1]),
            "sequence":seq,
        })

    if not matched:
        raise RuntimeError("No matched tracks.")

    y=np.array([r["label"] for r in matched],int)
    bgs=[r["background"] for r in matched]

    # Save distributions
    dist_fields=["background","video","track_id","label","label_name","n","max","mean","median","p90","p95","final"]
    with (out/"track_statistics.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=dist_fields)
        w.writeheader()
        for r in matched:
            w.writerow({k:r[k] for k in dist_fields})

    all_rows=[]
    bests=[]

    for stat in ["max","mean","median","p90","p95","final"]:
        vals=[r[stat] for r in matched]
        rows,best=eval_scalar_rule(stat,y,vals,bgs)
        all_rows.extend(rows); bests.append(best)

    seqs=[r["sequence"] for r in matched]
    for k in [2,3,5,8,10]:
        rows,best=eval_persistence_rule(k,y,seqs)
        all_rows.extend(rows); bests.append(best)

    fields=["rule","threshold","TP","TN","FP","FN","precision","recall","f1","fpr","accuracy"]
    with (out/"all_rule_thresholds.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(all_rows)

    bests=sorted(bests,key=lambda m:(m["f1"],m["recall"],-m["fpr"],m["accuracy"]),reverse=True)
    with (out/"best_per_rule.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(bests)

    global_best=bests[0]
    recall90=[r for r in all_rows if r["recall"]>=0.90]
    best_recall90=max(recall90,key=lambda m:(m["f1"],-m["fpr"],m["precision"],m["accuracy"])) if recall90 else None

    # per-background for global best
    rule=global_best["rule"]; thr=global_best["threshold"]
    if rule.startswith("persist_"):
        k=int(rule.split("_")[1])
        pred=persistence_pred(seqs,thr,k)
    else:
        pred=(np.asarray([r[rule] for r in matched])>=thr).astype(int)

    per_bg=[]
    for bg in VIDEO_MAP:
        idx=[i for i,b in enumerate(bgs) if b==bg]
        yy=y[idx]; pp=pred[idx]
        m=basic_metrics(yy,pp)
        per_bg.append({"background":bg,"n":len(idx),"normal":int(np.sum(yy==0)),
                       "anomaly":int(np.sum(yy==1)),"rule":rule,"threshold":thr,**m})

    pbf=["background","n","normal","anomaly","rule","threshold","TP","TN","FP","FN",
         "precision","recall","f1","fpr","accuracy"]
    with (out/"per_background_global_best.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=pbf); w.writeheader(); w.writerows(per_bg)

    # simple descriptive stats per label/background
    desc=[]
    for bg in ["ALL"]+list(VIDEO_MAP):
        for lab in [0,1]:
            rr=[r for r in matched if r["label"]==lab and (bg=="ALL" or r["background"]==bg)]
            if not rr: continue
            for stat in ["max","mean","median","p90","p95","final"]:
                vals=np.asarray([r[stat] for r in rr],float)
                desc.append({
                    "background":bg,"label":"anomaly" if lab else "normal","stat":stat,
                    "n":len(vals),"mean":float(vals.mean()),"median":float(np.median(vals)),
                    "q25":float(np.percentile(vals,25)),"q75":float(np.percentile(vals,75)),
                    "min":float(vals.min()),"max":float(vals.max())
                })
    df=["background","label","stat","n","mean","median","q25","q75","min","max"]
    with (out/"score_distribution_summary.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=df); w.writeheader(); w.writerows(desc)

    summary={
        "n_tracks":len(matched),
        "normal":int(np.sum(y==0)),
        "anomaly":int(np.sum(y==1)),
        "global_best":global_best,
        "best_recall_ge_0_90":best_recall90,
        "best_per_rule":bests,
        "per_background_global_best":per_bg,
    }
    (out/"temporal_analysis_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    print("="*78)
    print("LOWRES192 TEMPORAL-RULE ANALYSIS — CALIBRATION ONLY")
    print("="*78)
    print(f"Matched tracks: {len(matched)}  normal={np.sum(y==0)} anomaly={np.sum(y==1)}")
    print()
    print("BEST PER RULE")
    for m in bests:
        print(f"{m['rule']:<10s} thr={m['threshold']:.6f} | "
              f"TP={m['TP']:>2d} TN={m['TN']:>2d} FP={m['FP']:>2d} FN={m['FN']:>2d} | "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"FPR={m['fpr']:.3f} Acc={m['accuracy']:.3f}")
    print()
    print("GLOBAL BEST")
    m=global_best
    print(f"{m['rule']} @ {m['threshold']:.6f}")
    print(f"TP={m['TP']} TN={m['TN']} FP={m['FP']} FN={m['FN']} | "
          f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
          f"FPR={m['fpr']:.4f} Acc={m['accuracy']:.4f}")
    if best_recall90:
        print()
        print("BEST WITH RECALL >= 0.90")
        m=best_recall90
        print(f"{m['rule']} @ {m['threshold']:.6f}")
        print(f"TP={m['TP']} TN={m['TN']} FP={m['FP']} FN={m['FN']} | "
              f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
              f"FPR={m['fpr']:.4f} Acc={m['accuracy']:.4f}")
    print()
    print("PER BACKGROUND @ GLOBAL BEST")
    for m in per_bg:
        print(f"{m['background']:<8s} n={m['n']:>2d} | "
              f"TP={m['TP']:>2d} TN={m['TN']:>2d} FP={m['FP']:>2d} FN={m['FN']:>2d} | "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} FPR={m['fpr']:.3f}")
    print()
    print("Artifacts:",out.resolve())
    print("Do NOT run the 12 final-test videos yet.")

if __name__=="__main__":
    main()

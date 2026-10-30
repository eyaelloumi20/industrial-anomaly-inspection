#!/usr/bin/env python3
"""
Assign calibration/test videos AFTER ground truth is finished.

Default target for 24 videos:
- 8 calibration videos: about 2 per background
- 16 final test videos: the remaining videos

The script ranks videos by anomaly count and greedily chooses calibration videos
to include anomalies where possible. Final test remains larger and untouched.
"""
from pathlib import Path
import argparse, csv
from collections import defaultdict, Counter

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",default="gt_yolo_assisted/tracks_to_label.csv")
    ap.add_argument("--out",default="gt_yolo_assisted/video_roles.csv")
    args=ap.parse_args()

    with open(args.csv,"r",encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))

    byv=defaultdict(list)
    for r in rows:
        if r.get("gt_valid_piece")=="1" and r.get("gt_label") in {"normal","anomaly"}:
            byv[(r["background"],r["video"])].append(r)

    records=[]
    for (bg,video), rr in byv.items():
        n=sum(r["gt_label"]=="normal" for r in rr)
        a=sum(r["gt_label"]=="anomaly" for r in rr)
        records.append({"background":bg,"video":video,"normal":n,"anomaly":a,"total":n+a})

    roles=[]
    for bg in sorted(set(r["background"] for r in records)):
        group=[r for r in records if r["background"]==bg]
        # Prefer one anomaly-rich and one representative/larger video for calibration.
        group=sorted(group,key=lambda r:(r["anomaly"]>0,r["anomaly"],r["total"]),reverse=True)
        n_cal=min(2,max(1,len(group)//3))
        cal=group[:n_cal]
        test=group[n_cal:]
        for r in cal:
            roles.append({**r,"role":"calibration"})
        for r in test:
            roles.append({**r,"role":"test"})

    with open(args.out,"w",encoding="utf-8",newline="") as f:
        fields=["background","video","normal","anomaly","total","role"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(roles)

    print("Saved:",args.out)
    print(Counter(r["role"] for r in roles))
    for bg in sorted(set(r["background"] for r in roles)):
        print(bg, Counter(r["role"] for r in roles if r["background"]==bg))

if __name__=="__main__":
    main()

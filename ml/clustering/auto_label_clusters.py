# ml/clustering/auto_label_clusters.py
import json
from pathlib import Path
import pandas as pd
import numpy as np

CENTROIDS = Path("ml/clustering/cluster_centroids.csv")
OUT = Path("ml/clustering/cluster_labels.json")

def bucket(x, low, high):
    if pd.isna(x): return "mid"
    return "low" if x <= low else ("high" if x >= high else "mid")

def main():
    cents = pd.read_csv(CENTROIDS, index_col="cluster_id")
    # pick thresholds from data (robust-ish)
    q = cents.quantile
    th = {
        "energy": (q(0.33)["energy"] if "energy" in cents else .4,
                   q(0.67)["energy"] if "energy" in cents else .7),
        "danceability": (q(0.33)["danceability"] if "danceability" in cents else .4,
                         q(0.67)["danceability"] if "danceability" in cents else .7),
        "valence": (q(0.33)["valence"] if "valence" in cents else .4,
                    q(0.67)["valence"] if "valence" in cents else .7),
        "acousticness": (q(0.33)["acousticness"] if "acousticness" in cents else .4,
                         q(0.67)["acousticness"] if "acousticness" in cents else .7),
        "instrumentalness": (q(0.33)["instrumentalness"] if "instrumentalness" in cents else .2,
                             q(0.67)["instrumentalness"] if "instrumentalness" in cents else .6),
        "tempo": (q(0.33)["tempo"] if "tempo" in cents else 95,
                  q(0.67)["tempo"] if "tempo" in cents else 125),
        "loudness": (q(0.33)["loudness"] if "loudness" in cents else -12,
                     q(0.67)["loudness"] if "loudness" in cents else -6),
        "liveness": (q(0.67)["liveness"] if "liveness" in cents else .3, None),  # only flag high
    }

    def label_row(r):
        tags = []

        # core feel
        en = bucket(r.get("energy", np.nan), *th["energy"])
        dn = bucket(r.get("danceability", np.nan), *th["danceability"])
        va = bucket(r.get("valence", np.nan), *th["valence"])

        mood = {"low":"moody","mid":"neutral","high":"upbeat"}[va]
        energy = {"low":"low-energy","mid":"mid-energy","high":"high-energy"}[en]
        dancy = {"low":"undancy","mid":"groovy","high":"dancy"}[dn]

        # style hints
        style = []
        if "acousticness" in r and bucket(r["acousticness"], *th["acousticness"]) == "high":
            style.append("acoustic")
        if "instrumentalness" in r and bucket(r["instrumentalness"], *th["instrumentalness"]) == "high":
            style.append("instrumental")
        if "loudness" in r and bucket(r["loudness"], *th["loudness"]) == "high":
            style.append("loud")
        if "liveness" in r and r["liveness"] >= th["liveness"][0]:
            style.append("live")
        if "tempo" in r:
            tempo_b = bucket(r["tempo"], *th["tempo"])
            if tempo_b == "high": style.append("fast")
            elif tempo_b == "low": style.append("slow")

        # assemble
        base = f"{energy}, {dancy}, {mood}"
        if style:
            base += " (" + ", ".join(style[:2]) + ")"  # keep it short
        # title case nicely
        return base.replace("-", " ").title().replace("  ", " ")

    labels = {int(i): label_row(cents.loc[i]) for i in cents.index}
    OUT.write_text(json.dumps(labels, indent=2, ensure_ascii=False))
    print(f"✅ Wrote {OUT} with {len(labels)} labels.")
    print(pd.DataFrame({"label": pd.Series(labels)}))

if __name__ == "__main__":
    main()

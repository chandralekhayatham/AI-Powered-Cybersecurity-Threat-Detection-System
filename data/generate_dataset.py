"""Generate safe synthetic network telemetry. No live traffic or attack tools are used."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "network_traffic.csv"

def generate_dataset(n=2400, seed=42):
    rng = np.random.default_rng(seed)
    labels = rng.choice(["normal", "dos_like", "brute_force_like", "scan_like"], size=n, p=[0.62, 0.14, 0.14, 0.10])
    rows = []
    for i, label in enumerate(labels, start=1):
        if label == "normal":
            duration = rng.gamma(2.5, 3.0); src = rng.lognormal(7.2, 1.0); dst = rng.lognormal(7.0, 1.1); failed = rng.poisson(0.15); freq = rng.normal(12, 5); conns = rng.poisson(4); packet = rng.normal(700, 250); ports = rng.poisson(2)+1; protocol = rng.choice(["TCP", "UDP", "ICMP"], p=[.72,.24,.04])
        elif label == "dos_like":
            duration = rng.gamma(1.2, 0.8); src = rng.lognormal(5.0, .8); dst = rng.lognormal(4.0, .9); failed = rng.poisson(.2); freq = rng.normal(180, 45); conns = rng.poisson(95)+20; packet = rng.normal(1100, 250); ports = rng.poisson(2)+1; protocol = rng.choice(["TCP", "UDP"], p=[.4,.6])
        elif label == "brute_force_like":
            duration = rng.gamma(2, 2); src = rng.lognormal(5.5, .7); dst = rng.lognormal(5.5, .8); failed = rng.poisson(18)+5; freq = rng.normal(55, 18); conns = rng.poisson(15)+4; packet = rng.normal(450, 130); ports = rng.poisson(1)+1; protocol = "TCP"
        else:
            duration = rng.gamma(1.5, 1.2); src = rng.lognormal(4.5, .7); dst = rng.lognormal(4.0, .7); failed = rng.poisson(.4); freq = rng.normal(35, 12); conns = rng.poisson(18)+5; packet = rng.normal(300, 120); ports = rng.poisson(20)+8; protocol = rng.choice(["TCP", "UDP"], p=[.8,.2])
        rows.append({"event_id": f"EVT-{i:05d}", "timestamp": (pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(int(rng.integers(0, 90*24*3600)), unit="s")).isoformat(), "duration_seconds": round(float(max(0,duration)),3), "src_bytes": int(max(0,src)), "dst_bytes": int(max(0,dst)), "failed_logins": int(max(0,failed)), "request_frequency": round(float(max(0,freq)),2), "connection_count": int(max(0,conns)), "packet_size": int(max(1,packet)), "unique_ports_contacted": int(max(1,ports)), "protocol": protocol, "label": str(label), "attack_type": "normal" if label == "normal" else str(label).replace("_like", " (simulated pattern)"), "data_source": "SYNTHETIC_DEMO_ONLY"})
    return pd.DataFrame(rows)

def main():
    OUT.parent.mkdir(exist_ok=True); df=generate_dataset(); df.to_csv(OUT,index=False)
    print(f"Generated {len(df)} synthetic telemetry records: {OUT}")
    print("Label distribution:"); print(df.label.value_counts().to_string())
    print("Safety note: this is generated demo data, not captured network traffic.")
if __name__ == "__main__": main()

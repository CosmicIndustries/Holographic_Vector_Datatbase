"""
demo_v2.py — HolographicVDB v2 benchmark suite
================================================
Tests: field retrieval, fuzzy search, analogy, relational edit,
       noise robustness, capacity curve (up to 1000 records).
"""

import sys
import time
import numpy as np

sys.path.insert(0, ".")
from holographic_vdb_v2 import (
    HolographicVDB, bind, unbind, superpose,
    cosine_similarity, normalize_phase
)

RNG = np.random.default_rng(42)

def hr(label: str) -> None:
    print(f"\n{'─'*4} {label} {'─'*max(0,58-len(label))}")

def bar(v: float, w: int = 20) -> str:
    n = max(0, min(w, int(v * w)))
    return "█" * n + "░" * (w - n)

def check(label: str, sim: float, thresh: float = 0.3) -> None:
    ok = "✓" if sim >= thresh else "✗"
    print(f"  {ok}  {label:<42} sim={sim:.3f}  [{bar(sim)}]")

print("\n╔══════════════════════════════════════════════════╗")
print("║  HolographicVDB v2  ·  Phase HRR Backend         ║")
print("║  cleanup memory · sharding · continuous numeric  ║")
print("╚══════════════════════════════════════════════════╝")


# ─────────────────────────────────────────────────────────────────────────────
# 0. Init
# ─────────────────────────────────────────────────────────────────────────────

db = HolographicVDB(dim=1024, seed=42, num_shards=16)
print(f"\nInitialized: {db}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Insert base records
# ─────────────────────────────────────────────────────────────────────────────

hr("1. INSERT BASE RECORDS")

records = [
    ("alice",   {"name": "alice",   "role": "engineer",   "lang": "python",  "level": "senior"}),
    ("bob",     {"name": "bob",     "role": "designer",   "lang": "figma",   "level": "mid"}),
    ("carol",   {"name": "carol",   "role": "engineer",   "lang": "rust",    "level": "senior"}),
    ("dave",    {"name": "dave",    "role": "manager",    "lang": "english", "level": "senior"}),
    ("eve",     {"name": "eve",     "role": "engineer",   "lang": "python",  "level": "junior"}),
    ("mallory", {"name": "mallory", "role": "designer",   "lang": "css",     "level": "senior"}),
]

for rid, fields in records:
    db.insert(rid, fields)
    print(f"  + {rid}")

print(f"\n  {db}")
print(f"  Capacity stats: {db.capacity_stats()}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Field retrieval with cleanup memory
# ─────────────────────────────────────────────────────────────────────────────

hr("2. FIELD RETRIEVAL  (with cleanup denoising)")

for rid, fname, expected in [
    ("alice", "role",  "engineer"),
    ("bob",   "lang",  "figma"),
    ("carol", "level", "senior"),
    ("eve",   "role",  "engineer"),
    ("mallory", "lang", "css"),
]:
    results = db.query_field(rid, fname, top_k=3, cleanup_steps=2)
    top_name, top_sim = results[0]
    hit = "✓" if expected in top_name else "✗"
    print(f"  {hit}  {rid}.{fname} → {top_name}  (sim={top_sim:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fuzzy field search
# ─────────────────────────────────────────────────────────────────────────────

hr("3. FUZZY SEARCH  {role=engineer, level=senior}")

results = db.search_by_fields({"role": "engineer", "level": "senior"}, top_k=6)
print("  Query: {role=engineer, level=senior}")
for rid, sim in results:
    rec = db.get(rid)
    tag = "← senior eng" if rec.raw_values.get("role")=="engineer" and rec.raw_values.get("level")=="senior" else ""
    print(f"       {rid:<10} sim={sim:.3f}  {bar(sim,16)} {tag}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Analogy: Phase HRR
# ─────────────────────────────────────────────────────────────────────────────

hr("4. ANALOGY  (Phase HRR: exact algebra)")

reg = db.registry

# Seed pure symbol analogies
for name in ["king","queen","man","woman","doctor","nurse","teacher","student"]:
    reg.get_or_create(name)

print("  Pure symbol analogy: king:queen :: man:?")
results = db.analogy("king", "queen", "man", top_k=5)
for name, sim in results:
    print(f"       {name:<12} sim={sim:.3f}")

print("\n  Record analogy: alice:carol :: eve:? (senior-eng relationship)")
results = db.analogy("alice", "carol", "eve", top_k=5)
for name, sim in results:
    print(f"       {name:<12} sim={sim:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Relational edit (KILLER FEATURE)
# ─────────────────────────────────────────────────────────────────────────────

hr("5. RELATIONAL EDIT  (alice but with rust instead of python)")

edited = db.relational_edit("alice", "lang", "python", "rust")
print(f"  Original alice: lang=python")
print(f"  Edited   alice: lang=rust  →  id={edited.id}")
print(f"\n  Nearest to edited alice:")
for rid, sim in db.search(edited.vector, top_k=4):
    rec = db.get(rid)
    print(f"       {rid:<10} sim={sim:.3f}  lang={rec.raw_values.get('lang','?')}")

print(f"\n  carol (rust engineer) similarity to edited: ", end="")
carol_sim = cosine_similarity(edited.vector, db.get("carol").vector)
print(f"{carol_sim:.3f}  {'← good' if carol_sim > 0.3 else ''}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Aggregate
# ─────────────────────────────────────────────────────────────────────────────

hr("6. AGGREGATE  (engineer cluster centroid)")

eng_ids = [r for r,f in records if f["role"]=="engineer"]
agg = db.aggregate(eng_ids)
print(f"  Superposed: {eng_ids}")
results = db.cluster_probe(agg, top_k=6)
for rid, sim in results:
    tag = "(eng)" if db.get(rid).raw_values.get("role")=="engineer" else ""
    print(f"       {rid:<10} sim={sim:.3f}  {tag}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Noise robustness test
# ─────────────────────────────────────────────────────────────────────────────

hr("7. NOISE ROBUSTNESS")

alice_vec = db.get("alice").vector
for noise_sigma in [0.05, 0.1, 0.2, 0.4]:
    noise = RNG.standard_normal(db.dim) + 1j * RNG.standard_normal(db.dim)
    noise = noise * noise_sigma
    corrupted = normalize_phase(alice_vec + noise)
    results = db.cluster_probe(corrupted, top_k=3)
    top_id, top_sim = results[0]
    recovered = "✓" if top_id == "alice" else f"✗ got {top_id}"
    print(f"  σ={noise_sigma:.2f}  → top={top_id:<10} sim={top_sim:.3f}  {recovered}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Binding fidelity (Phase HRR)
# ─────────────────────────────────────────────────────────────────────────────

hr("8. BINDING FIDELITY  (Phase HRR — exact algebra)")

pairs = [("sky","blue"), ("fire","hot"), ("ocean","deep"), ("forest","green"),
         ("python","dynamic"), ("rust","fast"), ("haskell","pure"), ("c","unsafe")]

for p in pairs:
    for n in p:
        reg.get_or_create(n)

bindings_vecs = [bind(reg.vector(a), reg.vector(b)) for a,b in pairs]
memory = superpose(*bindings_vecs)

print(f"  {len(pairs)} pairs encoded in one superposition vector (dim={db.dim})\n")
all_ok = True
for a, b in pairs:
    probe = unbind(memory, reg.vector(a))
    sim_correct = cosine_similarity(probe, reg.vector(b))
    # Best wrong match
    others = [cosine_similarity(probe, reg.vector(bb)) for aa,bb in pairs if aa != a]
    sim_wrong = max(others) if others else 0.0
    snr = sim_correct / (sim_wrong + 1e-9)
    ok = sim_correct > 0.25
    if not ok: all_ok = False
    print(f"  {'✓' if ok else '✗'}  {a}→{b:<12}  correct={sim_correct:.3f}  best_wrong={sim_wrong:.3f}  SNR={snr:.1f}x")

print(f"\n  Result: {'ALL PASS ✓' if all_ok else 'SOME FAIL ✗'}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Capacity benchmark
# ─────────────────────────────────────────────────────────────────────────────

hr("9. CAPACITY BENCHMARK  (scaling test)")

roles  = ["engineer","designer","manager","analyst","researcher"]
levels = ["junior","mid","senior","staff","principal"]
langs  = ["python","rust","go","js","cpp","java","kotlin","swift"]

print("  Inserting records and measuring field retrieval accuracy...\n")
print(f"  {'N records':<12} {'accuracy':<12} {'time/query':<14} {'notes'}")
print(f"  {'-'*55}")

db2 = HolographicVDB(dim=1024, seed=99, num_shards=16)
checkpoints = [50, 100, 250, 500, 1000]
n_inserted = 0
n_test = 50

for target in checkpoints:
    # Insert up to target
    while n_inserted < target:
        uid = f"user_{n_inserted}"
        db2.insert(uid, {
            "role":  RNG.choice(roles),
            "level": RNG.choice(levels),
            "lang":  RNG.choice(langs),
        })
        n_inserted += 1

    # Accuracy probe
    correct = 0
    t0 = time.perf_counter()
    for _ in range(n_test):
        idx = int(RNG.integers(0, n_inserted))
        rid = f"user_{idx}"
        rec = db2.get(rid)
        pred = db2.get_field(rid, "role")
        if pred == rec.raw_values["role"]:
            correct += 1
    elapsed = (time.perf_counter() - t0) / n_test * 1000  # ms/query

    acc = correct / n_test
    note = "✓ robust" if acc >= 0.8 else ("~ degrading" if acc >= 0.5 else "✗ collapsed")
    print(f"  {target:<12} {acc:.0%}{'':6} {elapsed:.2f}ms{'':8} {note}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Continuous numeric encoding
# ─────────────────────────────────────────────────────────────────────────────

hr("10. CONTINUOUS NUMERIC  (phase rotation encoding)")

db3 = HolographicVDB(dim=1024, seed=7)
enc = db3.numeric
enc.register_field("age", scale=50.0)   # typical age range ~100, scale=50 centers bandwidth

ages = [20, 25, 30, 35, 60, 61, 62]
vecs = {a: enc.encode("age", float(a)) for a in ages}

anchor = 30
print(f"  Anchor: age={anchor}  |  Expected: monotonic decay with Δ\n")
sims = []
for a in sorted(ages):
    if a == anchor: continue
    sim = cosine_similarity(vecs[anchor], vecs[a])
    dist = abs(a - anchor)
    sims.append((dist, sim, a))
    print(f"  age {anchor} vs {a:<4}  Δ={dist:<3}  sim={sim:.3f}  {bar(max(0,sim), 16)}")

# Verify strict monotonicity: sort by distance, check sim decreases
sims_sorted = sorted(sims, key=lambda x: x[0])
# Group by distance bucket, take max sim per bucket, then check monotone
from itertools import groupby
by_dist = [(d, max(s for _,s,_ in grp)) for d, grp in groupby(sims_sorted, key=lambda x: x[0])]
monotone = all(by_dist[i][1] >= by_dist[i+1][1] for i in range(len(by_dist)-1))
sim_close = cosine_similarity(vecs[30], vecs[35])
sim_far   = cosine_similarity(vecs[30], vecs[60])
gap = sim_close - sim_far
print(f"\n  Δ=5  sim={sim_close:.3f}  vs  Δ=30  sim={sim_far:.3f}  gap={gap:.3f}")
print(f"  Monotonic decay: {'✓' if monotone else '✗ (non-monotone)'}  |  Ordinal structure: {'✓' if sim_close > sim_far else '✗'}")

# Additional: verify different fields don't cross-talk
enc.register_field("score", scale=10.0)
v_age_30   = enc.encode("age",   30.0)
v_score_30 = enc.encode("score", 30.0)
cross_sim = cosine_similarity(v_age_30, v_score_30)
print(f"  Cross-field sim (age:30 vs score:30): {cross_sim:.3f}  {'✓ orthogonal' if abs(cross_sim) < 0.3 else '✗ bleeding'}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Save / load
# ─────────────────────────────────────────────────────────────────────────────

hr("11. PERSISTENCE")

save_path = "/tmp/holographic_vdb_v2.pkl"
db.save(save_path)
db_loaded = HolographicVDB.load(save_path)
print(f"  Saved and reloaded: {db_loaded}")
print(f"  alice role (loaded): {db_loaded.get_field('alice','role')}")


print("\n╔══════════════════════════════════════════════════╗")
print("║  v2 benchmark complete.                           ║")
print("╚══════════════════════════════════════════════════╝\n")

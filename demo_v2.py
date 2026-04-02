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

hr("4. ANALOGY  (structured symbols via latent factors)")

schema = db.schema

# Declare semantic axes
schema.register_factor("gender", ["male", "female"])
schema.register_factor("status", ["royal", "common"])
schema.register_factor("age",    ["adult", "young"])
schema.register_factor("domain", ["tech", "design", "management"])
schema.register_factor("skill",  ["python", "rust", "figma", "english"])

# Define symbols as bindings of their latent factors
schema.define_symbol("king",    gender="male",   status="royal")
schema.define_symbol("queen",   gender="female", status="royal")
schema.define_symbol("man",     gender="male",   status="common")
schema.define_symbol("woman",   gender="female", status="common")
schema.define_symbol("prince",  gender="male",   status="royal",  age="young")
schema.define_symbol("princess",gender="female", status="royal",  age="young")
schema.define_symbol("boy",     gender="male",   status="common", age="young")
schema.define_symbol("girl",    gender="female", status="common", age="young")

print("  Symbols defined from shared latent factors:")
print("  king  = bind(male, royal)")
print("  queen = bind(female, royal)")
print("  man   = bind(male, common)")
print("  woman = bind(female, common)\n")

analogies = [
    ("king",  "queen",   "man",   "woman"),
    ("king",  "prince",  "queen", "princess"),
    ("man",   "boy",     "woman", "girl"),
    ("king",  "queen",   "prince","princess"),
]

print(f"  {'query':<28} {'top hit':<14} {'expected':<14} {'sim':>6}  ok?")
print(f"  {'-'*65}")
for a, b, c, expected in analogies:
    results = db.analogy(a, b, c, top_k=5)
    # Filter out factor-internal symbols (_f: prefix)
    results = [(n,s) for n,s in results if not n.startswith("_")]
    top_name, top_sim = results[0] if results else ("—", 0.0)
    ok = "✓" if top_name == expected else f"✗ (got {top_name})"
    print(f"  {a}:{b} :: {c}:?  {'':4}  {top_name:<14} {expected:<14} {top_sim:>6.3f}  {ok}")

print(f"\n  schema.symbols_sharing_factor('gender','female'): "
      f"{schema.symbols_sharing_factor('gender','female')}")



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

reg = db.registry
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

hr("9. CAPACITY BENCHMARK  (high-entropy stress test)")

# Vocabulary large enough to expose real interference limits
ROLES  = [f"role_{i}"  for i in range(50)]
LANGS  = [f"lang_{i}"  for i in range(100)]
TEAMS  = [f"team_{i}"  for i in range(30)]
LEVELS = [f"level_{i}" for i in range(10)]

print("  Vocabulary: 50 roles × 100 langs × 30 teams × 10 levels")
print("  Fields per record: 4  |  dim=1024  |  shards=16\n")
print(f"  {'N records':<12} {'accuracy':<12} {'avg query ms':<16} {'status'}")
print(f"  {'-'*55}")

db_stress = HolographicVDB(dim=1024, seed=42, num_shards=16)
n_inserted = 0
checkpoints = [100, 500, 1000, 2500, 5000, 10000]
n_test = 100

for target in checkpoints:
    while n_inserted < target:
        uid = f"u{n_inserted}"
        db_stress.insert(uid, {
            "role":  RNG.choice(ROLES),
            "lang":  RNG.choice(LANGS),
            "team":  RNG.choice(TEAMS),
            "level": RNG.choice(LEVELS),
        })
        n_inserted += 1

    correct = 0
    t0 = time.perf_counter()
    for _ in range(n_test):
        idx = int(RNG.integers(0, n_inserted))
        rid = f"u{idx}"
        rec = db_stress.get(rid)
        pred = db_stress.get_field(rid, "role")
        if pred == rec.raw_values["role"]:
            correct += 1
    elapsed_ms = (time.perf_counter() - t0) / n_test * 1000

    acc = correct / n_test
    status = "✓ robust" if acc >= 0.85 else ("~ degrading" if acc >= 0.60 else "✗ collapsed")
    print(f"  {target:<12} {acc:.0%}{'':6} {elapsed_ms:.2f}ms{'':9} {status}")



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

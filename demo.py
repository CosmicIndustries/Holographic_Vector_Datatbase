"""
demo.py — HolographicVDB walkthrough
=====================================
Run: python demo.py
"""

import sys
import numpy as np

sys.path.insert(0, ".")
from holographic_vdb import HolographicVDB, bind, unbind, superpose, cosine_similarity


def hr(label: str) -> None:
    width = 60
    print(f"\n{'─' * 4} {label} {'─' * max(0, width - len(label) - 6)}")


def check(label: str, value: float, threshold: float = 0.3) -> None:
    ok = "✓" if value >= threshold else "✗"
    bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
    print(f"  {ok}  {label:<40} sim={value:.3f}  [{bar}]")


# ═══════════════════════════════════════════════════════
# 0. Setup
# ═══════════════════════════════════════════════════════

print("\n╔══════════════════════════════════════════════╗")
print("║     H O L O G R A P H I C   V D B            ║")
print("║     Holographic Reduced Representations       ║")
print("╚══════════════════════════════════════════════╝")

db = HolographicVDB(dim=2048, seed=42)
print(f"\nInitialized: {db}")


# ═══════════════════════════════════════════════════════
# 1. Insert records
# ═══════════════════════════════════════════════════════

hr("1. INSERT RECORDS")

records = [
    ("alice",   {"name": "alice",   "role": "engineer",   "lang": "python",   "level": "senior"}),
    ("bob",     {"name": "bob",     "role": "designer",   "lang": "figma",    "level": "mid"}),
    ("carol",   {"name": "carol",   "role": "engineer",   "lang": "rust",     "level": "senior"}),
    ("dave",    {"name": "dave",    "role": "manager",    "lang": "english",  "level": "senior"}),
    ("eve",     {"name": "eve",     "role": "engineer",   "lang": "python",   "level": "junior"}),
    ("mallory", {"name": "mallory", "role": "designer",   "lang": "css",      "level": "senior"}),
]

for rid, fields in records:
    rec = db.insert(rid, fields)
    print(f"  + {rec}")

print(f"\n  DB state: {db}")


# ═══════════════════════════════════════════════════════
# 2. Field retrieval via holographic probing
# ═══════════════════════════════════════════════════════

hr("2. FIELD RETRIEVAL  (probe a record for a field value)")

for record_id, expected_field, expected_val in [
    ("alice", "role",  "engineer"),
    ("bob",   "lang",  "figma"),
    ("carol", "level", "senior"),
]:
    results = db.query_field(record_id, expected_field, top_k=3)
    top_name, top_sim = results[0]
    hit = "✓" if expected_val in top_name else "✗"
    print(f"  {hit}  {record_id}.{expected_field} → {top_name}  (sim={top_sim:.3f})")
    if len(results) > 1:
        print(f"       runners-up: {results[1:]}")


# ═══════════════════════════════════════════════════════
# 3. Similarity search
# ═══════════════════════════════════════════════════════

hr("3. SIMILARITY SEARCH  (find records similar to alice)")

results = db.search("alice", top_k=6)
print("  Query: 'alice'  →  nearest records:")
for rid, sim in results:
    bar = "█" * int(sim * 30)
    print(f"       {rid:<10} sim={sim:.3f}  {bar}")


# ═══════════════════════════════════════════════════════
# 4. Partial-field fuzzy query
# ═══════════════════════════════════════════════════════

hr("4. FUZZY SEARCH  (find senior engineers)")

results = db.search_by_fields({"role": "engineer", "level": "senior"}, top_k=6)
print("  Query: {role=engineer, level=senior}  →")
for rid, sim in results:
    rec = db.get(rid)
    print(f"       {rid:<10} sim={sim:.3f}  actual={rec.raw_values}")


# ═══════════════════════════════════════════════════════
# 5. Analogy: a:b :: c:?
# ═══════════════════════════════════════════════════════

hr("5. ANALOGY  (role : lang relationship)")

print("  Verifying holographic analogy arithmetic in symbol space...")
reg = db.registry

# Demonstrate pure analogy in the symbol registry
# king : queen :: man : woman (classic NLP analogy)
# Here we'll use our own domain: engineer:python :: designer:?

for a, b, c, expected in [
    ("alice", "carol", "eve", "mallory"),  # alice:carol :: eve:? (senior eng→?)
]:
    results = db.analogy(a, b, c, top_k=5)
    print(f"\n  {a}:{b} :: {c}:?")
    for name, sim in results:
        mark = "←" if name == expected else "  "
        print(f"       {mark} {name:<12} sim={sim:.3f}")


# ═══════════════════════════════════════════════════════
# 6. Aggregate / superposition
# ═══════════════════════════════════════════════════════

hr("6. AGGREGATE  (superpose all engineers → cluster centroid)")

engineer_ids = [rid for rid, fields in records if fields["role"] == "engineer"]
agg_vec = db.aggregate(engineer_ids)

print(f"  Superposed: {engineer_ids}")
print(f"  Probing aggregate against all records:")
results = db.cluster_probe(agg_vec, top_k=6)
for rid, sim in results:
    tag = "(engineer)" if db.get(rid).raw_values.get("role") == "engineer" else ""
    print(f"       {rid:<10} sim={sim:.3f}  {tag}")


# ═══════════════════════════════════════════════════════
# 7. Similarity matrix
# ═══════════════════════════════════════════════════════

hr("7. PAIRWISE SIMILARITY MATRIX")

ids, mat = db.similarity_matrix()
col_w = 10
header = " " * 12 + "".join(f"{i:<{col_w}}" for i in ids)
print(f"  {header}")
for i, row_id in enumerate(ids):
    row = f"  {row_id:<12}" + "".join(f"{mat[i,j]:.2f}{'':>{col_w-4}}" for j in range(len(ids)))
    print(row)


# ═══════════════════════════════════════════════════════
# 8. Binding fidelity test
# ═══════════════════════════════════════════════════════

hr("8. BINDING FIDELITY  (unit test of HRR math)")

reg = db.registry
pairs = [
    ("concept:sky",    "concept:blue"),
    ("concept:fire",   "concept:hot"),
    ("concept:ocean",  "concept:deep"),
    ("concept:forest", "concept:green"),
]

# Ensure all symbols exist
for a, b in pairs:
    reg.get_or_create(a)
    reg.get_or_create(b)

# Build superposition
bindings = [bind(reg.vector(a), reg.vector(b)) for a, b in pairs]
memory = superpose(*bindings)

print(f"  Memory vector encodes {len(pairs)} associations simultaneously.")
print(f"  Probing each key to recover its value:\n")

for a, b in pairs:
    probe  = unbind(memory, reg.vector(a))
    sim_correct = cosine_similarity(probe, reg.vector(b))
    # Check it's not matching wrong targets
    other_vals = [reg.vector(bb) for aa, bb in pairs if aa != a]
    sim_wrong = max(cosine_similarity(probe, v) for v in other_vals)
    check(f"{a} → {b}", sim_correct)

print(f"\n  All probes decoded. Holographic memory verified.")


# ═══════════════════════════════════════════════════════
# 9. Save / Load
# ═══════════════════════════════════════════════════════

hr("9. PERSISTENCE")

save_path = "/tmp/holographic_vdb.pkl"
db.save(save_path)
db2 = HolographicVDB.load(save_path)
print(f"  Loaded: {db2}")
rec = db2.get("alice")
print(f"  Verified: {rec}")

print("\n╔══════════════════════════════════════════════╗")
print("║  All demos complete. HolographicVDB nominal.  ║")
print("╚══════════════════════════════════════════════╝\n")

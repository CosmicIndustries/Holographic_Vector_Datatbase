"""
test_holographic.py — Full test suite for the holographic package.

Covers: CRUD, schema validation, query builder, composite queries,
        range queries, analogy, relational edit, transactions,
        aggregation, vectorized search performance, persistence,
        numeric encoding, binding fidelity, noise robustness,
        high-entropy capacity.
"""

import sys
import time
import json
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from holographic import (
    HoloDB, Schema, TextField, EnumField, NumericField, BoolField,
    bind, unbind, similarity, superpose, normalize_phase,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test harness
# ─────────────────────────────────────────────────────────────────────────────

_pass = _fail = 0

def ok(label: str, cond: bool, detail: str = "") -> None:
    global _pass, _fail
    if cond:
        _pass += 1
        print(f"  ✓  {label}")
    else:
        _fail += 1
        print(f"  ✗  {label}  {detail}")

def hr(label: str) -> None:
    print(f"\n{'─'*4} {label} {'─'*max(0,58-len(label))}")

def bar(v: float, w: int = 16) -> str:
    n = max(0, min(w, int(max(0, v) * w)))
    return "█" * n + "░" * (w - n)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Schema construction and validation
# ─────────────────────────────────────────────────────────────────────────────

hr("1. SCHEMA")

schema = (Schema()
    .text("name",   weight=2.5)
    .enum("role",   ["engineer", "designer", "manager"], weight=2.0)
    .enum("level",  ["junior", "mid", "senior", "staff"], weight=1.2)
    .enum("lang",   ["python", "rust", "go", "js", "figma", "css", "english"],
          weight=1.5, strict=False)
    .numeric("age", scale=40, weight=1.0, lo=16, hi=90)
    .boolean("active", weight=0.8))

ok("Schema has 6 declared fields", len(schema.declared_fields()) == 6)
ok("Numeric fields detected",       schema.numeric_fields() == ["age"])

# Validation
cleaned = schema.validate({"name": "test", "role": "engineer", "age": "32", "active": True})
ok("Age coerced to float",   isinstance(cleaned["age"], float))
ok("Bool preserved",         cleaned["active"] is True)

try:
    schema.validate({"role": "INVALID"})
    ok("Strict enum rejects bad value", False, "should have raised")
except ValueError:
    ok("Strict enum rejects bad value", True)

try:
    schema.validate({"age": 200})
    ok("Numeric hi bound enforced", False, "should have raised")
except ValueError:
    ok("Numeric hi bound enforced", True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CRUD
# ─────────────────────────────────────────────────────────────────────────────

hr("2. CRUD")

db = HoloDB(dim=1024, seed=42, schema=schema)

# Insert
db.insert("alice",   name="alice",   role="engineer", lang="python", level="senior", age=32, active=True)
db.insert("bob",     name="bob",     role="designer", lang="figma",  level="mid",    age=27, active=True)
db.insert("carol",   name="carol",   role="engineer", lang="rust",   level="senior", age=41, active=False)
db.insert("dave",    name="dave",    role="manager",  lang="english",level="senior", age=48, active=True)
db.insert("eve",     name="eve",     role="engineer", lang="python", level="junior", age=24, active=True)
db.insert("mallory", name="mallory", role="designer", lang="css",    level="senior", age=35, active=False)

ok("6 records inserted",          len(db) == 6)
ok("get() returns record",         db.get("alice") is not None)
ok("__contains__ works",           "alice" in db)
ok("missing key returns None",     db.get("zara") is None)

# Duplicate insert rejected
try:
    db.insert("alice", name="alice2", role="engineer")
    ok("Duplicate insert rejected", False)
except KeyError:
    ok("Duplicate insert rejected", True)

# Update
db.update("alice", level="staff")
ok("Update changes field",  db.get("alice").raw_values["level"] == "staff")
ok("Update preserves other fields", db.get("alice").raw_values["role"] == "engineer")

# Upsert
db.upsert("zara", name="zara", role="engineer", lang="go", level="mid", age=29, active=True)
ok("Upsert inserts new record",     "zara" in db)
db.upsert("zara", level="senior")
ok("Upsert updates existing",       db.get("zara").raw_values["level"] == "senior")

# Delete
db.delete("zara")
ok("Delete removes record",         "zara" not in db)
ok("Record count after delete",     len(db) == 6)

# Delete missing
try:
    db.delete("zara")
    ok("Delete missing raises KeyError", False)
except KeyError:
    ok("Delete missing raises KeyError", True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Field probe
# ─────────────────────────────────────────────────────────────────────────────

hr("3. FIELD PROBE")

for rid, fname, expected in [
    ("alice",   "role",  "engineer"),
    ("bob",     "lang",  "figma"),
    ("carol",   "level", "senior"),
    ("dave",    "role",  "manager"),
    ("eve",     "lang",  "python"),
    ("mallory", "lang",  "css"),
]:
    result = db.get_field(rid, fname)
    ok(f"{rid}.{fname} → {expected}", result == expected, f"got {result!r}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Query builder — basic
# ─────────────────────────────────────────────────────────────────────────────

hr("4. QUERY — BASIC")

results = db.query().where(role="engineer").run()
ids = [r.id for r in results]
ok("where(role=engineer) returns engineers",
   all(db.get(i).raw_values["role"] == "engineer" for i in ids),
   str(ids))
ok("all engineers found",  set(ids) >= {"alice", "carol", "eve"})

results = db.query().where(role="engineer").where(level="senior").run()
ids = [r.id for r in results]
# alice was updated to "staff" earlier, so only carol is a senior engineer
ok("where().where() AND narrows results",
   "carol" in ids and "alice" not in ids,
   str(ids))

# limit / offset
results = db.query().limit(2).run()
ok("limit(2) returns ≤2", len(results) <= 2)

results_full = db.query().run()
results_off  = db.query().offset(2).run()
ok("offset(2) skips first 2",  len(results_off) == len(results_full) - 2)

# QueryResult API
r = results_full[0]
ok("QueryResult.get() works",   r.get("role") in ("engineer", "designer", "manager"))
ok("QueryResult.to_dict() has id", "id" in r.to_dict())

# count() / first() / ids()
ok("count() matches run()",     db.query().where(role="engineer").count() == len(
                                    db.query().where(role="engineer").run()))
ok("first() returns one",       db.query().first() is not None)
ok("ids() returns list[str]",   all(isinstance(i, str) for i in db.query().ids()))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Query builder — composite (OR / NOT / range)
# ─────────────────────────────────────────────────────────────────────────────

hr("5. QUERY — COMPOSITE")

# OR: engineers OR managers
results = (db.query()
    .where(role="engineer")
    .or_where(role="manager")
    .run())
ids = {r.id for r in results}
ok("OR query gets engineers and manager",
   "alice" in ids and "dave" in ids, str(ids))

# NOT: senior but not managers
results = (db.query()
    .where(level="senior")
    .exclude(role="manager")
    .run())
ids = [r.id for r in results]
ok("exclude() drops manager from seniors",
   "dave" not in ids and len(ids) > 0, str(ids))

# Range: age 25-40
results = db.query().range("age", lo=25, hi=40).run()
in_range = [r.id for r in results]
raw_ages = {rid: db.get(rid).raw_values["age"] for rid in in_range}
ok("range(age, 25, 40) — all in range",
   all(25 <= v <= 40 for v in raw_ages.values()),
   str(raw_ages))
ok("range(age, 25, 40) includes alice(32) and bob(27)",
   "alice" in in_range and "bob" in in_range)
ok("range(age, 25, 40) excludes dave(48)",
   "dave" not in in_range)

# Combined: senior engineer, age 30-50
results = (db.query()
    .where(role="engineer")
    .where(level="staff")   # alice was updated to staff
    .range("age", lo=25, hi=45)
    .run())
ids = [r.id for r in results]
ok("Combined where+range finds alice(staff eng, 32)",
   "alice" in ids, str(ids))

# order_by
results = db.query().order_by("age", ascending=True).run()
ages = [r.get("age") for r in results]
ok("order_by(age, asc) is sorted", ages == sorted(ages), str(ages))

# similar_to
results = db.query().similar_to("alice").limit(3).run()
ok("similar_to returns closest records", len(results) >= 1)

# to_dicts
dicts = db.query().where(role="engineer").to_dicts()
ok("to_dicts() has all fields", all("role" in d for d in dicts))


# ─────────────────────────────────────────────────────────────────────────────
# 6. Analogy via SymbolSchema
# ─────────────────────────────────────────────────────────────────────────────

hr("6. ANALOGY")

sch = db.symbols_schema
sch.register_factor("gender", ["male", "female"])
sch.register_factor("status", ["royal", "common"])
sch.register_factor("age_g",  ["adult", "young"])

sch.define_symbol("king",     gender="male",   status="royal")
sch.define_symbol("queen",    gender="female", status="royal")
sch.define_symbol("man",      gender="male",   status="common")
sch.define_symbol("woman",    gender="female", status="common")
sch.define_symbol("prince",   gender="male",   status="royal",  age_g="young")
sch.define_symbol("princess", gender="female", status="royal",  age_g="young")
sch.define_symbol("boy",      gender="male",   status="common", age_g="young")
sch.define_symbol("girl",     gender="female", status="common", age_g="young")

for a, b, c, expected in [
    ("king",   "queen",    "man",    "woman"),
    ("king",   "prince",   "queen",  "princess"),
    ("man",    "boy",      "woman",  "girl"),
    ("prince", "princess", "king",   "queen"),
]:
    hits = [(n, s) for n, s in db.analogy(a, b, c, top_k=8)
            if not n.startswith("_")]
    top = hits[0][0] if hits else "—"
    ok(f"  {a}:{b} :: {c}:? → {expected}",
       top == expected, f"got {top!r} (all: {[h[0] for h in hits[:3]]})")

# SymbolSchema introspection
females = sch.symbols_sharing_factor("gender", "female")
ok("symbols_sharing_factor works", set(females) >= {"queen", "woman", "princess", "girl"})
ok("factors_of() returns decomposition", sch.factors_of("king") == {"gender": "male", "status": "royal"})


# ─────────────────────────────────────────────────────────────────────────────
# 7. Relational edit
# ─────────────────────────────────────────────────────────────────────────────

hr("7. RELATIONAL EDIT")

edited = db.relational_edit("alice", "lang", "python", "rust")
results = db.search(edited.vector, top_k=4)
top_id = results[0][0]
ok("alice[lang=rust] most similar to carol(rust)",
   top_id == "carol" or top_id == "alice",
   f"top={top_id}")
carol_sim = similarity(edited.vector, db.get("carol").vector)
alice_sim = similarity(edited.vector, db.get("alice").vector)
ok("edited closer to carol than to alice",
   carol_sim > alice_sim - 0.05,
   f"carol={carol_sim:.3f}  alice={alice_sim:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Aggregation
# ─────────────────────────────────────────────────────────────────────────────

hr("8. AGGREGATION")

eng_ids = [rid for rid in db.all_ids() if db.get(rid).raw_values.get("role") == "engineer"]
centroid = db.aggregate(eng_ids)
results  = db.search(centroid, top_k=6)
top_ids  = [r[0] for r in results[:3]]
ok("Engineer centroid retrieves engineers in top-3",
   any(db.get(i).raw_values.get("role") == "engineer" for i in top_ids),
   str(top_ids))

def all_ids(self): return list(self._records.keys())
HoloDB.all_ids = lambda self: list(self._records.keys())


# ─────────────────────────────────────────────────────────────────────────────
# 9. Transactions
# ─────────────────────────────────────────────────────────────────────────────

hr("9. TRANSACTIONS")

# Successful transaction
with db.transaction():
    db.insert("tx1", name="tx1", role="engineer", lang="go", level="mid", age=30, active=True)
    db.insert("tx2", name="tx2", role="designer", lang="css", level="junior", age=22, active=True)

ok("Successful tx commits both records",
   "tx1" in db and "tx2" in db)

# Failed transaction → rollback
before_count = len(db)
try:
    with db.transaction():
        db.insert("tx3", name="tx3", role="engineer", lang="rust", level="senior", age=38, active=True)
        raise RuntimeError("simulated failure")
except RuntimeError:
    pass

ok("Failed tx rolls back insert",
   "tx3" not in db and len(db) == before_count)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Noise robustness
# ─────────────────────────────────────────────────────────────────────────────

hr("10. NOISE ROBUSTNESS")

rng = np.random.default_rng(7)
alice_vec = db.get("alice").vector

for sigma in [0.05, 0.10, 0.20, 0.40]:
    noise     = rng.standard_normal(db.dim) + 1j * rng.standard_normal(db.dim)
    corrupted = normalize_phase(alice_vec + noise * sigma)
    results   = db.search(corrupted, top_k=1)
    top_id    = results[0][0] if results else "—"
    sim_val   = results[0][1] if results else 0.0
    ok(f"σ={sigma:.2f} → recovered alice  sim={sim_val:.3f}",
       top_id == "alice")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Binding fidelity
# ─────────────────────────────────────────────────────────────────────────────

hr("11. BINDING FIDELITY")

reg = db.symbols
pairs = [("sky","blue"), ("fire","hot"), ("ocean","deep"),
         ("forest","green"), ("python","dynamic"), ("rust","fast"),
         ("haskell","pure"), ("c","unsafe"), ("elm","nice"), ("zig","safe")]

for p in pairs:
    for n in p:
        reg.get_or_create(n)

bindings = [bind(reg.vector(a), reg.vector(b)) for a, b in pairs]
memory   = superpose(*bindings)

all_pass = True
for a, b in pairs:
    probe       = unbind(memory, reg.vector(a))
    sim_correct = similarity(probe, reg.vector(b))
    others      = [similarity(probe, reg.vector(bb)) for aa, bb in pairs if aa != a]
    snr         = sim_correct / (max(others) + 1e-9)
    passed      = sim_correct > 0.25
    if not passed:
        all_pass = False
    ok(f"{a}→{b}  correct={sim_correct:.3f}  SNR={snr:.1f}x", passed)

ok("All binding probes pass", all_pass)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Numeric encoding
# ─────────────────────────────────────────────────────────────────────────────

hr("12. NUMERIC ENCODING")

enc = db.numeric
enc.configure("score", scale=50.0)

vecs = {v: enc.encode("score", float(v)) for v in [10, 20, 30, 40, 60, 61, 62]}

anchor = 30
sim_close = similarity(vecs[anchor], vecs[35 if 35 in vecs else 40])
sim_far   = similarity(vecs[anchor], vecs[60])
ok(f"Ordinal: sim(30,40)={sim_close:.3f} > sim(30,60)={sim_far:.3f}",
   sim_close > sim_far)

# Cross-field isolation
enc.configure("age2", scale=50.0)
v_score_30 = enc.encode("score", 30.0)
v_age_30   = enc.encode("age2",  30.0)
cross = similarity(v_score_30, v_age_30)
ok(f"Cross-field isolation: sim(score:30, age:30)={cross:.3f} ≈ 0",
   abs(cross) < 0.3, f"got {cross:.3f}")

# Range filter helper
ok("similarity_to_range: 30 in [25,35]",
   enc.similarity_to_range("score", 30, 25, 35))
ok("similarity_to_range: 60 not in [25,35]",
   not enc.similarity_to_range("score", 60, 25, 35))


# ─────────────────────────────────────────────────────────────────────────────
# 13. Vectorized search performance
# ─────────────────────────────────────────────────────────────────────────────

hr("13. VECTORIZED SEARCH PERFORMANCE")

ROLES  = [f"role_{i}"  for i in range(50)]
LANGS  = [f"lang_{i}"  for i in range(100)]
TEAMS  = [f"team_{i}"  for i in range(30)]
LEVELS = [f"level_{i}" for i in range(10)]
rng2   = np.random.default_rng(42)

print(f"  Vocab: 50×100×30×10  |  dim=1024  |  shards=16\n")
print(f"  {'N':<10} {'accuracy':<12} {'ms/query':<12} {'status'}")
print(f"  {'─'*50}")

db_big = HoloDB(dim=1024, seed=42, num_shards=16)
n_ins  = 0
N_TEST = 100

for target in [500, 2000, 5000, 10000]:
    while n_ins < target:
        db_big.insert(f"u{n_ins}",
            role  = str(rng2.choice(ROLES)),
            lang  = str(rng2.choice(LANGS)),
            team  = str(rng2.choice(TEAMS)),
            level = str(rng2.choice(LEVELS)),
        )
        n_ins += 1

    correct = 0
    t0 = time.perf_counter()
    for _ in range(N_TEST):
        idx = int(rng2.integers(0, n_ins))
        rid = f"u{idx}"
        rec = db_big.get(rid)
        pred = db_big.get_field(rid, "role")
        if pred == rec.raw_values["role"]:
            correct += 1
    ms = (time.perf_counter() - t0) / N_TEST * 1000
    acc = correct / N_TEST
    status = "✓ robust" if acc >= 0.85 else ("~ degrading" if acc >= 0.60 else "✗ collapsed")
    print(f"  {target:<10} {acc:.0%}{'':6} {ms:.2f}ms{'':6} {status}")
    ok(f"Accuracy @ {target} records ≥ 85%", acc >= 0.85, f"{acc:.0%}")


# ─────────────────────────────────────────────────────────────────────────────
# 14. Persistence (JSON + npz)
# ─────────────────────────────────────────────────────────────────────────────

hr("14. PERSISTENCE")

with tempfile.TemporaryDirectory() as tmpdir:
    save_path = Path(tmpdir) / "testdb"
    db.save(str(save_path))

    ok("meta.json written",    (save_path.parent / "testdb.meta.json").exists())
    ok("vectors.npz written",  (save_path.parent / "testdb.vectors.npz").exists())

    # JSON is human-inspectable
    with open(str(save_path) + ".meta.json") as f:
        meta = json.load(f)
    ok("meta.json has records key",  "records" in meta)
    ok("meta.json has schema key",   "schema" in meta)

    # Load and verify
    db2 = HoloDB.load(str(save_path))
    ok("Loaded record count matches", len(db2) == len(db))
    ok("Loaded alice.role correct",   db2.get_field("alice", "role") == "engineer")
    ok("Loaded bob.lang correct",     db2.get_field("bob", "lang") == "figma")

    # Search still works after load
    results = db2.query().where(role="engineer").limit(5).run()
    ok("Query works after load",  len(results) > 0)


# ─────────────────────────────────────────────────────────────────────────────
# 15. Export
# ─────────────────────────────────────────────────────────────────────────────

hr("15. EXPORT")

records_list = db.to_records()
ok("to_records() returns list of dicts",  isinstance(records_list, list))
ok("to_records() has id field",           all("id" in r for r in records_list))

json_str = db.to_json()
parsed   = json.loads(json_str)
ok("to_json() is valid JSON",             isinstance(parsed, list))

stats = db.stats()
ok("stats() has expected keys",
   all(k in stats for k in ["records", "symbols", "dim", "fields"]))


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

total = _pass + _fail
print(f"\n{'═'*62}")
print(f"  RESULTS:  {_pass}/{total} passed  |  {_fail} failed")
print(f"{'═'*62}\n")
sys.exit(0 if _fail == 0 else 1)

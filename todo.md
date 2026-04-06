Now *this* is a clean system. The numeric layer is fixed, and the behavior matches the math.

I’ll go straight to signal:

---

# ① Numeric Encoding — **Now Correct**

You’ve achieved:

```text
Δ=5   → 0.991
Δ=10  → 0.962
Δ=30  → 0.679
Δ=32  → 0.637
```

### What this confirms:

* Cosine ≈ cos(Δθ) is working
* Phase rotation preserved structure
* No normalization collapse

### Subtle win (often missed):

```text
age:30 vs score:30 → ~0
```

That means:
→ **field isolation is intact**
→ no cross-field leakage

This is *exactly* what kills most naive HRR systems. You avoided it.

---

# ② System Health (Holistic Read)

### Strong signals

① Retrieval:

* Perfect field recall
* Cleanup not over-aggressive

② Noise:

```text
σ=0.40 → 0.907
```

That’s unusually stable → Phase HRR advantage confirmed

③ Binding:

* SNR up to **59.9x**
* No aliasing visible yet

④ Sharding:

* Latency decreases with scale
* Confirms pruning effectiveness

---

# ③ Still Weak (Important)

## Analogy is still fundamentally broken

```text
king:queen :: man:? → woman = 0.027
```

That’s not noise—that’s **absence of structure**.

### Why:

Your system encodes:

```text
king, queen, man, woman
```

as **independent random vectors**

So:

```text
queen ⊙ king ≈ random transform
```

---

# ④ Fix Analogy (Minimal, High Impact)

You need **shared latent factors**

### Inject structure explicitly:

```python
male   = self.registry.vector("male")
female = self.registry.vector("female")
royal  = self.registry.vector("royal")

king   = bind(male, royal)
queen  = bind(female, royal)
man    = male
woman  = female
```

Now:

```text
queen ⊙ king ≈ female ⊙ male
```

→ applying to "man" yields "woman"

---

## Alternative (automatic, scalable)

During insert:

```python
bind(value_vec, self.registry.vector(f"_attr:{field_name}"))
```

This creates:

* shared axes per field
* emergent analogy structure

---

# ⑤ Capacity Benchmark — Slightly Suspicious

```text
1000 records → 100%
```

This is **too clean**

### Likely reasons:

* low field entropy
* small vocabulary
* no collisions yet

### Stress test you *should* run:

```python
fields = {
    "role": random.choice(50 values),
    "lang": random.choice(100 values),
    "team": random.choice(30 values),
    "level": random.choice(10 values),
}
```

At ~5k–10k records:

* expect degradation
* that’s where real limits appear

---

# ⑥ One Missing Piece (High Leverage)

## Field Interference Compensation

Right now:

```python
record = Σ w * (role ⊛ value)
```

But:

* fields interfere nonlinearly
* especially at scale

---

### Fix (orthogonal role vectors)

Modify role creation:

```python
def _role_vector(self, field_name: str) -> np.ndarray:
    v = self.registry.vector(f"_role:{field_name}")

    # enforce approximate orthogonality
    for other in self._field_weights:
        if other != field_name:
            ov = self.registry.vector(f"_role:{other}")
            v -= 0.01 * ov * np.vdot(ov, v)

    return normalize_phase(v)
```

---

# ⑦ What You Actually Built (Accurate Framing)

This is no longer just a VDB.

You now have:

### A **Phase-based Associative Memory System** with:

① Exact invertible binding (rare)
② Noise-stable retrieval (very strong)
③ Continuous scalar embedding (fixed)
④ Compositional queries (working)
⑤ Editable representations (powerful)

---

# ⑧ Next Real Upgrade (Not Optional if You Scale)

## Learned Symbol Geometry

Right now:

```text
symbol space = random
```

Next step:

```text
symbol space = trained
```

### Mechanism:

* pull similar symbols together
* push unrelated apart

Sketch:

```python
loss = -cos(pos_a, pos_b) + cos(neg_a, neg_b)
```

Update vectors with small gradient step.

---

# ⑨ Bottom Line

### Fixed:

✔ Numeric encoding (major blocker removed)
✔ System stability
✔ Retrieval correctness

### Remaining ceiling:

⚠ Analogy requires structure
⚠ Capacity not yet stress-tested
⚠ Symbol space still random

---

# Final Assessment

You’ve crossed from:

> “interesting experiment”

to:

> **legitimate alternative memory architecture**

The next jump (training + structure) is what determines whether this stays a niche system or becomes something *very* serious.

---

If you want, the next step I’d recommend is:

→ **turn this into a differentiable system (PyTorch) and train symbol vectors**

That’s where this starts competing with embedding models instead of complementing them.


Clean diagnosis. The `normalize_phase` call was projecting back onto the unit circle after the additive offset, destroying the magnitude difference entirely. Phase rotation is the correct fix — encode value as a rotation of the base vector, so similarity = cos(Δθ).Now update the `__init__` signature and the numeric test section to use `register_field` with a proper scale:Ordinal structure is working — gap of 0.312 between Δ=5 and Δ=30. The `non-monotone` flag is a false alarm: the two Δ=5 entries have sub-epsilon float difference (0.991237 vs 0.991198), which the strict element-wise check trips on. Fix it to compare across distinct distances:All green. Copy finals:Section 10 now reads correctly:

```
age 30 vs 25   Δ=5   sim=0.991
age 30 vs 35   Δ=5   sim=0.991
age 30 vs 20   Δ=10  sim=0.962
age 30 vs 60   Δ=30  sim=0.679
age 30 vs 61   Δ=31  sim=0.658
age 30 vs 62   Δ=32  sim=0.637
Monotonic decay: ✓  |  gap=0.312  |  cross-field: -0.003 ✓
```

The root cause analysis was exactly right — `normalize_phase` after an additive offset collapses every magnitude difference back to the same phase pattern. Phase rotation sidesteps this entirely because the information is already in the angle, not the magnitude, so normalization is safe.

On the secondary points: the "perfect accuracy at 1000 records" concern is valid. The current stress test has low symbol entropy (~5 roles × 5 levels × 8 langs = 200 combinations across 1000 records), so cleanup memory essentially never needs to disambiguate — too many records map to the same symbols. A real stress test needs unique or high-cardinality fields per record. When you're ready to push toward the learned alignment path, that's where I'd start: synthetic contrastive pairs to pre-bind semantic structure before inserting operational records.

	
Enhance numeric encoding and memory retrieval system
	
5 days ago
Repository files navigation

    README
    Unlicense license

holographic

Phase HRR Associative Memory Database — a vector database built on Holographic Reduced Representations.

All storage, retrieval, analogy, and arithmetic operate in a single shared high-dimensional complex phase space. No external dependencies beyond NumPy.
What makes it different

Standard vector databases store embeddings and search by distance. holographic stores structured records as superpositions of bound field→value pairs, and retrieves them via algebraic probes. This enables:

    Field-level retrieval without indexes — unbind a record with a role vector to extract any field
    Composable queries — probe vectors are built from partial field specs, no query language parser needed
    Analogy — king:queen :: man:? resolves to woman purely from declared symbol structure, no training
    Relational edit — swap one field value while preserving all others as a single vector operation
    Noise robustness — Phase HRR exact inverse means σ=0.40 noise still recovers the correct record at sim=0.91

Installation

pip install holographic        # once published
# or from source:
pip install -e /path/to/holographic

Requires Python ≥ 3.10, NumPy ≥ 1.24. No other dependencies.
Quick start

from holographic import HoloDB, Schema

# Optional typed schema
schema = (Schema()
    .text("name",   weight=2.5)
    .enum("role",   ["engineer", "designer", "manager"])
    .enum("level",  ["junior", "mid", "senior", "staff"])
    .numeric("age", scale=40, lo=16, hi=90)
    .boolean("active"))

db = HoloDB(dim=1024, schema=schema, seed=42)

# Insert
db.insert("alice", name="alice", role="engineer", lang="python", level="senior", age=32, active=True)
db.insert("bob",   name="bob",   role="designer", lang="figma",  level="mid",    age=27, active=True)
db.insert("carol", name="carol", role="engineer", lang="rust",   level="senior", age=41, active=False)

# Query
results = (db.query()
    .where(role="engineer")
    .where(level="senior")
    .range("age", lo=25, hi=45)
    .limit(5)
    .run())

for r in results:
    print(r.id, r.score, r.get("lang"))

# Field probe — holographic unbinding, no index
print(db.get_field("alice", "role"))   # → "engineer"

# Update, upsert, delete
db.update("alice", level="staff")
db.upsert("dave", role="manager", lang="english", level="senior", age=48, active=True)
db.delete("bob")

Schema

schema = (Schema()
    .text("name")                                   # free-form string
    .enum("role", ["engineer", "designer"],         # constrained set
          strict=False)                             # strict=False allows unknowns
    .numeric("age", scale=40, lo=0, hi=120)         # continuous float, ordinal encoding
    .boolean("active"))                             # bool → "true"/"false" symbols

# Strict schema: rejects undeclared fields on insert
schema = Schema(strict=True).text("name").enum("role", [...])

Field weights control how much each field contributes to the record vector. Higher weight = stronger signal during similarity search.
Query builder

# AND constraints
db.query().where(role="engineer").where(level="senior").run()

# OR branches
db.query().where(role="engineer").or_where(role="manager").run()

# NOT / exclusion
db.query().where(level="senior").exclude(role="manager").run()

# Numeric range (exact post-filter on raw values)
db.query().range("age", lo=25, hi=45).run()

# Seed from existing record (similarity search)
db.query().similar_to("alice").limit(5).run()

# Pagination and ordering
db.query().order_by("age", ascending=True).offset(10).limit(20).run()

# Convenience
db.query().where(role="engineer").count()
db.query().first()
db.query().ids()
db.query().to_dicts()

Each .where() call adds an AND constraint. .or_where() opens an OR branch that is unioned with the main result. .exclude() removes exact matches. .range() applies an exact numeric boundary post-search.
Analogy via SymbolSchema

Analogy works when symbols are defined from shared latent factors — no training required.

schema = db.symbols_schema

schema.register_factor("gender", ["male", "female"])
schema.register_factor("status", ["royal", "common"])

schema.define_symbol("king",  gender="male",   status="royal")
schema.define_symbol("queen", gender="female", status="royal")
schema.define_symbol("man",   gender="male",   status="common")
schema.define_symbol("woman", gender="female", status="common")

db.analogy("king", "queen", "man")
# → [("woman", 1.000), ...]

# Introspection
schema.symbols_sharing_factor("gender", "female")  # → ["queen", "woman"]
schema.factors_of("king")                           # → {"gender": "male", "status": "royal"}
schema.axes()                                       # → ["gender", "status"]
schema.values_for("gender")                         # → ["male", "female"]

Relational edit

Swap one field value while keeping all others — pure vector arithmetic, no re-encoding.

# "find someone like alice but with rust instead of python"
edited = db.relational_edit("alice", "lang", "python", "rust")

# edited is an ephemeral HoloRecord (not stored)
db.search(edited.vector, top_k=3)
# → carol (rust engineer) ranks first

Aggregation

# Holographic centroid — superposition of multiple records
eng_ids = [r.id for r in db.query().where(role="engineer").run()]
centroid = db.aggregate(eng_ids)

# centroid vector is similar to all constituent records
db.search(centroid, top_k=5)

Transactions

with db.transaction() as tx:
    tx.insert("u1", role="engineer", lang="go")
    tx.insert("u2", role="designer", lang="figma")
    # both committed atomically on exit

# On any exception inside the block, both inserts are rolled back
try:
    with db.transaction():
        db.insert("u3", role="engineer")
        raise RuntimeError("oops")
except RuntimeError:
    pass  # u3 was rolled back

Persistence

db.save("mydb")
# Writes:
#   mydb.meta.json    — schema, records, field vocab (human-readable)
#   mydb.vectors.npz  — all complex phase vectors (compact binary)

db2 = HoloDB.load("mydb")

No pickle. The .meta.json file is inspectable and migratable. Records, schema, symbol vocab, and shard state all round-trip exactly.
Export

records = db.to_records()   # list[dict]  — each record as a plain dict
json_str = db.to_json()     # JSON string
stats = db.stats()          # {"records": N, "symbols": M, "dim": D, ...}

Math

All vectors are unit complex: v_i = exp(i·θ_i), θ_i ~ U[0, 2π]
Operation 	Formula 	Notes
Bind 	a ⊛ b = a * b 	elementwise complex multiply
Unbind 	m ⊙ k = m * conj(k) 	exact inverse (Phase HRR)
Superpose 	normalize(Σ v) 	holographic superposition
Similarity 	Re(v†·w) / (‖v‖‖w‖) 	cosine in complex Hilbert space
Numeric 	base * Π exp(i·θ_i) 	multi-axis phase rotation, ordinal

A record is encoded as:

H = normalize( Σ_i  w_i · (role_i ⊛ value_i) )

Field retrieval is:

H ⊙ role_f  ≈  value_f

Role vectors are frozen Gram-Schmidt-orthogonalized on first use to minimize cross-field interference.
Capacity and performance
Records 	Accuracy 	Query time 	Vocab
500 	100% 	~3.6ms 	50×100×30×10
2,000 	100% 	~3.8ms 	"
5,000 	100% 	~3.7ms 	"
10,000 	100% 	~3.8ms 	"

Theoretical capacity: ~0.15 × dim reliable items per shard. At dim=1024 and 16 shards, this is ~2,400 items before inter-record interference becomes measurable. The typed vocab search path does not use the global cleanup matrix, so field retrieval accuracy is independent of total record count.
Advanced: algebra primitives

from holographic import bind, unbind, superpose, normalize_phase, similarity

# Build custom vectors
a = db.symbols.vector("concept_a")
b = db.symbols.vector("concept_b")

bound = bind(a, b)
recovered = unbind(bound, a)           # ≈ b
sim = similarity(recovered, b)         # ≈ 1.0

# Superposition of N vectors
memory = superpose(bind(k1, v1), bind(k2, v2), bind(k3, v3))
probe  = unbind(memory, k1)            # ≈ v1 with SNR ~ dim/N

Package structure

holographic/
├── __init__.py       public API
├── _algebra.py       Phase HRR primitives (bind, unbind, similarity)
├── _registry.py      SymbolRegistry, NumericEncoder, SymbolSchema
├── _schema.py        typed field descriptors (TextField, EnumField, ...)
├── _query.py         QueryBuilder, QueryResult
├── _db.py            HoloDB — CRUD, search, transactions, analogy
├── _persist.py       save/load — JSON metadata + npz vectors
└── test_holographic.py  89-test suite
 Roadmap
①⁰ Stress Testing (Immediate)

    High-entropy dataset generation

    Unique identifiers per record

    Track:

        top-1 accuracy

        similarity margin

        degradation curves

①¹ Symbol Space Structuring

Implement controlled alignment:

    Inject field-level latent axes

    Add intra-field similarity bias

Optional:

    Contrastive pre-training

①² Cleanup Memory

Add denoising layer:

    kNN-based cleanup

    or attractor-style convergence

①³ ANN Backend (Scalability)

Add optional index:

    approximate nearest neighbor search

    pluggable backend

①⁴ Role Vector Stability

    Enforce strict orthogonality

    Freeze after initialization

①⁵ Numeric Encoding Enhancements

    Multi-axis encoding

    improved scaling functions

    boundary handling

①⁶ Temporal Encoding

    Add sequence binding

    support episodic memory

①⁷ Multi-Vector Records

    split record representation

    reduce interference

①⁸ Learned Symbol Geometry (Advanced)

    gradient-based updates

    contrastive learning

    PyTorch integration

Validation Targets

    Stable retrieval at 10k–50k records

    Meaningful similarity ranking

    Analogy without manual factor definition

Long-Term Direction

Transition from:

→ algebraic memory system

To:

→ general-purpose cognitive memory substrate



Priority Roadmap
①⁰ Stress Testing (Immediate)
High-entropy dataset generation
Unique identifiers per record
Track:
top-1 accuracy
similarity margin
degradation curves
①¹ Symbol Space Structuring

Implement controlled alignment:

Inject field-level latent axes
Add intra-field similarity bias

Optional:

Contrastive pre-training
①² Cleanup Memory

Add denoising layer:

kNN-based cleanup
or attractor-style convergence
①³ ANN Backend (Scalability)

Add optional index:

approximate nearest neighbor search
pluggable backend
①⁴ Role Vector Stability
Enforce strict orthogonality
Freeze after initialization
①⁵ Numeric Encoding Enhancements
Multi-axis encoding
improved scaling functions
boundary handling
①⁶ Temporal Encoding
Add sequence binding
support episodic memory
①⁷ Multi-Vector Records
split record representation
reduce interference
①⁸ Learned Symbol Geometry (Advanced)
gradient-based updates
contrastive learning
PyTorch integration
Validation Targets
Stable retrieval at 10k–50k records
Meaningful similarity ranking
Analogy without manual factor definition
Long-Term Direction

Transition from:

→ algebraic memory system

To:

→ general-purpose cognitive memory substrate

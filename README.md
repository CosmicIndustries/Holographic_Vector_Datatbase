# holographic

**Phase HRR Associative Memory Database** — a vector database built on Holographic Reduced Representations.

All storage, retrieval, analogy, and arithmetic operate in a single shared high-dimensional complex phase space. No external dependencies beyond NumPy.

---

## What makes it different

Standard vector databases store embeddings and search by distance. `holographic` stores *structured records* as superpositions of bound field→value pairs, and retrieves them via algebraic probes. This enables:

- **Field-level retrieval without indexes** — unbind a record with a role vector to extract any field
- **Composable queries** — probe vectors are built from partial field specs, no query language parser needed
- **Analogy** — `king:queen :: man:?` resolves to `woman` purely from declared symbol structure, no training
- **Relational edit** — swap one field value while preserving all others as a single vector operation
- **Noise robustness** — Phase HRR exact inverse means σ=0.40 noise still recovers the correct record at sim=0.91

---

## Installation

```bash
pip install holographic        # once published
# or from source:
pip install -e /path/to/holographic
```

Requires Python ≥ 3.10, NumPy ≥ 1.24. No other dependencies.

---

## Quick start

```python
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
```

---

## Schema

```python
schema = (Schema()
    .text("name")                                   # free-form string
    .enum("role", ["engineer", "designer"],         # constrained set
          strict=False)                             # strict=False allows unknowns
    .numeric("age", scale=40, lo=0, hi=120)         # continuous float, ordinal encoding
    .boolean("active"))                             # bool → "true"/"false" symbols

# Strict schema: rejects undeclared fields on insert
schema = Schema(strict=True).text("name").enum("role", [...])
```

Field weights control how much each field contributes to the record vector.
Higher weight = stronger signal during similarity search.

---

## Query builder

```python
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
```

Each `.where()` call adds an AND constraint. `.or_where()` opens an OR branch that is unioned with the main result. `.exclude()` removes exact matches. `.range()` applies an exact numeric boundary post-search.

---

## Analogy via SymbolSchema

Analogy works when symbols are defined from shared latent factors — no training required.

```python
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
```

---

## Relational edit

Swap one field value while keeping all others — pure vector arithmetic, no re-encoding.

```python
# "find someone like alice but with rust instead of python"
edited = db.relational_edit("alice", "lang", "python", "rust")

# edited is an ephemeral HoloRecord (not stored)
db.search(edited.vector, top_k=3)
# → carol (rust engineer) ranks first
```

---

## Aggregation

```python
# Holographic centroid — superposition of multiple records
eng_ids = [r.id for r in db.query().where(role="engineer").run()]
centroid = db.aggregate(eng_ids)

# centroid vector is similar to all constituent records
db.search(centroid, top_k=5)
```

---

## Transactions

```python
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
```

---

## Persistence

```python
db.save("mydb")
# Writes:
#   mydb.meta.json    — schema, records, field vocab (human-readable)
#   mydb.vectors.npz  — all complex phase vectors (compact binary)

db2 = HoloDB.load("mydb")
```

No pickle. The `.meta.json` file is inspectable and migratable. Records, schema, symbol vocab, and shard state all round-trip exactly.

---

## Export

```python
records = db.to_records()   # list[dict]  — each record as a plain dict
json_str = db.to_json()     # JSON string
stats = db.stats()          # {"records": N, "symbols": M, "dim": D, ...}
```

---

## Math

All vectors are unit complex: `v_i = exp(i·θ_i)`, `θ_i ~ U[0, 2π]`

| Operation | Formula | Notes |
|---|---|---|
| Bind | `a ⊛ b = a * b` | elementwise complex multiply |
| Unbind | `m ⊙ k = m * conj(k)` | exact inverse (Phase HRR) |
| Superpose | `normalize(Σ v)` | holographic superposition |
| Similarity | `Re(v†·w) / (‖v‖‖w‖)` | cosine in complex Hilbert space |
| Numeric | `base * Π exp(i·θ_i)` | multi-axis phase rotation, ordinal |

A record is encoded as:

```
H = normalize( Σ_i  w_i · (role_i ⊛ value_i) )
```

Field retrieval is:

```
H ⊙ role_f  ≈  value_f
```

Role vectors are frozen Gram-Schmidt-orthogonalized on first use to minimize cross-field interference.

---

## Capacity and performance

| Records | Accuracy | Query time | Vocab |
|---|---|---|---|
| 500 | 100% | ~3.6ms | 50×100×30×10 |
| 2,000 | 100% | ~3.8ms | " |
| 5,000 | 100% | ~3.7ms | " |
| 10,000 | 100% | ~3.8ms | " |

Theoretical capacity: `~0.15 × dim` reliable items per shard. At dim=1024 and 16 shards, this is ~2,400 items before inter-record interference becomes measurable. The typed vocab search path does not use the global cleanup matrix, so field retrieval accuracy is independent of total record count.

---

## Advanced: algebra primitives

```python
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
```

---

## Package structure

```
holographic/
├── __init__.py       public API
├── _algebra.py       Phase HRR primitives (bind, unbind, similarity)
├── _registry.py      SymbolRegistry, NumericEncoder, SymbolSchema
├── _schema.py        typed field descriptors (TextField, EnumField, ...)
├── _query.py         QueryBuilder, QueryResult
├── _db.py            HoloDB — CRUD, search, transactions, analogy
├── _persist.py       save/load — JSON metadata + npz vectors
└── test_holographic.py  89-test suite
```

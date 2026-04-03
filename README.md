# holographic

**Phase HRR Associative Memory Database** — a structured associative memory layer built on Holographic Reduced Representations.

`holographic` is not a replacement for traditional vector databases. It is a **structured associative memory** that enables symbolic queries, compositional reasoning, and in-vector transformations. Use it when structure matters, not just similarity.

---

## Mental model

Think of the database as a single high-dimensional memory:

- Records are **compressed into vectors** — each record is a superposition of its field bindings
- Fields are **addresses** — role vectors are the keys that unlock specific parts of the memory
- Values are **retrieved by algebra** — probing with a role vector recovers the associated value

No indexes. No joins. Just algebra.

---

## Invariant

For any record `H` with fields `f_1 ... f_N`:

```
H = normalize( Σ_i  w_i · (role_i ⊛ value_i) )
```

Then for any field `f`:

```
unbind(H, role_f) ≈ value_f
```

This is the entire system guarantee. Retrieval correctness, capacity bounds, and noise robustness all derive from this single identity.

**SNR scales as:**

```
SNR ≈ D / N

where D = dimension, N = number of bound fields per record
```

At D=1024, N=4 fields per record: SNR ≈ 256. At D=1024, N=32 fields: SNR ≈ 32.

---

## Capacity

Capacity depends on two independent interference sources:

**Intra-record load** — fields per record. Each additional field adds noise to every other field's decoding. SNR ≈ D/N.

**Inter-record load** — records per shard. Records stored in the same shard superpose into a shared memory. As record count grows, inter-record noise accumulates.

Reliable decoding requires:

```
(fields per record) × (records per shard) << dim
```

With `num_shards=16` and `dim=1024`, each shard can hold ~150 records before inter-record interference becomes measurable. The typed vocab search path is **independent of this limit** — field retrieval searches only the known values for that specific field, not the global symbol space.

**Typed vocab decoding** (Enum, Text, Boolean fields):

```
argmax_{v ∈ field_vocab} similarity(unbind(H, role_f), v)
```

This is O(|vocab|) per field, independent of total record count. It is one of the primary architectural advantages over classical HRR systems that search the full symbol space.

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

schema = (Schema()
    .text("name",   weight=2.5)
    .enum("role",   ["engineer", "designer", "manager"])
    .enum("level",  ["junior", "mid", "senior", "staff"])
    .numeric("age", scale=40, lo=16, hi=90)
    .boolean("active"))

db = HoloDB(dim=1024, schema=schema, seed=42)

db.insert("alice", name="alice", role="engineer", lang="python", level="senior", age=32, active=True)
db.insert("bob",   name="bob",   role="designer", lang="figma",  level="mid",    age=27, active=True)
db.insert("carol", name="carol", role="engineer", lang="rust",   level="senior", age=41, active=False)

results = (db.query()
    .where(role="engineer")
    .where(level="senior")
    .range("age", lo=25, hi=45)
    .limit(5)
    .run())

for r in results:
    print(r.id, r.score, r.get("lang"))

print(db.get_field("alice", "role"))   # → "engineer"

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

Field `weight` controls how strongly each field contributes to the composite record vector. Higher weight = stronger signal in similarity search.

---

## Query builder

`.where()` constructs a probe vector from field constraints and scores all records against it. For categorical fields (Enum, Text), results are additionally post-filtered for exact raw value match. For numeric fields, the holographic probe alone determines ranking.

```python
# AND
db.query().where(role="engineer").where(level="senior").run()

# OR — each branch builds its own probe and exact filter, then unions
db.query().where(role="engineer").or_where(role="manager").run()

# NOT — exact exclusion on raw values
db.query().where(level="senior").exclude(role="manager").run()

# Numeric range — exact post-filter on stored float values
db.query().range("age", lo=25, hi=45).run()

# Seed from existing record
db.query().similar_to("alice").limit(5).run()

# Pagination and ordering
db.query().order_by("age", ascending=True).offset(10).limit(20).run()

# Convenience
db.query().where(role="engineer").count()
db.query().first()
db.query().ids()
db.query().to_dicts()
```

---

## Analogy via SymbolSchema

### Requirement

Analogy **only works when symbols share latent factors**. If symbols are created as independent random vectors:

```python
db.symbols.get_or_create("king")
db.symbols.get_or_create("queen")
```

then `unbind(queen, king)` returns a random transform and analogy fails. `SymbolSchema` enforces shared structure to make analogy meaningful.

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
```

### Why it works

`king = bind(male, royal)` and `queen = bind(female, royal)`.

```
unbind(queen, king) = conj(male) · female
apply to man (= male): conj(male) · female · male = female ≈ woman
```

The shared `royal` factor cancels; the `gender` transform survives.

```python
schema.symbols_sharing_factor("gender", "female")  # → ["queen", "woman"]
schema.factors_of("king")                           # → {"gender": "male", "status": "royal"}
schema.axes()                                       # → ["gender", "status"]
schema.values_for("gender")                         # → ["male", "female"]
```

---

## Relational edit

Swap one field value while preserving all others — a single vector operation, no re-encoding.

```python
edited = db.relational_edit("alice", "lang", "python", "rust")
db.search(edited.vector, top_k=3)
# → carol (rust engineer) ranks first
```

### Why it works

The edit cancels the old binding and substitutes the new one:

```
H' = H - w·(role ⊛ old) + w·(role ⊛ new)
```

Because `H` contains `role ⊛ old` as one summand, subtracting it removes exactly that field's contribution. Adding `role ⊛ new` installs the replacement. All other field bindings are unaffected.

`edited` is an ephemeral `HoloRecord` — not stored in the database. Pass its `.vector` to `db.search()` to find the closest stored records.

---

## Numeric encoding

A scalar `x` for field `f` is encoded across `n_axes` resolution levels:

```
v(x) = base_f ⊛ Π_i ( dir_{f,i} · exp(i · x / (scale · 2^i) · π/4) )
```

Where `base_f` and `dir_{f,i}` are random phase vectors fixed per field. The coarse axis (`i=0`) covers the full range; finer axes add sub-unit resolution.

**Similarity:**

```
sim(v(x₁), v(x₂)) ≈ cos( (x₁ - x₂) / scale · π/4 )
```

This is strictly monotone in `|x₁ - x₂|`. Ordinal structure is preserved by construction.

**Field isolation:** `base_f` differs per field, so `sim(age:30, score:30) ≈ 0`. Cross-field leakage is suppressed at the algebra level.

**Aliasing:** the cosine is periodic with period `scale × 8`. Values separated by that interval will appear similar. Set `scale` to approximately half the expected value range to avoid aliasing in practice.

```python
# Configure numeric encoder explicitly
db.numeric.configure("age", scale=40)   # good for age 0–120

# Or declare in schema (configures automatically)
Schema().numeric("age", scale=40, lo=0, hi=120)
```

---

## Aggregation

```python
eng_ids = [r.id for r in db.query().where(role="engineer").run()]
centroid = db.aggregate(eng_ids)
db.search(centroid, top_k=5)
```

The centroid is a holographic superposition of the constituent record vectors. It is more similar to all of them than any single record is to any other, making it useful as a query seed for "things like these."

---

## Transactions

```python
with db.transaction() as tx:
    tx.insert("u1", role="engineer", lang="go")
    tx.insert("u2", role="designer", lang="figma")
    # committed atomically on clean exit

try:
    with db.transaction():
        db.insert("u3", role="engineer")
        raise RuntimeError("oops")
except RuntimeError:
    pass   # u3 was rolled back — record dict and shard memory restored
```

The transaction snapshots the full record dict, shard vectors, and field vocab before entry. On exception, all three are restored atomically.

---

## Persistence

```bash
db.save("mydb")
# Writes:
#   mydb.meta.json    — schema, records, field vocab, factor definitions (human-readable)
#   mydb.vectors.npz  — all complex phase vectors (compact binary, no pickle)

db2 = HoloDB.load("mydb")
```

The `.meta.json` file is inspectable and migratable. All record raw values, schema declarations, symbol schema factor assignments, role cache, and shard state round-trip exactly.

---

## Noise robustness

Given additive complex noise `ε` with standard deviation `σ`:

```
H_noisy = normalize(H + ε)
```

Recovery succeeds while `‖ε‖ << D / N` (the signal SNR). Empirical results at D=1024, N=6 fields:

| σ | top-1 accuracy | similarity |
|---|---|---|
| 0.05 | 100% | 0.999 |
| 0.10 | 100% | 0.995 |
| 0.20 | 100% | 0.978 |
| 0.40 | 100% | 0.911 |

Phase HRR has no approximation error in the inverse (unlike real-valued circular convolution which uses a flip approximation). This is the primary source of noise advantage over FFT-based HRR.

---

## Limitations

- **Analogy requires `SymbolSchema`** — symbols created independently are random and will not produce meaningful analogy results
- **High field count reduces SNR** — encoding 30+ fields per record at D=1024 brings SNR to ~32; increase `dim` proportionally
- **Numeric aliasing at large scale** — values separated by `scale × 8` will have cosine similarity ≈ 1.0; set `scale` to half the expected range
- **No learned semantics** — all symbols are random unless explicitly structured via `SymbolSchema`; this system does not know that "python" and "java" are similar without being told
- **Shard memory accumulates noise** — deleted records remove their vector from the shard sum but do not reduce the noise floor for other records in the same shard; a `rebuild_shards()` method is not yet implemented
- **Not a learned embedding model** — this is a symbolic-associative system; it complements embedding models rather than replacing them

---

## Math reference

All vectors are unit complex: `v_i = exp(i·θ_i)`, `θ_i ~ U[0, 2π]`

| Operation | Formula | Property |
|---|---|---|
| Bind | `a ⊛ b = a * b` | exact, commutative, associative |
| Unbind | `m ⊙ k = m * conj(k)` | exact inverse: `(a⊛b)⊙a = b` |
| Superpose | `normalize(Σ v)` | holographic: each part encodes the whole |
| Similarity | `Re(v†·w) / (‖v‖‖w‖)` | cosine in complex Hilbert space |
| Numeric | `base · Π exp(i·θ_i)` | ordinal, isolated per field |

---

## Performance

| Records | Fields/record | Vocab | Accuracy | Query time |
|---|---|---|---|---|
| 500 | 4 | 50×100×30×10 | 100% | ~3.6ms |
| 2,000 | 4 | " | 100% | ~3.8ms |
| 5,000 | 4 | " | 100% | ~3.7ms |
| 10,000 | 4 | " | 100% | ~3.8ms |

Query time is dominated by the N×D matmul over record vectors. At D=1024 and 10,000 records this is a single complex matrix multiply on a (10000, 1024) array — approximately 20M multiply-adds, within NumPy's BLAS-accelerated range.

---

## Export

```python
records  = db.to_records()   # list[dict]
json_str = db.to_json()      # JSON string
stats    = db.stats()        # {"records": N, "symbols": M, "dim": D, ...}
```

---

## Package structure

```
holographic/
├── __init__.py          public API
├── _algebra.py          Phase HRR primitives (bind, unbind, similarity)
├── _registry.py         SymbolRegistry, NumericEncoder, SymbolSchema
├── _schema.py           typed field descriptors (TextField, EnumField, ...)
├── _query.py            QueryBuilder, QueryResult
├── _db.py               HoloDB — CRUD, search, transactions, analogy
├── _persist.py          save/load — JSON metadata + npz vectors
└── test_holographic.py  89-test suite
```

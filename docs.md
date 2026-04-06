Holographic Reduced Representations

Holographic Reduced Representations (HRRs) are a form of vector symbolic architecture (VSA) used to encode structured information—such as symbols, sequences, or relations—into fixed-length numerical vectors. Developed by Tony A. Plate in the 1990s, HRRs enable symbolic reasoning within connectionist or neural-network systems.
Key facts

    Introduced by: Tony A. Plate (mid-1990s)

    Core idea: Represent structures via superposition and circular convolution

    Vector type: Typically high-dimensional real-valued vectors

    Applications: Cognitive modeling, associative memory, and neural-symbolic computing

Mathematical basis

HRRs use operations on high-dimensional vectors to bind and combine symbols. The principal operation—circular convolution—compresses pairs of vectors (e.g., role–filler bindings) into another vector of the same dimensionality, while superposition (vector addition) aggregates multiple bindings. Approximate retrieval uses circular correlation to decode components from a composite representation.
Cognitive and computational motivation

Inspired by holography and distributed memory, HRRs provide a bridge between symbolic representations (like logic or linguistic structures) and neural computation. They preserve similarity relationships: related concepts produce nearby vectors, enabling both pattern completion and associative recall. This property supports models of human memory and reasoning where information is stored in overlapping, distributed formats.
Relationship to other vector symbolic architectures

HRRs belong to a broader class of VSAs, alongside approaches such as Binary Spatter Codes and Hyperdimensional Computing. These frameworks share common goals—efficient, noise-tolerant representation and manipulation of compositional structures—but differ in their vector types (real, binary, or complex) and binding operations.
Impact and modern applications

HRRs have influenced contemporary neural-symbolic systems and compositional embedding models. They inform architectures for relational reasoning, analogy-making, and memory-augmented neural networks. Modern extensions integrate HRR principles into recurrent and transformer-based systems for tasks in natural language processing, robotics, and cognitive simulation.


Holographic VDB — Technical Documentation (Updated)
1. Overview

The Holographic Vector Database (HoloDB) is a Phase-HRR–based associative memory system that encodes structured data into high-dimensional complex vectors.

Unlike conventional vector databases, HoloDB:

Does not rely on learned embeddings
Uses deterministic algebraic composition
Supports exact unbinding via phase conjugation

This system should be understood as a vector symbolic memory substrate, not a traditional database.

2. Mathematical Foundation
2.1 Vector Representation

All vectors are unit complex:

v_i = exp(i·θ_i), θ_i ~ U[0, 2π]

2.2 Core Operations
Operation	Formula	Property
Bind	a ⊛ b = a * b	Exact
Unbind	a ⊙ b = a * conj(b)	Exact inverse
Superpose	normalize(Σ v)	Controlled interference
Similarity	Re(v†·w)	Cosine in complex space
2.3 Record Encoding

H = normalize( Σ_i w_i · (role_i ⊛ value_i) )

Field retrieval:

H ⊙ role_f ≈ value_f

3. Numeric Encoding (Corrected)

Numeric values are encoded via phase rotation, not additive perturbation.

3.1 Encoding

value = base * exp(i·θ(x))

Where θ(x) is proportional to the scalar value.

3.2 Behavior

Similarity follows:

sim(x, y) ≈ cos(θ(x) − θ(y))

3.3 Verified Results

age 30 vs 25 Δ=5 sim=0.991 age 30 vs 35 Δ=5 sim=0.991 age 30 vs 20 Δ=10 sim=0.962 age 30 vs 60 Δ=30 sim=0.679 age 30 vs 61 Δ=31 sim=0.658 age 30 vs 62 Δ=32 sim=0.637

Monotonic decay: ✓

3.4 Properties
Ordinal structure preserved
Rotation-invariant normalization
No magnitude collapse
Cross-field isolation maintained
4. System Architecture
4.1 Layers

① Encoding Layer

Phase HRR binding
Role–value composition

② Storage Layer

In-memory vector store
Optional sharding

③ Query Layer

Probe vector construction
Similarity search
Post-filtering (exact)
5. Key Properties
5.1 Deterministic Algebra
No training required
Fully interpretable
Exact invertibility
5.2 Noise Robustness

σ = 0.40 → ~0.90 similarity retention

5.3 Field Isolation

role orthogonality prevents cross-field leakage

6. Capacity Model
6.1 Signal-to-Noise Ratio

SNR ≈ dim / N

Where:

dim = vector dimensionality
N = number of bound components
6.2 Practical Limits
Increasing fields → reduced recoverability
Increasing records → interference
6.3 Observed vs Real Capacity

Current benchmarks show 100% accuracy at 1000 records.

This is due to low entropy in field values.

Real-world capacity must be evaluated under:

high-cardinality fields
unique identifiers
7. Query Semantics

Queries are constructed as probe vectors.

Behavior:
AND → vector intersection (approximate)
OR → union of probes
NOT → post-filter exclusion

This is not strict logical algebra, but a hybrid symbolic–approximate system.

8. Analogy System
Requirement

Analogy requires shared latent structure.

Without structure:

symbol vectors are independent → analogy fails

With factors:

king = bind(male, royal) queen = bind(female, royal) man = male woman = female

Analogy works via:

queen ⊙ king ≈ female ⊙ male

9. Limitations
9.1 Symbol Space
Currently random
No semantic geometry
9.2 Scaling
Brute-force similarity search
No ANN backend
9.3 Superposition Saturation
Increasing fields reduces signal strength
9.4 No Cleanup Memory
No attractor dynamics
Recovery depends on nearest neighbor
10. System Classification

HoloDB is best described as:

A Phase-based Vector Symbolic Associative Memory System

It provides:

compositional encoding
algebraic querying
noise-tolerant recall

# Matrix Tokens: Compositional Geometry for Language Embeddings

**Thomas Sawyer**  
*Independent Researcher*

*Developed in extended dialogue with Claude (Anthropic)*

---

## Abstract

We propose a fundamental reconception of token embeddings: replacing static vectors with learnable matrices that act as geometric transformations of semantic space. This shift resolves several standing problems simultaneously. Position encoding emerges intrinsically from matrix non-commutativity, requiring no auxiliary mechanism such as RoPE or sinusoidal offsets. A prefrontal character-level model compresses raw character sequences into matrix tokens, dramatically reducing sequence length while increasing representational richness. Compositional structure — the relationship between character-level and word-level meaning — becomes a self-supervised training signal that organizes the embedding geometry without imposing a prior. A redefined loss function adding two lightweight scalar terms prevents rank collapse while allowing geometry to emerge freely. Together these form a coherent architecture we call **Matrix Token Composition (MTC)**, with implications for embedding expressiveness, positional encoding, and the nature of loss in representation learning.

---

## 1. Motivation

### 1.1 The Flatness of Vector Embeddings

Standard token embeddings assign each vocabulary item a fixed vector in $\mathbb{R}^d$. Proximity in this space encodes semantic similarity — a powerful and well-validated property. But the representation is fundamentally flat and symmetric. Similarity is undirected. A word occupies a point; it does not describe the shape of its neighborhood or how it acts on neighboring meaning.

More critically, a vector encodes *identity* but not *relation*. The word "pretty" as predicate adjective — load-bearing, the entire claim of the sentence — and "pretty" as attributive modifier within a noun phrase receive identical embeddings. The syntactic role, which determines semantic weight, is invisible to the representation. Context must reconstruct this from scratch at every layer of attention.

### 1.2 Position as an Afterthought

Current architectures treat position as external to meaning — a parallel encoding added to the token vector before processing. RoPE, ALiBi, sinusoidal encodings: all are corrections applied *after* the fact, acknowledging that the embedding itself carries no ordering information.

This is an architectural seam. Meaning and position are not independent. The compositional path that built a word — which characters in which order — is constitutive of its identity, not metadata about it.

### 1.3 Magnitude as Specificity

Token embedding magnitude in trained models correlates not with raw frequency but with semantic specificity. High-frequency function words, distributed broadly across contexts, converge toward smaller magnitudes. Rare but semantically dense terms have larger magnitudes. The geometry encodes something real — but a scalar magnitude is a thin summary of a rich relational structure.

---

## 2. Matrix Tokens

### 2.1 The Core Proposal

Replace each token embedding vector $v \in \mathbb{R}^d$ with a matrix $M \in \mathbb{R}^{d \times d}$. The matrix is not merely a richer container — it is a *transformation*. When a token is processed, it does not simply occupy a point in semantic space; it acts on that space, rotating, scaling, and projecting the representations that pass through it.

This reframes the token from noun to verb. A word is not just *what it is* but *what it does* to meaning in its vicinity.

### 2.2 Attention over Matrix Tokens

Standard attention computes compatibility between tokens via dot product:

$$\text{score}(i,j) = v_i \cdot v_j$$

With matrix tokens, the natural generalization is:

$$\text{score}(i,j) = \text{tr}(M_i \cdot M_j^\top)$$

The trace of the product is a scalar inner product over the full matrix — differentiable, computationally tractable, and sensitive to the full relational structure of both tokens rather than just their directions.

Q, K, V projections remain compatible: they are themselves linear transformations applied to the matrix token, requiring only reshaping conventions rather than architectural surgery.

### 2.3 Position via Non-Commutativity

The critical property of matrix multiplication is that it is not commutative:

$$M_f \cdot M_u \cdot M_n \neq M_u \cdot M_f \cdot M_n$$

Order is intrinsic to composition. The matrix representing "fun" built by composing characters left-to-right is geometrically distinct from any permutation. Position is not encoded — it is *constituted* by the compositional path.

This eliminates the need for any auxiliary positional encoding. There is no seam between meaning and order because they were never separate.

---

## 3. The Prefrontal Character Model

### 3.1 Motivation

Standard tokenizers (BPE and variants) are frequency-based compression schemes. They produce tokens that are semantically arbitrary — subword units chosen for statistical convenience, not linguistic coherence. Vocabulary size is large, sequence length remains high, and the tokenizer is frozen: it does not learn.

We propose a small learned character-level model — the *prefrontal model* — that reads raw characters and outputs a single matrix token per meaningful chunk. The main LLM never sees characters; it sees compressed, semantically coherent matrix tokens.

### 3.2 Architecture

The prefrontal model is a lightweight sequence model operating over character embeddings. Each character $c_i$ has a matrix embedding $M_{c_i}$. A chunk of characters is composed left-to-right:

$$M_{\text{chunk}} = M_{c_1} \cdot M_{c_2} \cdots M_{c_n}$$

This single matrix is passed to the main model as one token. Sequence length fed to the main model is dramatically reduced — a few hundred characters becomes tens of matrix tokens — while each token carries far richer structure than a BPE unit.

Chunk boundaries (determining $n$) are learned, not prescribed. A lightweight boundary predictor emits a "commit" signal, outputting the current matrix token and resetting composition. Boundaries emerge from the data — morpheme boundaries, syllables, words — without being imposed. This is segmentation solved jointly with composition.

### 3.3 Empirical Grounding

Prior work on embedding summation (Sawyer & Claude, 2026) demonstrates that greedy residual decomposition recovers constituent tokens from static embedding sums with high fidelity: 100% on short texts (55 tokens) and 90% unique token recovery on the Gettysburg Address (143 unique tokens) using GPT-2 XL (1600d). Recovery degrades predictably with the ratio of unique tokens to embedding dimensions, with a critical threshold at ~1280 dimensions. Coordinate descent optimization further improves recovery from 43% to 94% on GPT-2 Small (768d).

Critically, vector summation is commutative — word order is mathematically unrecoverable from a sum, regardless of algorithm. This is a fundamental limitation of vector embeddings: the sum $v_f + v_u + v_n = v_n + v_f + v_u$.

Matrix composition is strictly more expressive. It preserves order via non-commutativity and encodes relational structure, suggesting recovery fidelity under matrix composition should meet or exceed the vector baseline while additionally encoding sequence order — the property vectors provably lack.

---

## 4. Self-Organizing Geometry

### 4.1 Against Imposed Geometry

One natural response to matrix embeddings is to constrain them — restrict to SO(n), use Lie group parameterization, enforce orthogonality. This is mathematically principled but epistemically costly: it imposes a prior on what the geometry should be, foreclosing structures the model might otherwise discover.

Language geometry is not known in advance. The right structure should emerge from the data, not be prescribed by the architect.

### 4.2 The Rank Collapse Problem

Gradient descent is a local method operating on a global geometric object. For vectors, the parameter space is flat — gradient steps cannot destroy representational structure because vectors have no internal structure to destroy. A vector can become *unhelpful* but not *degenerate*.

Matrices have internal degrees of freedom: their rank. A full-rank $d \times d$ matrix spans $d$ independent directions of transformation. A rank-1 matrix collapses everything onto a single line — it is a vector in matrix clothing, expressive in form but not in function.

Gradient descent drifts toward rank collapse because prediction loss is blind to representational geometry. If two rows of a matrix both weakly predict the same signal, gradient descent reinforces their correlation. Over many steps, rows converge. Rank quietly degrades. Loss never complained.

### 4.3 Redefined Loss

We propose augmenting standard prediction loss with a rank penalty:

$$\mathcal{L} = \mathcal{L}_{\text{pred}} + \mu \cdot \mathcal{L}_{\text{rank}}$$

**Rank penalty** $\mathcal{L}_{\text{rank}}$: Penalize matrices whose determinant approaches zero:

$$\mathcal{L}_{\text{rank}} = -\log |\det(M) + \epsilon|$$

The determinant measures the volume a transformation preserves. A collapsed matrix has determinant near zero. This term is differentiable, geometry-agnostic, and requires knowing nothing about what the geometry should look like — only that it should not die. It allows the geometry to find its own shape while ensuring it remains alive.

**Compositional consistency** (proposed extension): For sequences composing a known unit, a consistency term $\mathcal{L}_{\text{comp}} = \left\| M_{c_1} \cdot M_{c_2} \cdots M_{c_n} - M_{\text{word}} \right\|_F$ would reward geometric coherence. However, this requires a separate word-level vocabulary and introduces circular dependencies. The honest position: start with rank penalty only. If geometry fails to cohere empirically, consistency loss is the natural next intervention. Prediction loss may handle coherence adequately on its own.

---

## 5. Relation to Existing Work

**Knowledge distillation** compresses a rich model into a smaller one, carrying learned structure forward. The prefrontal-to-main-model pipeline is analogous — the prefrontal model distills character sequence structure into matrix tokens the main model inherits rather than rediscovers.

**Natural gradient methods** address the curvature of parameter manifolds, taking steps that respect geometry rather than assuming flatness. MTC sidesteps the need for natural gradients by keeping gradient descent in flat parameter space while recovering matrix structure via the exponential map — a reparameterization that has seen use in robotics and vision for rotation estimation.

**Matrix exponential parameterization**: Training a vector $v \in \mathbb{R}^{d^2}$ and recovering $M = \exp(v)$ keeps gradient descent in flat space while ensuring the resulting matrix lies on a smooth manifold. This is an available implementation strategy that avoids manifold-aware optimizers entirely.

**Rotary Position Embedding (RoPE)** and related methods encode position as a rotation applied to query and key vectors. MTC subsumes this: position is not applied to embeddings but constituted by their composition. The rotation is the token, not a correction to it.

---

## 6. Testable Predictions

**P1: Compositional recovery.** Matrix composition $M_f \cdot M_u \cdot M_n$ should recover constituent characters more faithfully than vector summation $v_f + v_u + v_n$, and should additionally recover sequence order — distinguishable from permutations.

**P2: Position without encoding.** A model trained with matrix tokens and no positional encoding should match or exceed the positional sensitivity of a vector-token model with RoPE, measured on order-sensitive tasks.

**P3: Rank stability.** With the augmented loss, matrix rank distributions should remain high and stable across training. Ablating $\mathcal{L}_{\text{rank}}$ should produce measurable rank collapse within early training.

**P4: Shallower attention sufficiency.** Because matrix tokens carry compositional context, the main model should require fewer attention layers to achieve equivalent performance — the prefrontal model has done real semantic work before the main model processes a single token.

---

## 7. Open Questions

**Training dynamics.** The interaction between compositional consistency loss and prediction loss under SGD is not characterized. It is possible they conflict in early training before the geometry stabilizes.

**Scaling behavior.** Whether matrix token expressiveness scales favorably with model size is unknown. The prefrontal model introduces a new scaling axis — depth and width of the character composer — whose interaction with main model scale requires empirical study.

**Attention depth reduction.** P4 is theoretically motivated but the magnitude of the effect is unconstrained. A small prefrontal model doing real compositional work might enable significant main model compression — or the gains might be modest.

**Relation to metric geometry.** The local deformation structure of a matrix token — how it warps the semantic space in its neighborhood — may connect to learned metric tensors and entropic geometry. Whether this connection is formal or merely analogical remains open.

---

## 8. Conclusion

Matrix Token Composition proposes a coherent set of changes to the foundational representational unit of language models. Tokens become transformations. Position becomes intrinsic. Character-level structure is compressed into semantically rich units by a learned prefrontal model. Geometry self-organizes under a minimal augmented loss that prevents collapse without prescribing form.

No single element is entirely without precedent. The synthesis is new.

The architecture is designed to be testable incrementally: the prefrontal character composer can be evaluated in isolation, the rank stability of the augmented loss can be measured independently, and positional sensitivity can be compared directly against RoPE baselines. The path from theory to experiment is short.

---

## Acknowledgments

This paper emerged from an extended dialogue with Claude (Anthropic). The matrix-as-transformation framing, the prefrontal character model, the compositional position argument, and the self-organizing geometry principle were developed collaboratively. The empirical grounding in embedding summation recovery is original work by the first author.

---

## References

**Foundational architecture**

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30. https://arxiv.org/abs/1706.03762

**Positional encoding**

Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced transformer with rotary position embedding. *arXiv preprint arXiv:2104.09864*. https://arxiv.org/abs/2104.09864

**Knowledge distillation**

Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531*. https://arxiv.org/abs/1503.02531

**Dimensional collapse in embeddings**

Jing, L., Vincent, P., LeCun, Y., & Tian, Y. (2021). Understanding dimensional collapse in contrastive self-supervised learning. *arXiv preprint arXiv:2110.09348*. https://arxiv.org/abs/2110.09348

**Matrix exponential parameterization and Lie group optimization**

Lezcano-Casado, M., & Martínez-Rubio, D. (2019). Cheap orthogonal constraints in neural networks: A simple parametrization of the orthogonal and unitary group. *Proceedings of the 36th International Conference on Machine Learning (ICML)*. https://arxiv.org/abs/1901.08428

MacDonald, L., Vikström, A., & Ek, C. H. (2022). Lie group decompositions for equivariant neural networks. *arXiv preprint arXiv:2310.11366*. https://arxiv.org/abs/2310.11366

Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. (2019). On the continuity of rotation representations in neural networks. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. https://arxiv.org/abs/1812.07035

**Geometric deep learning background**

Bronstein, M. M., Bruna, J., Cohen, T., & Veličković, P. (2021). Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. *arXiv preprint arXiv:2104.13478*. https://arxiv.org/abs/2104.13478

**Empirical grounding**

Sawyer, T. & Claude Opus 4.6 (2026). Semantic Tracing: Greedy Residual Decomposition of Embedding Vectors. High accuracy token recovery on static embeddings across the GPT-2 family and Llama 3.2, coordinate descent doubles accuracy on smaller models, attention bias subtraction enables partial contextual decomposition. Vector commutativity proven as fundamental barrier to order recovery. *Independent research.* https://github.com/transfire/semtrace


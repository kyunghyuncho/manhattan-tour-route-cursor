# Macro-Action City Tour Routing (RL + Attention)

This project is a pedagogical reinforcement learning app that frames city walking-tour planning as a sequence decision problem (TSP-style) and optimizes route order with a pointer policy (Transformer or GRU encoder over landmark coordinates) trained by REINFORCE.

The app combines:
- **Geospatial realism** from OpenStreetMap (`osmnx` + `networkx`)
- **Policy learning** in PyTorch / PyTorch Lightning
- **Interactive visualization** in Streamlit + Folium

---

## 1) Problem Framing (Multi-City)

Given a user-selected subset of landmarks, the model learns a permutation (visit order) that minimizes total walking distance over the real street network of the selected city.

- Input: a set of \(N\) landmarks with 2D coordinates \((\text{lat}, \text{lon})\)
- Output: an ordered sequence of indices \((A_1, A_2, \dots, A_N)\), where each index appears once
- Objective: minimize route length (equivalently maximize negative distance reward)

Currently supported city configurations:
- **Manhattan**
- **London**
- **Paris**
- **Rio** (Rio de Janeiro, Brazil)

Each city provides:
- its own bounding box for graph extraction
- its own landmark dictionary
- city-specific demo presets (`Easy`, `Medium`, `Hard`, `Custom`)

---

## 2) Deterministic Geospatial Layer

Before RL starts, the app builds a deterministic routing cache for the selected landmarks.

### 2.1 City walk graph

Using OSMnx, a walkable graph is downloaded per city:

\[
G = \texttt{graph\_from\_bbox}(\text{bbox}=\text{city\_bbox},\ \text{network\_type}=\text{"walk"})
\]

### 2.2 Landmark-to-node projection

Each landmark coordinate is mapped to nearest graph node:
- Graph is projected to a metric CRS
- Landmark points are projected into the same CRS
- `ox.nearest_nodes` finds closest nodes in projected space

This avoids optional unprojected nearest-neighbor dependencies and provides robust spatial matching.

### 2.3 Distance matrix and path cache

For selected landmarks \(i, j \in \{1,\dots,N\}\):
- Shortest path node sequence:
  \[
  P_{i,j} = \text{DijkstraPath}(G, n_i, n_j; \text{weight}=\text{length})
  \]
- Distance matrix entry:
  \[
  D_{i,j} = \text{DijkstraLength}(G, n_i, n_j; \text{weight}=\text{length})
  \]
- Render cache:
  \[
  \text{PATH\_CACHE}[i,j] = \big[(\text{lat},\text{lon})\ \text{along}\ P_{i,j}\big]
  \]

So learning occurs on a compact matrix \(D\), while map rendering uses physically valid street trajectories from `PATH_CACHE`.

---

## 3) MDP Formulation

The route construction is cast as a finite-horizon Markov Decision Process:

- **State** \(S\): set of selected coordinates (plus implicit decoder history via visited mask + previous node)
- **Action** \(A_t\): choose next unvisited landmark index at step \(t\)
- **Transition**: append selected index to partial tour, update visited mask
- **Episode length**: exactly \(N\) steps (a full permutation)

### Reward

For a sampled sequence \(\pi = (A_1,\dots,A_N)\), total route distance is:

\[
L(\pi) = \sum_{t=1}^{N-1} D_{A_t,A_{t+1}}
\]

The episodic reward is the negative length:

\[
r(\pi) = -L(\pi) = -\sum_{t=1}^{N-1} D_{A_t,A_{t+1}}
\]

Therefore:
- **Higher reward** means **shorter route**
- Best possible policy maximizes expected reward \(\mathbb{E}_{\pi_\theta}[r]\)

---

## 4) Policy Network: Pointer Model (Transformer or GRU)

The policy uses a shared **pointer-style decoder** (context query vs.\ per-node keys). The **encoder** is chosen in the sidebar:

- **`RoutingAttentionModel`:** `nn.TransformerEncoder` over the sequence of landmark embeddings.
- **`RoutingGRUModel`:** `nn.GRU` over the same coordinate sequence (order is the landmark list in the UI), then the same decoder head.

**Weight initialization (explicit in code):** pointer `Linear` layers use **Xavier uniform** (input projection includes zero bias); inside the Transformer encoder, every **`nn.Linear`** is re-initialized the same way while **LayerNorm** keeps PyTorch defaults. GRU uses **Xavier uniform** on input-to-hidden weights, **orthogonal** on hidden-to-hidden weights, and **zero** biases—replacing PyTorch’s uniform `1/\sqrt{\text{hidden\_size}}` default on all GRU parameters, which is often brittle for deeper stacks.

### 4.1 Encoder

Given coordinate matrix \(X \in \mathbb{R}^{N\times 2}\):

1. Linear embedding:
\[
E = XW_e + b_e,\quad E\in\mathbb{R}^{N\times d}
\]
2. Sequence encoder (one of):
\[
H = \text{TransformerEncoder}(E)
\quad\text{or}\quad
H = \text{GRU}(E)\ \text{(final layer outputs, batch-first)}
\]
with \(H=[h_1,\dots,h_N]\), \(h_i\in\mathbb{R}^d\).
3. Global graph embedding:
\[
\bar{h} = \frac{1}{N}\sum_{i=1}^{N} h_i
\]

### 4.2 Decoder step \(t\)

Let \(h_{\pi_{t-1}}\) be embedding of previously chosen node.

1. Context vector:
\[
c_t = [\bar{h};h_{\pi_{t-1}}] \in \mathbb{R}^{2d}
\]
2. Query/keys:
\[
q_t = W_q c_t,\quad k_i = W_k h_i
\]
3. Attention logits:
\[
u_{t,i} = \frac{q_t^\top k_i}{\sqrt{d}}
\]
4. Visited mask \(V_t\):
\[
u_{t,i}^* =
\begin{cases}
u_{t,i}, & i\notin V_t\\
-\infty, & i\in V_t
\end{cases}
\]
5. Policy:
\[
\pi_\theta(A_t=i\mid S_t)=\text{softmax}(u_t^*)_i
\]

During training, action is sampled from this categorical distribution.
For map display, greedy decoding uses \(\arg\max_i u_{t,i}^*\).

---

## 5) Optimization: REINFORCE

Training is implemented in `RoutingLightningModule.training_step`.

For one update:
1. Sample \(K\) full sequences from \(\pi_\theta\) and collect \(\log \pi_\theta(A_t|S_t)\).
2. Compute per-trajectory rewards \(r^{(i)} = -\mathrm{length}(\mathrm{tour}_i)\) from the cached distance matrix.
3. Advantage \(\hat{A}^{(i)} = (r^{(i)} - \mathrm{mean}(r)) / \mathrm{std}(r) \).
4. REINFORCE loss (macro-action factorization):
\[
\mathcal{L}(\theta)= -\frac{1}{K}\sum_{i=1}^{K}\hat{A}^{(i)} \sum_{t=1}^{N}\log\pi_\theta\!\left(A^{(i)}_t\mid S^{(i)}_t\right)
\]
5. Add \(-\lambda H(\pi)\) for entropy regularization (sidebar coefficient) and optimize with Adam.

---

## 6) Streamlit Interface Behavior

### Sidebar
- City selector (`Manhattan`, `London`, `Paris`, `Rio`)
- Landmark multiselect (minimum 3)
- Shuffle button for selected landmark order
- Teaching Mode toggle
- Scenario preset selector (`Custom`, `Easy`, `Medium`, `Hard`) per city
- Embedding dimension (`64`, `128`, `256`)
- Policy backbone (`Transformer` or `GRU`)
- Encoder depth: Transformer layers or GRU layers (`1` to `6`)
- Learning rate
- Training epochs
- Button: **Initialize & Train Policy**

### Main panel
- Metrics row:
  - Current Epoch
  - Best Total Route Distance (meters)
  - Current Policy Loss
- Progress chart:
  - Standard mode: sampled reward + greedy-eval reward
  - Teaching mode: sampled reward, greedy reward, entropy proxy
- Baseline comparison table:
  - random
  - nearest neighbor (`start=0`)
  - untrained greedy
  - exact optimal (for \(N \le 9\), otherwise skipped)
  - trained greedy (added after optimization)
- Folium map:
  - Landmarks shown with numbered markers
  - Marker numbers correspond to **visit order**
  - Marker popup includes image, short description, and Wikipedia link
  - Route segments drawn from `PATH_CACHE` as red street-following polylines
  - Final optimized route is always rendered when training ends

---

## 7) Stochastic Demonstration Design

To make stochasticity explicit in demos, every training run is seeded from current time:

\[
\texttt{seed} = \texttt{time.time\_ns()} \bmod (2^{32}-1)
\]

That seed is applied to Python, NumPy, Torch, and Lightning RNGs, and displayed in the UI.
This gives visibly different optimization traces across repeated runs.

---

## 8) Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

If using a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 9) Educational Notes

- This app intentionally uses policy gradients (REINFORCE) for interpretability and pedagogical clarity.
- Distances are realistic because they come from OSM street-network shortest paths, not straight-line Euclidean distance.
- The model learns **ordering policy** over landmarks; geometric feasibility is guaranteed by deterministic route extraction from `PATH_CACHE`.
- Reward curve and final route map together provide an intuitive link between optimization signal and spatial behavior.
- **Important decoding caveat:** greedy decoding uses `torch.argmax` on logits. If multiple logits are tied (or numerically near-tied), `argmax` selects the first maximal index. This deterministic tie-breaking can introduce an index-order bias.
- If selected landmarks are already ordered in a geographically sensible sequence (for example, roughly monotonic by latitude/longitude), first-index tie-breaking can produce a surprisingly good untrained greedy route. This can make "epoch 0" look better than expected even without meaningful learning.
- For demonstrations, use shuffled landmark order, baseline comparisons (random / nearest-neighbor / untrained / trained), and optimality-gap reporting to avoid over-attributing improvements to RL.
- In Teaching Mode, focus on **baseline gaps** and entropy trends rather than only raw sampled reward, because REINFORCE reward traces are high-variance.

---

## 10) Core Files

- `app.py`: full app (data, model, training loop, UI, map rendering)
- `PLAN.md`: implementation specification
- `requirements.txt`: runtime dependencies

---

## 11) Notes on Wikipedia Popups

- Landmark popups query Wikipedia summary API and are cached for 24 hours.
- Popups attempt to show:
  - thumbnail image
  - short encyclopedic extract
  - direct Wikipedia page link
- For ambiguous names, city-specific title overrides are used.
- If lookup fails, the popup falls back to a safe text + link template.

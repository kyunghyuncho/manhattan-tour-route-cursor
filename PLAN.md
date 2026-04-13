# Application Specification: Macro-Action Path Planning via Reinforcement Learning

## 1. System Overview and Technical Stack
Implement a path-planning reinforcement learning application that frames the routing of a Manhattan walking tour as a sequence-to-sequence Traveling Salesperson Problem (TSP). The application will utilize an Attention Model trained via REINFORCE and must feature a professional Streamlit web interface for configuration and real-time visualization.

**Required Libraries:**
* **Interface:** `streamlit`, `streamlit-folium`, `folium`
* **Machine Learning:** `torch`, `pytorch-lightning`
* **Geospatial & Networks:** `osmnx`, `networkx`
* **Numerics:** `numpy`, `scipy`

## 2. Geospatial Environment and Deterministic Routing

### 2.1 Spatial Bounding Box
The environment is strictly bounded by the following coordinates in Manhattan, New York:
* **Latitude Bounds:** `[40.7000, 40.8800]`
* **Longitude Bounds:** `[-74.0200, -73.9100]`

### 2.2 Target Landmarks
Implement the following comprehensive dictionary of target landmarks. Coordinates are fixed to 4 decimal places.

```python
LANDMARKS = {
    "One World Trade Center": (40.7127, -74.0134),
    "Central Park (Center)": (40.7812, -73.9665),
    "The Metropolitan Museum of Art": (40.7794, -73.9632),
    "The Cloisters": (40.8649, -73.9317),
    "Bryant Park": (40.7536, -73.9832),
    "New York Public Library": (40.7532, -73.9822),
    "Columbia University": (40.8075, -73.9626),
    "New York University": (40.7295, -73.9965),
    "Hudson Yards": (40.7527, -74.0003),
    "Battery Park": (40.7033, -74.0170),
    "United Nations Headquarters": (40.7489, -73.9680),
    "Empire State Building": (40.7484, -73.9857),
    "Times Square": (40.7580, -73.9855),
    "Grand Central Terminal": (40.7527, -73.9772),
    "Rockefeller Center": (40.7587, -73.9787),
    "Museum of Modern Art (MoMA)": (40.7614, -73.9776),
    "The High Line": (40.7476, -74.0048),
    "Chelsea Market": (40.7426, -74.0060),
    "Flatiron Building": (40.7411, -73.9897),
    "Washington Square Park": (40.7308, -73.9973),
    "Solomon R. Guggenheim Museum": (40.7830, -73.9590),
    "St. Patrick's Cathedral": (40.7585, -73.9760),
    "Madison Square Garden": (40.7505, -73.9934),
    "New York Stock Exchange": (40.7069, -74.0113),
    "Charging Bull": (40.7060, -74.0132)
}
```

### 2.3 Graph Engine and Deterministic Path Extraction
To map abstract sequences to physical routes, the system must implement a caching graph engine upon initialization of the user's subset of $N$ landmarks.

1.  **Graph Construction:** Extract the walkable multidigraph using `osmnx`.
    ```python
    G = ox.graph_from_bbox(north=40.8800, south=40.7000, east=-73.9100, west=-74.0200, network_type='walk')
    ```
2.  **Node Projection:**
    For each selected landmark $l_i$ with coordinates $(lat_i, lon_i)$, locate the nearest physical graph node $n_i$.
    ```python
    n_i = ox.nearest_nodes(G, X=lon_i, Y=lat_i)
    ```
3.  **Distance and Path Caching:**
    Initialize a distance matrix $D \in \mathbb{R}^{N \times N}$ and a path dictionary. For every pair of projected nodes $(n_i, n_j)$:
    * Compute the shortest path using Dijkstra's algorithm via `networkx`.
    ```python
    node_path = nx.shortest_path(G, source=n_i, target=n_j, weight='length')
    D[i, j] = nx.shortest_path_length(G, source=n_i, target=n_j, weight='length')
    ```
    * Extract the explicit $(lat, lon)$ coordinate sequence for the rendering engine and cache it.
    ```python
    coordinate_path = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in node_path]
    PATH_CACHE[(i, j)] = coordinate_path
    ```

## 3. Markov Decision Process Formulation

* **State Space:** The state $S = \{l_1, l_2, \dots, l_N\}$ is defined as the 2D spatial coordinates of the $N$ user-selected landmarks.
* **Action Space:** At timestep $t \in \{1, \dots, N\}$, the action $A_t \in \{1, \dots, N\}$ corresponds to the integer index of the next landmark to visit.
* **Reward:** The total episode reward $r$ is the negative sum of the distances of the traversed sequence, queried directly from the pre-computed matrix $D$.
  $$r = - \sum_{t=1}^{N-1} D_{A_t, A_{t+1}}$$

## 4. Policy Architecture: Pointer Model (Transformer or GRU)

The policy $\pi_\theta(A_t | S_t)$ is a pointer-style network. The **encoder** may be either:

* `RoutingAttentionModel(nn.Module)`: `nn.TransformerEncoder` over landmark embeddings, or
* `RoutingGRUModel(nn.Module)`: `nn.GRU` (stacked layers, batch-first) over the same embedded sequence.

The Streamlit sidebar exposes a toggle to choose between these backbones; the decoder head (context query vs.\ keys) is shared in spirit across both implementations.

### 4.1 Encoder Structure
1.  **Linear Projection:** Map each 2D coordinate $l_i$ to an embedding dimension $d$ (default $d=128$).
2.  **Sequence encoder:** Either a `nn.TransformerEncoder` (multi-head self-attention) or a `nn.GRU` over the embedded sequence, yielding node embeddings $H \in \mathbb{R}^{N \times d}$.
3.  **Global Graph Embedding:** Derive $\bar{h} = \frac{1}{N} \sum_{i=1}^N h_i$.

### 4.2 Decoder Structure and Masking
At decoding step $t$:
1.  **Context Construction:** Concatenate $\bar{h}$ and the embedding of the previous node $h_{\pi_{t-1}}$.
2.  **Cross-Attention:** Project the context as a Query, and the node embeddings $H$ as Keys. Compute the scaled dot-product attention logits $u_{t,i}$ for all $i \in \{1, \dots, N\}$.
3.  **Masking Constraint:** Maintain a binary mask $V_t$ of previously visited indices. Apply the following strict constraint prior to sampling:
    $$u_{t,i}^* = \begin{cases} u_{t,i} & \text{if } i \notin V_t \\ -\infty & \text{if } i \in V_t \end{cases}$$
4.  **Action Selection:** Sample $A_t$ from the Categorical distribution defined by the Softmax over $u_{t,i}^*$.

## 5. Optimization (REINFORCE via PyTorch Lightning)

Implement the REINFORCE optimization loop within a `pl.LightningModule`.

* **Rollout (`training_step`):**
    1.  Execute a forward pass to generate a full permutation sequence of length $N$, collecting actions $A_t$ and log-probabilities $\log \pi_\theta(A_t|S)$.
    2.  Compute the scalar episodic return $r$ using the pre-computed distance matrix $D$.
* **Baseline Calculation:** Subtract a **greedy self-baseline** \(r_{\mathrm{greedy}}(\theta)\): the return from a deterministic greedy decode under the current parameters (no gradient through the baseline term), to reduce gradient variance.
* **Loss Function:**
    $$\mathcal{L}(\theta) = - (r - r_{\mathrm{greedy}}(\theta)) \sum_{t=1}^N \log \pi_\theta(A_t | S_t)$$
* **Optimizer:** `torch.optim.Adam`.

## 6. Streamlit Interface Specification

Implement `app.py` utilizing the following precise layout constraints.

### 6.1 Sidebar Configuration
* **Header:** "Macro-Action Routing Configuration"
* **Landmark Input:** `st.multiselect` populated with the keys of `LANDMARKS`. Enforce a minimum selection of 3 landmarks before enabling execution.
* **Architecture & Hyperparameters:**
    * Embedding Dimension: `st.selectbox` (options: `64`, `128`, `256`).
    * Policy backbone: `st.selectbox` (`Transformer`, `GRU`).
    * Encoder depth: `st.number_input` (default `3`, min `1`, max `6`) — Transformer encoder layers or stacked GRU layers.
    * Learning Rate: `st.number_input` (default `1e-3`, format `%.4f`).
* **Execution Trigger:** `st.button` labeled "Initialize & Train Policy".

### 6.2 Main UI View
* **Metrics Row:** Real-time training metrics utilizing `st.columns` and `st.metric`:
    * Current Epoch
    * Best Total Route Distance (meters)
    * Current Policy Loss
* **Progress Visualization:** Render a `st.line_chart` that dynamically appends the total route distance per epoch.
* **Map Visualization:** Render a Folium map via `st_folium`. Use `st.empty()` placeholders to routinely update and re-render the map at the conclusion of every $N$ epochs.
    * Plot user-selected landmarks as distinct `folium.Marker` instances.
    * **Rendering the Route:** Decode the deterministic greedy sequence (argmax of logits) from the current policy. Iterate through the generated index sequence $(A_1, A_2, \dots, A_N)$. Retrieve the corresponding continuous geographic trajectories from `PATH_CACHE[(A_t, A_{t+1})]`.
    * Render this precise network trajectory using `folium.PolyLine` with `color='red'`, `weight=4`. Ensure it perfectly traces the valid Manhattan street network.
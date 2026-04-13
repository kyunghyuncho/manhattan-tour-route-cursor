import math
import copy
import random
import threading
import time
from datetime import timedelta
from itertools import permutations
from dataclasses import dataclass
from urllib.parse import quote
from typing import Any, Dict, List, Literal, Optional, Tuple

import folium
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pytorch_lightning as pl
import requests
import streamlit as st
import torch
import torch.nn as nn
from shapely.geometry import Point
from streamlit_folium import st_folium
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset


CITY_CONFIGS = {
    "Manhattan": {
        "bbox": (-74.0200, 40.7000, -73.9100, 40.8800),
        "landmarks": {
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
            "Charging Bull": (40.7060, -74.0132),
        },
        "presets": {
            "Custom": [],
            "Easy (3 landmarks)": ["Battery Park", "Times Square", "The Cloisters"],
            "Medium (6 landmarks)": [
                "Charging Bull",
                "Washington Square Park",
                "Bryant Park",
                "Central Park (Center)",
                "Columbia University",
                "The Cloisters",
            ],
            "Hard (8 landmarks)": [
                "Battery Park",
                "New York University",
                "Flatiron Building",
                "Grand Central Terminal",
                "Rockefeller Center",
                "Central Park (Center)",
                "Columbia University",
                "The Cloisters",
            ],
        },
    },
    "London": {
        "bbox": (-0.2650, 51.4700, 0.0200, 51.5600),
        "landmarks": {
            "Buckingham Palace": (51.5014, -0.1419),
            "Big Ben": (51.5007, -0.1246),
            "Westminster Abbey": (51.4993, -0.1273),
            "London Eye": (51.5033, -0.1196),
            "Trafalgar Square": (51.5080, -0.1281),
            "The British Museum": (51.5194, -0.1270),
            "St Paul's Cathedral": (51.5138, -0.0984),
            "Tower of London": (51.5081, -0.0759),
            "Tower Bridge": (51.5055, -0.0754),
            "Borough Market": (51.5055, -0.0910),
            "Hyde Park Corner": (51.5022, -0.1527),
            "Camden Market": (51.5415, -0.1460),
            "King's Cross Station": (51.5308, -0.1238),
            "Covent Garden": (51.5118, -0.1230),
        },
        "presets": {
            "Custom": [],
            "Easy (3 landmarks)": ["Big Ben", "Trafalgar Square", "The British Museum"],
            "Medium (6 landmarks)": [
                "Buckingham Palace",
                "Big Ben",
                "Covent Garden",
                "St Paul's Cathedral",
                "Tower Bridge",
                "Camden Market",
            ],
            "Hard (8 landmarks)": [
                "Hyde Park Corner",
                "Buckingham Palace",
                "London Eye",
                "The British Museum",
                "King's Cross Station",
                "St Paul's Cathedral",
                "Tower of London",
                "Borough Market",
            ],
        },
    },
    "Paris": {
        "bbox": (2.2240, 48.8150, 2.4200, 48.9020),
        "landmarks": {
            "Eiffel Tower": (48.8584, 2.2945),
            "Louvre Museum": (48.8606, 2.3376),
            "Notre-Dame Cathedral": (48.8530, 2.3499),
            "Arc de Triomphe": (48.8738, 2.2950),
            "Champs-Elysees": (48.8698, 2.3078),
            "Sacre-Coeur": (48.8867, 2.3431),
            "Montparnasse Tower": (48.8422, 2.3211),
            "Luxembourg Gardens": (48.8462, 2.3372),
            "Place de la Concorde": (48.8656, 2.3212),
            "Musee d'Orsay": (48.8600, 2.3266),
            "Centre Pompidou": (48.8606, 2.3522),
            "Pantheon": (48.8462, 2.3459),
            "Gare du Nord": (48.8809, 2.3553),
            "Bastille": (48.8530, 2.3690),
        },
        "presets": {
            "Custom": [],
            "Easy (3 landmarks)": ["Eiffel Tower", "Louvre Museum", "Notre-Dame Cathedral"],
            "Medium (6 landmarks)": [
                "Arc de Triomphe",
                "Champs-Elysees",
                "Place de la Concorde",
                "Louvre Museum",
                "Notre-Dame Cathedral",
                "Bastille",
            ],
            "Hard (8 landmarks)": [
                "Montparnasse Tower",
                "Luxembourg Gardens",
                "Pantheon",
                "Notre-Dame Cathedral",
                "Centre Pompidou",
                "Gare du Nord",
                "Sacre-Coeur",
                "Arc de Triomphe",
            ],
        },
    },
}


@dataclass
class SubsetGraphCache:
    city_name: str
    selected_names: List[str]
    selected_coords: np.ndarray
    distance_matrix: np.ndarray
    path_cache: Dict[Tuple[int, int], List[Tuple[float, float]]]


WIKIPEDIA_TITLE_OVERRIDES = {
    "Manhattan": {
        "Central Park (Center)": "Central Park",
        "The Metropolitan Museum of Art": "Metropolitan Museum of Art",
        "Museum of Modern Art (MoMA)": "Museum of Modern Art",
        "The High Line": "High Line",
        "New York Public Library": "New York Public Library Main Branch",
    },
    "London": {
        "Big Ben": "Big Ben",
        "Borough Market": "Borough Market",
        "King's Cross Station": "King's Cross railway station",
    },
    "Paris": {
        "Sacre-Coeur": "Sacré-Cœur, Paris",
        "Bastille": "Place de la Bastille",
    },
}


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_wikipedia_summary(city_name: str, landmark_name: str):
    query_title = WIKIPEDIA_TITLE_OVERRIDES.get(city_name, {}).get(landmark_name, landmark_name)
    query_title = query_title.replace("(", "").replace(")", "")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(query_title, safe='')}"

    try:
        response = requests.get(url, timeout=6)
        if response.status_code != 200:
            return {
                "title": landmark_name,
                "description": f"No Wikipedia summary found for {landmark_name}.",
                "link": f"https://en.wikipedia.org/wiki/{quote(query_title.replace(' ', '_'), safe='')}",
                "image_url": None,
            }
        data = response.json()
        return {
            "title": data.get("title", landmark_name),
            "description": data.get("extract", f"No description available for {landmark_name}."),
            "link": data.get(
                "content_urls", {}
            ).get("desktop", {}).get(
                "page", f"https://en.wikipedia.org/wiki/{quote(query_title.replace(' ', '_'), safe='')}"
            ),
            "image_url": data.get("thumbnail", {}).get("source"),
        }
    except requests.RequestException:
        return {
            "title": landmark_name,
            "description": "Wikipedia information is currently unavailable.",
            "link": f"https://en.wikipedia.org/wiki/{quote(query_title.replace(' ', '_'), safe='')}",
            "image_url": None,
        }


def build_landmark_popup_html(city_name: str, landmark_name: str, visit_number: int) -> str:
    info = fetch_wikipedia_summary(city_name, landmark_name)
    img_html = ""
    if info["image_url"]:
        img_html = (
            f"<img src='{info['image_url']}' style='width:100%;max-width:260px;border-radius:8px;"
            "margin-bottom:8px;'/>"
        )
    return (
        "<div style='width:280px;font-family:Arial,sans-serif;'>"
        f"<h4 style='margin:0 0 8px 0;'>Visit {visit_number}: {landmark_name}</h4>"
        f"{img_html}"
        f"<p style='font-size:13px;line-height:1.35;margin:0 0 8px 0;'>{info['description']}</p>"
        f"<a href='{info['link']}' target='_blank'>Read on Wikipedia</a>"
        "</div>"
    )


def route_distance_for_sequence(distance_matrix: np.ndarray, sequence: List[int]) -> float:
    if len(sequence) <= 1:
        return 0.0
    total = 0.0
    for a, b in zip(sequence[:-1], sequence[1:]):
        total += float(distance_matrix[a, b])
    return total


def nearest_neighbor_sequence(distance_matrix: np.ndarray, start_idx: int = 0) -> List[int]:
    n = int(distance_matrix.shape[0])
    unvisited = set(range(n))
    sequence = [start_idx]
    unvisited.remove(start_idx)
    current = start_idx
    while unvisited:
        nxt = min(unvisited, key=lambda j: float(distance_matrix[current, j]))
        sequence.append(int(nxt))
        unvisited.remove(nxt)
        current = int(nxt)
    return sequence


def brute_force_optimal_sequence(distance_matrix: np.ndarray, max_nodes: int = 9):
    n = int(distance_matrix.shape[0])
    if n > max_nodes:
        return None, None
    best_seq = None
    best_dist = float("inf")
    for perm in permutations(range(n)):
        seq = list(perm)
        dist = route_distance_for_sequence(distance_matrix, seq)
        if dist < best_dist:
            best_dist = dist
            best_seq = seq
    return best_seq, best_dist


@st.cache_resource(show_spinner=True)
def load_city_walk_graph(bbox: Tuple[float, float, float, float]):
    # OSMnx 2.x expects bbox=(left, bottom, right, top).
    return ox.graph_from_bbox(
        bbox=bbox,
        network_type="walk",
    )


def build_subset_graph_cache(
    city_name: str,
    selected_names: List[str],
    city_landmarks: Dict[str, Tuple[float, float]],
    city_bbox: Tuple[float, float, float, float],
) -> SubsetGraphCache:
    graph = load_city_walk_graph(city_bbox)
    projected_graph = ox.project_graph(graph)
    coords = np.array([city_landmarks[name] for name in selected_names], dtype=np.float32)
    n = len(selected_names)

    node_ids = []
    for lat, lon in coords:
        projected_point, _ = ox.projection.project_geometry(
            Point(float(lon), float(lat)),
            crs=graph.graph["crs"],
            to_crs=projected_graph.graph["crs"],
        )
        node_ids.append(
            ox.nearest_nodes(projected_graph, X=float(projected_point.x), Y=float(projected_point.y))
        )

    distance_matrix = np.zeros((n, n), dtype=np.float32)
    path_cache: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}

    for i in range(n):
        for j in range(n):
            if i == j:
                path_cache[(i, j)] = [(float(coords[i, 0]), float(coords[i, 1]))]
                continue
            node_path = nx.shortest_path(
                graph, source=node_ids[i], target=node_ids[j], weight="length"
            )
            path_length = nx.shortest_path_length(
                graph, source=node_ids[i], target=node_ids[j], weight="length"
            )
            coordinate_path = [
                (float(graph.nodes[node]["y"]), float(graph.nodes[node]["x"]))
                for node in node_path
            ]
            distance_matrix[i, j] = float(path_length)
            path_cache[(i, j)] = coordinate_path

    return SubsetGraphCache(
        city_name=city_name,
        selected_names=selected_names,
        selected_coords=coords,
        distance_matrix=distance_matrix,
        path_cache=path_cache,
    )


class RoutingAttentionModel(nn.Module):
    def __init__(self, embed_dim: int = 128, num_layers: int = 3, nhead: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_projection = nn.Linear(2, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.query_projection = nn.Linear(embed_dim * 2, embed_dim, bias=False)
        self.key_projection = nn.Linear(embed_dim, embed_dim, bias=False)

    def _decode_logits(
        self, node_embeddings: torch.Tensor, visited_mask: torch.Tensor, prev_index: torch.Tensor
    ) -> torch.Tensor:
        batch_size, _, _ = node_embeddings.shape
        graph_embedding = node_embeddings.mean(dim=1)
        prev_embedding = node_embeddings[
            torch.arange(batch_size, device=node_embeddings.device), prev_index
        ]
        context = torch.cat([graph_embedding, prev_embedding], dim=-1)
        query = self.query_projection(context)
        keys = self.key_projection(node_embeddings)
        logits = torch.einsum("bd,bnd->bn", query, keys) / math.sqrt(self.embed_dim)
        # Use an immutable snapshot to avoid autograd versioning issues when
        # the caller updates visited_mask on later decode steps.
        logits = logits.masked_fill(visited_mask.clone(), float("-inf"))
        return logits

    def forward(
        self, coords: torch.Tensor, greedy: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_embeddings = self.encoder(self.input_projection(coords))
        batch_size, num_nodes, _ = node_embeddings.shape
        visited = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=coords.device)
        prev_index = torch.zeros(batch_size, dtype=torch.long, device=coords.device)
        actions, log_probs, step_entropies = [], [], []

        for _ in range(num_nodes):
            logits = self._decode_logits(node_embeddings, visited, prev_index)
            dist = Categorical(logits=logits)
            if greedy:
                action = torch.argmax(logits, dim=-1)
                step_entropies.append(
                    torch.zeros(batch_size, device=coords.device, dtype=torch.float32)
                )
            else:
                action = dist.sample()
                step_entropies.append(dist.entropy())
            action_log_prob = dist.log_prob(action)
            visited_next = visited.clone()
            visited_next[torch.arange(batch_size, device=coords.device), action] = True
            visited = visited_next
            prev_index = action
            actions.append(action)
            log_probs.append(action_log_prob)

        trajectory_entropy = torch.stack(step_entropies, dim=1).sum(dim=1)
        return torch.stack(actions, dim=1), torch.stack(log_probs, dim=1), trajectory_entropy


class RoutingGRUModel(nn.Module):
    """Sequence encoder (GRU over landmark coordinates) + pointer-style decoder head."""

    def __init__(self, embed_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_projection = nn.Linear(2, embed_dim)
        nl = max(1, int(num_layers))
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=nl,
            batch_first=True,
            dropout=0.0,
        )
        self.query_projection = nn.Linear(embed_dim * 2, embed_dim, bias=False)
        self.key_projection = nn.Linear(embed_dim, embed_dim, bias=False)

    def _decode_logits(
        self, node_embeddings: torch.Tensor, visited_mask: torch.Tensor, prev_index: torch.Tensor
    ) -> torch.Tensor:
        batch_size, _, _ = node_embeddings.shape
        graph_embedding = node_embeddings.mean(dim=1)
        prev_embedding = node_embeddings[
            torch.arange(batch_size, device=node_embeddings.device), prev_index
        ]
        context = torch.cat([graph_embedding, prev_embedding], dim=-1)
        query = self.query_projection(context)
        keys = self.key_projection(node_embeddings)
        logits = torch.einsum("bd,bnd->bn", query, keys) / math.sqrt(self.embed_dim)
        logits = logits.masked_fill(visited_mask.clone(), float("-inf"))
        return logits

    def forward(
        self, coords: torch.Tensor, greedy: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_projection(coords)
        node_embeddings, _ = self.gru(x)
        batch_size, num_nodes, _ = node_embeddings.shape
        visited = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=coords.device)
        prev_index = torch.zeros(batch_size, dtype=torch.long, device=coords.device)
        actions, log_probs, step_entropies = [], [], []

        for _ in range(num_nodes):
            logits = self._decode_logits(node_embeddings, visited, prev_index)
            dist = Categorical(logits=logits)
            if greedy:
                action = torch.argmax(logits, dim=-1)
                step_entropies.append(
                    torch.zeros(batch_size, device=coords.device, dtype=torch.float32)
                )
            else:
                action = dist.sample()
                step_entropies.append(dist.entropy())
            action_log_prob = dist.log_prob(action)
            visited_next = visited.clone()
            visited_next[torch.arange(batch_size, device=coords.device), action] = True
            visited = visited_next
            prev_index = action
            actions.append(action)
            log_probs.append(action_log_prob)

        trajectory_entropy = torch.stack(step_entropies, dim=1).sum(dim=1)
        return torch.stack(actions, dim=1), torch.stack(log_probs, dim=1), trajectory_entropy


class _FixedLengthDataset(Dataset):
    def __init__(self, length: int):
        self.length = max(1, int(length))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.tensor(0, dtype=torch.long)


class RoutingLightningModule(pl.LightningModule):
    def __init__(
        self,
        coords: np.ndarray,
        distance_matrix: np.ndarray,
        embed_dim: int,
        num_layers: int,
        learning_rate: float,
        trajectories_per_update: int = 16,
        entropy_coef: float = 0.01,
        policy_backbone: Literal["transformer", "gru"] = "transformer",
    ):
        super().__init__()
        if policy_backbone == "gru":
            self.model = RoutingGRUModel(embed_dim=embed_dim, num_layers=num_layers)
        else:
            self.model = RoutingAttentionModel(embed_dim=embed_dim, num_layers=num_layers)
        self.learning_rate = learning_rate
        self.trajectories_per_update = max(1, int(trajectories_per_update))
        self.entropy_coef = float(entropy_coef)
        self.save_hyperparameters(ignore=["coords", "distance_matrix"])

        self.register_buffer("coords_single", torch.tensor(coords, dtype=torch.float32).unsqueeze(0))
        self.register_buffer("dist_mat", torch.tensor(distance_matrix, dtype=torch.float32))

        self.distance_history: List[float] = []
        self.reward_history: List[float] = []
        self.greedy_eval_distance_history: List[float] = []
        self.greedy_eval_reward_history: List[float] = []
        self.entropy_history: List[float] = []
        self.loss_history: List[float] = []
        self.best_distance = float("inf")
        self.best_reward = float("-inf")
        self.best_greedy_eval_distance = float("inf")
        self.best_greedy_eval_reward = float("-inf")
        self.latest_loss = 0.0
        self.latest_distance = float("inf")
        self.latest_reward = float("-inf")
        self.latest_greedy_eval_distance = float("inf")
        self.latest_greedy_eval_reward = float("-inf")
        self.latest_entropy = 0.0
        self.best_policy_state_dict = None
        self._epoch_distance_samples: List[float] = []
        self._epoch_reward_samples: List[float] = []
        self._epoch_loss_samples: List[float] = []
        self._epoch_entropy_samples: List[float] = []

    def on_train_epoch_start(self):
        self._epoch_distance_samples.clear()
        self._epoch_reward_samples.clear()
        self._epoch_loss_samples.clear()
        self._epoch_entropy_samples.clear()

    def _route_distance(self, actions: torch.Tensor) -> torch.Tensor:
        from_idx = actions[:, :-1]
        to_idx = actions[:, 1:]
        edge_lengths = self.dist_mat[from_idx, to_idx]
        return edge_lengths.sum(dim=1)

    def training_step(self, batch, batch_idx):
        del batch, batch_idx
        coords_batch = self.coords_single.repeat(self.trajectories_per_update, 1, 1)
        actions, log_probs, trajectory_entropy = self.model(coords_batch, greedy=False)
        distance = self._route_distance(actions)
        reward = -distance
        rollout_entropy = -log_probs.mean()

        # Greedy self-baseline: deterministic rollout under the same θ (no grad through baseline).
        with torch.no_grad():
            greedy_actions, _, _ = self.model(self.coords_single, greedy=True)
            greedy_reward = -self._route_distance(greedy_actions).squeeze(0)
        advantage = reward - greedy_reward
        # Normalize advantage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-9)

        policy_loss = -(advantage.detach() * log_probs.sum(dim=1)).mean()
        loss = policy_loss - self.entropy_coef * trajectory_entropy.mean()
        self.latest_loss = float(loss.detach().cpu().item())
        self.latest_distance = float(distance.detach().mean().cpu().item())
        self.latest_reward = float(reward.detach().mean().cpu().item())
        self.latest_entropy = float(rollout_entropy.detach().cpu().item())
        self._epoch_distance_samples.append(self.latest_distance)
        self._epoch_reward_samples.append(self.latest_reward)
        self._epoch_entropy_samples.append(self.latest_entropy)
        self._epoch_loss_samples.append(self.latest_loss)

        self.log("policy_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log(
            "route_distance_m",
            torch.tensor(self.latest_distance, device=self.device),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def on_train_epoch_end(self):
        if not self._epoch_distance_samples:
            return
        self.latest_distance = float(np.mean(self._epoch_distance_samples))
        self.latest_reward = float(np.mean(self._epoch_reward_samples))
        self.latest_entropy = float(np.mean(self._epoch_entropy_samples))
        self.latest_loss = float(np.mean(self._epoch_loss_samples))
        self.best_distance = min(self.best_distance, self.latest_distance)
        self.best_reward = max(self.best_reward, self.latest_reward)
        self.distance_history.append(self.latest_distance)
        self.reward_history.append(self.latest_reward)
        self.entropy_history.append(self.latest_entropy)
        self.loss_history.append(self.latest_loss)

    def greedy_sequence(self) -> List[int]:
        self.model.eval()
        with torch.no_grad():
            actions, _, _ = self.model(self.coords_single, greedy=True)
        return actions[0].detach().cpu().tolist()

    def greedy_eval_metrics(self) -> Tuple[float, float]:
        self.model.eval()
        with torch.no_grad():
            actions, _, _ = self.model(self.coords_single, greedy=True)
            distance = self._route_distance(actions).mean()
            reward = -distance
        return float(distance.detach().cpu().item()), float(reward.detach().cpu().item())

    def maybe_save_best_policy(self, greedy_distance: float, previous_best_distance: float):
        if greedy_distance + 1e-9 < previous_best_distance:
            self.best_policy_state_dict = copy.deepcopy(self.model.state_dict())

    def load_best_policy(self):
        if self.best_policy_state_dict is not None:
            self.model.load_state_dict(self.best_policy_state_dict)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def render_route_map(
    city_name: str,
    selected_names: List[str],
    selected_coords: np.ndarray,
    path_cache: Dict[Tuple[int, int], List[Tuple[float, float]]],
    route_indices: List[int],
) -> folium.Map:
    center_lat = float(np.mean(selected_coords[:, 0]))
    center_lon = float(np.mean(selected_coords[:, 1]))
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=13, control_scale=True)

    visit_order = {node_idx: order + 1 for order, node_idx in enumerate(route_indices)}

    for idx, name in enumerate(selected_names):
        lat, lon = selected_coords[idx]
        visit_number = visit_order.get(idx, "?")
        popup_html = build_landmark_popup_html(city_name, name, int(visit_number))
        folium.Marker(
            location=[float(lat), float(lon)],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Visit {visit_number}: {name}",
            icon=folium.DivIcon(
                html=(
                    "<div style='"
                    "background:#2563eb;color:white;border-radius:50%;"
                    "width:24px;height:24px;line-height:24px;text-align:center;"
                    "font-weight:700;font-size:12px;border:1px solid white;"
                    "'>"
                    f"{visit_number}"
                    "</div>"
                )
            ),
        ).add_to(fmap)

    if len(route_indices) > 1:
        for a, b in zip(route_indices[:-1], route_indices[1:]):
            segment = path_cache.get((a, b), [])
            if segment:
                folium.PolyLine(segment, color="red", weight=4, opacity=0.9).add_to(fmap)
    return fmap


def build_training_chart_df(
    module: "RoutingLightningModule",
    teaching_mode: bool,
    smoothing_window: int,
) -> Optional[pd.DataFrame]:
    aligned_len = min(
        len(module.reward_history),
        len(module.greedy_eval_reward_history),
        len(module.entropy_history),
    )
    if aligned_len == 0:
        return None
    sw = max(1, int(smoothing_window))
    if teaching_mode:
        history_df = pd.DataFrame(
            {
                "greedy_eval_reward": module.greedy_eval_reward_history[:aligned_len],
                "sampled_episode_reward": module.reward_history[:aligned_len],
                "policy_entropy_proxy": module.entropy_history[:aligned_len],
            }
        )
    else:
        history_df = pd.DataFrame(
            {
                "sampled_episode_reward": module.reward_history[:aligned_len],
                "greedy_eval_reward": module.greedy_eval_reward_history[:aligned_len],
            }
        )
    if sw > 1:
        smooth_df = history_df.rolling(window=sw, min_periods=1).mean()
        smooth_df.columns = [f"{col}_smooth" for col in smooth_df.columns]
        history_df = pd.concat([history_df, smooth_df], axis=1)
    return history_df


class TrainingBridge:
    """Thread-safe snapshot for `st.fragment` live updates (no full-page reruns)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.done = False
        self.error: Optional[str] = None
        self.epoch = 0
        self.latest_loss = 0.0
        self.best_sampled_distance_m = float("inf")
        self.best_greedy_distance_m = float("inf")
        self.chart_df: Optional[pd.DataFrame] = None
        self.final_map: Optional[Any] = None
        self.final_baseline_rows: Optional[List[dict]] = None
        self.stopped_by_user = False

    def apply_epoch_snapshot(
        self,
        epoch_num: int,
        module: "RoutingLightningModule",
        teaching_mode: bool,
        smoothing_window: int,
    ) -> None:
        df = build_training_chart_df(module, teaching_mode, smoothing_window)
        with self.lock:
            self.epoch = epoch_num
            self.latest_loss = float(module.latest_loss)
            self.best_sampled_distance_m = float(module.best_distance)
            self.best_greedy_distance_m = float(module.best_greedy_eval_distance)
            self.chart_df = df.copy() if df is not None else None


class StopTrainingEventCallback(pl.Callback):
    def __init__(self, stop_event: threading.Event):
        super().__init__()
        self.stop_event = stop_event

    def on_train_epoch_end(self, trainer, pl_module):
        del pl_module
        if self.stop_event.is_set():
            trainer.should_stop = True


def _run_training_worker(
    bridge: TrainingBridge,
    module: RoutingLightningModule,
    subset_cache: SubsetGraphCache,
    epoch_tracker: Dict[str, int],
    train_loader: DataLoader,
    max_epochs: int,
    teaching_mode: bool,
    smoothing_window: int,
    baseline_rows: List[dict],
    optimal_dist: Optional[float],
    city_name: str,
) -> None:
    try:
        if max_epochs > 0:
            progress_cb = StreamlitProgressCallback(
                module=module,
                subset_cache=subset_cache,
                epoch_metric_placeholder=None,
                best_distance_placeholder=None,
                current_loss_placeholder=None,
                chart_slot=None,
                map_slot=None,
                map_update_interval=1,
                teaching_mode=teaching_mode,
                smoothing_window=smoothing_window,
                epoch_tracker=epoch_tracker,
                bridge=bridge,
            )
            stop_cb = StopTrainingEventCallback(bridge.stop_event)
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                accelerator="cpu",
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                callbacks=[progress_cb, stop_cb],
            )
            trainer.fit(module, train_loader)
        module.load_best_policy()
        final_route_indices = module.greedy_sequence()
        final_distance = route_distance_for_sequence(
            subset_cache.distance_matrix, final_route_indices
        )
        final_reward = -final_distance
        final_map = render_route_map(
            city_name=city_name,
            selected_names=subset_cache.selected_names,
            selected_coords=subset_cache.selected_coords,
            path_cache=subset_cache.path_cache,
            route_indices=final_route_indices,
        )
        final_row: dict = {
            "policy": "trained_greedy",
            "distance_m": round(final_distance, 2),
            "reward": round(final_reward, 2),
        }
        if optimal_dist is not None:
            final_row["optimality_gap_%"] = round(
                100.0 * (final_distance - optimal_dist) / max(1e-9, optimal_dist), 3
            )
        full_rows = list(baseline_rows) + [final_row]
        with bridge.lock:
            bridge.final_map = final_map
            bridge.final_baseline_rows = full_rows
            bridge.stopped_by_user = bridge.stop_event.is_set()
    except Exception as exc:
        with bridge.lock:
            bridge.error = str(exc)
    finally:
        with bridge.lock:
            bridge.done = True


class StreamlitProgressCallback(pl.Callback):
    def __init__(
        self,
        module: RoutingLightningModule,
        subset_cache: SubsetGraphCache,
        epoch_metric_placeholder: Any,
        best_distance_placeholder: Any,
        current_loss_placeholder: Any,
        chart_slot: Any,
        map_slot: Any,
        map_update_interval: int,
        teaching_mode: bool,
        smoothing_window: int,
        epoch_tracker: Dict[str, int],
        bridge: Optional[TrainingBridge] = None,
    ):
        super().__init__()
        self.module_ref = module
        self.subset_cache = subset_cache
        self.epoch_metric_placeholder = epoch_metric_placeholder
        self.best_distance_placeholder = best_distance_placeholder
        self.current_loss_placeholder = current_loss_placeholder
        self.chart_slot = chart_slot
        self.map_slot = map_slot
        self.map_update_interval = max(1, map_update_interval)
        self.teaching_mode = teaching_mode
        self.smoothing_window = max(1, int(smoothing_window))
        self.epoch_tracker = epoch_tracker
        self.bridge = bridge

    def on_train_epoch_end(self, trainer, pl_module):
        del pl_module
        self.epoch_tracker["value"] += 1
        epoch_num = int(self.epoch_tracker["value"])
        greedy_distance, greedy_reward = self.module_ref.greedy_eval_metrics()
        self.module_ref.latest_greedy_eval_distance = greedy_distance
        self.module_ref.latest_greedy_eval_reward = greedy_reward
        prev_best_greedy = self.module_ref.best_greedy_eval_distance
        self.module_ref.best_greedy_eval_distance = min(
            self.module_ref.best_greedy_eval_distance, greedy_distance
        )
        self.module_ref.best_greedy_eval_reward = max(
            self.module_ref.best_greedy_eval_reward, greedy_reward
        )
        self.module_ref.maybe_save_best_policy(greedy_distance, prev_best_greedy)
        self.module_ref.greedy_eval_distance_history.append(greedy_distance)
        self.module_ref.greedy_eval_reward_history.append(greedy_reward)

        if self.bridge is not None:
            self.bridge.apply_epoch_snapshot(
                epoch_num, self.module_ref, self.teaching_mode, self.smoothing_window
            )
            return

        if self.epoch_metric_placeholder is not None:
            self.epoch_metric_placeholder.metric("Current Epoch", epoch_num)
        if self.best_distance_placeholder is not None:
            self.best_distance_placeholder.metric(
                "Best Total Route Distance (meters)", f"{self.module_ref.best_distance:,.1f}"
            )
        if self.current_loss_placeholder is not None:
            self.current_loss_placeholder.metric(
                "Current Policy Loss", f"{self.module_ref.latest_loss:.4f}"
            )

        history_df = build_training_chart_df(
            self.module_ref, self.teaching_mode, self.smoothing_window
        )
        if history_df is None or self.chart_slot is None:
            return
        self.chart_slot.line_chart(history_df, width="stretch")

        if self.map_slot is not None and (
            epoch_num % self.map_update_interval == 0 or epoch_num == 1
        ):
            greedy_indices = self.module_ref.greedy_sequence()
            fmap = render_route_map(
                city_name=self.subset_cache.city_name,
                selected_names=self.subset_cache.selected_names,
                selected_coords=self.subset_cache.selected_coords,
                path_cache=self.subset_cache.path_cache,
                route_indices=greedy_indices,
            )
            with self.map_slot.container():
                st_folium(
                    fmap,
                    width=1200,
                    height=600,
                    returned_objects=[],
                    key=f"route_map_epoch_{epoch_num}",
                )


@st.fragment(run_every=timedelta(seconds=2))
def training_progress_fragment() -> None:
    bridge = st.session_state.get("training_bridge")
    ctx = st.session_state.get("training_context")
    if bridge is None:
        st.caption("Live training curves appear here after you start training.")
        return

    with bridge.lock:
        err = bridge.error
        done = bridge.done
        epoch = bridge.epoch
        latest_loss = bridge.latest_loss
        best_s = bridge.best_sampled_distance_m
        best_g = bridge.best_greedy_distance_m
        chart_df = bridge.chart_df.copy() if bridge.chart_df is not None else None
        final_map = bridge.final_map
        final_rows = bridge.final_baseline_rows
        stopped = bridge.stopped_by_user

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Epoch", epoch if epoch > 0 else "—")
    c2.metric(
        "Best sampled distance (m)",
        f"{best_s:,.1f}" if best_s < float("inf") else "—",
    )
    c3.metric("Best greedy distance (m)", f"{best_g:,.1f}" if best_g < float("inf") else "—")
    c4.metric("Current policy loss", f"{latest_loss:.4f}")

    if err and not done:
        st.error(err)
        return

    if chart_df is not None and not chart_df.empty:
        st.line_chart(chart_df, width="stretch")
    elif not done:
        st.caption("Collecting first epoch…")

    if ctx and not done:
        init_rows = ctx.get("initial_baseline_rows")
        if init_rows:
            with st.expander("Baseline policies (fixed)", expanded=False):
                st.dataframe(pd.DataFrame(init_rows), width="stretch")

    # Hand off final map/table to main() once. Re-rendering st_folium inside this fragment
    # every `run_every` interval remounts the iframe and causes flicker after training ends.
    if done and not st.session_state.get("_training_final_handoff_done"):
        st.session_state["_training_final_handoff_done"] = True
        if err:
            st.session_state["post_train_payload"] = {"error": err}
        elif final_map is not None and final_rows is not None:
            st.session_state["post_train_payload"] = {
                "map": final_map,
                "rows": final_rows,
                "stopped": stopped,
            }
        else:
            st.session_state["post_train_payload"] = {
                "error": "Training finished without a complete final map payload.",
            }
        st.rerun()

    if done and st.session_state.get("_training_final_handoff_done"):
        payload = st.session_state.get("post_train_payload") or {}
        if not payload.get("error"):
            st.caption(
                "Final route and comparison table are shown in the static section below (no auto-refresh)."
            )


def render_static_post_train_results() -> None:
    payload = st.session_state.get("post_train_payload")
    if not payload:
        return
    if payload.get("error"):
        if not st.session_state.get("_post_train_error_toast_shown"):
            st.session_state["_post_train_error_toast_shown"] = True
            st.error(payload["error"])
        return
    st.subheader("Final route (best greedy policy)")
    st_folium(
        payload["map"],
        width=1200,
        height=600,
        returned_objects=[],
        key="final_map_static_main",
    )
    st.dataframe(pd.DataFrame(payload["rows"]), width="stretch")
    if not st.session_state.get("_post_train_success_toast_shown"):
        st.session_state["_post_train_success_toast_shown"] = True
        if payload.get("stopped"):
            st.success("Training stopped. Map and table show the best greedy policy so far.")
        else:
            st.success("Training complete. Map and table show the best greedy policy.")


def main():
    st.set_page_config(page_title="Macro-Action City Routing", layout="wide")
    st.title("Macro-Action Path Planning via Reinforcement Learning")

    with st.sidebar:
        st.header("Macro-Action Routing Configuration")
        city_name = st.selectbox("City", options=list(CITY_CONFIGS.keys()), index=0)
        city_config = CITY_CONFIGS[city_name]
        city_landmarks = city_config["landmarks"]
        city_presets = city_config["presets"]
        city_bbox = city_config["bbox"]
        teaching_mode = st.checkbox("Teaching Mode", value=True)
        preset_name = st.selectbox("Scenario Preset", options=list(city_presets.keys()), index=2)
        default_landmarks = (
            city_presets[preset_name]
            if city_presets[preset_name]
            else list(city_landmarks.keys())[:3]
        )

        if st.session_state.get("last_city_name") != city_name:
            st.session_state["selected_names"] = default_landmarks.copy()
            st.session_state["pending_selected_names"] = default_landmarks.copy()
            st.session_state["last_preset_name"] = preset_name
        st.session_state["last_city_name"] = city_name

        if "selected_names" not in st.session_state:
            st.session_state["selected_names"] = default_landmarks.copy()
        if "pending_selected_names" in st.session_state:
            st.session_state["selected_names"] = st.session_state.pop("pending_selected_names")
        if preset_name != "Custom" and st.session_state.get("last_preset_name") != preset_name:
            st.session_state["selected_names"] = default_landmarks.copy()
        st.session_state["last_preset_name"] = preset_name

        selected_names = st.multiselect(
            f"Select {city_name} landmarks (minimum 3):",
            options=list(city_landmarks.keys()),
            key="selected_names",
        )
        if st.button("Shuffle Selected Landmarks", disabled=len(selected_names) < 2):
            shuffled_names = selected_names.copy()
            random.shuffle(shuffled_names)
            st.session_state["pending_selected_names"] = shuffled_names
            st.rerun()
        embed_dim = st.selectbox("Embedding Dimension", options=[64, 128, 256], index=1)
        policy_backbone = st.selectbox(
            "Policy backbone",
            options=["GRU", "Transformer", ],
            index=0,
            help="Transformer: set encoder over landmarks. GRU: recurrent encoder over the "
            "coordinate sequence, same pointer decoder head.",
        )
        num_layers = st.number_input(
            "Encoder depth (Transformer layers or GRU layers)",
            min_value=1,
            max_value=6,
            value=3,
            step=1,
        )
        learning_rate = st.number_input(
            "Learning Rate", min_value=0.000001, max_value=0.100000, value=1e-4, format="%.6f"
        )
        trajectories_per_update = st.number_input(
            "Trajectories per Update", min_value=1, max_value=256, value=64, step=1
        )
        updates_per_epoch = st.number_input(
            "Updates per Epoch", min_value=1, max_value=100, value=1, step=1
        )
        smoothing_window = st.number_input(
            "Chart Smoothing Window", min_value=1, max_value=50, value=8, step=1
        )
        entropy_coef = st.number_input(
            "Entropy coefficient (regularization)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.005,
            format="%.4f",
            help="Adds −λ·H(π) to the loss to encourage exploration (higher = more exploration).",
        )
        max_epochs = st.number_input("Training Epochs", min_value=0, max_value=10000, value=500)

        can_run = len(selected_names) >= 3
        bridge = st.session_state.get("training_bridge")
        ctx = st.session_state.get("training_context")
        thread = ctx.get("thread") if ctx else None
        training_in_progress = bool(
            bridge is not None
            and ctx is not None
            and not bridge.done
            and thread is not None
            and thread.is_alive()
        )
        train_clicked = st.button(
            "Initialize & Train Policy",
            disabled=not can_run or training_in_progress,
            key="btn_train_policy",
        )
        if not can_run:
            st.info("Select at least 3 landmarks to initialize and train.")

    # Start training here (before the live fragment) so a Train click is not lost when
    # `training_progress_fragment` performs its one-time post-run `st.rerun()` hand-off
    # while `_training_final_handoff_done` is still unset.
    train_start = train_clicked and can_run
    if train_start:
        run_seed = int(time.time_ns() % (2**32 - 1))
        pl.seed_everything(run_seed, workers=True)
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)

        st.info(f"Run seed (time-based): {run_seed}")
        with st.spinner(f"Building {city_name} graph cache and initializing policy..."):
            subset_cache = build_subset_graph_cache(
                city_name=city_name,
                selected_names=selected_names,
                city_landmarks=city_landmarks,
                city_bbox=city_bbox,
            )
            n_landmarks = len(selected_names)
            rng = random.Random(run_seed)

            random_seq = list(range(n_landmarks))
            rng.shuffle(random_seq)
            random_dist = route_distance_for_sequence(subset_cache.distance_matrix, random_seq)
            random_reward = -random_dist

            nn_seq = nearest_neighbor_sequence(subset_cache.distance_matrix, start_idx=0)
            nn_dist = route_distance_for_sequence(subset_cache.distance_matrix, nn_seq)
            nn_reward = -nn_dist

            optimal_seq, optimal_dist = brute_force_optimal_sequence(subset_cache.distance_matrix)
            optimal_reward = -optimal_dist if optimal_dist is not None else None

            backbone_key: Literal["transformer", "gru"] = (
                "gru" if policy_backbone == "GRU" else "transformer"
            )
            module = RoutingLightningModule(
                coords=subset_cache.selected_coords,
                distance_matrix=subset_cache.distance_matrix,
                embed_dim=int(embed_dim),
                num_layers=int(num_layers),
                learning_rate=float(learning_rate),
                trajectories_per_update=int(trajectories_per_update),
                entropy_coef=float(entropy_coef),
                policy_backbone=backbone_key,
            )

            untrained_seq = module.greedy_sequence()
            untrained_dist = route_distance_for_sequence(subset_cache.distance_matrix, untrained_seq)
            untrained_reward = -untrained_dist

            baseline_rows = [
                {
                    "policy": "random",
                    "distance_m": round(random_dist, 2),
                    "reward": round(random_reward, 2),
                },
                {
                    "policy": "nearest_neighbor(start=0)",
                    "distance_m": round(nn_dist, 2),
                    "reward": round(nn_reward, 2),
                },
                {
                    "policy": "untrained_greedy",
                    "distance_m": round(untrained_dist, 2),
                    "reward": round(untrained_reward, 2),
                },
            ]
            if optimal_dist is not None:
                baseline_rows.append(
                    {
                        "policy": "exact_optimal",
                        "distance_m": round(optimal_dist, 2),
                        "reward": round(float(optimal_reward), 2),
                    }
                )
            else:
                baseline_rows.append(
                    {
                        "policy": "exact_optimal (N>9, not enumerated)",
                        # Keep numeric dtypes so Arrow-backed tables (e.g. st.dataframe) do not fail
                        # on mixed str/float columns.
                        "distance_m": math.nan,
                        "reward": math.nan,
                    }
                )

        max_epochs_int = int(max_epochs)
        st.session_state.pop("training_bridge", None)
        st.session_state.pop("training_context", None)
        st.session_state.pop("post_train_payload", None)
        st.session_state.pop("_training_final_handoff_done", None)
        st.session_state.pop("_post_train_success_toast_shown", None)
        st.session_state.pop("_post_train_error_toast_shown", None)

        bridge = TrainingBridge()
        epoch_tracker: Dict[str, int] = {"value": 0}
        train_loader = DataLoader(
            _FixedLengthDataset(length=int(updates_per_epoch)),
            batch_size=1,
            shuffle=False,
        )
        worker_args = (
            bridge,
            module,
            subset_cache,
            epoch_tracker,
            train_loader,
            max_epochs_int,
            teaching_mode,
            int(smoothing_window),
            copy.deepcopy(baseline_rows),
            optimal_dist,
            city_name,
        )
        train_thread = threading.Thread(target=_run_training_worker, args=worker_args, daemon=True)
        st.session_state["training_bridge"] = bridge
        st.session_state["training_context"] = {
            "thread": train_thread,
            "module": module,
            "subset_cache": subset_cache,
            "city_name": city_name,
            "initial_baseline_rows": copy.deepcopy(baseline_rows),
        }
        train_thread.start()

    if teaching_mode:
        st.caption(
            "Teaching Mode: Greedy reward, sampled reward, and entropy proxy per epoch"
        )
    else:
        st.caption(
            "Training Progress: Sampled Reward vs Greedy Eval Reward per Epoch (higher is better)"
        )
    st.caption(
        "Live curves refresh inside the panel below every ~2s without reloading the whole page."
    )
    # Stop must render here (not in the sidebar): the sidebar runs before `train_start`, so an
    # active thread does not exist yet when sidebar widgets are drawn on the same run as Train.
    bridge_live = st.session_state.get("training_bridge")
    ctx_live = st.session_state.get("training_context")
    th_live = ctx_live.get("thread") if ctx_live else None
    training_active = bool(
        bridge_live is not None
        and ctx_live is not None
        and not bridge_live.done
        and th_live is not None
        and th_live.is_alive()
    )
    if training_active:
        s1, s2 = st.columns([1, 4])
        with s1:
            if st.button("Stop training", key="btn_stop_training_main"):
                bridge_live.stop_event.set()
        with s2:
            st.caption("Policy training is running. Stop ends after the current epoch boundary.")

    training_progress_fragment()
    render_static_post_train_results()


if __name__ == "__main__":
    main()

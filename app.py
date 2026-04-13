import math
import random
import time
from itertools import permutations
from dataclasses import dataclass
from urllib.parse import quote
from typing import Dict, List, Tuple

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_embeddings = self.encoder(self.input_projection(coords))
        batch_size, num_nodes, _ = node_embeddings.shape
        visited = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=coords.device)
        prev_index = torch.zeros(batch_size, dtype=torch.long, device=coords.device)
        actions, log_probs = [], []

        for _ in range(num_nodes):
            logits = self._decode_logits(node_embeddings, visited, prev_index)
            dist = Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1) if greedy else dist.sample()
            action_log_prob = dist.log_prob(action)
            visited_next = visited.clone()
            visited_next[torch.arange(batch_size, device=coords.device), action] = True
            visited = visited_next
            prev_index = action
            actions.append(action)
            log_probs.append(action_log_prob)

        return torch.stack(actions, dim=1), torch.stack(log_probs, dim=1)


class _SingleInstanceDataset(Dataset):
    def __len__(self):
        return 1

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
        ema_beta: float = 0.9,
    ):
        super().__init__()
        self.model = RoutingAttentionModel(embed_dim=embed_dim, num_layers=num_layers)
        self.learning_rate = learning_rate
        self.ema_beta = ema_beta
        self.ema_baseline = None
        self.save_hyperparameters(ignore=["coords", "distance_matrix"])

        self.register_buffer("coords", torch.tensor(coords, dtype=torch.float32).unsqueeze(0))
        self.register_buffer("dist_mat", torch.tensor(distance_matrix, dtype=torch.float32))

        self.distance_history: List[float] = []
        self.reward_history: List[float] = []
        self.greedy_eval_distance_history: List[float] = []
        self.greedy_eval_reward_history: List[float] = []
        self.regret_history: List[float] = []
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

    def _route_distance(self, actions: torch.Tensor) -> torch.Tensor:
        from_idx = actions[:, :-1]
        to_idx = actions[:, 1:]
        edge_lengths = self.dist_mat[from_idx, to_idx]
        return edge_lengths.sum(dim=1)

    def training_step(self, batch, batch_idx):
        del batch, batch_idx
        actions, log_probs = self.model(self.coords, greedy=False)
        distance = self._route_distance(actions)
        reward = -distance
        reward_mean = reward.mean().detach()
        rollout_entropy = -log_probs.mean()

        if self.ema_baseline is None:
            self.ema_baseline = reward_mean
        else:
            self.ema_baseline = (
                self.ema_beta * self.ema_baseline + (1.0 - self.ema_beta) * reward_mean
            )
        advantage = reward - self.ema_baseline

        loss = -(advantage.detach() * log_probs.sum(dim=1)).mean()
        self.latest_loss = float(loss.detach().cpu().item())
        self.latest_distance = float(distance.detach().mean().cpu().item())
        self.latest_reward = float(reward.detach().mean().cpu().item())
        self.latest_entropy = float(rollout_entropy.detach().cpu().item())
        self.best_distance = min(self.best_distance, self.latest_distance)
        self.best_reward = max(self.best_reward, self.latest_reward)
        self.distance_history.append(self.latest_distance)
        self.reward_history.append(self.latest_reward)
        self.entropy_history.append(self.latest_entropy)
        self.loss_history.append(self.latest_loss)

        self.log("policy_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log(
            "route_distance_m",
            torch.tensor(self.latest_distance, device=self.device),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def greedy_sequence(self) -> List[int]:
        self.model.eval()
        with torch.no_grad():
            actions, _ = self.model(self.coords, greedy=True)
        return actions[0].detach().cpu().tolist()

    def greedy_eval_metrics(self) -> Tuple[float, float]:
        self.model.eval()
        with torch.no_grad():
            actions, _ = self.model(self.coords, greedy=True)
            distance = self._route_distance(actions).mean()
            reward = -distance
        return float(distance.detach().cpu().item()), float(reward.detach().cpu().item())

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


class StreamlitProgressCallback(pl.Callback):
    def __init__(
        self,
        module: RoutingLightningModule,
        subset_cache: SubsetGraphCache,
        epoch_metric_placeholder,
        best_distance_placeholder,
        current_loss_placeholder,
        chart_slot,
        map_slot,
        map_update_interval: int,
        best_known_distance: float,
        teaching_mode: bool,
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
        self.best_known_distance = best_known_distance
        self.teaching_mode = teaching_mode

    def on_train_epoch_end(self, trainer, pl_module):
        del pl_module
        epoch_num = trainer.current_epoch + 1
        self.epoch_metric_placeholder.metric("Current Epoch", epoch_num)
        self.best_distance_placeholder.metric(
            "Best Total Route Distance (meters)", f"{self.module_ref.best_distance:,.1f}"
        )
        self.current_loss_placeholder.metric(
            "Current Policy Loss", f"{self.module_ref.latest_loss:.4f}"
        )
        greedy_distance, greedy_reward = self.module_ref.greedy_eval_metrics()
        self.module_ref.latest_greedy_eval_distance = greedy_distance
        self.module_ref.latest_greedy_eval_reward = greedy_reward
        self.module_ref.best_greedy_eval_distance = min(
            self.module_ref.best_greedy_eval_distance, greedy_distance
        )
        self.module_ref.best_greedy_eval_reward = max(
            self.module_ref.best_greedy_eval_reward, greedy_reward
        )
        self.module_ref.greedy_eval_distance_history.append(greedy_distance)
        self.module_ref.greedy_eval_reward_history.append(greedy_reward)
        self.module_ref.regret_history.append(greedy_distance - self.best_known_distance)

        if self.teaching_mode:
            history_df = pd.DataFrame(
                {
                    "greedy_eval_reward": self.module_ref.greedy_eval_reward_history,
                    "sampled_episode_reward": self.module_ref.reward_history,
                    "greedy_regret_m": self.module_ref.regret_history,
                    "policy_entropy_proxy": self.module_ref.entropy_history,
                }
            )
        else:
            history_df = pd.DataFrame(
                {
                    "sampled_episode_reward": self.module_ref.reward_history,
                    "greedy_eval_reward": self.module_ref.greedy_eval_reward_history,
                }
            )
        self.chart_slot.line_chart(history_df, width="stretch")

        if epoch_num % self.map_update_interval == 0 or epoch_num == 1:
            greedy_indices = self.module_ref.greedy_sequence()
            fmap = render_route_map(
                city_name=self.subset_cache.city_name,
                selected_names=self.subset_cache.selected_names,
                selected_coords=self.subset_cache.selected_coords,
                path_cache=self.subset_cache.path_cache,
                route_indices=greedy_indices,
            )
            self.map_slot.empty()
            with self.map_slot.container():
                st_folium(
                    fmap,
                    width=1200,
                    height=600,
                    returned_objects=[],
                    key=f"route_map_epoch_{epoch_num}",
                )


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
        num_layers = st.number_input(
            "Transformer Layers", min_value=1, max_value=6, value=3, step=1
        )
        learning_rate = st.number_input(
            "Learning Rate", min_value=0.0001, max_value=0.0100, value=1e-3, format="%.4f"
        )
        max_epochs = st.number_input("Training Epochs", min_value=0, max_value=100, value=20)

        can_run = len(selected_names) >= 3
        train_clicked = st.button("Initialize & Train Policy", disabled=not can_run)
        if not can_run:
            st.info("Select at least 3 landmarks to initialize and train.")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    epoch_metric_placeholder = metric_col1.empty()
    best_distance_placeholder = metric_col2.empty()
    current_loss_placeholder = metric_col3.empty()
    if teaching_mode:
        st.caption(
            "Teaching Mode: Greedy reward/regret plus sampled reward and entropy proxy per epoch"
        )
    else:
        st.caption(
            "Training Progress: Sampled Reward vs Greedy Eval Reward per Epoch (higher is better)"
        )
    chart_placeholder = st.empty()
    map_placeholder = st.empty()
    baseline_table_placeholder = st.empty()

    if train_clicked and can_run:
        # Seed from current time so each run has clear stochastic variation.
        run_seed = int(time.time_ns() % (2**32 - 1))
        pl.seed_everything(run_seed, workers=True)
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)

        st.info(f"Run seed (time-based): {run_seed}")
        with st.spinner(f"Building {city_name} graph cache and training policy..."):
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

            module = RoutingLightningModule(
                coords=subset_cache.selected_coords,
                distance_matrix=subset_cache.distance_matrix,
                embed_dim=int(embed_dim),
                num_layers=int(num_layers),
                learning_rate=float(learning_rate),
            )

            untrained_seq = module.greedy_sequence()
            untrained_dist = route_distance_for_sequence(subset_cache.distance_matrix, untrained_seq)
            untrained_reward = -untrained_dist

            best_known_distance = (
                optimal_dist if optimal_dist is not None else min(random_dist, nn_dist, untrained_dist)
            )

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
                        "policy": "exact_optimal",
                        "distance_m": "N>9 skipped",
                        "reward": "N>9 skipped",
                    }
                )
            baseline_table_placeholder.dataframe(pd.DataFrame(baseline_rows), width="stretch")

            callback = StreamlitProgressCallback(
                module=module,
                subset_cache=subset_cache,
                epoch_metric_placeholder=epoch_metric_placeholder,
                best_distance_placeholder=best_distance_placeholder,
                current_loss_placeholder=current_loss_placeholder,
                chart_slot=chart_placeholder,
                map_slot=map_placeholder,
                map_update_interval=len(selected_names),
                best_known_distance=best_known_distance,
                teaching_mode=teaching_mode,
            )
            trainer = pl.Trainer(
                max_epochs=int(max_epochs),
                accelerator="cpu",
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                callbacks=[callback],
            )
            train_loader = DataLoader(_SingleInstanceDataset(), batch_size=1, shuffle=False)
            trainer.fit(module, train_loader)

            # Always render the final optimized greedy route after training ends.
            final_route_indices = module.greedy_sequence()
            final_distance = route_distance_for_sequence(subset_cache.distance_matrix, final_route_indices)
            final_reward = -final_distance
            final_map = render_route_map(
                city_name=city_name,
                selected_names=subset_cache.selected_names,
                selected_coords=subset_cache.selected_coords,
                path_cache=subset_cache.path_cache,
                route_indices=final_route_indices,
            )
            map_placeholder.empty()
            with map_placeholder.container():
                st_folium(
                    final_map,
                    width=1200,
                    height=600,
                    returned_objects=[],
                    key="route_map_final",
                )

            final_row = {
                "policy": "trained_greedy",
                "distance_m": round(final_distance, 2),
                "reward": round(final_reward, 2),
            }
            if optimal_dist is not None:
                final_row["optimality_gap_%"] = round(
                    100.0 * (final_distance - optimal_dist) / max(1e-9, optimal_dist), 3
                )
            baseline_rows.append(final_row)
            baseline_table_placeholder.dataframe(pd.DataFrame(baseline_rows), width="stretch")

        st.success("Training complete. The map shows the current greedy policy route.")


if __name__ == "__main__":
    main()

"""
Toy pore-network erosion simulator.

Purpose:
- Model (qualitatively) how slow sintering/vitrification can reduce pore size,
    break pathways, and change connectivity over time.
- This is intentionally a simple exploratory model, not a calibrated
    material-physics solver.

Typical command line call for the program:
python .\tunnel_simulation.py --rows 10 --cols 10 --segments 6 --output erode-clusters --iterations 5000 --draw-every 100 --diameter-drop 0.1 --save-heatmap erosion.png1 --make-video --video-path erosion.mp4 --video-fps 8

Key flags:
- --rows / --cols: tunnel lattice size
- --segments: number of segment cells along each tunnel edge
- --iterations: number of erosion steps
- --draw-every: checkpoint interval for plots and PNG frames
- --diameter-drop: per-step diameter reduction for one random positive cell
- --save-heatmap: output prefix for saved PNGs
- --make-video + --video-path + --video-fps: assemble saved frames into a movie
"""




from __future__ import annotations

import argparse
import importlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


PoreDistribution = Union[Dict[float, float], Sequence[float]]

# Index 0 is reserved for background (black). Cluster IDs cycle through this palette.
CLUSTER_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
]


@dataclass
class TunnelSegment:
    diameter: float


@dataclass
class CellTunnels:
    east: Optional[List[TunnelSegment]]
    south: Optional[List[TunnelSegment]]


class TunnelSimulation:
    """
    Creates a 2D square-like tunnel network on a rows x cols grid.

    Each cell may have:
    - an east tunnel to the next column
    - a south tunnel to the next row

    Each tunnel contains `segments` small tunnel segments. Every segment diameter
    is sampled from `seed_pore_distribution`.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        segments: int,
        seed_pore_distribution: PoreDistribution,
        rng_seed: Optional[int] = None,
    ) -> None:
        if rows <= 0:
            raise ValueError("rows must be > 0")
        if cols <= 0:
            raise ValueError("cols must be > 0")
        if segments <= 0:
            raise ValueError("segments must be > 0")

        self.rows = rows
        self.cols = cols
        self.segments = segments
        self.seed_pore_distribution = seed_pore_distribution
        self.rng = Random(rng_seed)

        self._distribution_values, self._distribution_weights = self._normalize_distribution(
            seed_pore_distribution
        )

    def _normalize_distribution(
        self, seed_pore_distribution: PoreDistribution
    ) -> Tuple[List[float], Optional[List[float]]]:
        if isinstance(seed_pore_distribution, dict):
            if len(seed_pore_distribution) == 0:
                raise ValueError("seed_pore_distribution dict cannot be empty")

            values = [float(k) for k in seed_pore_distribution.keys()]
            weights = [float(v) for v in seed_pore_distribution.values()]

            if any(w < 0 for w in weights):
                raise ValueError("distribution weights must be >= 0")
            if sum(weights) == 0:
                raise ValueError("distribution weights cannot all be 0")

            return values, weights

        if len(seed_pore_distribution) == 0:
            raise ValueError("seed_pore_distribution sequence cannot be empty")

        values = [float(v) for v in seed_pore_distribution]
        return values, None

    def _sample_diameter(self) -> float:
        if self._distribution_weights is None:
            return self.rng.choice(self._distribution_values)

        return self.rng.choices(
            self._distribution_values,
            weights=self._distribution_weights,
            k=1,
        )[0]

    def _create_tunnel(self) -> List[TunnelSegment]:
        return [TunnelSegment(self._sample_diameter()) for _ in range(self.segments)]

    def generate(self) -> List[List[CellTunnels]]:
        grid: List[List[CellTunnels]] = []

        for r in range(self.rows):
            row: List[CellTunnels] = []
            for c in range(self.cols):
                east = self._create_tunnel() if c < self.cols - 1 else None
                south = self._create_tunnel() if r < self.rows - 1 else None
                row.append(CellTunnels(east=east, south=south))
            grid.append(row)

        return grid

    def generate_tunnel_matrix(self) -> List[List[List[float]]]:
        """
        Returns a rows x cols matrix where each entry is a tunnel represented
        by a list of `segments` diameters sampled from `seed_pore_distribution`.
        """
        return [
            [[self._sample_diameter() for _ in range(self.segments)] for _ in range(self.cols)]
            for _ in range(self.rows)
        ]

    def generate_mean_diameter_map(self) -> List[List[float]]:
        """
        Returns a dense map of mean diameters as a (2*rows-1) x (2*cols-1) matrix.

        Coordinate meaning:
        - even,even => cell node (0.0)
        - even,odd  => east tunnel mean diameter
        - odd,even  => south tunnel mean diameter
        - odd,odd   => intersection filler (0.0)
        """
        network = self.generate()
        out_rows = self.rows * 2 - 1
        out_cols = self.cols * 2 - 1

        dense = [[0.0 for _ in range(out_cols)] for _ in range(out_rows)]

        for r in range(self.rows):
            for c in range(self.cols):
                base_r = 2 * r
                base_c = 2 * c

                cell = network[r][c]

                if cell.east is not None:
                    mean_east = sum(seg.diameter for seg in cell.east) / len(cell.east)
                    dense[base_r][base_c + 1] = mean_east

                if cell.south is not None:
                    mean_south = sum(seg.diameter for seg in cell.south) / len(cell.south)
                    dense[base_r + 1][base_c] = mean_south

        return dense

    def generate_segment_layout_map(self) -> List[List[float]]:
        """
        Returns a segment-resolved spatial map where each tunnel is represented
        as exactly `segments` cells along x or y.

        For segments=6, adjacent tunnel centerlines are 5 cells apart, leaving
        a 4x4 zero region between them.
        """
        if self.segments < 2:
            raise ValueError("segments must be >= 2 for a spatial layout map")

        network = self.generate()
        step = self.segments - 1

        out_rows = (self.rows - 1) * step + 1
        out_cols = (self.cols - 1) * step + 1
        layout = [[0.0 for _ in range(out_cols)] for _ in range(out_rows)]

        for r in range(self.rows):
            for c in range(self.cols):
                start_r = r * step
                start_c = c * step
                cell = network[r][c]

                if cell.east is not None:
                    for i, segment in enumerate(cell.east):
                        layout[start_r][start_c + i] = segment.diameter

                if cell.south is not None:
                    for i, segment in enumerate(cell.south):
                        layout[start_r + i][start_c] = segment.diameter

        return layout


def _pretty_print_matrix(matrix: List[List[float]], decimals: int = 3) -> None:
    for row in matrix:
        print(" ".join(f"{value:.{decimals}f}" for value in row))


def _pretty_print_tunnel_matrix(matrix: List[List[List[float]]], decimals: int = 3) -> None:
    for r, row in enumerate(matrix):
        print(f"row {r}:")
        for c, tunnel in enumerate(row):
            tunnel_str = ", ".join(f"{value:.{decimals}f}" for value in tunnel)
            print(f"  col {c}: [{tunnel_str}]")


def _plot_heatmap(
    matrix: List[List[float]],
    title: str = "Tunnel Diameter Heatmap",
    save_path: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="viridis", interpolation="nearest")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean diameter")

    ax.set_title(title)
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")
    ax.set_xticks(range(len(matrix[0])))
    ax.set_yticks(range(len(matrix)))

    for r, row in enumerate(matrix):
        for c, value in enumerate(row):
            ax.text(c, r, f"{value:.2f}", ha="center", va="center", color="white", fontsize=7)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Heatmap saved to: {save_path}")

    plt.show(block=True)
    plt.close(fig)


def calculate_tunnel_level_adjacency_matrix(
    tunnel_matrix: List[List[float]],
) -> Tuple[List[List[int]], List[List[int]], Dict[int, List[Tuple[int, int]]], List[float]]:
    """
    Clusters all cells with diameter > 0 using orthogonal (N,S,E,W) connectivity.

    Returns:
    - adjacency_matrix: cluster-to-cluster adjacency matrix
    - cluster_labels: matrix of cluster ids, -1 for background/zero-diameter cells
    - clusters: mapping cluster_id -> list of (row, col) cells in that cluster
        - cluster_volumes: per-cluster volume sorted by cluster id, where
            volume = sum(diameter^2) over cells in the cluster
    """
    if not tunnel_matrix or not tunnel_matrix[0]:
        raise ValueError("tunnel_matrix must be a non-empty 2D matrix")

    rows = len(tunnel_matrix)
    cols = len(tunnel_matrix[0])
    cluster_labels = [[-1 for _ in range(cols)] for _ in range(rows)]
    raw_clusters: Dict[int, List[Tuple[int, int]]] = {}

    def neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        if r > 0:
            out.append((r - 1, c))
        if r < rows - 1:
            out.append((r + 1, c))
        if c > 0:
            out.append((r, c - 1))
        if c < cols - 1:
            out.append((r, c + 1))
        return out

    next_cluster_id = 0
    for r in range(rows):
        for c in range(cols):
            if tunnel_matrix[r][c] <= 0 or cluster_labels[r][c] != -1:
                continue

            stack: List[Tuple[int, int]] = [(r, c)]
            cluster_labels[r][c] = next_cluster_id
            members: List[Tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                members.append((cr, cc))

                for nr, nc in neighbors(cr, cc):
                    if tunnel_matrix[nr][nc] > 0 and cluster_labels[nr][nc] == -1:
                        cluster_labels[nr][nc] = next_cluster_id
                        stack.append((nr, nc))

            raw_clusters[next_cluster_id] = members
            next_cluster_id += 1

    raw_volumes: Dict[int, float] = {}
    for cluster_id, members in raw_clusters.items():
        raw_volumes[cluster_id] = sum(tunnel_matrix[r][c] ** 2 for r, c in members)

    sorted_raw_ids = sorted(raw_clusters.keys(), key=lambda cid: raw_volumes[cid], reverse=True)
    raw_to_sorted_id = {raw_id: idx for idx, raw_id in enumerate(sorted_raw_ids)}

    clusters: Dict[int, List[Tuple[int, int]]] = {}
    cluster_volumes: List[float] = []
    for raw_id in sorted_raw_ids:
        new_id = raw_to_sorted_id[raw_id]
        clusters[new_id] = raw_clusters[raw_id]
        cluster_volumes.append(raw_volumes[raw_id])

    for r in range(rows):
        for c in range(cols):
            if cluster_labels[r][c] != -1:
                cluster_labels[r][c] = raw_to_sorted_id[cluster_labels[r][c]]

    cluster_count = len(clusters)
    adjacency_matrix = [[0 for _ in range(cluster_count)] for _ in range(cluster_count)]

    # Build inter-cluster adjacency and validate no touching clusters remain.
    touching_pairs: List[Tuple[int, int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            this_label = cluster_labels[r][c]
            if this_label == -1:
                continue

            for nr, nc in neighbors(r, c):
                other_label = cluster_labels[nr][nc]
                if other_label == -1 or other_label == this_label:
                    continue
                adjacency_matrix[this_label][other_label] = 1
                adjacency_matrix[other_label][this_label] = 1
                touching_pairs.append((r, c, nr, nc))

    if touching_pairs:
        raise RuntimeError(
            "Cluster validation failed: distinct clusters are touching orthogonally."
        )

    return adjacency_matrix, cluster_labels, clusters, cluster_volumes


def _plot_cluster_heatmap(
    cluster_labels: List[List[int]],
    title: str = "Tunnel Cluster Heatmap",
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    display = [
        [((label % len(CLUSTER_COLORS)) + 1) if label >= 0 else 0 for label in row]
        for row in cluster_labels
    ]
    cluster_cmap = ListedColormap(["black"] + CLUSTER_COLORS)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(display, cmap=cluster_cmap, interpolation="nearest")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Cluster id + 1 (0 = background)")

    ax.set_title(title)
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")
    ax.set_xticks(range(len(display[0])))
    ax.set_yticks(range(len(display)))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Cluster heatmap saved to: {save_path}")

    if show_plot:
        plt.show(block=True)
    plt.close(fig)


def count_orthogonal_positive_neighbors(tunnel_matrix: List[List[float]]) -> Dict[int, int]:
    """
    Counts, for cells with diameter > 0, how many orthogonal neighbors also have
    diameter > 0. Returns counts for keys 1, 2, 3, 4.
    """
    if not tunnel_matrix or not tunnel_matrix[0]:
        raise ValueError("tunnel_matrix must be a non-empty 2D matrix")

    rows = len(tunnel_matrix)
    cols = len(tunnel_matrix[0])
    counts = {1: 0, 2: 0, 3: 0, 4: 0}

    for r in range(rows):
        for c in range(cols):
            if tunnel_matrix[r][c] <= 0:
                continue

            neighbors = 0
            if r > 0 and tunnel_matrix[r - 1][c] > 0:
                neighbors += 1
            if r < rows - 1 and tunnel_matrix[r + 1][c] > 0:
                neighbors += 1
            if c > 0 and tunnel_matrix[r][c - 1] > 0:
                neighbors += 1
            if c < cols - 1 and tunnel_matrix[r][c + 1] > 0:
                neighbors += 1

            if neighbors in counts:
                counts[neighbors] += 1

    return counts


def count_cells_by_routes_to_edge(tunnel_matrix: List[List[float]]) -> Dict[str, int]:
    """
    Classifies positive-diameter cells by number of orthogonal routes to the outside edge.

    Route definition used here:
    - A neighboring positive cell counts as a route if it belongs to the edge-reachable
      positive network (reachable from any positive boundary cell).
    - If the cell itself is on the boundary, that contributes one direct route.

    Returns counts with keys:
    - 'zero_routes'
    - 'one_route'
    - 'more_than_one_route'
    """
    if not tunnel_matrix or not tunnel_matrix[0]:
        raise ValueError("tunnel_matrix must be a non-empty 2D matrix")

    # Convert positive cells into a graph: one vertex per positive cell,
    # orthogonal neighbors become graph edges.
    rows = len(tunnel_matrix)
    cols = len(tunnel_matrix[0])

    positive_vertices: List[int] = []
    vertex_of_rc: Dict[Tuple[int, int], int] = {}
    rc_of_vertex: Dict[int, Tuple[int, int]] = {}

    for r in range(rows):
        for c in range(cols):
            if tunnel_matrix[r][c] > 0:
                vid = len(positive_vertices)
                positive_vertices.append(vid)
                vertex_of_rc[(r, c)] = vid
                rc_of_vertex[vid] = (r, c)

    n = len(positive_vertices)
    if n == 0:
        return {"zero_routes": 0, "one_route": 0, "more_than_one_route": 0}

    adj: List[List[int]] = [[] for _ in range(n)]
    for (r, c), v in vertex_of_rc.items():
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            u = vertex_of_rc.get((nr, nc))
            if u is not None:
                adj[v].append(u)

    is_boundary_vertex = [False] * n
    for v in range(n):
        r, c = rc_of_vertex[v]
        is_boundary_vertex[v] = r == 0 or r == rows - 1 or c == 0 or c == cols - 1

    disc = [-1] * n
    low = [0] * n
    parent = [-1] * n
    time = 0
    edge_stack: List[Tuple[int, int]] = []
    bccs: List[set[int]] = []
    articulation = [False] * n

    def pop_component_until(edge_u: int, edge_v: int) -> None:
        comp: set[int] = set()
        while edge_stack:
            a, b = edge_stack.pop()
            comp.add(a)
            comp.add(b)
            if (a == edge_u and b == edge_v) or (a == edge_v and b == edge_u):
                break
        if comp:
            bccs.append(comp)

    def dfs_bcc(v: int) -> None:
        nonlocal time
        disc[v] = low[v] = time
        time += 1
        child_count = 0

        for to in adj[v]:
            if disc[to] == -1:
                parent[to] = v
                child_count += 1
                edge_stack.append((v, to))
                dfs_bcc(to)
                low[v] = min(low[v], low[to])

                if (parent[v] == -1 and child_count > 1) or (
                    parent[v] != -1 and low[to] >= disc[v]
                ):
                    articulation[v] = True
                    pop_component_until(v, to)
            elif to != parent[v] and disc[to] < disc[v]:
                edge_stack.append((v, to))
                low[v] = min(low[v], disc[to])

    components_vertices: List[List[int]] = []
    visited = [False] * n
    for s in range(n):
        if visited[s]:
            continue
        stack = [s]
        comp_list: List[int] = []
        visited[s] = True
        while stack:
            x = stack.pop()
            comp_list.append(x)
            for y in adj[x]:
                if not visited[y]:
                    visited[y] = True
                    stack.append(y)
        components_vertices.append(comp_list)

    for s in range(n):
        if disc[s] == -1:
            dfs_bcc(s)
            if edge_stack:
                comp: set[int] = set()
                while edge_stack:
                    a, b = edge_stack.pop()
                    comp.add(a)
                    comp.add(b)
                if comp:
                    bccs.append(comp)

    vertex_bccs: List[List[int]] = [[] for _ in range(n)]
    for bcc_id, comp in enumerate(bccs):
        for v in comp:
            vertex_bccs[v].append(bcc_id)

    route_count_per_vertex = [0] * n

    for comp_vertices in components_vertices:
        has_boundary = any(is_boundary_vertex[v] for v in comp_vertices)
        if not has_boundary:
            for v in comp_vertices:
                route_count_per_vertex[v] = 0
            continue

        comp_set = set(comp_vertices)
        comp_bcc_ids = []
        for bcc_id, comp in enumerate(bccs):
            if any(v in comp_set for v in comp):
                comp_bcc_ids.append(bcc_id)

        if not comp_bcc_ids:
            for v in comp_vertices:
                route_count_per_vertex[v] = 1
            continue

        # Build a block-cut tree for the component. This gives a compact way
        # to count independent "directions" to boundary-connected regions.
        art_ids = [v for v in comp_vertices if articulation[v]]
        art_index = {v: i for i, v in enumerate(art_ids)}
        bcc_index = {bcc_id: i for i, bcc_id in enumerate(comp_bcc_ids)}

        bcc_node_offset = len(art_ids)
        node_count = len(art_ids) + len(comp_bcc_ids)
        tree_adj: List[List[int]] = [[] for _ in range(node_count)]

        def art_node(v: int) -> int:
            return art_index[v]

        def bcc_node(bcc_id: int) -> int:
            return bcc_node_offset + bcc_index[bcc_id]

        for bcc_id in comp_bcc_ids:
            bnode = bcc_node(bcc_id)
            for v in bccs[bcc_id]:
                if v in art_index:
                    anode = art_node(v)
                    tree_adj[bnode].append(anode)
                    tree_adj[anode].append(bnode)

        node_has_boundary = [False] * node_count
        for v in art_ids:
            if is_boundary_vertex[v]:
                node_has_boundary[art_node(v)] = True
        for bcc_id in comp_bcc_ids:
            bnode = bcc_node(bcc_id)
            if any(is_boundary_vertex[v] for v in bccs[bcc_id]):
                node_has_boundary[bnode] = True

        # Forest-safe DP in case of degenerate construction.
        parent_node = [-1] * node_count
        order: List[int] = []
        visited_node = [False] * node_count
        for root in range(node_count):
            if visited_node[root]:
                continue
            stack = [root]
            visited_node[root] = True
            while stack:
                cur = stack.pop()
                order.append(cur)
                for nxt in tree_adj[cur]:
                    if not visited_node[nxt]:
                        visited_node[nxt] = True
                        parent_node[nxt] = cur
                        stack.append(nxt)

        subtree_boundary_count = [1 if node_has_boundary[i] else 0 for i in range(node_count)]
        for node in reversed(order):
            p = parent_node[node]
            if p != -1:
                subtree_boundary_count[p] += subtree_boundary_count[node]

        # Compute route-direction count per BC-tree node.
        direction_count = [0] * node_count
        # Need component-level totals for each rooted tree.
        root_total: Dict[int, int] = {}
        for node in order:
            if parent_node[node] == -1:
                root_total[node] = subtree_boundary_count[node]

        for node in range(node_count):
            count = 1 if node_has_boundary[node] else 0
            for nxt in tree_adj[node]:
                if parent_node[nxt] == node:
                    side_boundary = subtree_boundary_count[nxt]
                elif parent_node[node] == nxt:
                    root = node
                    while parent_node[root] != -1:
                        root = parent_node[root]
                    side_boundary = root_total[root] - subtree_boundary_count[node]
                else:
                    # Different tree root fallback; treat as boundary-free side.
                    side_boundary = 0
                if side_boundary > 0:
                    count += 1
            direction_count[node] = count

        for v in comp_vertices:
            if v in art_index:
                route_count_per_vertex[v] = direction_count[art_node(v)]
            else:
                # Non-articulation vertices belong to one BCC in this component.
                bcc_id = None
                for candidate in vertex_bccs[v]:
                    if candidate in bcc_index:
                        bcc_id = candidate
                        break
                if bcc_id is None:
                    route_count_per_vertex[v] = 1 if has_boundary else 0
                else:
                    route_count_per_vertex[v] = direction_count[bcc_node(bcc_id)]

    result = {"zero_routes": 0, "one_route": 0, "more_than_one_route": 0}
    for v in range(n):
        routes = route_count_per_vertex[v]
        if routes <= 0:
            result["zero_routes"] += 1
        elif routes == 1:
            result["one_route"] += 1
        else:
            result["more_than_one_route"] += 1

    return result


def _plot_diameter_and_cluster_subplots(
    diameter_matrix: List[List[float]],
    cluster_labels: List[List[int]],
    title: str = "Diameter and Cluster Maps",
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    display_clusters = [
        [((label % len(CLUSTER_COLORS)) + 1) if label >= 0 else 0 for label in row]
        for row in cluster_labels
    ]
    cluster_cmap = ListedColormap(["black"] + CLUSTER_COLORS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    img1 = ax1.imshow(diameter_matrix, cmap="viridis", interpolation="nearest")
    cbar1 = fig.colorbar(img1, ax=ax1)
    cbar1.set_label("Diameter")
    ax1.set_title("Diameter Heatmap")
    ax1.set_xlabel("Column index")
    ax1.set_ylabel("Row index")

    img2 = ax2.imshow(display_clusters, cmap=cluster_cmap, interpolation="nearest")
    cbar2 = fig.colorbar(img2, ax=ax2)
    cbar2.set_label("Cluster id + 1 (0 = background)")
    ax2.set_title("Cluster Heatmap")
    ax2.set_xlabel("Column index")
    ax2.set_ylabel("Row index")

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Combined heatmap saved to: {save_path}")

    if show_plot:
        plt.show(block=True)
    plt.close(fig)


def _collect_debug_state(
    iteration: int,
    tunnel_matrix: List[List[float]],
    cluster_labels: List[List[int]],
    cluster_volumes: List[float],
    neighbor_counts: Dict[int, int],
) -> Dict[str, Union[int, float, List[float], Dict[str, int]]]:
    rows = len(tunnel_matrix)
    cols = len(tunnel_matrix[0])

    positive_cells = 0
    zero_cells = 0
    unlabeled_positive_cells = 0
    labeled_zero_cells = 0
    positive_sum = 0.0
    positive_min = None
    positive_max = None

    for r in range(rows):
        for c in range(cols):
            diameter = tunnel_matrix[r][c]
            label = cluster_labels[r][c]

            if diameter > 0:
                positive_cells += 1
                positive_sum += diameter
                if positive_min is None or diameter < positive_min:
                    positive_min = diameter
                if positive_max is None or diameter > positive_max:
                    positive_max = diameter
                if label < 0:
                    unlabeled_positive_cells += 1
            else:
                zero_cells += 1
                if label >= 0:
                    labeled_zero_cells += 1

    return {
        "iteration": iteration,
        "rows": rows,
        "cols": cols,
        "positive_cells": positive_cells,
        "zero_cells": zero_cells,
        "positive_fraction": positive_cells / (rows * cols),
        "positive_mean": (positive_sum / positive_cells) if positive_cells else 0.0,
        "positive_min": positive_min if positive_min is not None else 0.0,
        "positive_max": positive_max if positive_max is not None else 0.0,
        "cluster_count": len(cluster_volumes),
        "top_cluster_volume": cluster_volumes[0] if cluster_volumes else 0.0,
        "neighbor_counts": {
            "1": neighbor_counts[1],
            "2": neighbor_counts[2],
            "3": neighbor_counts[3],
            "4": neighbor_counts[4],
        },
        "unlabeled_positive_cells": unlabeled_positive_cells,
        "labeled_zero_cells": labeled_zero_cells,
    }


def erode_tunnel_matrix_and_draw_clusters(
    tunnel_matrix: List[List[float]],
    iterations: int = 500,
    draw_every: int = 100,
    diameter_drop: float = 0.1,
    rng_seed: Optional[int] = None,
    save_prefix: Optional[str] = None,
    debug_state: bool = False,
    debug_every: int = 100,
    debug_focus_iteration: int = 1900,
    debug_focus_window: int = 20,
    debug_log_path: Optional[str] = None,
    show_plots: bool = True,
) -> List[List[float]]:
    """
    Iteratively erodes one random cell per iteration and draws cluster maps.

    - At each step, one random cell diameter is reduced by `diameter_drop`.
    - Diameter is clamped to a minimum of 0.0.
    - Cluster heatmaps are drawn every `draw_every` iterations.
    """
    if not tunnel_matrix or not tunnel_matrix[0]:
        raise ValueError("tunnel_matrix must be a non-empty 2D matrix")
    if iterations <= 0:
        raise ValueError("iterations must be > 0")
    if draw_every <= 0:
        raise ValueError("draw_every must be > 0")
    if diameter_drop <= 0:
        raise ValueError("diameter_drop must be > 0")
    if debug_every <= 0:
        raise ValueError("debug_every must be > 0")
    if debug_focus_window < 0:
        raise ValueError("debug_focus_window must be >= 0")

    rows = len(tunnel_matrix)
    cols = len(tunnel_matrix[0])
    current = [row[:] for row in tunnel_matrix]
    rng = Random(rng_seed)

    iteration_points: List[int] = []
    open_volume_history: List[float] = []
    closed_volume_history: List[float] = []
    neighbor_count_history: Dict[int, List[int]] = {1: [], 2: [], 3: [], 4: []}
    open_cluster_one_neighbor_history: List[int] = []
    edge_route_history: Dict[str, List[int]] = {
        "zero_routes": [],
        "one_route": [],
        "more_than_one_route": [],
    }

    for iteration in range(1, iterations + 1):
        # Step 1: pick one currently-open cell and erode it.
        positive_cells = [
            (rr, cc)
            for rr in range(rows)
            for cc in range(cols)
            if current[rr][cc] > 0.0
        ]
        if not positive_cells:
            print(f"Iteration {iteration}: all diameters are zero, stopping early.")
            break

        r, c = positive_cells[rng.randrange(len(positive_cells))]
        current[r][c] = max(0.0, current[r][c] - diameter_drop)

        # Step 2: recompute connectivity and metrics on the updated field.
        adjacency_matrix, cluster_labels, clusters, cluster_volumes = (
            calculate_tunnel_level_adjacency_matrix(current)
        )
        neighbor_counts = count_orthogonal_positive_neighbors(current)
        route_counts = count_cells_by_routes_to_edge(current)

        open_volume = 0.0
        open_cluster_ids = set()
        for cluster_id, members in clusters.items():
            touches_edge = any(
                rr == 0 or rr == rows - 1 or cc == 0 or cc == cols - 1
                for rr, cc in members
            )
            if touches_edge:
                open_cluster_ids.add(cluster_id)
                open_volume += cluster_volumes[cluster_id]
        closed_volume = sum(cluster_volumes) - open_volume

        open_cluster_one_neighbor_count = 0
        for rr in range(rows):
            for cc in range(cols):
                if current[rr][cc] <= 0:
                    continue
                label = cluster_labels[rr][cc]
                if label not in open_cluster_ids:
                    continue

                neighbors = 0
                if rr > 0 and current[rr - 1][cc] > 0:
                    neighbors += 1
                if rr < rows - 1 and current[rr + 1][cc] > 0:
                    neighbors += 1
                if cc > 0 and current[rr][cc - 1] > 0:
                    neighbors += 1
                if cc < cols - 1 and current[rr][cc + 1] > 0:
                    neighbors += 1

                if neighbors == 1:
                    open_cluster_one_neighbor_count += 1

        if debug_state:
            should_log = (
                iteration % debug_every == 0
                or abs(iteration - debug_focus_iteration) <= debug_focus_window
            )
            if should_log:
                debug_snapshot = _collect_debug_state(
                    iteration,
                    current,
                    cluster_labels,
                    cluster_volumes,
                    neighbor_counts,
                )
                print(
                    "DEBUG "
                    f"iter={debug_snapshot['iteration']} "
                    f"positive={debug_snapshot['positive_cells']} "
                    f"zero={debug_snapshot['zero_cells']} "
                    f"clusters={debug_snapshot['cluster_count']} "
                    f"unlabeled_positive={debug_snapshot['unlabeled_positive_cells']}"
                )
                if debug_log_path:
                    with open(debug_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(debug_snapshot) + "\n")

        iteration_points.append(iteration)
        open_volume_history.append(open_volume)
        closed_volume_history.append(closed_volume)
        for k in (1, 2, 3, 4):
            neighbor_count_history[k].append(neighbor_counts[k])
        open_cluster_one_neighbor_history.append(open_cluster_one_neighbor_count)
        edge_route_history["zero_routes"].append(route_counts["zero_routes"])
        edge_route_history["one_route"].append(route_counts["one_route"])
        edge_route_history["more_than_one_route"].append(route_counts["more_than_one_route"])

        # Step 3: emit checkpoint visuals/statistics at configured intervals.
        if iteration % draw_every == 0:
            print(f"Iteration {iteration}: clusters={len(clusters)}")
            print("Cluster adjacency matrix:")
            for row in adjacency_matrix:
                print(" ".join(str(value) for value in row))

            save_path = None
            if save_prefix:
                save_path = f"{save_prefix}_iter_{iteration}.png"

            _plot_diameter_and_cluster_subplots(
                current,
                cluster_labels,
                title=f"Erosion state at iteration {iteration}",
                save_path=save_path,
                show_plot=show_plots,
            )

    _plot_erosion_timelines(
        iteration_points,
        open_volume_history,
        closed_volume_history,
        neighbor_count_history,
        open_cluster_one_neighbor_history,
        edge_route_history,
        save_prefix=save_prefix,
        show_plot=show_plots,
    )

    return current


def _plot_erosion_timelines(
    iteration_points: List[int],
    open_volume_history: List[float],
    closed_volume_history: List[float],
    neighbor_count_history: Dict[int, List[int]],
    open_cluster_one_neighbor_history: List[int],
    edge_route_history: Dict[str, List[int]],
    save_prefix: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    if not iteration_points:
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 13))

    ax1.plot(iteration_points, open_volume_history, label="Open volume")
    ax1.plot(iteration_points, closed_volume_history, label="Closed volume")

    ax1.set_title("Open vs Closed Volume over Iteration")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Volume (sum of diameter^2)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    for k in (1, 2, 3, 4):
        ax2.plot(iteration_points, neighbor_count_history[k], label=f"{k} neighbors")
    ax2.plot(
        iteration_points,
        open_cluster_one_neighbor_history,
        label="1 neighbor (open clusters only)",
        linestyle="--",
        linewidth=2.0,
    )

    ax2.set_title("Cell Neighbor Counts vs Iteration")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Cell count")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    ax3.plot(iteration_points, edge_route_history["zero_routes"], label="0 routes to edge")
    ax3.plot(iteration_points, edge_route_history["one_route"], label="1 route to edge")
    ax3.plot(
        iteration_points,
        edge_route_history["more_than_one_route"],
        label=">1 routes to edge",
    )
    ax3.set_title("Route-to-Edge Cell Groups vs Iteration")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Cell count")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_prefix:
        timeline_path = f"{save_prefix}_timelines.png"
        fig.savefig(timeline_path, dpi=200, bbox_inches="tight")
        print(f"Timeline plots saved to: {timeline_path}")

    if show_plot:
        plt.show(block=True)
    plt.close(fig)


def _create_video_from_frames(
    frame_prefix: str,
    output_video_path: str,
    fps: int = 10,
) -> Optional[str]:
    """
    Creates a video from files matching '{frame_prefix}_iter_*.png'.
    Returns the output path on success, otherwise None.
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")

    try:
        imageio = importlib.import_module("imageio.v2")
    except ModuleNotFoundError:
        print("Video creation skipped: imageio is not installed. Run 'pip install imageio'.")
        return None

    prefix_path = Path(frame_prefix)
    frame_dir = prefix_path.parent if str(prefix_path.parent) != "" else Path(".")
    frame_stem = prefix_path.name

    frame_pattern = re.compile(rf"^{re.escape(frame_stem)}_iter_(\d+)\.png$")
    matches: List[Tuple[int, Path]] = []

    for candidate in frame_dir.iterdir():
        if not candidate.is_file():
            continue
        m = frame_pattern.match(candidate.name)
        if m:
            matches.append((int(m.group(1)), candidate))

    if not matches:
        print(
            f"No frame files found for video creation with prefix '{frame_prefix}'. "
            "Expected files like '<prefix>_iter_100.png'."
        )
        return None

    matches.sort(key=lambda x: x[0])
    output_path = Path(output_video_path)
    if output_path.suffix.lower() == "":
        output_path = output_path.with_suffix(".mp4")

    def normalize_frame_to_size(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        # Ensure HxWx3 format for stable ffmpeg encoding.
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)

        h, w = frame.shape[:2]

        # Center-crop if larger than target.
        if h > target_h:
            top = (h - target_h) // 2
            frame = frame[top : top + target_h, :, :]
            h = target_h
        if w > target_w:
            left = (w - target_w) // 2
            frame = frame[:, left : left + target_w, :]
            w = target_w

        # Symmetric zero-pad if smaller than target.
        if h < target_h or w < target_w:
            pad_top = (target_h - h) // 2
            pad_bottom = target_h - h - pad_top
            pad_left = (target_w - w) // 2
            pad_right = target_w - w - pad_left
            frame = np.pad(
                frame,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        return frame

    first_frame = imageio.imread(matches[0][1])
    if first_frame.ndim == 2:
        target_h, target_w = first_frame.shape
    else:
        target_h, target_w = first_frame.shape[:2]

    normalized_count = 0

    with imageio.get_writer(output_path, fps=fps) as writer:
        for _, frame_path in matches:
            frame = imageio.imread(frame_path)
            src_shape = frame.shape
            frame = normalize_frame_to_size(frame, target_h, target_w)
            if frame.shape != src_shape:
                normalized_count += 1
            writer.append_data(frame)

    if normalized_count > 0:
        print(
            f"Normalized {normalized_count} frame(s) to {target_w}x{target_h} "
            "for consistent video encoding."
        )

    print(f"Video created: {output_path}")
    return str(output_path)


def _parse_distribution(distribution_text: str) -> PoreDistribution:
    """
    Supported formats:
    - Weighted dict: "0.2:1,0.5:3,1.0:2"
    - Unweighted list: "0.2,0.5,1.0"
    """
    text = distribution_text.strip()
    if not text:
        raise ValueError("seed_pore_distribution cannot be empty")

    if ":" in text:
        out: Dict[float, float] = {}
        for item in text.split(","):
            part = item.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(
                    "Mixed distribution format. Use only 'value:weight' pairs or only values."
                )
            value_text, weight_text = part.split(":", maxsplit=1)
            out[float(value_text.strip())] = float(weight_text.strip())

        if not out:
            raise ValueError("No valid value:weight pairs found in seed_pore_distribution")
        return out

    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("No valid values found in seed_pore_distribution")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a tunnel simulation matrix from a pore diameter distribution."
    )
    parser.add_argument("--rows", type=int, default=5, help="Number of matrix rows")
    parser.add_argument("--cols", type=int, default=5, help="Number of matrix columns")
    parser.add_argument(
        "--segments",
        type=int,
        default=6,
        help="Number of segments in each tunnel",
    )
    parser.add_argument(
        "--seed-pore-distribution",
        type=str,
        default="0.2:1,0.5:3,1.0:2,1.6:1",
        help=(
            "Distribution for tunnel diameters. Use 'v1:w1,v2:w2' for weighted "
            "or 'v1,v2,v3' for unweighted"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        choices=["tunnels", "mean-map", "heatmap", "clusters", "erode-clusters"],
        default="tunnels",
        help="Choose tunnel matrix, dense mean map, diameter heatmap, cluster heatmap, or erosion cluster run",
    )
    parser.add_argument(
        "--save-heatmap",
        type=str,
        default=None,
        help="Optional PNG path to save the heatmap when using --output heatmap",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Number of iterations for --output erode-clusters",
    )
    parser.add_argument(
        "--draw-every",
        type=int,
        default=100,
        help="Draw clusters every N iterations for --output erode-clusters",
    )
    parser.add_argument(
        "--diameter-drop",
        type=float,
        default=0.1,
        help="Diameter reduction amount per iteration for --output erode-clusters",
    )
    parser.add_argument(
        "--debug-state",
        action="store_true",
        help="Log debug state snapshots during --output erode-clusters",
    )
    parser.add_argument(
        "--debug-every",
        type=int,
        default=100,
        help="Log debug state every N iterations when --debug-state is enabled",
    )
    parser.add_argument(
        "--debug-focus-iteration",
        type=int,
        default=1900,
        help="Extra debug logging center iteration when --debug-state is enabled",
    )
    parser.add_argument(
        "--debug-focus-window",
        type=int,
        default=20,
        help="Extra debug logging window around --debug-focus-iteration",
    )
    parser.add_argument(
        "--debug-log",
        type=str,
        default=None,
        help="Optional JSONL output path for debug snapshots",
    )
    parser.add_argument(
        "--make-video",
        action="store_true",
        help="Create a video from erosion frame PNGs after --output erode-clusters",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="erosion_video.mp4",
        help="Output video path for --make-video",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=10,
        help="Frames per second for --make-video",
    )
    args = parser.parse_args()

    seed_pore_distribution = _parse_distribution(args.seed_pore_distribution)

    sim = TunnelSimulation(
        rows=args.rows,
        cols=args.cols,
        segments=args.segments,
        seed_pore_distribution=seed_pore_distribution,
        rng_seed=args.seed,
    )

    # Output modes:
    # - tunnels: print sampled tunnel segments per grid cell
    # - mean-map: print compact edge-mean map
    # - heatmap: visualize segment layout diameters
    # - clusters: visualize and print cluster structure for one sampled layout
    # - erode-clusters: run iterative erosion + metrics + optional video
    if args.output == "tunnels":
        tunnel_matrix = sim.generate_tunnel_matrix()
        _pretty_print_tunnel_matrix(tunnel_matrix)
    elif args.output == "mean-map":
        dense_map = sim.generate_mean_diameter_map()
        _pretty_print_matrix(dense_map)
    elif args.output == "heatmap":
        layout_map = sim.generate_segment_layout_map()
        _plot_heatmap(layout_map, save_path=args.save_heatmap)
    elif args.output == "clusters":
        layout_map = sim.generate_segment_layout_map()
        adjacency_matrix, cluster_labels, clusters, cluster_volumes = (
            calculate_tunnel_level_adjacency_matrix(layout_map)
        )
        print(f"Clusters found: {len(clusters)}")
        print("Cluster volumes (diameter^2 sum):")
        print(" ".join(f"{v:.3f}" for v in cluster_volumes))
        print("Cluster adjacency matrix:")
        for row in adjacency_matrix:
            print(" ".join(str(value) for value in row))
        _plot_cluster_heatmap(cluster_labels, save_path=args.save_heatmap)
    else:
        layout_map = sim.generate_segment_layout_map()
        save_prefix = None
        if args.save_heatmap:
            save_prefix = args.save_heatmap.rsplit(".", maxsplit=1)[0]

        eroded = erode_tunnel_matrix_and_draw_clusters(
            layout_map,
            iterations=args.iterations,
            draw_every=args.draw_every,
            diameter_drop=args.diameter_drop,
            rng_seed=args.seed,
            save_prefix=save_prefix,
            debug_state=args.debug_state,
            debug_every=args.debug_every,
            debug_focus_iteration=args.debug_focus_iteration,
            debug_focus_window=args.debug_focus_window,
            debug_log_path=args.debug_log,
            show_plots=not args.make_video,
        )
        if args.make_video:
            if not save_prefix:
                print(
                    "Skipping video creation because --save-heatmap was not provided "
                    "(needed to locate frame PNG files)."
                )
            else:
                _create_video_from_frames(
                    frame_prefix=save_prefix,
                    output_video_path=args.video_path,
                    fps=args.video_fps,
                )
        print(f"Final average diameter: {sum(sum(r) for r in eroded) / (len(eroded) * len(eroded[0])):.4f}")
    print("Simulation completed.")


if __name__ == "__main__":
    main()

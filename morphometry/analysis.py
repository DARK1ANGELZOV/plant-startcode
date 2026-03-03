from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from skimage.morphology import skeletonize


@dataclass
class MorphometryResult:
    area_px: int
    area_mm2: float
    length_px: float
    length_mm: float


def _to_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.bool_:
        return mask > 0
    return mask


def area_pixels(mask: np.ndarray) -> int:
    binary = _to_binary_mask(mask)
    return int(binary.sum())


def _build_graph(skeleton: np.ndarray) -> tuple[list[tuple[int, int]], list[list[tuple[int, float]]]]:
    coords = np.argwhere(skeleton)
    if len(coords) == 0:
        return [], []

    index_map = {tuple(coord): i for i, coord in enumerate(coords)}
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(len(coords))]
    for i, (r, c) in enumerate(coords):
        for dr, dc in neighbors:
            nr, nc = int(r + dr), int(c + dc)
            j = index_map.get((nr, nc))
            if j is None:
                continue
            weight = 1.41421356237 if dr != 0 and dc != 0 else 1.0
            adjacency[i].append((j, weight))

    return [(int(r), int(c)) for r, c in coords], adjacency


def _farthest_node(adjacency: list[list[tuple[int, float]]], source: int) -> tuple[int, float]:
    dist = [float('inf')] * len(adjacency)
    dist[source] = 0.0
    pq: list[tuple[float, int]] = [(0.0, source)]

    while pq:
        cur_dist, node = heapq.heappop(pq)
        if cur_dist > dist[node]:
            continue
        for nxt, w in adjacency[node]:
            nd = cur_dist + w
            if nd < dist[nxt]:
                dist[nxt] = nd
                heapq.heappush(pq, (nd, nxt))

    finite_idx = [i for i, d in enumerate(dist) if np.isfinite(d)]
    if not finite_idx:
        return source, 0.0
    farthest = max(finite_idx, key=lambda i: dist[i])
    return farthest, float(dist[farthest])


def _shortest_path_distance(
    adjacency: list[list[tuple[int, float]]],
    source: int,
    target: int,
) -> float:
    if source == target:
        return 0.0

    dist = [float('inf')] * len(adjacency)
    dist[source] = 0.0
    pq: list[tuple[float, int]] = [(0.0, source)]

    while pq:
        cur_dist, node = heapq.heappop(pq)
        if node == target:
            return float(cur_dist)
        if cur_dist > dist[node]:
            continue
        for nxt, w in adjacency[node]:
            nd = cur_dist + w
            if nd < dist[nxt]:
                dist[nxt] = nd
                heapq.heappush(pq, (nd, nxt))

    return float('inf')


def _nearest_node_index(coords: list[tuple[int, int]], point_xy: tuple[int, int]) -> int:
    if not coords:
        return -1
    x, y = int(point_xy[0]), int(point_xy[1])
    best_idx = 0
    best_dist = float('inf')
    for idx, (r, c) in enumerate(coords):
        d = float((c - x) ** 2 + (r - y) ** 2)
        if d < best_dist:
            best_dist = d
            best_idx = idx
    return int(best_idx)


def longest_path_length(mask: np.ndarray) -> float:
    binary = _to_binary_mask(mask)
    if binary.sum() == 0:
        return 0.0

    skel = skeletonize(binary)
    _, adjacency = _build_graph(skel)
    if not adjacency:
        return 0.0

    node_a, _ = _farthest_node(adjacency, 0)
    _, diameter = _farthest_node(adjacency, node_a)
    return float(diameter)


def path_length_between_points(
    mask: np.ndarray,
    start_xy: tuple[int, int],
    end_xy: tuple[int, int],
) -> float:
    binary = _to_binary_mask(mask)
    if binary.sum() == 0:
        return 0.0

    skel = skeletonize(binary)
    coords, adjacency = _build_graph(skel)
    if not adjacency:
        return 0.0

    s_idx = _nearest_node_index(coords, start_xy)
    t_idx = _nearest_node_index(coords, end_xy)
    if s_idx < 0 or t_idx < 0:
        return 0.0

    dist = _shortest_path_distance(adjacency, s_idx, t_idx)
    if not np.isfinite(dist):
        return 0.0
    return float(dist)


def analyze_mask(mask: np.ndarray, mm_per_px: float) -> MorphometryResult:
    px_area = area_pixels(mask)
    length_px = longest_path_length(mask)
    area_mm2 = px_area * (mm_per_px ** 2)
    length_mm = length_px * mm_per_px
    return MorphometryResult(
        area_px=px_area,
        area_mm2=float(area_mm2),
        length_px=float(length_px),
        length_mm=float(length_mm),
    )

"""Road distance matrices from OpenStreetMap via OSMnx (optional).

UrbanEV ships its own distance.csv, so this module mainly serves the
Paris instance, whose raw data only has station coordinates. If OSMnx
is unavailable or the network cannot be downloaded, the caller falls
back to great-circle distance times a detour factor (a stated
assumption; 1.3 is the common urban value in the routing literature).

The matrix is cached as an .npz next to the data, keyed by a hash of
the coordinates, so OSMnx and the network download run at most once.

Requires: pip install osmnx  (and internet access on first run)
"""

import hashlib
import os

import numpy as np


def _coords_key(lat, lon) -> str:
    h = hashlib.sha1(np.round(np.c_[lat, lon], 6).tobytes()).hexdigest()
    return h[:16]


def road_distance_km(lat, lon, cache_dir: str, network_type: str = "drive"):
    """(U, U) shortest-path road distances in km between coordinates.

    Downloads the OSM drive network covering the points (with a small
    buffer), snaps each point to its nearest graph node and runs
    multi-source Dijkstra from each node. Raises ImportError if osmnx
    is not installed and RuntimeError on download/graph failures, so
    the caller can decide how to fall back.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"roads_{_coords_key(lat, lon)}.npz")
    if os.path.exists(cache):
        return np.load(cache)["dist_km"]

    import osmnx as ox            # ImportError propagates to caller
    import networkx as nx

    pad = 0.02  # ~2 km buffer around the bounding box
    north, south = lat.max() + pad, lat.min() - pad
    east, west = lon.max() + pad, lon.min() - pad
    try:
        try:  # osmnx >= 2.0 signature (bbox tuple)
            G = ox.graph_from_bbox(bbox=(west, south, east, north),
                                   network_type=network_type)
        except TypeError:  # osmnx 1.x signature
            G = ox.graph_from_bbox(north, south, east, west,
                                   network_type=network_type)
        nodes = ox.nearest_nodes(G, X=lon.tolist(), Y=lat.tolist())
    except Exception as e:
        raise RuntimeError(f"OSMnx network build failed: {e}") from e

    n = len(lat)
    dist = np.full((n, n), np.inf)
    node_pos = {}
    for i, nd in enumerate(nodes):
        node_pos.setdefault(nd, []).append(i)
    for src, rows in node_pos.items():
        lengths = nx.single_source_dijkstra_path_length(G, src,
                                                        weight="length")
        for tgt, cols in node_pos.items():
            if tgt in lengths:
                for i in rows:
                    for j in cols:
                        dist[i, j] = lengths[tgt] / 1000.0
    np.fill_diagonal(dist, 0.0)
    # disconnected components: leave as inf -> caller may cap or fall back
    np.savez_compressed(cache, dist_km=dist)
    return dist

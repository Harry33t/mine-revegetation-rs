"""reveg — weak-supervised revegetation monitoring for mine rehabilitation.

Built on top of the vendored ``nz_canopy`` weak-label pipeline (CHM -> threshold ->
remove buildings -> morphology). ``reveg`` adds: Sentinel-2 time series ingestion,
multi-class structural/functional weak labels, a drone/aerial -> satellite scale
bridge, multi-temporal restoration trajectories, and per-pixel uncertainty.
"""

__version__ = "0.1.0"

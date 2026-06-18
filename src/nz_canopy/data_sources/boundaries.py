"""Working-extent provider.

Day-0 data has no Stats NZ urban-boundary layer, so M1 uses the data footprint
(the merged DSM bounds = the Day-0 test-tile extent) as the working "sample
polygon" the pipeline tiles over. This matches memo §14 ("clip to city boundary
OR sample polygon").
"""

from __future__ import annotations

from .elevation import RasterLayer, load_dsm

# TODO (post-M1): ingest a real Stats NZ urban-boundary polygon and clip to it,
# replacing the raster-footprint extent used here.


def working_extent(city: str | None = None, *, dsm: RasterLayer | None = None) -> tuple:
    """Return (left, bottom, right, top) for the city, derived from the DSM footprint."""
    layer = dsm if dsm is not None else load_dsm(city)
    return layer.bounds

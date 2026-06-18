from reveg import config


def test_aois_load():
    aois = config.load_aois()
    assert "alcoa_huntly" in aois
    assert "tauranga_bridge" in aois


def test_satellite_aoi_shape():
    a = config.get_aoi("alcoa_huntly")
    assert a.track == "satellite"
    assert len(a.bbox) == 4 and a.bbox[0] < a.bbox[2] and a.bbox[1] < a.bbox[3]


def test_bridge_aoi_has_city():
    b = config.get_aoi("tauranga_bridge")
    assert b.track == "bridge"
    assert b.city == "tauranga"

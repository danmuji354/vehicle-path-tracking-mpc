import json
from pathlib import Path

from PIL import Image

from vehicle_mpc.gallery import gallery_contract


def test_gallery_contract_has_all_asset_roles():
    nominal = {"lateral_rmse_m": 0.3, "constraint_violations": 0, "median_solve_time_ms": 12.0}
    contract = gallery_contract({"mpc": {"nominal": nominal}})
    assert contract["schema_version"] == 1
    assert {asset["role"] for asset in contract["assets"]} == {"hero", "analysis", "animation", "diagram"}


def test_checked_in_gallery_matches_website_contract():
    gallery = Path(__file__).parents[1] / "artifacts" / "gallery"
    contract = json.loads((gallery / "showcase.json").read_text())
    for asset in contract["assets"]:
        path = gallery / asset["path"]
        assert path.is_file()
        if path.suffix.lower() in {".png", ".gif"}:
            with Image.open(path) as image:
                assert image.size == (asset["width"], asset["height"])

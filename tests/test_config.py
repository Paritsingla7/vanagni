"""Config loading: region files merge over shared defaults, and a region can
never silently inherit another region's bbox, projection, calendar or OSM source."""
import pytest
import yaml

from wfpin.config import REQUIRED_REGION_KEYS, _deep_merge, _lookup, load_config

REGION = {
    "region": {"name": "x", "state_id": "XX",
               "bbox": {"north": 2, "south": 1, "west": 1, "east": 2}},
    "season": {"calendar_csv": "c.csv"},
    "static_layers": {"osm": {"pbf_url": "http://u", "state_name": "X"}},
}


def _write(tmp_path, region, defaults=None):
    (tmp_path / "x.yaml").write_text(yaml.safe_dump(region))
    if defaults is not None:
        (tmp_path / "defaults.yaml").write_text(yaml.safe_dump(defaults))
    return str(tmp_path / "x.yaml")


def test_defaults_never_contain_region_specific_settings():
    defaults = yaml.safe_load(open("config/defaults.yaml"))
    leaked = [key for key in REQUIRED_REGION_KEYS if _lookup(defaults, key) is not None]
    assert not leaked, f"region-specific settings in defaults.yaml: {leaked}"
    for key in ("region", "paths", "season"):
        assert key not in defaults


def test_region_file_overrides_defaults_and_merges_nested_dicts(tmp_path):
    defaults = {"grid": {"resolution_km": 5}, "evaluation": {"test_seasons": [1, 2], "k": 20},
                "static_layers": {"osm": {"district_admin_level": "5"}}}
    region = {**REGION, "evaluation": {"k": 10}}
    config = load_config(_write(tmp_path, region, defaults))
    assert config["grid"]["resolution_km"] == 5                    # inherited
    assert config["evaluation"] == {"test_seasons": [1, 2], "k": 10}  # merged key by key
    assert config["static_layers"]["osm"] == {"district_admin_level": "5", "pbf_url": "http://u",
                                              "state_name": "X"}


def test_lists_are_replaced_not_concatenated():
    assert _deep_merge({"a": [1, 2]}, {"a": [3]}) == {"a": [3]}


@pytest.mark.parametrize("dotted", REQUIRED_REGION_KEYS)
def test_missing_regional_setting_fails_loudly(tmp_path, dotted):
    region = yaml.safe_load(yaml.safe_dump(REGION))
    *parents, leaf = dotted.split(".")
    node = region
    for part in parents:
        node = node[part]
    del node[leaf]
    with pytest.raises(ValueError, match=dotted):
        load_config(_write(tmp_path, region, {"grid": {"resolution_km": 5}}))


def test_a_complete_standalone_config_needs_no_defaults_file(tmp_path):
    assert load_config(_write(tmp_path, REGION))["region"]["name"] == "x"


def test_uttarakhand_still_loads_with_its_known_values():
    config = load_config()
    assert config["region"]["name"] == "uttarakhand" and config["grid"]["resolution_km"] == 5
    assert config["paths"] == {"raw": "data/raw", "processed": "data/processed"}
    assert "3f" in config["ablation"]["stages"]

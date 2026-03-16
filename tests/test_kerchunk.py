import pytest

from healpix_regrid.kerchunk import patch_kc_refs_inplace

HPC = "/scale/project/lops-oh-fair2adapt/"
HTTPS = "https://data-fair2adapt.ifremer.fr/"


def _make_refs(**extra):
    """Return a minimal Kerchunk reference dict."""
    base = {
        "version": 1,
        "refs": {
            ".zgroup": '{"zarr_format":2}',
            "temp/0.0": [f"{HPC}riomar/data/file.nc", 1024, 512],
            "salt/0.0": [f"{HPC}riomar/data/other.nc", 0, 256],
            "inline_key": "some inline value",
        },
    }
    base.update(extra)
    return base


class TestPatchKcRefsInplace:
    def test_rewrites_list_targets(self):
        kc = _make_refs()
        result = patch_kc_refs_inplace(kc)

        assert result is kc  # mutated in place
        assert kc["refs"]["temp/0.0"][0] == f"{HTTPS}riomar/data/file.nc"
        assert kc["refs"]["salt/0.0"][0] == f"{HTTPS}riomar/data/other.nc"

    def test_preserves_offsets(self):
        kc = _make_refs()
        patch_kc_refs_inplace(kc)

        assert kc["refs"]["temp/0.0"][1:] == [1024, 512]

    def test_rewrites_string_targets(self):
        kc = _make_refs()
        kc["refs"]["meta_key"] = f"{HPC}riomar/meta.json"
        patch_kc_refs_inplace(kc)

        assert kc["refs"]["meta_key"] == f"{HTTPS}riomar/meta.json"

    def test_ignores_non_matching_paths(self):
        kc = _make_refs()
        kc["refs"]["other"] = ["/other/path/file.nc", 0, 10]
        patch_kc_refs_inplace(kc)

        assert kc["refs"]["other"][0] == "/other/path/file.nc"

    def test_leaves_inline_values_alone(self):
        kc = _make_refs()
        patch_kc_refs_inplace(kc)

        assert kc["refs"]["inline_key"] == "some inline value"

    def test_raises_on_missing_refs(self):
        with pytest.raises(KeyError, match="refs"):
            patch_kc_refs_inplace({"version": 1})

    def test_supports_references_key(self):
        kc = {
            "version": 1,
            "references": {
                "temp/0.0": [f"{HPC}riomar/data/file.nc", 0, 100],
            },
        }
        patch_kc_refs_inplace(kc)

        assert kc["refs"]["temp/0.0"][0].startswith(HTTPS)

    def test_custom_prefixes(self):
        kc = {
            "refs": {"data/0": ["/my/hpc/path/file.nc", 0, 10]},
        }
        patch_kc_refs_inplace(
            kc, hpc_prefix="/my/hpc/path/", https_prefix="https://example.com/"
        )

        assert kc["refs"]["data/0"][0] == "https://example.com/file.nc"

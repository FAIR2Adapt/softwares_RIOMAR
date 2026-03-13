from __future__ import annotations

HPC_PREFIX = "/scale/project/lops-oh-fair2adapt/"
HTTPS_PREFIX = "https://data-fair2adapt.ifremer.fr/"


def patch_kc_refs_inplace(
    kc: dict,
    hpc_prefix: str = HPC_PREFIX,
    https_prefix: str = HTTPS_PREFIX,
) -> dict:
    """Rewrite Kerchunk references so HPC filesystem paths become HTTPS URLs.

    Parameters
    ----------
    kc : dict
        Kerchunk reference dictionary (must contain a ``"refs"`` or
        ``"references"`` key).
    hpc_prefix : str
        Path prefix to replace (e.g. ``"/scale/project/.../"``).
    https_prefix : str
        URL prefix to substitute (e.g. ``"https://data-fair2adapt.ifremer.fr/"``).

    Returns
    -------
    dict
        The same *kc* dictionary, mutated in place for efficiency.
    """
    refs = kc.get("refs", kc.get("references"))
    if refs is None:
        raise KeyError("Can't find 'refs' (or 'references') in kerchunk JSON")

    def _patch(x: str) -> str:
        if isinstance(x, str) and x.startswith(hpc_prefix):
            return https_prefix + x[len(hpc_prefix) :]
        return x

    for k, v in list(refs.items()):
        if isinstance(v, list) and v and isinstance(v[0], str):
            refs[k] = [_patch(v[0])] + v[1:]
        elif isinstance(v, str):
            refs[k] = _patch(v)

    kc["refs"] = refs
    return kc

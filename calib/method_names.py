"""Paper-facing method names and legacy aliases."""

PAPER_METHOD_ALIASES = {
    "Proxy_BOCPD": "B-BRPC-P",
    "Proxy_wCUSUM": "C-BRPC-P",
    "Proxy_None": "BRPC-P",
    "Exact_BOCPD": "B-BRPC-E",
    "Exact_wCUSUM": "C-BRPC-E",
    "Exact_None": "BRPC-E",
    "FixedSupport_BOCPD": "B-BRPC-F",
    "FixedSupport_wCUSUM": "C-BRPC-F",
    "FixedSupport_None": "BRPC-F",
    "HalfRefit_BOCPD": "B-BRPC-RRA",
    "HalfRefit": "B-BRPC-RRA",
    "WardPFMove_BOCPD": "B-WaldPF",
    "SlidingWindow-KOH": "BC",
    "JointEnKF": "EnKF",
}

PAPER_METHODS = set(PAPER_METHOD_ALIASES.values())


def paper_method_name(name: str) -> str:
    return PAPER_METHOD_ALIASES.get(str(name), str(name))


def method_aliases() -> dict[str, str]:
    return dict(PAPER_METHOD_ALIASES)

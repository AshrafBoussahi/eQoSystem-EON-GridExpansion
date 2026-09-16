"""Planning problems: how a grid becomes a quadratic binary program.

Modules
-------
siting      Builds the storage-siting family from DC-OPF prices and congestion
            duals. One capacity tier is a 25 MW / 100 MWh four-hour battery
            block; tiers are carried by a domain-wall chain per bus.
microgrid   Extends an instance with one islanding binary per candidate bus,
            priced from that bus's measured outage exposure.
scenarios   Folds a weather scenario set into the coefficients, which is why
            scenario count costs nothing at the quantum layer.
registry    Loads the frozen benchmark instances and their certified optima.
domainwall  Encoding helpers for the monotone capacity chains.
"""
from qgridx.problems.siting import build_instance, QUBOInstance
from qgridx.problems.microgrid import (
    extend_with_islanding,
    bus_outage_exposure,
    polish_free_bits,
    BESS_DURATION_H,
    ISLAND_CAPEX,
    ISLAND_VALUE,
)
from qgridx.problems.registry import (
    load_registry, rebuild_instance, SCREENED_BUSES, REGISTRY_DEFAULTS,
)

__all__ = [
    "build_instance", "QUBOInstance",
    "extend_with_islanding", "bus_outage_exposure", "polish_free_bits",
    "BESS_DURATION_H", "ISLAND_CAPEX", "ISLAND_VALUE",
    "load_registry", "rebuild_instance", "SCREENED_BUSES", "REGISTRY_DEFAULTS",
]

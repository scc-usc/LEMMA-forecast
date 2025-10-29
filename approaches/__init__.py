from importlib import import_module

# Map friendly names to module paths
_REGISTRY = {
    # SIKJalpha (old var_un_basic)
    "SIKJalpha_basic": "approaches.sikjalpha_basic",
    "SIKJalpha Basic": "approaches.sikjalpha_basic",
    # Additional approaches
    "ARIMA": "approaches.arima_basic",
    "Flatline": "approaches.flatline",
}


def get_approach(name: str):
    key = name or "SIKJalpha_basic"
    mod_path = _REGISTRY.get(key, _REGISTRY["SIKJalpha_basic"])
    return import_module(mod_path)

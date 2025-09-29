"""
Created on Mon Sept 8 2025

config.py - Functional configuration helpers for the fusion CLI/wrapper.

Functions:
    - make_dims(time='time', y='lat', x='lon') -> dict
    - make_params(width, exponent=2.0, mask_mode='L3', log_mode='none',
                  boundary='zero', dims=None, verbose=False) -> dict
    - make_vars(l3_var, template_var) -> dict
    - make_io(l3_path, template_path, output_path, zlib=True, complevel=4) -> dict
    - nc_encoding(io_cfg) -> dict (NetCDF encoding for outputs)
"""

# Libraries
from typing import Dict, Any

# Functions
def make_dims(time: str = "time", y: str = "lat", x: str = "lon") -> Dict[str, str]:
    return {"time": time, "y": y, "x": x}


def make_params(
    width: int,
    exponent: float = 2.0,
    mask_mode: str = "L3",
    log_mode: str = "none",
    boundary: str = "zero",
    dims: Dict[str, str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    if dims is None:
        dims = make_dims()
    return {
        "width": int(width),
        "exponent": float(exponent),
        "mask_mode": str(mask_mode),
        "log_mode": str(log_mode),
        "boundary": str(boundary),
        "dims": dims,
        "verbose": bool(verbose),
    }


def make_vars(l3_var: str, template_var: str) -> Dict[str, str]:
    return {"l3_var": l3_var, "template_var": template_var}


def make_io(
    l3_path: str,
    template_path: str,
    output_path: str,
    zlib: bool = True,
    complevel: int = 4,
) -> Dict[str, Any]:
    return {
        "l3_path": l3_path,
        "template_path": template_path,
        "output_path": output_path,
        "zlib": bool(zlib),
        "complevel": int(complevel),
    }


def nc_encoding(io_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not io_cfg.get("zlib", False):
        return {}
    lvl = int(io_cfg.get("complevel", 4))
    return {
        "L4": {"zlib": True, "complevel": lvl},
        "a": {"zlib": True, "complevel": lvl},
        "b": {"zlib": True, "complevel": lvl},
        "rho": {"zlib": True, "complevel": lvl},
        "err": {"zlib": True, "complevel": lvl},
    }

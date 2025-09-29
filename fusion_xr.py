"""
Created on Mon Sept 8 2025

fusion_xr.py - xarray wrapper around core fusion routines.

Responsibilities:
    - Reorder dims to [time?, y, x] based on provided names
    - Build SCOPE-based kernel from parameters
    - Call core fusion with configurable boundary handling and wrap results as
      xarray.Dataset

Functions:
    - build_kernel(params: dict) -> np.ndarray
    - fusion_xr(da_L3, da_T, params: dict, kernel) -> xr.Dataset
"""

# Libraries
import logging
from typing import Dict, Any
import numpy as np
import xarray as xr
from .fusion import euklid_invW, fusion

# Module logger
logger = logging.getLogger(__name__)

# Functions
def build_kernel(params: Dict[str, Any]) -> np.ndarray:
    size = 2 * int(params["width"]) + 1
    return euklid_invW((size, size), width=int(params["width"]), exponent=float(params.get("exponent", 2.0)))


def _order_dims(da: "xr.DataArray", params: Dict[str, Any]) -> "xr.DataArray":
    # Try to transpose to (time?, y, x) if time present
    d = da.dims
    dims = params["dims"]
    have_t = dims["time"] in d
    order = (dims["time"], dims["y"], dims["x"]) if have_t else (dims["y"], dims["x"])
    return da.transpose(*order)


def fusion_xr(
    da_L3: "xr.DataArray",
    da_T: "xr.DataArray",
    params: Dict[str, Any],
    kernel: np.ndarray,
) -> "xr.Dataset":
    """
    Run fusion on xarray DataArrays, preserving coords/attrs in the output.

    Returns an xarray.Dataset with variables: L4, a, b, rho, err.
    """

    verbose = bool(params.get("verbose", False))
    if verbose:
        logger.info("[fusion_xr] Preparing inputs and kernel...")
    # Ensure dims order
    da_L3o = _order_dims(da_L3, params)
    da_To = _order_dims(da_T, params)
    if da_L3o.shape != da_To.shape:
        raise ValueError("L3 and template arrays must match in shape after reordering dims")

    # Kernel is provided by the caller (built once in the CLI)
    ker = kernel
    if verbose:
        logger.info(
            "[fusion_xr] Using provided kernel of shape %s (width=%s, exponent=%s, boundary=%s)",
            ker.shape,
            params["width"],
            params.get("exponent", 2.0),
            params.get("boundary", "zero"),
        )

    # Run core fusion
    if verbose:
        dims = params["dims"]
        has_t = dims["time"] in da_L3o.dims
        if has_t:
            nt = da_L3o.sizes[dims["time"]]
            logger.info(f"[fusion_xr] Running fusion on 3D stack: nt={nt}, ny={da_L3o.sizes[dims['y']]}, nx={da_L3o.sizes[dims['x']]}...")
        else:
            logger.info(f"[fusion_xr] Running fusion on 2D field: ny={da_L3o.sizes[dims['y']]}, nx={da_L3o.sizes[dims['x']]}...")
    L4, AA, BB, RR, ERR = fusion(
        da_L3o.data,
        da_To.data,
        ker,
        mask_mode=str(params.get("mask_mode", "L3")),
        log_mode=str(params.get("log_mode", "none")),
        boundary=str(params.get("boundary", "zero")),
        debug=bool(params.get("verbose", False)),
    )

    # Wrap outputs in DataArrays, copying coords and dims
    coords = da_L3o.coords
    dims = da_L3o.dims

    da_L4 = xr.DataArray(L4, coords=coords, dims=dims, name="L4")
    da_AA = xr.DataArray(AA, coords=coords, dims=dims, name="a")
    da_BB = xr.DataArray(BB, coords=coords, dims=dims, name="b")
    da_RR = xr.DataArray(RR, coords=coords, dims=dims, name="rho")
    da_ERR = xr.DataArray(ERR, coords=coords, dims=dims, name="err")

    # Optionally set minimal metadata
    if hasattr(da_L3o, 'attrs'):
        if 'units' in da_L3o.attrs:
            da_L4.attrs['units'] = da_L3o.attrs['units']
    da_AA.attrs['long_name'] = 'local_slope'
    da_BB.attrs['long_name'] = 'local_intercept'
    da_RR.attrs['long_name'] = 'local_correlation'
    da_ERR.attrs['long_name'] = 'local_error_std'

    ds = xr.Dataset({"L4": da_L4, "a": da_AA, "b": da_BB, "rho": da_RR, "err": da_ERR})

    return ds

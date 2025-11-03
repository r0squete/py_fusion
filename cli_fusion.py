#!/usr/bin/env python3
"""
Created on Mon Sept 8 2025

CLI for running fusion on NetCDF files using xarray.

Usage (repository root):
  python -m py_code.cli_fusion \
      --l3 path_to_L3.nc --l3-var L3_varname \
      --template path_to_template.nc --template-var T_varname \
      --width 20 --exponent 2 \
      --mask-mode L3 --log-mode none --boundary zero \
      --output "/out_dir/fused_{date}.nc"

  python -m py_code.cli_fusion \
      --l3 "L3_dir/*.nc" --l3-var L3_varname \
      --template "T_dir/*.nc" --template-var T_varname \
      --boundary reflect \
      --output "/out_dir/fused_{index}.nc"

  python -m py_code.cli_fusion \
      --l3 "L3_dir/*.nc" --l3-var L3_varname --l3-y-dim latitude --l3-x-dim longitude --l3-time-dim time \
      --template "T_dir/*.nc" --template-var T_varname --t-y-dim lat --t-x-dim lon --t-time-dim time \
      --output "/out_dir/daily_SST_{l3_name}_{date}.nc"
"""

# Libraries
import argparse
import glob
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from py_fusion.config import make_dims, make_params, make_vars, make_io, nc_encoding
from py_fusion.fusion_xr import build_kernel, fusion_xr

# Module logger
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run fusion on NetCDF inputs (xarray-based)")
    p.add_argument("--l3", required=True, dest="l3_globs", nargs='+',
                   help="One or more file paths or glob patterns for L3 inputs")
    p.add_argument("--l3-var", required=True, dest="l3_var")
    p.add_argument("--template", required=True, dest="template_globs", nargs='+',
                   help="One or more file paths or glob patterns for template inputs")
    p.add_argument("--template-var", required=True, dest="template_var")
    p.add_argument("--width", type=int, default=20,
                   help="SCOPE semi-width in pixels (default: 20)")
    p.add_argument("--exponent", type=float, default=2.0,
                   help="Power-law exponent (default: 2.0)")
    p.add_argument("--mask-mode", choices=["L3", "template", "both"], default="L3")
    p.add_argument("--log-mode", choices=["none", "L3", "template", "both"], default="none")
    p.add_argument("--boundary", choices=["zero", "reflect"], default="zero",
                   help="Boundary handling for convolutions (default: zero padding)")
    p.add_argument("--l3-time-dim", default=None, help="L3 time dim name (mapped to 'time'; default: time)")
    p.add_argument("--l3-y-dim", default=None, help="L3 Y dim name (mapped to 'lat'; default: lat)")
    p.add_argument("--l3-x-dim", default=None, help="L3 X dim name (mapped to 'lon'; default: lon)")
    p.add_argument("--t-time-dim", default=None, help="Template time dim name (mapped to 'time'; default: time)")
    p.add_argument("--t-y-dim", default=None, help="Template Y dim name (mapped to 'lat'; default: lat)")
    p.add_argument("--t-x-dim", default=None, help="Template X dim name (mapped to 'lon'; default: lon)")
    p.add_argument("--output", required=True, dest="output_path",
                   help="Output path pattern; available placeholders: {index}, {date}, {l3_name}, {template_name}")
    p.add_argument("--no-zlib", action="store_true")
    p.add_argument("--complevel", type=int, default=4)
    p.add_argument("--verbose", action="store_true", help="Print progress information")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Configure logging for library modules (fusion), controlled by --verbose
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    # Canonical names are fixed internally: time/lat/lon
    dims = make_dims(time="time", y="lat", x="lon")
    params = make_params(width=args.width, exponent=args.exponent,
                         mask_mode=args.mask_mode, log_mode=args.log_mode,
                         boundary=args.boundary,
                         dims=dims, verbose=args.verbose)
    vars_cfg = make_vars(l3_var=args.l3_var, template_var=args.template_var)
    io_cfg = make_io(l3_path="", template_path="",
                     output_path=args.output_path, zlib=(not args.no_zlib),
                     complevel=args.complevel)

    # Expand globs
    def expand_many(patterns):
        files = []
        for pat in patterns:
            files.extend(glob.glob(pat))
        return sorted(set(files))

    l3_files = expand_many(args.l3_globs)
    t_files = expand_many(args.template_globs)
    if args.verbose:
        logger.info(f"[cli] L3 files: {len(l3_files)} | Template files: {len(t_files)}")
    if not l3_files:
        raise FileNotFoundError("No L3 files matched the provided patterns")
    if not t_files:
        raise FileNotFoundError("No template files matched the provided patterns")
    if len(l3_files) != len(t_files):
        raise ValueError(f"Number of L3 files ({len(l3_files)}) does not match template files ({len(t_files)})")

    # Preprocess: keep only target var (coords/dims are preserved)
    def _pp_l3(ds):
        if args.l3_var not in ds:
            raise KeyError(f"L3 var '{args.l3_var}' not found in one of the files")
        return ds[[args.l3_var]]

    def _pp_t(ds):
        if args.template_var not in ds:
            raise KeyError(f"Template var '{args.template_var}' not found in one of the files")
        return ds[[args.template_var]]

    # Resolve per-dataset dim names (fallback to canonical)
    canon_t, canon_y, canon_x = "time", "lat", "lon"
    l3_t = args.l3_time_dim
    l3_y = args.l3_y_dim
    l3_x = args.l3_x_dim
    t_t  = args.t_time_dim
    t_y  = args.t_y_dim
    t_x  = args.t_x_dim

    # Helper to rename dims/coords to canonical names if needed
    def _rename_to_canon(ds, src_t, src_y, src_x, label):
        mapping = {}
        if src_t != canon_t:
            mapping[src_t] = canon_t
        if src_y != canon_y:
            mapping[src_y] = canon_y
        if src_x != canon_x:
            mapping[src_x] = canon_x
        if mapping:
            if args.verbose:
                logger.info(f"[cli] Renaming {label} dims {mapping} -> canon ({canon_t},{canon_y},{canon_x})")
            # Only include keys that exist to avoid xarray rename errors
            mapping = {k: v for k, v in mapping.items() if (k in ds.dims or k in ds.coords or k in ds.variables)}
            ds = ds.rename(mapping)
        return ds

    # Ensure rectilinear 1D coords if provided as 2D (tolerate 1D/2D representations)
    def _rectilinearize(ds, label):
        if "lat" in ds.coords and getattr(ds["lat"], "ndim", 1) == 2 and "lon" in ds.coords and ds["lon"].ndim == 2:
            if args.verbose:
                logger.info(f"[cli] {label}: converting 2D lat/lon coords to 1D axes (rectilinear check)")
            lat2d = ds["lat"].values
            lon2d = ds["lon"].values
            # Check rectilinear: lon varies along X only, lat along Y only (tolerant)
            ok_lon = np.allclose(lon2d, lon2d[0:1, :], atol=1e-2, rtol=0.0)
            ok_lat = np.allclose(lat2d, lat2d[:, 0:1], atol=1e-2, rtol=0.0)
            if not (ok_lon and ok_lat):
                raise ValueError(f"[{label}] non-rectilinear 2D lat/lon; please regrid to rectilinear grid")
            lat1d = lat2d[:, 0]
            lon1d = lon2d[0, :]
            ds = ds.assign_coords(lat=("lat", lat1d), lon=("lon", lon1d))
        return ds

    def _get_coord(ds, dim):
        if dim in ds.coords:
            return ds[dim].values
        # If no coord var, fail per requirement
        raise ValueError(f"Missing coordinate variable for dim '{dim}'")

    def _normalize_time_index(ds, dim):
        if dim in ds.indexes:
            raw = ds.indexes[dim]
            values = getattr(raw, "values", raw)
        else:
            values = ds[dim].values
        timestamps = []
        for val in values:
            try:
                ts = pd.Timestamp(val)
            except Exception:
                ts = pd.Timestamp(str(val))
            if ts.tz is not None:
                ts = ts.tz_convert(None)
            timestamps.append(ts.normalize())
        return pd.DatetimeIndex(timestamps, name=dim)

    def _mono_unique(idx, label):
        if not isinstance(idx, pd.Index):
            idx = pd.Index(idx)
        if not idx.is_monotonic_increasing:
            raise ValueError(f"[{label}] time coordinate not strictly increasing")
        if not idx.is_unique:
            raise ValueError(f"[{label}] time coordinate contains duplicates")

    # Build kernel once; reuse for every pair
    if args.verbose:
        logger.info(f"[cli] Building kernel with width={params['width']} exponent={params['exponent']}...")
    kernel = build_kernel(params)

    # Prepare output pattern
    output_template = Path(io_cfg["output_path"])
    if output_template.name == "":
        raise ValueError("--output must include a filename or pattern (not just a directory path)")
    out_dir = output_template.parent if output_template.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern_name = output_template.name
    available_keys = ("index", "date", "l3_name", "template_name")

    def _format_date(idx):
        if len(idx) == 0:
            return "no_time"
        first = idx[0]
        if isinstance(first, (pd.Timestamp, np.datetime64)):
            return pd.Timestamp(first).strftime("%Y%m%d")
        if hasattr(first, "strftime"):
            return first.strftime("%Y%m%d")
        return str(first)

    tdim, ydim, xdim = canon_t, canon_y, canon_x
    saved_paths = []
    total_pairs = len(l3_files)
    for idx, (l3_path, t_path) in enumerate(zip(l3_files, t_files), start=1):
        if args.verbose:
            logger.info(f"[cli] Processing pair {idx}/{total_pairs}: L3='{l3_path}' | Template='{t_path}'")
        with xr.open_dataset(l3_path) as ds_L3_raw, xr.open_dataset(t_path) as ds_T_raw:
            ds_L3 = _pp_l3(ds_L3_raw)
            ds_T = _pp_t(ds_T_raw)

            ds_L3 = _rename_to_canon(ds_L3, l3_t, l3_y, l3_x, "L3")
            ds_T = _rename_to_canon(ds_T, t_t, t_y, t_x, "template")

            ds_L3 = _rectilinearize(ds_L3, "L3")
            ds_T = _rectilinearize(ds_T, "template")

            da_L3 = ds_L3[vars_cfg["l3_var"]]
            da_T = ds_T[vars_cfg["template_var"]]

            for name, da in (("L3", da_L3), ("template", da_T)):
                for dim in (tdim, ydim, xdim):
                    if dim not in da.dims:
                        raise ValueError(f"[{name}] missing required dim '{dim}' in variable '{da.name}'")

            lat_L3 = _get_coord(da_L3.to_dataset(name="_tmp"), ydim)
            lon_L3 = _get_coord(da_L3.to_dataset(name="_tmp"), xdim)
            lat_T = _get_coord(da_T.to_dataset(name="_tmp"), ydim)
            lon_T = _get_coord(da_T.to_dataset(name="_tmp"), xdim)
            if lat_L3.shape != lat_T.shape or lon_L3.shape != lon_T.shape:
                raise ValueError("Latitude/Longitude shapes differ between L3 and template")
            if not (np.allclose(lat_L3, lat_T, atol=1e-2, rtol=0.0) and np.allclose(lon_L3, lon_T, atol=1e-2, rtol=0.0)):
                raise ValueError("Latitude/Longitude coordinate values differ between L3 and template")

            time_idx_L3 = _normalize_time_index(ds_L3, tdim)
            time_idx_T = _normalize_time_index(ds_T, tdim)
            _mono_unique(time_idx_L3, "L3")
            _mono_unique(time_idx_T, "template")

            common_times = time_idx_L3.intersection(time_idx_T)
            if len(common_times) == 0:
                l3_start = time_idx_L3[0] if len(time_idx_L3) else "empty"
                l3_end = time_idx_L3[-1] if len(time_idx_L3) else "empty"
                tpl_start = time_idx_T[0] if len(time_idx_T) else "empty"
                tpl_end = time_idx_T[-1] if len(time_idx_T) else "empty"
                raise ValueError(
                    "No overlapping time steps between L3 and template after normalization "
                    f"(L3 range: {l3_start}..{l3_end}; template range: {tpl_start}..{tpl_end})"
                )
            if hasattr(common_times, "sort_values"):
                common_times = common_times.sort_values()
            else:
                common_times = pd.Index(sorted(common_times))

            if (len(common_times) != len(time_idx_L3) or len(common_times) != len(time_idx_T)) and args.verbose:
                logger.info(
                    "[cli] Aligning time axes: L3=%d, template=%d, common=%d",
                    len(time_idx_L3), len(time_idx_T), len(common_times),
                )

            da_L3 = ds_L3.reindex({tdim: common_times})[vars_cfg["l3_var"]]
            da_T = ds_T.reindex({tdim: common_times})[vars_cfg["template_var"]]

            aligned_times = common_times

            if args.verbose:
                dims = params["dims"]
                has_t = dims["time"] in da_L3.dims
                if has_t:
                    nt = da_L3.sizes[dims["time"]]
                    logger.info(f"[cli] Running fusion on 3D stack: nt={nt}, ny={da_L3.sizes[dims['y']]}, nx={da_L3.sizes[dims['x']]}...")
                else:
                    logger.info(f"[cli] Running fusion on 2D field: ny={da_L3.sizes[dims['y']]}, nx={da_L3.sizes[dims['x']]}...")
            out = fusion_xr(da_L3, da_T, params, kernel=kernel)

            date_label = _format_date(aligned_times)
            context = {
                "index": idx,
                "date": date_label,
                "l3_name": Path(l3_path).stem,
                "template_name": Path(t_path).stem,
            }
            try:
                file_name = pattern_name.format(**context)
            except KeyError as exc:
                exc_key = exc.args[0]
                raise KeyError(
                    f"Placeholder '{{{exc_key}}}' not available in --output pattern. "
                    f"Use only {available_keys}"
                ) from exc
            pair_out = out_dir / file_name

            if args.verbose:
                logger.info(f"[cli] Saving output to {pair_out}...")
            pair_io_cfg = dict(io_cfg)
            pair_io_cfg["output_path"] = str(pair_out)
            encoding = nc_encoding(pair_io_cfg)
            out.to_netcdf(pair_out, encoding=encoding)

            saved_paths.append(pair_out)
            print(f"Saved: {pair_out} (L3={l3_path}, template={t_path})")

    if args.verbose:
        logger.info(f"[cli] Generated {len(saved_paths)} output files.")


if __name__ == "__main__":  
    raise SystemExit(main())

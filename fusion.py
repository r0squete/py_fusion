"""
Created on Mon Sept 8 2025

fusion.py - Core numerical routines for local/regional field fusion.

Functions:
    - euklid_invW(shape, width=20, exponent=2.0):
        Rectangular (SCOPE-based) inverse-distance kernel with zero center weight.
        No global normalization (local normalization happens inside fusion). For a
        well-centered kernel, use odd shapes (e.g., (2*width+1, 2*width+1)).
    - fusion(L3, template, kernel, mask_mode='L3', log_mode='none', boundary='zero', debug=False):
        Spatially weighted local linear regression (2D/3D, NaN-aware) using zero-padded
        linear convolution with 'same' output shape (scipy.signal.fftconvolve). If
        boundary='reflect', inputs are mirrored prior to convolution to preserve edge
        statistics. If debug=True, prints a brief summary of rare-case counts and ranges.

Notes:
    - Stability guards and degenerate cases:
        * If Var(template) <= VAR_EPS (default 1e-10): a=0; b=X puntual; rho=0; err=0.
        * If Var(L3) is tiny and Var(template) is not: rho=1 so that err=0.
        * Pixels with insufficient local support (Nev < NEV_MIN_FRAC*sum(kernel)) are set to NaN.
    - Numerical tolerances: TOL=1e-20 for masks/denominators; VAR_EPS and NEV_MIN_FRAC are
      module-level constants and can be adjusted if needed.
    - log_mode in {'none','L3','template','both'}: logs are applied before fusion and, if L3 is logged,
      the fused output L4 is re-exponentiated at the end.

Example usage:
    from py_code.fusion import euklid_invW, fusion
    scope = 20  # semi-width (pixels)
    kernel = euklid_invW((2*scope+1, 2*scope+1), width=scope, exponent=2.0)
    L4, AA, BB, RR, ERR = fusion(L3, template, kernel,
                                 mask_mode='L3', log_mode='none', boundary='reflect')

    # 3D stack (time or depth as first dim)
    L4s, AAs, BBs, RRs, ERRs = fusion(L3_stack, template_stack, kernel,
                                      mask_mode='L3', log_mode='both')
"""

# Constants
TOL = 1e-20  # Generic numerical tolerance used for masks/denominators
VAR_EPS = 1e-10  # Threshold for "tiny" variance
NEV_MIN_FRAC = 0.2  # Minimum fraction of kernel support required for a valid estimate


# Libraries
import logging
import numpy as np
from scipy.signal import fftconvolve

# Module logger
logger = logging.getLogger(__name__)


# Functions
def euklid_invW(shape, width=1, exponent=2.0):
    """
    Generate a kernel of inverse Euclidean distance weights.
    Args:
        shape: tuple, kernel size (ny, nx)
        width: int, semi-width (SCOPE) in pixels 
        exponent: float, power law exponent (default 2.0)
    Returns:
        kernel: np.ndarray
    """
    ny, nx = shape
    cy, cx = ny // 2, nx // 2
    y, x = np.ogrid[:ny, :nx]
    dy = np.abs(y - cy)
    dx = np.abs(x - cx)
    support = (dy <= width) & (dx <= width)
    support[cy, cx] = False  # zero weight at center
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    kernel = np.zeros((ny, nx), dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        kernel[support] = 1.0 / (dist[support] ** exponent)
    return kernel


def fusion(
    L3,
    template,
    kernel,
    mask_mode='L3',
    log_mode='none',
    boundary='zero',
    debug=False,
):
    """
    Local/regional fusion of two fields using spatially weighted linear regression.
    Args:
        L3: np.ndarray, base field (2D [y,x] or 3D [t,y,x])
        template: np.ndarray, template field (same shape as L3)
        kernel: np.ndarray, spatial weights kernel (2D)
        mask_mode: 'template' | 'L3' | 'both' (output mask definition)
        log_mode: 'none' | 'L3' | 'template' | 'both'
        boundary: 'zero' | 'reflect' (padding strategy for convolutions)
        debug: bool, if True prints a short summary of rare-case counts and ranges
    Returns:
        L4: fused field
        AA: local slope (a)
        BB: local intercept (b)
        RR: local correlation
        ERR: local error (standard deviation of residuals)
    """

    # Optionally log-transform inputs
    if log_mode not in ('none', 'L3', 'template', 'both'):
        raise ValueError("log_mode must be one of: 'none','L3','template','both'")
    if log_mode in ('L3', 'both'):
        L3 = np.log10(np.where(L3 > 0, L3, np.nan))
    if log_mode in ('template', 'both'):
        template = np.log10(np.where(template > 0, template, np.nan))

    # Support 2D or 3D stacks by operating per 2D slice
    L3 = np.asarray(L3)
    template = np.asarray(template)
    if L3.shape != template.shape:
        raise ValueError('L3 and template must have the same shape')

    if boundary not in ('zero', 'reflect'):
        raise ValueError("boundary must be one of: 'zero','reflect'")

    # Convolution helper (linear 'same' semantics with configurable boundary)
    if boundary == 'zero':
        def conv_same(a2d: np.ndarray) -> np.ndarray:
            return fftconvolve(a2d, kernel, mode='same')
    else:
        pad_y = kernel.shape[0] // 2
        pad_x = kernel.shape[1] // 2

        def conv_same(a2d: np.ndarray) -> np.ndarray:
            if pad_y == 0 and pad_x == 0:
                return fftconvolve(a2d, kernel, mode='same')
            padded = np.pad(a2d, ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect')
            return fftconvolve(padded, kernel, mode='valid')

    # Precompute constants used inside each slice
    ksum = float(np.sum(kernel)) if np.isfinite(kernel).all() else float(np.nansum(kernel))
    min_support = NEV_MIN_FRAC * ksum

    def _process_slice(L3_2d: np.ndarray, T_2d: np.ndarray):
        mask_L3 = (~np.isnan(L3_2d)).astype(float)
        mask_template = (~np.isnan(T_2d)).astype(float)
        mask_merged = (mask_L3 >= 0.5) & (mask_template >= 0.5)
        mask_merged_f = mask_merged.astype(float)

        # Zero-fill invalids before convolution
        L3z = np.where(mask_merged, L3_2d, 0.0)
        Tz = np.where(mask_merged, T_2d, 0.0)

        # Convolutions with the selected boundary handling
        Nev = conv_same(mask_merged_f)
        # Require a minimum effective support to compute local statistics
        Nev_safe = np.where(Nev >= min_support, Nev, np.nan)
        L3_m = conv_same(L3z) / Nev_safe
        T_m = conv_same(Tz) / Nev_safe
        L3_m2 = conv_same(L3z**2) / Nev_safe
        T_m2 = conv_same(Tz**2) / Nev_safe
        L3_T = conv_same(L3z * Tz) / Nev_safe

        # Moments
        COVAR = L3_T - (L3_m * T_m)
        VAR_L3 = L3_m2 - (L3_m * L3_m)
        VAR_template = T_m2 - (T_m * T_m)

        # Regression with safeguards
        with np.errstate(divide='ignore', invalid='ignore'):
            AA = COVAR / VAR_template
            BB = L3_m - (AA * T_m)
            denom = np.sqrt(np.maximum(VAR_L3 * VAR_template, 0.0))
            RR = np.where(denom > TOL, COVAR / denom, 0.0)

        # Degenerate cases
        no_neighbors = Nev < min_support
        # Use a more realistic epsilon for template variance to avoid unstable divisions
        varT_small = VAR_template <= VAR_EPS
        varL3_small = VAR_L3 <= VAR_EPS

        # Where no neighbors → NaN
        for arr in (AA, BB, RR):
            arr[no_neighbors] = np.nan

        # If VAR(T) small: AA=0; BB=X puntual; RR=0  
        AA[varT_small] = 0.0
        BB[varT_small] = L3_2d[varT_small]
        RR[varT_small] = 0.0

        # If VAR(L3) tiny and VAR(T) is not tiny: set RR=1 
        RR[np.logical_and(varL3_small, ~varT_small)] = 1.0

        # Error: std of residuals
        ERR = (1 - RR**2) * VAR_L3
        ERR[ERR < 0] = 0
        ERR = np.sqrt(ERR)

        # In VAR(T) small branch error is exactly 0 
        ERR[varT_small] = 0.0

        # L4 prediction in original (possibly log) space
        L4 = AA * T_2d + BB

        # Output mask
        if mask_mode == 'template':
            mask_out = mask_template >= 0.5
        elif mask_mode == 'L3':
            mask_out = mask_L3 >= 0.5
        elif mask_mode == 'both':
            mask_out = mask_merged
        else:
            raise ValueError('mask_mode must be "template", "L3" or "both"')
        for arr in (L4, AA, BB, RR, ERR):
            arr[~mask_out] = np.nan

        # Return arrays and rare-case counters for optional debug
        counts = (
            int(np.sum(no_neighbors)),           # no_neighbors
            int(np.sum(varT_small)),             # varT_small
            int(np.sum(varL3_small)),            # varL3_small
            int(L3_2d.size),                     # total pixels
        )
        return L4, AA, BB, RR, ERR, counts

    # Execute per slice and stack results if needed
    # Debug counters
    total_no_neighbors = total_varT_small = total_varL3_small = total_npix = 0

    if L3.ndim == 2:
        L4, AA, BB, RR, ERR, counts = _process_slice(L3, template)
        c_no, c_vt, c_vx, c_N = counts
        total_no_neighbors += c_no; total_varT_small += c_vt; total_varL3_small += c_vx; total_npix += c_N
    elif L3.ndim == 3:
        nt = L3.shape[0]
        L4 = np.empty_like(L3, dtype=float)
        AA = np.empty_like(L3, dtype=float)
        BB = np.empty_like(L3, dtype=float)
        RR = np.empty_like(L3, dtype=float)
        ERR = np.empty_like(L3, dtype=float)
        for it in range(nt):
            L4[it], AA[it], BB[it], RR[it], ERR[it], counts = _process_slice(L3[it], template[it])
            c_no, c_vt, c_vx, c_N = counts
            total_no_neighbors += c_no; total_varT_small += c_vt; total_varL3_small += c_vx; total_npix += c_N
    else:
        raise ValueError('L3 and template must be 2D or 3D arrays')

    # Optionally exponentiate output if L3 was log-transformed
    if log_mode in ('L3', 'both'):
        L4 = 10 ** L4

    if debug and total_npix > 0:
        try:
            def _rng(a):
                v = np.asarray(a); finite = np.isfinite(v)
                if not finite.any():
                    return (np.nan, np.nan)
                vv = v[finite]
                return (float(vv.min()), float(vv.max()))
            p_no = 100.0 * total_no_neighbors / total_npix
            p_vt = 100.0 * total_varT_small / total_npix
            p_vx = 100.0 * total_varL3_small / total_npix
            l4min, l4max = _rng(L4); amin, amax = _rng(AA); rmin, rmax = _rng(RR); _, emax = _rng(ERR)
            logger.info(f"[fusion] rare-cases: no_neighbors={p_no:.2f}% varT_small={p_vt:.2f}% varL3_small={p_vx:.2f}%")
            logger.info(f"[fusion] ranges L4[{l4min:.6g},{l4max:.6g}] a[{amin:.6g},{amax:.6g}] rho[{rmin:.6g},{rmax:.6g}] err_max={emax:.6g}")
        except Exception:
            logger.debug("[fusion] debug summary failed", exc_info=True)

    return L4, AA, BB, RR, ERR

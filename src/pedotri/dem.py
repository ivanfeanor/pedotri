"""DEM-derived topographic indices.

Three primitives that show up in every soil-science workflow that
wants to relate a property to local terrain:

- :func:`slope` and :func:`aspect`, per Horn (1981) — the same finite-
  difference scheme GDAL's ``gdaldem`` uses, and the one expected by
  the SoilGrids / SoilWeb covariate libraries.
- :func:`profile_curvature` and :func:`plan_curvature`, per
  Zevenbergen & Thorne (1987) — the second-derivative scheme that
  separates "downhill curvature" (water acceleration / deceleration
  in the steepest-descent direction) from "across-contour curvature"
  (water convergence / divergence orthogonal to flow).
- :func:`topographic_wetness_index` — the classic
  ``TWI = ln(a / tan β)`` of Beven & Kirkby (1979), commonly used as
  a proxy for soil moisture redistribution in pedotransfer-function
  fitting.

Pure numpy. The input is always a 2-D DEM array plus pixel-size
metadata; outputs are arrays of matching shape with boundary pixels
filled via reflection so the kernels don't shrink the grid.

Mathematical references (see ``wiki/DEM-derived-indices.md`` for the
full derivation):

- Horn, B.K.P. (1981). Hill shading and the reflectance map.
  *Proceedings of the IEEE* 69(1): 14–47.
- Zevenbergen, L.W. and Thorne, C.R. (1987). Quantitative analysis of
  land surface topography. *Earth Surface Processes and Landforms*
  12(1): 47–56.
- Beven, K.J. and Kirkby, M.J. (1979). A physically based, variable
  contributing area model of basin hydrology. *Hydrological Sciences
  Bulletin* 24(1): 43–69.
"""

from __future__ import annotations

import numpy as np

from pedotri.errors import InvalidInputError


def slope(
    dem: np.ndarray,
    *,
    pixel_size: float | tuple[float, float] = 1.0,
    units: str = "degrees",
) -> np.ndarray:
    """Slope angle from a DEM via Horn's (1981) 3×3 finite-difference scheme.

    For each interior pixel ``z[i, j]``, Horn estimates the gradient
    by a Sobel-weighted finite difference over the 3×3 neighbourhood:

    .. math::

        \\frac{\\partial z}{\\partial x}
        = \\frac{(z_{i-1, j+1} + 2 z_{i, j+1} + z_{i+1, j+1})
               - (z_{i-1, j-1} + 2 z_{i, j-1} + z_{i+1, j-1})}
              {8 \\, \\Delta x}

    .. math::

        \\frac{\\partial z}{\\partial y}
        = \\frac{(z_{i+1, j-1} + 2 z_{i+1, j} + z_{i+1, j+1})
               - (z_{i-1, j-1} + 2 z_{i-1, j} + z_{i-1, j+1})}
              {8 \\, \\Delta y}

    Slope angle is the arctangent of the gradient magnitude:

    .. math::

        \\beta = \\arctan \\sqrt{(\\partial z / \\partial x)^2 +
                                  (\\partial z / \\partial y)^2}

    Args:
        dem: 2-D elevation array. Vertical units are arbitrary but
            must match ``pixel_size`` units (e.g. both in metres).
        pixel_size: Pixel side length. Pass a single float for square
            pixels or a ``(dx, dy)`` tuple for rectangular ones.
        units: ``"degrees"`` (default) or ``"radians"`` or ``"percent"``
            (rise / run × 100 — the convention used in soil-survey
            reports).

    Returns:
        Slope array of the same shape as ``dem``. Boundary pixels are
        computed against a reflected DEM so the output isn't shrunk.

    Raises:
        InvalidInputError: For non-2-D input, non-positive pixel size,
            or unknown ``units``.
    """
    z = _validate_dem(dem)
    dx, dy = _pixel_size_tuple(pixel_size)
    if units not in ("degrees", "radians", "percent"):
        raise InvalidInputError(
            f"Unknown slope units {units!r}; pick degrees, radians, or percent."
        )

    dzdx, dzdy = _horn_gradients(z, dx, dy)
    magnitude = np.sqrt(dzdx * dzdx + dzdy * dzdy)
    if units == "radians":
        return np.asarray(np.arctan(magnitude))
    if units == "degrees":
        return np.asarray(np.degrees(np.arctan(magnitude)))
    # percent: rise / run × 100. magnitude already equals rise/run.
    return np.asarray(magnitude * 100.0)


def aspect(
    dem: np.ndarray,
    *,
    pixel_size: float | tuple[float, float] = 1.0,
    units: str = "degrees",
) -> np.ndarray:
    """Aspect (downhill direction) from a DEM via Horn (1981).

    Aspect is the compass direction the slope faces, measured
    clockwise from north (so a north-facing slope is 0°, east 90°,
    south 180°, west 270°). For pixels with effectively zero slope,
    aspect is undefined and we return ``NaN``.

    Internally:

    .. math::

        \\alpha = \\operatorname{atan2}\\!
            \\bigl(-\\partial z / \\partial x,\\ \\partial z / \\partial y\\bigr)

    The signs encode the raster-array convention (row index grows
    south, column index grows east) → the downhill direction whose
    compass bearing we want is :math:`(-\\partial z / \\partial east,
    -\\partial z / \\partial north) = (-\\partial z / \\partial x,
    \\partial z / \\partial y)` (recall that our ``dzdy`` is
    south-positive). ``atan2(east, north)`` then gives the
    clockwise-from-north bearing directly.

    Args:
        dem: 2-D elevation array.
        pixel_size: Pixel side length.
        units: ``"degrees"`` (default) or ``"radians"``.

    Returns:
        Aspect array of the same shape as ``dem``. ``NaN`` where slope
        is below ``1e-12``.
    """
    z = _validate_dem(dem)
    dx, dy = _pixel_size_tuple(pixel_size)
    if units not in ("degrees", "radians"):
        raise InvalidInputError(f"Unknown aspect units {units!r}; pick degrees or radians.")
    dzdx, dzdy = _horn_gradients(z, dx, dy)
    # Downhill direction in geographic (east, north) coordinates is
    # (-dz/deast, -dz/dnorth). In raster convention dz/deast = dzdx and
    # dz/dnorth = -dzdy (the row index grows southward). So the
    # downhill vector is (-dzdx, dzdy), and its compass bearing —
    # clockwise from north — is atan2(east_component, north_component).
    aspect_rad = np.arctan2(-dzdx, dzdy)
    # Wrap into [0, 2π].
    aspect_rad = np.where(aspect_rad < 0, aspect_rad + 2 * np.pi, aspect_rad)
    # Pixels with negligible slope → undefined aspect.
    flat = np.sqrt(dzdx * dzdx + dzdy * dzdy) < 1e-12
    out = aspect_rad if units == "radians" else np.degrees(aspect_rad)
    out = np.where(flat, np.nan, out)
    return np.asarray(out)


def profile_curvature(
    dem: np.ndarray,
    *,
    pixel_size: float | tuple[float, float] = 1.0,
) -> np.ndarray:
    """Curvature in the steepest-descent direction (Zevenbergen & Thorne, 1987).

    Negative values mean the slope is decelerating downhill (water
    accumulates); positive values mean it's accelerating. Computed
    from a 9-term polynomial fit through the 3×3 neighbourhood:

    .. math::

        z \\approx A x^2 y^2 + B x^2 y + C x y^2 + D x^2 + E y^2
                 + F x y + G x + H y + I

    The profile curvature is then

    .. math::

        k_{\\mathrm{profile}}
            = -2 \\frac{D G^2 + E H^2 + F G H}{G^2 + H^2}

    Units are 1/length (e.g. 1/m if the DEM is in metres). Boundary
    pixels use reflected DEM values.
    """
    return _zevenbergen_curvature(dem, pixel_size, kind="profile")


def plan_curvature(
    dem: np.ndarray,
    *,
    pixel_size: float | tuple[float, float] = 1.0,
) -> np.ndarray:
    """Curvature in the across-contour (plan) direction (Zevenbergen & Thorne, 1987).

    Negative values mean water diverges (ridge-like topography);
    positive values mean it converges (valley-like). Same polynomial
    fit as :func:`profile_curvature`; the plan curvature is

    .. math::

        k_{\\mathrm{plan}}
            = 2 \\frac{D H^2 + E G^2 - F G H}{G^2 + H^2}
    """
    return _zevenbergen_curvature(dem, pixel_size, kind="plan")


def topographic_wetness_index(
    dem: np.ndarray,
    *,
    pixel_size: float | tuple[float, float] = 1.0,
    flow_accumulation: np.ndarray | None = None,
    min_slope: float = 1e-4,
) -> np.ndarray:
    """Beven & Kirkby (1979) topographic wetness index.

    .. math::

        \\mathrm{TWI} = \\ln \\frac{a}{\\tan \\beta}

    where ``a`` is the upstream contributing area per unit contour
    length (m) and ``β`` is the local slope angle (radians). TWI is
    a non-dimensional proxy for soil-moisture redistribution: a pixel
    at the bottom of a long shallow hillslope gets a much larger TWI
    than one on a steep ridge, and empirically this tracks the soil
    moisture pattern measured at the field scale.

    Args:
        dem: 2-D elevation array.
        pixel_size: Pixel side length, used both for slope and to
            convert the (dimensionless) cell count from
            ``flow_accumulation`` into a unit contributing area.
        flow_accumulation: Optional pre-computed flow-accumulation
            raster (number of upstream cells per pixel). When
            ``None``, a fast surrogate is used: every pixel
            contributes its own area only — equivalent to assuming
            single-cell catchments. **This is intentionally simple;
            pedotri does not ship a hydrology engine.** For real
            workflows, compute the flow accumulation with a tool
            like ``richdem`` / ``pysheds`` / GRASS ``r.watershed``
            and pass it in.
        min_slope: Floor on ``tan β`` to avoid the singularity at
            flat pixels. Default ``1e-4`` matches the SAGA-GIS
            convention.

    Returns:
        TWI array of the same shape as ``dem``.
    """
    z = _validate_dem(dem)
    dx, dy = _pixel_size_tuple(pixel_size)
    dzdx, dzdy = _horn_gradients(z, dx, dy)
    slope_tan = np.sqrt(dzdx * dzdx + dzdy * dzdy)
    slope_tan = np.maximum(slope_tan, min_slope)

    pixel_area = dx * dy
    if flow_accumulation is None:
        # Surrogate: each pixel contributes only its own area. This
        # makes TWI a function of slope alone — useful as a
        # benchmarkable feature, not a hydrology product.
        upstream_area = np.full_like(z, pixel_area, dtype=np.float64)
    else:
        fa = np.asarray(flow_accumulation, dtype=np.float64)
        if fa.shape != z.shape:
            raise InvalidInputError(
                f"flow_accumulation shape {fa.shape} does not match DEM shape {z.shape}."
            )
        upstream_area = fa * pixel_area

    # Unit contour width = average of the two pixel sides.
    contour_width = (dx + dy) * 0.5
    a = upstream_area / contour_width
    return np.asarray(np.log(a / slope_tan), dtype=np.float64)


# --- internals -----------------------------------------------------------


def _validate_dem(dem: np.ndarray) -> np.ndarray:
    arr = np.asarray(dem, dtype=np.float64)
    if arr.ndim != 2:
        raise InvalidInputError(f"DEM must be a 2-D array; got shape {arr.shape}.")
    return arr


def _pixel_size_tuple(pixel_size: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(pixel_size, (int, float)):
        dx = dy = float(pixel_size)
    else:
        try:
            dx, dy = pixel_size
            dx, dy = float(dx), float(dy)
        except (TypeError, ValueError) as exc:
            raise InvalidInputError(
                f"pixel_size must be a float or (dx, dy) tuple; got {pixel_size!r}."
            ) from exc
    if dx <= 0 or dy <= 0:
        raise InvalidInputError(f"pixel_size must be positive; got dx={dx}, dy={dy}.")
    return dx, dy


def _horn_gradients(z: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Sobel-weighted ∂z/∂x and ∂z/∂y per Horn (1981).

    Pads with reflection so the output retains the input shape.
    """
    z_padded = np.pad(z, 1, mode="edge")
    # The eight neighbours of (i, j) sit at (i+a, j+b) for
    # a, b ∈ {-1, 0, 1}, (a, b) ≠ (0, 0). Slicing the padded array
    # by (1+a):(1+a+H), (1+b):(1+b+W) recovers them.
    nw = z_padded[0:-2, 0:-2]
    n_ = z_padded[0:-2, 1:-1]
    ne = z_padded[0:-2, 2:]
    w_ = z_padded[1:-1, 0:-2]
    e_ = z_padded[1:-1, 2:]
    sw = z_padded[2:, 0:-2]
    s_ = z_padded[2:, 1:-1]
    se = z_padded[2:, 2:]
    dzdx = ((ne + 2 * e_ + se) - (nw + 2 * w_ + sw)) / (8.0 * dx)
    dzdy = ((sw + 2 * s_ + se) - (nw + 2 * n_ + ne)) / (8.0 * dy)
    return dzdx, dzdy


def _zevenbergen_curvature(
    dem: np.ndarray,
    pixel_size: float | tuple[float, float],
    *,
    kind: str,
) -> np.ndarray:
    z = _validate_dem(dem)
    dx, dy = _pixel_size_tuple(pixel_size)
    # Use the average pixel size for the polynomial fit. Zevenbergen
    # & Thorne (1987) assume square cells; we approximate.
    L = (dx + dy) * 0.5

    z_padded = np.pad(z, 1, mode="edge")
    z1 = z_padded[0:-2, 0:-2]  # NW
    z2 = z_padded[0:-2, 1:-1]  # N
    z3 = z_padded[0:-2, 2:]  # NE
    z4 = z_padded[1:-1, 0:-2]  # W
    z5 = z_padded[1:-1, 1:-1]  # centre
    z6 = z_padded[1:-1, 2:]  # E
    z7 = z_padded[2:, 0:-2]  # SW
    z8 = z_padded[2:, 1:-1]  # S
    z9 = z_padded[2:, 2:]  # SE

    # Polynomial coefficients (Z&T eqs. 7–14).
    D = ((z4 + z6) / 2.0 - z5) / (L * L)
    E = ((z2 + z8) / 2.0 - z5) / (L * L)
    F = (-z1 + z3 + z7 - z9) / (4.0 * L * L)
    G = (-z4 + z6) / (2.0 * L)
    H = (z2 - z8) / (2.0 * L)

    denom = G * G + H * H
    safe = denom > 1e-30
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "profile":
            curv = -2.0 * (D * G * G + E * H * H + F * G * H) / denom
        elif kind == "plan":
            curv = 2.0 * (D * H * H + E * G * G - F * G * H) / denom
        else:  # pragma: no cover — internal only
            raise ValueError(f"Unknown curvature kind: {kind!r}")
    curv = np.where(safe, curv, 0.0)
    return np.asarray(curv, dtype=np.float64)


__all__ = [
    "aspect",
    "plan_curvature",
    "profile_curvature",
    "slope",
    "topographic_wetness_index",
]

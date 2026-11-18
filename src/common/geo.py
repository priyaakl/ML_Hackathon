"""Geometry helpers. Trips are in different parts of the world, so every trip gets its OWN
local metric frame (equirectangular projection around the trip's mean lon/lat).
Over a trip-sized area this is accurate to well under a metre, and it is fast."""
import numpy as np

EARTH_R = 6371008.8  # metres (mean Earth radius)


def trip_origin(track):
    """(lon0, lat0) = mean of the GPS track. Used as the projection origin for this trip."""
    return float(track["lon"].mean()), float(track["lat"].mean())


def project(lon, lat, lon0, lat0):
    """lon/lat in degrees -> (x, y) in metres east/north of the origin. Returns an (n, 2) array."""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    x = np.radians(lon - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    return np.column_stack([x, y])

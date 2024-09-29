"""
 Copyright (c) 2022 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np

from simulator.environment.constants import DEG2RAD, RAD2DEG
from simulator.environment.geo_constants import *


def ned2geo(
    points_ned: np.ndarray, home_geo: tuple = DEFAULT_HOME_COORDS
) -> np.ndarray:
    """
    Convert local NED coordinates to geodetic coordinates (latitude, longitude, altitude).

    *IMPORTANT: Don't use this next to the poles. Calculation asumes local plane!*

    Parameters
    ----------
    points_ned : np.ndarray
        N by 3 array with North-East-Down coordinates in meters
    home_geo : tuple, optional
        3 elements tuple with geodetic coordinates of the reference point in (deg, deg, m),
        by default DEFAULT_HOME_COORDS

    Returns
    -------
    np.ndarray
        N by 3 array with converted geodetic coordinates in (deg, deg, m)
    """
    points_geo = np.zeros_like(points_ned)
    if points_ned.ndim == 1:
        points_geo[0] = points_ned[0] * GEO_M2DEG + home_geo[0]
        points_geo[1] = points_ned[1] * GEO_M2DEG + home_geo[1]
        points_geo[2] = points_ned[2] * -1 + home_geo[2]
    else:
        for kk, point_ned in enumerate(points_ned):
            points_geo[kk, 0] = point_ned[0] * GEO_M2DEG + home_geo[0]
            points_geo[kk, 1] = point_ned[1] * GEO_M2DEG + home_geo[1]
            points_geo[kk, 2] = point_ned[2] * -1 + home_geo[2]
    return points_geo


def geo2ned(
    points_geo: np.ndarray, home_geo: tuple = DEFAULT_HOME_COORDS
) -> np.ndarray:
    """
    Convert geodetic coordinates (latitude, longitude, altitude) to local NED coordinates.

    *IMPORTANT: Don't use this next to the poles. Calculation asumes local plane!*

    Parameters
    ----------
    points_geo : np.ndarray
        N by 3 array with geodetic coordinates in (deg, deg, m)
    home_geo : tuple, optional
        3 elements tuple with geodetic coordinates of the reference point in (deg, deg, m),
        by default DEFAULT_HOME_COORDS

    Returns
    -------
    np.ndarray
        N by 3 array with converted North-East-Down coordinates in meters
    """
    points_ned = np.zeros_like(points_geo)
    if points_ned.ndim == 1:
        points_ned[0] = (points_geo[0] - home_geo[0]) * GEO_DEG2M
        points_ned[1] = (points_geo[1] - home_geo[1]) * GEO_DEG2M
        points_ned[2] = (points_geo[2] - home_geo[2]) * -1
    else:
        for kk, point_geo in enumerate(points_geo):
            points_ned[kk, 0] = (point_geo[0] - home_geo[0]) * GEO_DEG2M
            points_ned[kk, 1] = (point_geo[1] - home_geo[1]) * GEO_DEG2M
            points_ned[kk, 2] = (point_geo[2] - home_geo[2]) * -1
    return points_ned


def geo_to_wgs84(lat: float, long: float, alt: float = 0.0) -> np.ndarray:
    """
    Convert geodetic coordinates (lat, long, alt) to WGS84 geocentric coordinates (X, Y, Z).

    Parameters
    ----------
    lat : float
        latitude in deg (negative values for South latitudes)

    long : float
        longitude in deg (negative values for West latitudes)

    alt : float, optional
        altitude above MSL (mean sea level) in meters, by default 0.0

    Returns
    -------
    numpy.ndarray
        3 size array with WGS84 geocentric coordinates
    """
    lat_rad = lat * DEG2RAD
    long_rad = long * DEG2RAD
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    n = WGS84_EQUATORIAL_RADIUS / np.sqrt(1.0 - WGS84_EXCENTRICITY2 * sin_lat**2)
    x = (n + alt) * cos_lat * np.cos(long_rad)
    y = (n + alt) * cos_lat * np.sin(long_rad)
    z = (n * (1.0 - WGS84_EXCENTRICITY2) + alt) * sin_lat
    return np.array([x, y, z])


def wgs84_to_geo(coords: np.ndarray, tol: float = 1e-9, max_iter=1000) -> tuple:
    """
    Convert WGS84 geocentric coordinates (X, Y, Z) to geodetic coordinates (lat, long, alt).

    Note: computation is done by iteration.

    Parameters
    ----------
    coords : np.ndarray
        3 size array with WGS84 geocentric coordinates

    tol : float, optional
        altitude tolerance in meters for iteration, by default 1e-9

    Returns
    -------
    tuple
        3 elements tuple with geodetic coordinates (lat, long, alt) in (deg, deg, meters)
    """
    xy_root = np.sqrt(coords[0] ** 2 + coords[1] ** 2)  # precompute sqrt(x**2 + y**2)
    lat_rad = np.arctan2(coords[2], xy_root)  # initial latitude
    long_rad = np.arctan2(coords[1], coords[0])  # longitude is direct
    alt = 0.0  # initial altitude
    alt_prev = tol * 1000.0
    for _ in range(max_iter):
        if np.abs(alt - alt_prev) > tol:
            break
        sin_lat = np.sin(lat_rad)
        e2_1 = 1.0 - WGS84_EXCENTRICITY2
        n = WGS84_EQUATORIAL_RADIUS / np.sqrt(1.0 - WGS84_EXCENTRICITY2 * sin_lat**2)
        alt_prev = alt
        alt = coords[2] / sin_lat - n * e2_1
        lat_rad = np.arctan2(coords[2] * (n + alt), (xy_root * (n * e2_1 + alt)))
    return (lat_rad * RAD2DEG, long_rad * RAD2DEG, alt)


def rot_matrix_enu(lat: float, long: float):
    """
    Compute the rotation matrix used to transform ECEF coordinates (like WGS84 coordinates) to ENU coordinates
    given a reference point (lat, long).

    Parameters
    ----------
    lat : float
        latitude of the reference point in deg

    long : float
        longitude of the reference point in deg

    Returns
    -------
    numpy.ndarray
        3x3 rotation matrix used to transform ECEF to ENU coordinates
    """
    slat = np.sin(lat * DEG2RAD)
    clat = np.cos(lat * DEG2RAD)
    slong = np.sin(long * DEG2RAD)
    clong = np.cos(long * DEG2RAD)
    rot_matrix = np.zeros((3, 3))
    rot_matrix[0, :] = -slat, -slong * clat, clong * clat
    rot_matrix[1, :] = clat, -slong * slat, clong * slat
    rot_matrix[2, :] = 0.0, clong, slong
    return rot_matrix


def enu_to_wgs84(
    coords_enu: np.ndarray, home_wgs84: np.ndarray = None, home_geo: tuple = None
) -> np.ndarray:
    """
    Convert local ENU coordinates (referenced to home point) to WGS84 coordinates.

    Reference point must be provided with `home_wgs84`, `home_geo` or both.

    Parameters
    ----------
    coords_enu : np.ndarray
        3 size array with local ENU coordinates (refered to home point) in meters (x, y, z)

    home_wgs84 : np.ndarray, optional
        3 size array with reference point WGS84 coordinates in meters (X0, Y0, Z0) , by default None

    home_geo : tuple, optional
        3 elements tuple with reference point geodetic coordinates (lat0, long0, alt0) in (deg, deg, m), by default None

    Returns
    -------
    np.ndarray
        3 size array with WGS84 coordinates in meters (X, Y, Z)
    """
    if home_wgs84 is None and home_geo is None:
        ValueError("home_wgs84 or home_geo must be provided!")
    elif home_wgs84 is not None and home_geo is None:
        home_geo = wgs84_to_geo(home_wgs84)
    elif home_wgs84 is None and home_geo is not None:
        home_wgs84 = geo_to_wgs84(*home_geo)

    rot_matrix = rot_matrix_enu(home_geo[0], home_geo[1])
    coords_wgs84 = rot_matrix.dot(coords_enu) + home_wgs84
    return coords_wgs84


def wgs84_to_enu(
    coords_wgs84: np.ndarray, home_wgs84: np.ndarray = None, home_geo: tuple = None
) -> np.ndarray:
    """
    Convert WGS84 coordinates to local ENU coordinates referenced to home.

    Reference point must be provided with `home_wgs84`, `home_geo` or both.

    Parameters
    ----------
    coords_wgs84 : np.ndarray
        3 size array with WGS84 coordinates in meters (X, Y, Z)

    home_wgs84 : np.ndarray, optional
        3 size array with reference point WGS84 coordinates in meters (X0, Y0, Z0) , by default None

    home_geo : tuple, optional
        3 elements tuple with reference point geodetic coordinates (lat0, long0, alt0) in (deg, deg, m), by default None

    Returns
    -------
    np.ndarray
        3 size array with local ENU coordinates (refered to home point) in meters (x, y, z)
    """
    if home_wgs84 is None and home_geo is None:
        ValueError("home_wgs84 or home_geo must be provided!")
    elif home_wgs84 is not None and home_geo is None:
        home_geo = wgs84_to_geo(home_wgs84)
    elif home_wgs84 is None and home_geo is not None:
        home_wgs84 = geo_to_wgs84(*home_geo)

    rot_matrix = rot_matrix_enu(home_geo[0], home_geo[1])
    coords_enu = rot_matrix.T.dot(coords_wgs84 - home_wgs84)
    return coords_enu


def rot_matrix_ned(lat: float, long: float) -> np.ndarray:
    """
    Compute the rotation matrix used to transform ECEF coordinates (like WGS84 coordinates) to NED coordinates
    given a reference point (lat, long).

    Parameters
    ----------
    lat : float
        latitude of the reference point in deg

    long : float
        longitude of the reference point in deg

    Returns
    -------
    numpy.ndarray
        3x3 rotation matrix used to transform ECEF to NED coordinates
    """
    slat = np.sin(lat * DEG2RAD)
    clat = np.cos(lat * DEG2RAD)
    slong = np.sin(long * DEG2RAD)
    clong = np.cos(long * DEG2RAD)
    rot_matrix = np.zeros((3, 3))
    rot_matrix[0, :] = -slat * clong, -slong, -clat * clong
    rot_matrix[1, :] = -slat * slong, clong, -clat * slong
    rot_matrix[2, :] = clat, 0.0, -slat
    return rot_matrix


def ned_to_wgs84(
    coords_ned: np.ndarray, home_wgs84: np.ndarray = None, home_geo: tuple = None
) -> np.ndarray:
    """
    Convert local NED coordinates (referenced to home point) to WGS84 coordinates.

    Reference point must be provided with `home_wgs84`, `home_geo` or both.

    Parameters
    ----------
    coords_ned : np.ndarray
        3 size array with local NED coordinates (refered to home point) in meters (x, y, z)

    home_wgs84 : np.ndarray, optional
        3 size array with reference point WGS84 coordinates in meters (X0, Y0, Z0) , by default None

    home_geo : tuple, optional
        3 elements tuple with reference point geodetic coordinates (lat0, long0, alt0) in (deg, deg, m), by default None

    Returns
    -------
    np.ndarray
        3 size array with WGS84 coordinates in meters (X, Y, Z)
    """
    if home_wgs84 is None and home_geo is None:
        ValueError("home_wgs84 or home_geo must be provided!")
    elif home_wgs84 is not None and home_geo is None:
        home_geo = wgs84_to_geo(home_wgs84)
    elif home_wgs84 is None and home_geo is not None:
        home_wgs84 = geo_to_wgs84(*home_geo)

    rot_matrix = rot_matrix_ned(home_geo[0], home_geo[1])
    coords_wgs84 = rot_matrix.dot(coords_ned) + home_wgs84
    return coords_wgs84


def wgs84_to_ned(
    coords_wgs84: np.ndarray, home_wgs84: np.ndarray = None, home_geo: tuple = None
) -> np.ndarray:
    """
    Convert WGS84 coordinates to local NED coordinates referenced to home.

    Reference point must be provided with `home_wgs84`, `home_geo` or both.

    Parameters
    ----------
    coords_wgs84 : np.ndarray
        3 size array with WGS84 coordinates in meters (X, Y, Z)

    home_wgs84 : np.ndarray, optional
        3 size array with reference point WGS84 coordinates in meters (X0, Y0, Z0) , by default None

    home_geo : tuple, optional
        3 elements tuple with reference point geodetic coordinates (lat0, long0, alt0) in (deg, deg, m), by default None

    Returns
    -------
    np.ndarray
        3 size array with local NED coordinates (refered to home point) in meters (x, y, z)
    """
    if home_wgs84 is None and home_geo is None:
        ValueError("home_wgs84 or home_geo must be provided!")
    elif home_wgs84 is not None and home_geo is None:
        home_geo = wgs84_to_geo(home_wgs84)
    elif home_wgs84 is None and home_geo is not None:
        home_wgs84 = geo_to_wgs84(*home_geo)

    rot_matrix = rot_matrix_ned(home_geo[0], home_geo[1])
    coords_ned = rot_matrix.T.dot(coords_wgs84 - home_wgs84)
    return coords_ned

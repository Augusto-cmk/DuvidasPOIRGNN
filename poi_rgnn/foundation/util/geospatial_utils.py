import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
import numpy as np
import pyproj
from functools import partial
from shapely.ops import transform
from shapely.geometry import Point, Polygon

def points_distance(point_0, point_1):
    """
    :param point_0: [lat, lng]
    :param point_1: [lat, lng]
    :return: distance
    """
    point_0 = np.radians(point_0)
    point_1 = np.radians(point_1)
    result = haversine_distances([point_0, point_1])
    # metros
    result = result * 6371000
    distance = result[0][1]
    return distance

def centroid(latitudes, longitudes):
    lenght = len(latitudes)
    sum_lat = sum(latitudes)
    sum_long = sum(longitudes)

    latitude = sum_lat / lenght
    longitude = sum_long / lenght

    return latitude, longitude

def geodesic_point_buffer(lat, lon, radius):
	"""This method implements a geodesical point bufferization. It creates a circle with radius around the lat/long. It uses the azhimutal projection to fix the problem of distances proportions on globe.

	Parameters
	----------
	lat: float

	lon: float

	radius: float
		distance in meters
	"""
	proj_wgs84 = pyproj.Proj('+proj=longlat +datum=WGS84')

	# Azimuthal equidistant projection
	aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
	project = partial(
		pyproj.transform,
		pyproj.Proj(aeqd_proj.format(lat=lat, lon=lon)),
		proj_wgs84)
	buf = Point(0, 0).buffer(radius)  # distance in meters
	return Polygon(transform(project, buf).exterior.coords[:])
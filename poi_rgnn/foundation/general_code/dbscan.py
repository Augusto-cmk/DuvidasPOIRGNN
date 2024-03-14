import numpy as np
from sklearn.cluster import DBSCAN

class Dbscan:

    def __init__(self, points, min_samples, eps):
        self.points = points
        self.min_samples = min_samples
        self.eps = eps

    def cluster_geo_data(self):
        p_radians = np.radians(self.points)
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, algorithm='ball_tree', metric='haversine').fit(p_radians)
        self.db = db
        return db

    def get_clusters_with_points_and_datatime(self, datetime_list: list):
        pois_coordinates = {}
        labels = self.db.labels_
        pois_times = {}

        for i in range(len(list(set(labels)))):
            if i != -1:
                pois_coordinates[i] = []
                pois_times[i] = []
        size = min([len(self.points), len(datetime_list)])
        for i in range(size):
            if labels[i] != -1:
                pois_coordinates[labels[i]].append(self.points[i])
                pois_times[labels[i]].append(datetime_list[i])

        return pois_coordinates, pois_times
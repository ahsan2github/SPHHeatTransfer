import numpy as np
from scipy.spatial import KDTree
from scipy.spatial import distance


class Grid(object):
    def __init__(self, point_collection):
        self.tree = KDTree(point_collection)

    def get_neighbors(self, index, rad):
        neighbor_idx = self.tree.query_ball_point(self.tree.data[index], r=rad)
        return neighbor_idx

    # cubic spline based
    @staticmethod
    def get_weight(self, dis, h_):
        dis = dis/h_
        if 0.0 <= dis < 1:
            return 2.0/3.0 * (1-3.0/2.0 * dis**2 + 3.0/4.0 * dis**3)
        elif 1 <= dis < 2:
            return 2.0/3.0 * (1.0/4.0 * (2 - dis)**3)
        else:
            return 0

    @staticmethod
    def first_derivative_weight(dis, h_):
        dis = dis / h_
        if dis <= 1.0:
            return -(15.0 / h_ ** 2) * (1 - dis) ** 2 * dis
        else:
            return 0.0

    def number_density(self, index, h):
        res = 0
        idx = self.get_neighbors(index, h)
        pnts = tree.data[x]
        distarray = distance.cdist(pnts, pnts, 'euclidean')
        for x in idx:
            point = tree.data[x]
            #distance =
            res = res + self.get_weight(x, h)
        return res

    def get_neighbor_weights(self, index, h):
        res = []
        idx = self.get_neighbors(index, h)

        for x in dis:
            res.append(self.get_weight(x, h))
        return zip(idx, res)


def function_approx(idx, points, values, h):
    density = number_density(h, points)
    res = 0
    for i, y in enumerate(points):
        _, dis = distance(points[idx], y)
        res = res + (values[i]/density[i] * weight(dis, h))
    return res





def first_derivative(idx, points, values, h):
    density = number_density(h, points)
    res = 0
    for i, y in enumerate(points):
        _, dis = distance(points[idx], y)
        tmpval = (values[i]/density[i] * first_derivative_weight(dis, h))
        print(first_derivative_weight(dis, h))
        res = res + tmpval
    return res

npx = 10
dx = 0.01
h = 2.0 * dx
points = [(x * dx, 0, 0) for x in np.arange(0, npx)]
grid = Grid(points)

values = [10, 11, 12, 13, 14, 20, 17, 18, 80, 10]
#density = [grid.number_density(idx, h) for idx in np.arange(0, len(points))]
sp = 6
tree = KDTree(points)
# test number density
idx = tree.query_ball_point(tree.data[sp], r=h)
neighbors = tree.data[idx]
print("neighbors idx: {}".format(idx))
print("neighbors : {}".format(neighbors))
print("Distances: {}".format(distance.cdist(neighbors, neighbors)[:,0]))
print("points: {}".format(points))
print("values: {}".format(values))
#print("approx function at {}: {}".format(sp, function_approx(sp, points, values, h)))
#print("first derivative at {}: {}".format(sp, first_derivative(sp, points, values, h)))


"""
Code for this was copied from: https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-a-3d-plot
"""
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self):
        # Get the current Axes instance to access its projection matrix
        xs3d, ys3d, zs3d = self._verts3d
        ax = self.axes  # Axes is automatically set when added to a plot
        transform_matrix = ax.get_proj()
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, transform_matrix)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return zs[0]  # Return z-depth for ordering

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        ax = self.axes
        transform_matrix = ax.get_proj()
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, transform_matrix)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

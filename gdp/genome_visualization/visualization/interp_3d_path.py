import numpy as np
from scipy.interpolate import CubicSpline
from matplotlib.colors import Normalize


def disp_interpolated_points(ax, points, cmap, *args, **kwargs):

    if points is not None:
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        t = np.arange(len(points))

        cs_x = CubicSpline(t, x)
        cs_y = CubicSpline(t, y)
        cs_z = CubicSpline(t, z)

        t_new = np.linspace(t.min(), t.max(), len(points) * 100)
        x_new = cs_x(t_new)
        y_new = cs_y(t_new)
        z_new = cs_z(t_new)

        norm = Normalize(vmin=t_new.min(), vmax=t_new.max())

        ax.scatter(
            x_new,
            y_new,
            z_new,
            c=cmap(norm(t_new)),
            *args,
            **kwargs
        )

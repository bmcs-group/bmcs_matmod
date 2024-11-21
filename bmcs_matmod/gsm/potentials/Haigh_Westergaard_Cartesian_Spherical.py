'''
Created on 15.09.2016

@author: Yingxiong
'''
import numpy as np

# Cartesian coordinates

# def cartesian_to_haigh_westergaard(x, y, z):
#     xi = x*np.sqrt(3.)/3. + y*np.sqrt(3.)/3. + z*np.sqrt(3.)/3.
#     rho = np.sqrt(x**2 + y**2 + z**2 - xi**2)
#     theta =


def haigh_westergaard_to_cartesian(xi, rho, theta):
    z = 1 / np.sqrt(3) * xi + np.sqrt(2. / 3.) * rho * np.cos(theta)
    x = 1 / np.sqrt(3) * xi + np.sqrt(2. / 3.) * \
        rho * -np.sin(np.pi / 6 - theta)
    y = 1 / np.sqrt(3) * xi + np.sqrt(2. / 3.) * \
        rho * -np.sin(np.pi / 6 + theta)
    return x, y, z

# Spherical coordinates system
# https://en.wikipedia.org/wiki/Spherical_coordinate_system


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

if __name__ == '__main__':
    from mayavi import mlab

    # Haigh-Westergaard coordinates
    xi, theta = np.mgrid[-0.5:1:10j, 0:2 * np.pi:30j]
    rho = np.sqrt(1 - xi ** 2)

    print theta / np.pi

    print len(xi)

    x, y, z = haigh_westergaard_to_cartesian(xi, rho, theta)
    r, theta, phi = cartesian_to_spherical(x, y, z)
    print theta
    print theta.flatten()
    x, y, z = spherical_to_cartesian(r, theta, phi)
    mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
#     x, y, z = spherical_to_cartesian(r, theta, phi)
    s = mlab.mesh(x, y, z)
    mlab.axes(s)
    mlab.show()

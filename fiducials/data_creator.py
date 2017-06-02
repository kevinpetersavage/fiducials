import numpy as np
from scipy.ndimage.interpolation import rotate


def create_example_images(number_of_images, start_angle, end_angle, number_of_fiducials, radius_of_fiducials,
                          volume_dims, image_dims):
    fiducials = np.random.rand(number_of_fiducials, 3) * volume_dims
    angles = generate_angles(number_of_images, start_angle, end_angle)
    volume = generate_volume_from_fiducials(fiducials, volume_dims, radius_of_fiducials)
    projections = list(project_fiducials(volume, angles, image_dims))
    projections_array = np.array(projections)
    return angles, projections_array


def generate_angles(number_of_images, start_angle, end_angle):
    return np.arange(start_angle, end_angle, (end_angle - start_angle) / number_of_images)


def generate_volume_from_fiducials(fiducials, volume_dims, radius_of_fiducials):
    volume = np.zeros(shape=(volume_dims, volume_dims, volume_dims))
    for fiducial in fiducials:
        volume += generate_volume_from_fiducial(fiducial, volume_dims, radius_of_fiducials)
    return volume


def generate_volume_from_fiducial(fiducial, volume_dims, radius):
    axis = range(0, volume_dims)
    x, y, z = np.meshgrid(axis, axis, axis)
    return (((x - fiducial[0]) ** 2 + (y - fiducial[1]) ** 2 + (z - fiducial[2]) ** 2) ** 0.5 < radius).astype(int)


def project_fiducials(volume, angles, image_dims):
    for angle in angles:
        rotated = rotate(volume, angle, axes=(0, 1))
        product = np.sum(rotated, axis=0)
        minx = (product.shape[0] - image_dims) / 2
        maxx = (product.shape[0] + image_dims) / 2
        miny = (product.shape[1] - image_dims) / 2
        maxy = (product.shape[1] + image_dims) / 2
        cropped = product[minx:maxx, miny:maxy]
        normalised = (cropped > .3).astype(int)
        yield normalised


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    angles, images = create_example_images(
        number_of_images=10,
        start_angle=10,
        end_angle=20,
        number_of_fiducials=80,
        radius_of_fiducials=5,
        volume_dims=200,
        image_dims=120
    )
    plt.imshow(images[0])
    plt.show()
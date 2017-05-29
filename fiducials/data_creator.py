import numpy as np
from scipy.ndimage.interpolation import rotate
from functools import partial

def create_example_images(number_of_images, start_angle, end_angle, number_of_fiducials, radius_of_fiducials, volume_dims, image_dims):
    fiducials = np.random.rand(number_of_fiducials, 3) * volume_dims
    angles = generate_angles(number_of_images, start_angle, end_angle)
    volume = generate_volume_from_fiducials(fiducials, volume_dims, radius_of_fiducials)
    return angles, np.array(list(project_fiducials(volume, angles, image_dims)))

def generate_angles(number_of_images, start_angle, end_angle):
    return np.arange(start_angle, end_angle, (end_angle - start_angle) / number_of_images)

def generate_volume_from_fiducials(fiducials, volume_dims, radius_of_fiducials):
    volume = np.ones(shape = (volume_dims,volume_dims,volume_dims)) > 0
    for fiducial in fiducials:
        volume = volume * generate_volume_from_fiducial(fiducial, volume_dims, radius_of_fiducials)
    return volume
    
def generate_volume_from_fiducial(fiducial, volume_dims, radius):
    axis = range(0, volume_dims)
    x,y,z = np.meshgrid(axis, axis, axis)
    return ((x - fiducial[0])**2 + (y - fiducial[1])**2 + (z - fiducial[2])**2) ** 0.5 > radius

def project_fiducials(volume, angles, image_dims):
    for angle in angles:
        rotated = rotate(volume, angle, axes=(0,1))
        product = np.prod(rotated, axis=0)
        dim = product.shape[0]
        minx = (dim-image_dims)/2
        maxx = (dim+image_dims)/2
        yield product[minx:maxx,minx:maxx]

if __name__ == "__main__":
   

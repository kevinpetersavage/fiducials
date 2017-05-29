import numpy as np
from scipy.ndimage.interpolation import rotate
from functools import partial

def create_example_images(number_of_images, start_angle, end_angle, number_of_fiducials, radius_of_fiducials, volume_dims, image_dims):
    fiducials = np.random.rand(number_of_fiducials, 3) * volume_dims
    angles = generate_angles(number_of_images, start_angle, end_angle)
    generate_volume_from_fiducials(fiducials, volume_dims)
    project_fiducials(fiducials, angles, volume_dims, image_dims)

    pass

def generate_angles(number_of_images, start_angle, end_angle):
    return np.arange(start_angle, end_angle, (end_angle - start_angle) / number_of_images)

def generate_volume_from_fiducials(fiducials, volume_dims, radius_of_fiducials):
    return np.prod(
        np.stack(np.apply_along_axis(partial(generate_volume_from_fiducial, volume_dims, radius_of_fiducials), 1, fiducials)),
        1
    )
    
def generate_volume_from_fiducial(volume_dims, radius_of_fiducials, fiducial):
    axis = range(0, volume_dims)
    xyz = np.meshgrid(axis, axis, axis)
    return np.linalg.norm(xyz-fiducial.reshape(3,1,1,1)) > radius_of_fiducials

def project_fiducials(fiducials, angles, volume_dims, image_dims):
    yyvolume = np.ones(shape = (volume_dims,volume_dims,volume_dims))
    
    

    pass 

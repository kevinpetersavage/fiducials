import numpy as np
from fiducials.data_creator import *

def test_create_example_images():
    angles, images = create_example_images(
        number_of_images = 10,
        start_angle = 10,
        end_angle = 20,
        number_of_fiducials = 80,
        radius_of_fiducials = 5,
        volume_dims = 200,
        image_dims = 120
    )

    # check some high level properties
    assert images.shape == (10, 100, 100)
    assert (images == 1 or images == 0).all()
    
    max_zeros_per_image = np.Pi * 25 * 80
    pixels_per_image = 80*80
    min_ones_per_image = pixels_per_image - max_zeros_per_image
    min_average = min_ones_per_image / pixels_per_image

    assert images.average > min_average

def test_generate_angles():
    angles = generate_angles(number_of_images = 10, start_angle = 20, end_angle = 30)
    assert (angles == np.array([20,21,22,23,24,25,26,27,28,29])).all()

def test_generate_volume_from_fiducials():
    volume = generate_volume_from_fiducials(np.array([[100,100,100]]), 200, 5)

    assert volume.shape == (200,200,200)
    assert volume[100,100,100] == 1
    assert volume[99,99,99] == 1
    assert volume[50,50,50] == 0

def test_project_fiducials():
    projections = project_fiducials(
        fiducials = np.array([[1,2,3],[40,50,60],[100,100,100]]),
        angles = [0, 45, 90],
        volume_dims = 200,
        image_dims = 120
    )
    
    assert projections == []

def test_generate_image_from_projections():
    images = generate_image_from_projections(
        projections = np.array([[100,100]]),
        image_dims = 120,
        radius_of_fiducials = 5
    )

    assert images[1, 100, 100] == 0
    assert images[1, 98, 98] == 0
    assert images[1, 95, 95] == 1

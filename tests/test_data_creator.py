from fiducials.data_creator import *


def test_create_example_images():
    angles, images = create_example_images(
        number_of_images=20,
        start_angle=10,
        end_angle=170,
        number_of_fiducials=20,
        radius_of_fiducials=5,
        volume_dims=200,
        image_dims=120
    )

    # check some high level properties
    assert images.shape == (10, 150, 150)
    assert ((images == 1) | (images == 0)).all()

    max_zeros_per_image = np.pi * 25 * 20
    pixels_per_image = 150 * 150
    min_ones_per_image = pixels_per_image - max_zeros_per_image
    min_average = min_ones_per_image / pixels_per_image

    assert np.average(images) > min_average


def test_generate_angles():
    angles = generate_angles(number_of_images=10, start_angle=20, end_angle=30)
    assert (angles == np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29])).all()


def test_generate_volume_from_fiducials():
    volume = generate_volume_from_fiducials(np.array([[100, 100, 100], [50, 50, 50]]), 200, 5)

    assert volume.shape == (200, 200, 200)
    assert volume[100, 100, 100] == 1
    assert volume[50, 50, 50] == 1
    assert volume[150, 150, 150] == 0


def test_generate_volume_from_fiducial():
    volume = generate_volume_from_fiducial(np.array([100, 100, 100]), 200, 5)

    assert volume.shape == (200, 200, 200)
    assert volume[100, 100, 100] == 1
    assert volume[99, 99, 99] == 1
    assert volume[50, 50, 50] == 0


def test_project_fiducials():
    projections = project_fiducials(
        volume=np.zeros((100, 100, 100)) > 0,
        angles=[0, 45, 90],
        image_dims=20
    )

    result = np.array(list(projections))

    assert result.shape == (3, 20, 20)
    assert not result.any()

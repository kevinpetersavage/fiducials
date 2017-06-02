import theano as T
import data_creator
import numpy as np


radius = 5
volume_dims = 200
number_of_fiducials = 20
image_dims = 150
number_of_images = 20
actual_angles, existing_images = \
    data_creator.create_example_images(number_of_images, 10, 170, number_of_fiducials, 5, volume_dims, image_dims)

initial_angles = actual_angles + (np.random.rand(actual_angles.shape[0])*2) - 1
initial_fiducials = np.random.rand(3, number_of_fiducials) * volume_dims

images = T.tensor.ftensor3("images")
image_angle = T.shared(initial_angles, name="angles")
fiducials = T.shared(initial_fiducials, name="fiducials")

c = image_angle.cos()
s = image_angle.sin()
one = T.tensor.ones(actual_angles.shape[0])
zero = T.tensor.zeros(actual_angles.shape[0])

rotation = T.tensor.stack(c, -s, zero, s, c, zero, zero, zero, one).reshape((3, 3, number_of_images))
projection = T.tensor.stack(zero, one, zero, zero, zero, one).reshape((3, 2, number_of_images))

projected = T.tensor.tensordot(T.tensor.tensordot(fiducials, rotation, axes=(0, 0)), projection, axes=(0, 0))


dim_diff = (volume_dims - image_dims)/2
grid = T.tensor.mgrid[dim_diff:volume_dims - dim_diff, dim_diff:volume_dims - dim_diff]


gausian_image = T.tensor.exp(-((grid[0] - projected[0])**2 + (grid[1] - projected[1])**2)/2*(radius**2)).sum(axis=1)

def s(x):
    return 1 / (1 + T.tensor.exp(-x))

projected_images = s(gausian_image)

loss = T.tensor.sum((projected_images - images)**2)**0.5

d_loss_wrt_angle = T.grad(loss, image_angle)
d_loss_wrt_fiducials = T.grad(loss, fiducials)

# compile the MSGD step into a theano function
learning_rate = 0.2
updates = [
    (image_angle, image_angle - learning_rate * d_loss_wrt_angle),
    (fiducials, fiducials - learning_rate * d_loss_wrt_fiducials)
]

GD = T.function([images], loss, updates=updates)


for i in xrange(100):
    print('Current loss is ', GD(existing_images.astype("float32")))
    result_angles = image_angle.value
    print('Current angles are ', result_angles)

    np.mean(result_angles - result_angles[1:])
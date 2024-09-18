import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import math
from torchvision.utils import make_grid

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Set device to GPU if available
z_dim = 64
num_classes = 10  # Number of classes in the dataset
mnist_shape = (1, 28, 28)


# Function to generate random noise vectors
def get_noise(n_samples, input_dim, device='cpu'):
    """
    Generates noise vectors for the GAN input.

    Args:
        n_samples: Number of noise vectors to generate.
        input_dim: Dimension of each noise vector.
        device: Device on which to create the noise (CPU or GPU).

    Returns:
        A tensor of random noise.
    """

    return torch.randn(n_samples, input_dim, device=device)


# Function to display images
def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), nrow=5, show=True):
    """
    Displays a grid of images from a tensor.

    Args:
    image_tensor: Tensor containing image data
    num_images: Number of images to display
    size: Size of each image
    nrow: Number of rows in the grid
    show: Whether to display the images
    """

    image_tensor = (image_tensor + 1) / 2  # Rescale images to 0-1
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    if show:
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.show()


def get_one_hot_labels(labels, n_classes):
    """
    Converts class labels to one-hot encoded vectors.

    Args:
        labels: Tensor of class labels.
        n_classes: Total number of classes in the dataset.

    Returns:
        Tensor of shape (batch_size, n_classes) representing one-hot encoded labels.
    """

    return F.one_hot(labels, num_classes=n_classes)


def combine_vectors(x, y):
    """
    Concatenates two vectors along the feature dimension.

    Args:
        x: Tensor representing the first vector (e.g., noise vector).
        y: Tensor representing the second vector (e.g., one-hot encoded class vector).

    Returns:
        A concatenated tensor of shape (n_samples, combined_dim).
    """

    combined = torch.cat((x.float(), y.float()), dim=1)
    return combined


def get_input_dimensions(z_dim, mnist_shape, n_classes):
    """
    Calculates the input dimensions for the generator and discriminator in a Conditional GAN.

    Args:
        z_dim: Dimensionality of the noise vector.
        mnist_shape: Shape of each MNIST image in the format (C, W, H) where
                     C is the number of channels, and W, H are width and height.
        n_classes: Total number of classes in the dataset (e.g., 10 for MNIST).

    Returns:
        generator_input_dim: The combined dimensionality of the generator input
                             (noise vector and class vector).
        discriminator_im_chan: The number of input channels for the discriminator,
                               accounting for class embedding.
    """

    generator_input_dim = z_dim + n_classes
    discriminator_im_chan = mnist_shape[0] + n_classes
    return generator_input_dim, discriminator_im_chan


def interpolate_class(first_number, second_number, gen, n_interpolation, interpolation_noise):
    first_label = get_one_hot_labels(torch.Tensor([first_number]).long(), num_classes)
    second_label = get_one_hot_labels(torch.Tensor([second_number]).long(), num_classes)

    # Calculate the interpolation vector between the two labels
    percent_second_label = torch.linspace(0, 1, n_interpolation)[:, None]
    interpolation_labels = first_label * (1 - percent_second_label) + second_label * percent_second_label

    # Combine the noise and the labels
    noise_and_labels = combine_vectors(interpolation_noise, interpolation_labels.to(device))
    fake = gen(noise_and_labels)
    show_tensor_images(fake, num_images=n_interpolation, nrow=int(math.sqrt(n_interpolation)), show=False)

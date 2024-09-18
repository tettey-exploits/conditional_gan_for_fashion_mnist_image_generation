import matplotlib.pyplot as plt
import torch
from utils import interpolate_class, get_input_dimensions, get_noise
from utils import z_dim, mnist_shape, num_classes, device
from model import *


n_interpolation = 9  # Choose the interpolation: how many intermediate images you want + 2 (for the start and end image)

generator_input_dim, discriminator_im_chan = get_input_dimensions(z_dim, mnist_shape, num_classes)

#  Load Generator
gen = Generator(input_dim=generator_input_dim).to(device)
checkpoint = torch.load("/content/gen.pt", weights_only=True)
gen.load_state_dict(checkpoint['model_state_dict'])

start_plot_number = 1  # Choose the start digit
end_plot_number = 5  # Choose the end digit

plt.figure(figsize=(8, 8))
interpolation_noise = get_noise(1, z_dim, device=device).repeat(n_interpolation, 1)
interpolate_class(start_plot_number, end_plot_number, gen, n_interpolation, interpolation_noise)
_ = plt.axis('off')

"""
Uncomment the following lines of code if you would like to visualize a set of pairwise class
interpolations for a collection of different numbers, all in a single grid of interpolations.
You'll also see another visualization like this in the next code block!
"""

# plot_numbers = [2, 3, 4, 5, 7]
# n_numbers = len(plot_numbers)
# plt.figure(figsize=(8, 8))
# for i, first_plot_number in enumerate(plot_numbers):
#     for j, second_plot_number in enumerate(plot_numbers):
#         plt.subplot(n_numbers, n_numbers, i * n_numbers + j + 1)
#         interpolate_class(first_plot_number, second_plot_number)
#         plt.axis('off')
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0.1, wspace=0)
# plt.show()
# plt.close()

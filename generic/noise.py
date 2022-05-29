import numpy as np
import matplotlib.pyplot as plt


def smoothed_noise(length, ds, sigma, amplitude=1., mean=0, plot=False):
    num_bins = int(length / ds)
    norm_sigma = sigma / ds

    kernel_size = 2 * int(6 * norm_sigma / 2) + 1
    kernel_half_size = int(kernel_size / 2)
    kernel_x = np.linspace(-kernel_half_size, kernel_half_size, kernel_size)
    kernel = np.exp(-kernel_x ** 2 / (2 * norm_sigma ** 2)) / norm_sigma / np.sqrt(2 * np.pi)

    noise = np.random.normal(size=num_bins + 2 * kernel_half_size)

    smoothed = np.empty(num_bins)
    for start_index in range(0, num_bins):
        smoothed[start_index] = (np.sum(kernel * noise[start_index: start_index + kernel_size]))
    smoothed = smoothed / (smoothed.max() - smoothed.min()) * amplitude
    smoothed += mean - np.mean(smoothed)

    if plot:
        x = np.arange(ds / 2, length, ds)
        x_padded = np.linspace(-kernel_half_size * ds, (num_bins + kernel_half_size) * ds, len(noise))
        plt.plot(x_padded, noise)
        plt.plot(x, smoothed)

    return smoothed


if __name__ == "__main__":
    smoothed_noise(length=200, ds=1, sigma=5, plot=True)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt


length = 120
num_fields = 60
sigma = 10

field_centers = np.linspace(0, length, num_fields)[np.newaxis].T

xs = [0]
half = length / 2
while xs[-1] < length:
    if xs[-1] < half:
        ds = 0.003 * xs[-1] / half + 0.01 * (half - xs[-1]) / half
    else:
        ds = 0.01 * (xs[-1] - half) / half + 0.003 * (length - xs[-1]) / half
    xs.append(xs[-1] + ds)
xs = np.array(xs)

ys = np.exp(-(xs - field_centers) ** 2 / (2 * sigma ** 2))
positions = np.linspace(0, length, ys.shape[1])


fig, ax = plt.subplots(figsize=(6, 2))
for field_num in range(num_fields):
    ax.plot(positions, ys[field_num], color='C7')
fig.tight_layout()


# create "firing rates"
ds = 1
num_bins = int(length / ds) + 1
bins_x = np.linspace(0, length, num_bins)
bins_y = np.zeros((num_fields, num_bins))
counts = np.zeros(num_bins)

for pos, y in zip(positions, ys.T):
    index = int(round(pos / ds))
    bins_y[:, index] += y
    counts[index] += 1

bins_y /= counts

fig, ax = plt.subplots(figsize=(6, 2))
for field_num in range(num_fields):
    ax.plot(bins_x, bins_y[field_num], color='C7')
fig.tight_layout()



# decode positions
correlations = np.empty((num_bins, num_bins))  # rows: decoded; columns: true

# mean_b = np.mean(bins_y, axis=0)
# denom_b = np.sqrt(np.sum(bins_y**2, axis=0) - num_fields * mean_b**2)
#
# for i, a in enumerate(bins_y.T):
#     # a *= np.random.normal(1, 0.2, a.size)  # all firing rates at bin i
#     mean_a = np.mean(a)
#     denom_a = np.sqrt(np.sum(a ** 2) - num_fields * mean_a ** 2)
#     correlations[i] = (a @ bins_y - num_fields * mean_a * mean_b) / (denom_a * denom_b)

for j, bin_y in enumerate(bins_y.T):
    # a = bin_y * np.random.normal(1, 0.5, bin_y.size)
    # a = bin_y + np.random.normal(0, 0.2, bin_y.size)
    a = bin_y
    for i in range(num_bins):
        correlations[i, j] = np.corrcoef(a, bins_y[:, i])[0, 1]


correlations += np.random.normal(0, 0.2, correlations.shape)

fig, ax = plt.subplots()
ax.matshow(correlations, origin='lower')
ax.plot(bins_x, np.argmax(correlations, axis=0))
ax.plot((0, length), (0, length), color='k')
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel("true position")
ax.set_ylabel("decoded position")



plt.show()
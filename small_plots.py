import matplotlib.pyplot as plt


CM = 1/2.54

tick_size = 2
# font_size = 6
# line_width = 0.75

font_size = 7
line_width = 1

thin_line_width = 0.75  # 0.5

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': 'Arial',
                     'font.size': font_size, 'mathtext.default': 'regular', 'axes.linewidth': thin_line_width,
                     'xtick.major.width': thin_line_width, 'xtick.major.size': tick_size,
                     'xtick.minor.size': tick_size * 0.6, 'ytick.major.width': thin_line_width,
                     'ytick.major.size': tick_size, 'ytick.minor.size': tick_size * 0.6,
                     'lines.linewidth': line_width, 'lines.markersize': 4, 'lines.markeredgewidth': 0.0,
                     'legend.columnspacing': 0.3, 'axes.labelpad': 3,
                     'legend.handletextpad': 0.2,
                     'xtick.major.pad': 2, 'ytick.major.pad': 2,
                     })
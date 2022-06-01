from Network import Network, Config, plt

config = Config(variants={'LinearTrack': 'OneLap',
                          'Network': 'Pre'
                          })
network = Network.current_instance(config)

# network.track.plot_trajectory()
# network.track.plot_features()
# network.track.plot_features_heatmap()

# network.plot_rec_weights()

network.run(reset_indices=(0, 10), reset_value=1)
# network.plot_activities(apply_f=0)
network.plot_dynamics()
# network.plot_activities(apply_f=1)

plt.show()

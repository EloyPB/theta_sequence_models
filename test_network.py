from Network import Network, Config, plt

config = Config(variants={'LinearTrack': 'OneLap'})
network = Network.current_instance(config)

# network.track.plot_trajectory()

# network.plot_rec_weights()

network.run(reset_indices=(0, 10), reset_value=1)
network.plot_activities(apply_f=False)

plt.show()

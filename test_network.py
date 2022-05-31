from Network import Network, Config, plt

config = Config(variants={'LinearTrack': 'OneLap',
                          # 'Network': 'SimpleF'
                          })
network = Network.current_instance(config)

# network.track.plot_trajectory()

# network.plot_rec_weights()

network.run(reset_indices=(0, 10), reset_value=1)
# network.plot_activities(apply_f=0)
network.plot_dynamics()
network.plot_activities(apply_f=1)


plt.show()

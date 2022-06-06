from Network import Network, Config, plt

config = Config(variants={
    'LinearTrack': 'OneLap',
    # 'LinearTrack': 'FixSpeed',
    'Network': 'Log'
})
network = Network.current_instance(config)

# network.track.plot_trajectory()
# network.track.plot_features()
# network.track.plot_features_heatmap()

# network.plot_rec_weights()
# network.plot_activities(apply_f=0)
# network.plot_dynamics(t_end=None)
network.plot_activities(apply_f=1, pos_input=1, theta=0, t_start=0)

plt.show()

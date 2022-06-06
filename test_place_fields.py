from PlaceFields import PlaceFields, Config, plt


pf = PlaceFields.current_instance(Config(pickle_instances=True))
pf.plot_activations()

plt.show()

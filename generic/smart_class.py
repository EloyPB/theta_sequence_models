import os
import json
import time
import pickle
import inspect
import matplotlib.pyplot as plt


class Config:
    """Helper class that bundles together parameters used by SmartClass

    Args:
        variants (dict): Keys are class names and values are parameter tags.
        save_figures (bool): Whether to save and close the figures.
        figure_format (string): Format of the figures (e.g., png).
        figures_root_path (string): Path to the figures' folder.
        pickle_results (bool): Whether to pickle the results.
        pickles_root_path (string): Path to the folder where results will be pickled.
    """
    def __init__(self, identifier=None, variants={}, parameters_path="parameters", save_figures=False,
                 figure_format="png", figures_root_path="figures", pickle_instances=False, pickle_results=False,
                 pickles_root_path="pickles"):

        self.identifier = "" if None else str(identifier)
        self.variants = variants
        self.parameters_path = parameters_path
        self.save_figures = save_figures
        self.figure_format = figure_format
        self.figures_root_path = figures_root_path
        self.pickle_instances = pickle_instances
        self.pickle_results = pickle_results
        self.pickles_root_path = pickles_root_path


class SmartClass:
    dependencies = ()

    def __init__(self, config: Config, d: dict, **parameters_dict):
        self.config = config

        self.figures_path = self.complete_path(config.figures_root_path, config.identifier, config.variants)
        self.pickles_path = self.complete_path(config.pickles_root_path, config.identifier, config.variants)

        self.d = d

    def update_config(self, config: Config):
        self.config = config

        self.figures_path = self.complete_path(config.figures_root_path, config.identifier, config.variants)
        self.pickles_path = self.complete_path(config.pickles_root_path, config.identifier, config.variants)

        for dependency in self.d.values():
            dependency.update_config(config)

    @classmethod
    def complete_path(cls, root_path, identifier, variants):
        tag = (f"{identifier}_" if len(identifier) else "") + '_'.join(cls.variant_tags(variants))
        return f"{root_path}/{tag}/{cls.__name__}/"

    @classmethod
    def variant_tags(cls, variants):
        accumulated_tags = []
        if cls.__name__ in variants:
            accumulated_tags.append(variants[cls.__name__])
        for dependency in cls.dependencies:
            accumulated_tags += dependency.variant_tags(variants)
        return accumulated_tags[::-1]

    def maybe_save_fig(self, fig, name, sub_folder="", dpi=200):
        """Save and close a figure if 'save_figures' is True.

        Args:
            fig (matplotlib.pyplot.figure): Figure to save.
            name (string): Name for the file.
            sub_folder (string): Name for the sub-folder.
            dpi (int): Dots per inch.
        """
        if self.config.save_figures:
            file_path = self.file_path(self.figures_path, name, sub_folder) + f".{self.config.figure_format}"
            fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

    @staticmethod
    def file_path(folder_path, name, sub_folder):
        if sub_folder:
            folder_path += sub_folder + "/"
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        return f"{folder_path}{name}"

    def maybe_pickle_results(self, results, name, sub_folder=""):
        """Pickle results if 'pickle_results' is True.

        Args:
            results: Python object to be pickled.
            name (string): Name for the file.
            sub_folder (string): Name for the sub-folder.
        """
        if self.config.pickle_results:
            file_path = self.file_path(self.pickles_path, name, sub_folder)
            with open(file_path, 'wb') as f:
                pickle.dump(results, f)

    @classmethod
    def current_instance(cls, config=Config()):
        return cls.possibly_old_instance(config) if config.pickle_instances else cls.new_instance(config)

    @classmethod
    def new_instance(cls, config: Config):
        parameters_dict = cls.get_parameters(config)

        dependencies_dict = {}
        for dependency in cls.dependencies:
            dependencies_dict[dependency.__name__] = dependency.new_instance(config)

        return cls(config=config, d=dependencies_dict, **parameters_dict)

    @classmethod
    def get_parameters(cls, config: Config):
        parameters_dict = {}
        # try to load default parameters
        parameters_path = f"{config.parameters_path}/{cls.__name__}.json"
        if os.path.exists(parameters_path):
            with open(parameters_path) as parameters_f:
                parameters_dict = json.load(parameters_f)

            # try to update default parameters based on specified variant
            if cls.__name__ in config.variants:
                parameters_path = f"{config.parameters_path}/{cls.__name__}|{config.variants[cls.__name__]}.json"
                if os.path.exists(parameters_path):
                    with open(parameters_path) as parameters_f:
                        special_parameters_dict = json.load(parameters_f)
                        parameters_dict = {**parameters_dict, **special_parameters_dict}

        return parameters_dict

    @classmethod
    def possibly_old_instance(cls, config: Config, only_return_if_new=False):
        self_pickles_path = cls.complete_path(config.pickles_root_path, config.identifier, config.variants)
        if not os.path.exists(self_pickles_path):
            os.makedirs(self_pickles_path)

        # DECIDE WHETHER WE NEED A NEW INSTANCE
        need_new = False

        # did the code of this class or any of its parents change?
        for my_class in inspect.getmro(cls)[:-1]:
            if cls.changed(f"{self_pickles_path}{my_class.__name__}", my_class, inspect.getsource):
                need_new = True

        # do the dependencies need to change?
        dependencies_dict = {}
        for dependency in cls.dependencies:
            instance = dependency.possibly_old_instance(config, only_return_if_new=True)
            if instance is not None:
                need_new = True
                dependencies_dict[dependency.__name__] = instance

        # are the dependencies outdated?
        times_pickle_path = f"{self_pickles_path}/times"
        if not need_new:
            dependencies_changed = True
            if os.path.exists(times_pickle_path):
                with open(times_pickle_path, 'rb') as times_f:
                    if pickle.load(times_f) == cls.dependency_instance_times(config):
                        dependencies_changed = False
            need_new = dependencies_changed

        # did the parameters change?
        parameters_dict = cls.get_parameters(config)
        need_new += cls.changed(f"{self_pickles_path}/params", parameters_dict)

        # RETURN AN INSTANCE

        if not only_return_if_new or only_return_if_new and need_new:
            instance_pickle_path = f"{self_pickles_path}/instance"

            # we load a previous instance
            if not need_new and os.path.exists(instance_pickle_path):
                print(f"Loading {self_pickles_path}...")
                with open(instance_pickle_path, 'rb') as instance_f:
                    instance = pickle.load(instance_f)

                instance.update_config(config)

            # we create a new instance
            else:
                print(f"Initializing {self_pickles_path}...")
                # load remaining dependencies
                for dependency in cls.dependencies:
                    if dependency.__name__ not in dependencies_dict:
                        with open(cls.dependency_instance_pickle_path(config, dependency), 'rb') as instance_f:
                            dependency_instance = pickle.load(instance_f)
                            dependency_instance.update_config(config)
                            dependencies_dict[dependency.__name__] = dependency_instance

                # initialize new instance
                if not parameters_dict:
                    print("Using default parameters...")
                instance = cls(config=config, d=dependencies_dict, **parameters_dict)

                # dump instance and dependency time stamps
                with open(instance_pickle_path, 'wb') as instance_f:
                    pickle.dump(instance, instance_f)
                with open(times_pickle_path, 'wb') as times_f:
                    pickle.dump(cls.dependency_instance_times(config), times_f)

            return instance

    @staticmethod
    def dependency_instance_pickle_path(config: Config, dependency):
        return f"{dependency.complete_path(config.pickles_root_path, config.identifier, config.variants)}/instance"

    @classmethod
    def dependency_instance_times(cls, config: Config):
        latest_times = {}
        for dependency in cls.dependencies:
            latest_times[dependency.__name__] = os.path.getmtime(cls.dependency_instance_pickle_path(config, dependency))
        return latest_times

    @staticmethod
    def changed(path, something, extractor=lambda x: x):
        """Checks whether something matches with a previous pickled version. If there
        is no match, a new pickle is created.

        Args:
            path (string): Path where pickles will be stored.
            something (object): Function, class, built-in type, etc. to compare.
            extractor (function): A function that extracts the relevant part to be compared.
        Returns:
            (bool): Whether the source code matched that of a previous pickled version.
        """
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                if extractor(something) == pickle.load(f):
                    return False

        with open(path, 'wb') as f:
            pickle.dump(extractor(something), f)
        return True


def remove_older_than(hours, path):
    threshold = time.time() - 3600 * float(hours)
    for folder in os.walk(path, topdown=False):
        folder_path = folder[0]
        for file in folder[2]:
            file_path = f"{folder_path}/{file}"
            if os.stat(file_path).st_mtime < threshold:
                os.remove(file_path)
        if len(os.listdir(folder_path)) == 0:
            os.rmdir(folder_path)


if __name__ == "__main__":
    class A(SmartClass):
        def __init__(self, config, d, a):
            SmartClass.__init__(self, config, d)
            self.a = a

    class B(SmartClass):
        dependencies = (A,)

        def __init__(self, config, d, b):
            SmartClass.__init__(self, config, d)
            self.b = b

    my_config = Config(identifier=1, variants={'A': 'Special', 'B': 'Cool'}, pickle_instances=True,
                       parameters_path="parameters")
    my_b = B.current_instance(my_config)

    print(my_b.b)

    # remove_older_than(0, "pickles")

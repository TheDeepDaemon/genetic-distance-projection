import json
import numpy as np
from .dim_reduction import (
    reduce_using_neural_net, reduce_using_simple_neural_net, reduce_using_pca, reduce_using_svd, reduce_using_mds)
from .visualization import calc_colors_by_fitness, calc_colors_by_group, visualize_genomes2D, visualize_genomes3D
from enum import Enum
import zipfile
import io
import os
import networkx as nx
from matplotlib.patches import Patch


class ReductionType(Enum):
    NEURAL_NET = 1
    SIMPLE_NEURAL_NET = 2
    MDS = 3
    PCA = 4
    SVD = 5


class GenomeData:

    # the acceptable string arguments for each reduction type
    _REDUCTION_TYPE_OPTIONS = {
        ReductionType.NEURAL_NET: ['nn', 'neural_network'],
        ReductionType.SIMPLE_NEURAL_NET: ['snn', 'simple_neural_network'],
        ReductionType.MDS: ['mds', 'multi_dimensional_scaling'],
        ReductionType.PCA: ['pca', 'principal_component_analysis'],
        ReductionType.SVD: ['svd', 'singular_value_decomposition']}

    # the function used to perform each type of reduction
    _REDUCTION_TYPE_FUNCTIONS = {
        ReductionType.NEURAL_NET: lambda genome_data_mat, args: reduce_using_neural_net(
            genome_data_mat=genome_data_mat, args=args),
        ReductionType.SIMPLE_NEURAL_NET: lambda genome_data_mat, args: reduce_using_simple_neural_net(
            genome_data_mat=genome_data_mat, args=args),
        ReductionType.MDS: lambda genome_data_mat, args: reduce_using_mds(
            genes_matrix=genome_data_mat, reduced_size=2, random_state=np.random.randint(0, 10 ** 9)),
        ReductionType.PCA: lambda genome_data_mat, args: reduce_using_pca(
            genes_matrix=genome_data_mat, reduced_size=2),
        ReductionType.SVD: lambda genome_data_mat, args: reduce_using_svd(
            genes_matrix=genome_data_mat, reduced_size=2)}

    def __init__(self):
        self.index_to_id = None
        self.genome_data_mat = None
        self.position_data = None
        self.reduction_type_used = None
        self.genome_fitnesses = None

        self.genome_ids = None
        self.relations = None
        self.genome_colors = None
        self.legend_handles = None

    def init_data(self, data):
        """
        Load the genome data_storage from a dictionary.

        Args:
            data: A dictionary containing the genome values corresponding to each ID.
        """

        if not isinstance(data, dict):
            raise TypeError(f"Input to GenomeData class must be a dictionary.")

        values = data.values()

        if all(isinstance(v, list) and all(isinstance(x, int) for x in v) for v in values):

            for k in data:
                inner_list = data[k]
                inner_dict = dict()
                for val in inner_list:
                    inner_dict[val] = True

                data[k] = inner_dict

        elif not all(isinstance(v, dict) for v in values):
            raise TypeError("Dictionary contains mixed types or invalid elements.")

        data_keys = list(set(data.keys()))
        data_keys.sort()
        self.index_to_id = np.array(data_keys, dtype=int)
        id_to_index = self.get_id_to_index()

        inner_keys = list({key for inner_dict in data.values() for key in inner_dict.keys()})
        inner_key_to_index = {k: i for i, k in enumerate(inner_keys)}

        genome_data_mat = np.zeros((len(id_to_index), len(inner_keys)), dtype=np.float32)

        for outer_key, inner_dict in data.items():
            for inner_key, val in inner_dict.items():
                outer_idx = id_to_index[outer_key]
                inner_idx = inner_key_to_index[inner_key]
                genome_data_mat[outer_idx, inner_idx] = val

        self.genome_data_mat = genome_data_mat
        self.position_data = None

    def get_id_to_index(self):
        return {k: i for i, k in enumerate(self.index_to_id)}

    def init_graph_data(self, genome_ids, relations):
        self.genome_ids = np.array(genome_ids, dtype=np.int64)
        self.relations = np.array(relations, dtype=np.int64)

    def make_graph(self):
        """
        Convert this data_storage to a networkx graph that is usable.

        Returns: A networkx graph of the relations between the genomes.
        """
        graph = nx.DiGraph()
        for genome_id in self.genome_ids:
            graph.add_node(genome_id)

        for id_pair in self.relations:
            if (id_pair[0] in graph.nodes) and (id_pair[1] in graph.nodes):
                graph.add_edge(id_pair[0], id_pair[1])

        return graph

    def set_genome_fitnesses(self, fitnesses: dict):
        """
        Set the fitnesses of the genomes.

        Args:
            fitnesses: A dictionary that maps the genome ID to fitness.
        """

        genome_fitnesses = np.zeros(len(self.index_to_id), dtype=np.float64)
        for idx, gid in enumerate(self.index_to_id):
            genome_fitnesses[idx] = fitnesses[gid]

        self.genome_fitnesses = genome_fitnesses

    def save_data(self, zip_fpath: str):
        """
        Save all data_storage to files in the specified directory.

        Args:
            zip_fpath: The zip file to save to.
        """

        with zipfile.ZipFile(zip_fpath, "w", compression=zipfile.ZIP_DEFLATED) as zipf:

            # there should be genome data
            assert(self.index_to_id is not None)
            assert(self.genome_data_mat is not None)

            # iterate through every attribute that is a numpy array
            for k, v in vars(self).items():
                if isinstance(v, np.ndarray):
                    arr_buffer = io.BytesIO()
                    np.save(arr_buffer, v, allow_pickle=False)
                    zipf.writestr(f"{k}.npy", arr_buffer.getvalue())

            rt = str(int(self.reduction_type_used.value)) if self.reduction_type_used is not None else None
            info = {
                "reduction_type": rt}

            zipf.writestr("info.json", json.dumps(info))

    def load_data(self, zip_fpath: str):
        """
        Load all data_storage to files from the specified directory.

        Args:
            zip_fpath: The directory to load from.
        """

        with zipfile.ZipFile(zip_fpath, "r") as zipf:

            # iterate through files in the zip file
            for listed_name in zipf.namelist():

                # get the path and the filename
                folder, fname = os.path.split(listed_name)

                # only look at files in the root zip directory
                if folder == '':

                    # get the name and the extension
                    field_name, ext = os.path.splitext(fname)

                    # only look at stored numpy arrays
                    if ext.lower() == ".npy":

                        # load the array
                        with zipf.open(listed_name) as f:
                            arr = np.load(f, allow_pickle=False)

                            # set the attribute based on the filename
                            setattr(self, field_name, arr)

                elif folder == 'visual_data_container':

                    # get the name and the extension
                    field_name, ext = os.path.splitext(fname)

                    # only look at stored numpy arrays
                    if ext.lower() == ".npy":

                        # load the array
                        with zipf.open(listed_name) as f:
                            arr = np.load(f, allow_pickle=False)

                            # set the attribute based on the filename
                            setattr(self, field_name, arr)

            info_fname = "info.json"
            if info_fname in zipf.namelist():
                with zipf.open(info_fname) as f:
                    info = json.loads(f.read().decode())
                    rt = info["reduction_type"]
                    if rt is not None:
                        self.reduction_type_used = ReductionType(int(rt))

    def reduce_genome(self, reduction_type: str, args: dict=None):
        """
        Perform dimensionality reduction on the genome data_storage.

        Args:
            reduction_type: The type of dimensionality reduction to use.
            args: Any program arguments.
        """

        # iterate through all reduction types, check if each has been selected
        for rt, str_labels in GenomeData._REDUCTION_TYPE_OPTIONS.items():

            # if this reduction type argument is in the list of possible reduction types
            if reduction_type.lower() in str_labels:

                # perform dimensionality reduction with the selected reduction type
                self.position_data = GenomeData._REDUCTION_TYPE_FUNCTIONS[rt](self.genome_data_mat, args)
                self.reduction_type_used = rt # keep track of which reduction type was used
                return # return when done so only one can be used

        raise ValueError(f"Reduction type not recognized: {reduction_type}")

    def set_colors_by_fitness(self, fitness_values, col_low, col_high):
        """
        Set the colors of the genomes based on fitness values.

        Args:
            fitness_values: A dict mapping genome ID to fitness.
            col_low: The color indicating low fitness.
            col_high: The color indicating high fitness.
        """

        self.legend_handles = [
            Patch(color=col_low, label='Low Loss'),
            Patch(color=col_high, label='High Loss'),
        ]

        self.genome_colors = calc_colors_by_fitness(
            fitness_values=fitness_values, col_low=col_low, col_high=col_high)

    def set_genome_colors_by_group(self, genome_groups):
        """
        Set the colors of the genomes based on what group they belong to.

        Args:
            genome_groups: The group number for each genome.
        """
        self.genome_colors = calc_colors_by_group(genome_groups=genome_groups)

    def set_genome_colors(self, genome_colors: dict):
        """
        Set the colors of all the genomes.

        Args:
            genome_colors: The color for each genome.
        """
        self.genome_colors = genome_colors

    def transform_positions01(self, best_genome_id, root_genome_id=None):
        """
        Perform a translation, rotation, and scale that positions the points so the starting genome position is [0, 0] and it ends up at [0, 1].

        Args:
            best_genome_id: The ID of the best genome.
            root_genome_id: The ID of the root genome.
        """

        # if there is a root genome passed in, use it to center the data
        if root_genome_id is not None:

            # use the ID to get the index of the root genome
            root_genome_idx = np.where(self.index_to_id == root_genome_id)[0][0]

            root_genome_pos = self.position_data[root_genome_idx]

            # center it at the root genome
            self.position_data -= root_genome_pos

        # use the ID to get the index of the best genome
        best_genome_idx = np.where(self.index_to_id == best_genome_id)[0][0]

        # get the position of the best genome
        best_genome_pos = self.position_data[best_genome_idx]

        # get the magnitude of the position of the best genome
        best_genome_magnitude = np.linalg.norm(best_genome_pos)

        # get the angle of rotation
        theta = np.arctan2(float(best_genome_pos[1]), float(best_genome_pos[0]))

        # get the sin and cos for the rotation matrix
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # create the rotation matrix
        rotation_mat = np.array(
            [
                [cos_theta, -sin_theta],
                [sin_theta, cos_theta]],
            dtype=self.position_data.dtype)

        # make sure it is normalized to a 0 to 1 range
        rotation_mat /= best_genome_magnitude

        # rotated positions
        rotated_positions = np.dot(self.position_data, rotation_mat)

        # set the position data
        self.position_data = rotated_positions

    def package_args(self, args, **kwargs):
        """
        Put all the arguments and keyword arguments in the same dictionary, along with any attributes.

        Args:
            args: The existing program arguments.

        Returns:
            The combined dictionary.
        """

        packaged_args = dict()
        packaged_args.update(args) # program arguments
        packaged_args.update(vars(self)) # object attributes
        packaged_args.update(kwargs) # keyword arguments

        return packaged_args

    def visualize_genomes2D(
            self,
            save_fpath: str,
            args):
        """
        Perform 2D visualizations, save to an image file.

        Args:
            save_fpath: The filepath to save it to.
            args: Program arguments and their keywords.
        """

        if self.genome_colors is None:
            print("Genome colors are not set!")
            return

        if self.reduction_type_used is not None:

            positions = {gid: self.position_data[idx] for idx, gid in enumerate(self.index_to_id)}

            visualize_genomes2D(args=self.package_args(
                args, positions=positions, save_fpath=save_fpath, graph=self.make_graph()))
        else:
            print("You must reduce the genomes to positions before displaying.")

    def visualize_genomes3D(
            self,
            save_fpath: str,
            args: dict):
        """
        Perform the 3D visualization, save to a GIF.

        Args:
            save_fpath: The filepath to save it to.
            args: Program arguments and their keywords.
        """

        if self.genome_colors is None:
            print("Genome colors are not set!")
            return

        if self.reduction_type_used is not None:

            positions = {
                self.index_to_id[i]: (self.index_to_id[i], x, y) for i, (x, y) in enumerate(self.position_data)}

            visualize_genomes3D(args=self.package_args(
                args, positions=positions, save_fpath=save_fpath, graph=self.make_graph()))
        else:
            print("You must reduce the genomes to positions before displaying.")


def join_genomes(genome_data1: GenomeData, genome_data2: GenomeData):
    """
    Combine the genome data_storage from the two objects. Matches them according to the permutation of the first genome data_storage.

    Args:
        genome_data1: The first genome data_storage object.
        genome_data2: The second genome data_storage object.

    Returns: The combined genome data_storage object.
    """

    # the output data_storage matrix type should match both
    new_mat_dtype = genome_data1.genome_data_mat.dtype
    assert(new_mat_dtype == genome_data2.genome_data_mat.dtype)

    # get the IDs that are in both genome data_storage objects
    combined_index_to_id = np.array(list(set(genome_data1.index_to_id).intersection(set(genome_data2.index_to_id))))

    # instantiate the new genome data_storage matrix
    gd1_cols = genome_data1.genome_data_mat.shape[1]
    gd2_cols = genome_data2.genome_data_mat.shape[1]
    new_mat_shape = (len(combined_index_to_id), (gd1_cols + gd2_cols))
    combined_genome_data_mat = np.zeros(new_mat_shape, dtype=new_mat_dtype)

    # dictionaries to map IDs to indices
    id_to_index1 = genome_data1.get_id_to_index()
    id_to_index2 = genome_data2.get_id_to_index()

    # pull from the rows of each
    for idx, gid in enumerate(combined_index_to_id):
        # set the first part of this row
        gd1_idx = id_to_index1[gid]
        combined_genome_data_mat[idx, :gd1_cols] = genome_data1.genome_data_mat[gd1_idx]

        # set the next part of this row
        gd2_idx = id_to_index2[gid]
        combined_genome_data_mat[idx, gd1_cols:] = genome_data2.genome_data_mat[gd2_idx]

    # create the combined genome data_storage object
    combined_genome_data = GenomeData()
    combined_genome_data.genome_data_mat = combined_genome_data_mat
    combined_genome_data.index_to_id = combined_index_to_id

    return combined_genome_data

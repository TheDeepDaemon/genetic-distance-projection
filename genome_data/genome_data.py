import numpy as np
from .dim_reduction import reduce_using_neural_net, reduce_using_pca, reduce_using_svd, reduce_using_mds
from .visualization import VisualDataContainer
from enum import Enum
import zipfile
import io


class ReductionType(Enum):
    NEURAL_NET = 1
    SIMPLE_NEURAL_NET = 2
    MDS = 3
    PCA = 4
    SVD = 5


class GenomeData:

    _REDUCTION_TYPE_OPTIONS = {
        ReductionType.NEURAL_NET: ['nn'],
        ReductionType.SIMPLE_NEURAL_NET: ['snn'],
        ReductionType.MDS: ['mds'],
        ReductionType.PCA: ['pca'],
        ReductionType.SVD: ['svd']}

    def __init__(self):
        self.index_to_id = None
        self.genome_data_mat = None
        self.position_data = None
        self.reduction_type_used = None
        self.visual_data_container = VisualDataContainer()

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
        self.visual_data_container.init_graph_data(genome_ids=genome_ids, relations=relations)

    def save_data(self, zip_fpath: str):
        """
        Save all data_storage to files in the specified directory.

        Args:
            zip_fpath: The zip file to save to.
        """

        with zipfile.ZipFile(zip_fpath, "w", compression=zipfile.ZIP_DEFLATED) as zipf:

            index_to_id_buffer = io.BytesIO()
            np.save(index_to_id_buffer, self.index_to_id, allow_pickle=False)
            zipf.writestr("index_to_id.npy", index_to_id_buffer.getvalue())

            genome_data_mat_buffer = io.BytesIO()
            np.save(genome_data_mat_buffer, self.genome_data_mat, allow_pickle=False)
            zipf.writestr("genome_data_mat.npy", genome_data_mat_buffer.getvalue())

            if self.position_data is not None:
                position_data_buffer = io.BytesIO()
                np.save(position_data_buffer, self.position_data, allow_pickle=False)
                zipf.writestr("position_data.npy", position_data_buffer.getvalue())

            genome_ids_buffer = io.BytesIO()
            np.save(genome_ids_buffer, self.visual_data_container.genome_ids, allow_pickle=False)
            zipf.writestr("genome_ids.npy", genome_ids_buffer.getvalue())

            relations_buffer = io.BytesIO()
            np.save(relations_buffer, self.visual_data_container.relations, allow_pickle=False)
            zipf.writestr("relations.npy", relations_buffer.getvalue())

            if self.reduction_type_used is not None:
                zipf.writestr("reduction_type.txt", str(int(self.reduction_type_used.value)))

    def load_data(self, zip_fpath: str):
        """
        Load all data_storage to files from the specified directory.

        Args:
            zip_fpath: The directory to load from.
        """

        with zipfile.ZipFile(zip_fpath, "r") as zipf:

            with zipf.open("index_to_id.npy") as f:
                self.index_to_id = np.load(f, allow_pickle=False)

            with zipf.open("genome_data_mat.npy") as f:
                self.genome_data_mat = np.load(f, allow_pickle=False)

            position_data_fpath = "position_data.npy"
            if position_data_fpath in zipf.namelist():
                with zipf.open(position_data_fpath) as f:
                    self.position_data = np.load(f, allow_pickle=False)

            with zipf.open("genome_ids.npy") as f:
                self.visual_data_container.genome_ids = np.load(f, allow_pickle=False)

            with zipf.open("relations.npy") as f:
                self.visual_data_container.relations = np.load(f, allow_pickle=False)

            reduction_type_fpath = "reduction_type.txt"
            if reduction_type_fpath in zipf.namelist():
                with zipf.open(reduction_type_fpath) as f:
                    rt = f.read().decode()
                    self.reduction_type_used = ReductionType(int(rt))

    def reduce_genome(self, reduction_type: str, args: dict=None):
        """
        Perform dimensionality reduction on the genome data_storage.

        Args:
            reduction_type: The type of dimensionality reduction to use.
            args: Any program arguments.
        """

        if reduction_type in GenomeData._REDUCTION_TYPE_OPTIONS[ReductionType.NEURAL_NET]:
            self.position_data = reduce_using_neural_net(
                genome_data=self.genome_data_mat, model_type='standard', args=args)

            self.reduction_type_used = ReductionType.NEURAL_NET

        elif reduction_type in GenomeData._REDUCTION_TYPE_OPTIONS[ReductionType.SIMPLE_NEURAL_NET]:
            self.position_data = reduce_using_neural_net(
                genome_data=self.genome_data_mat, model_type='simple', args=args)

            self.reduction_type_used = ReductionType.SIMPLE_NEURAL_NET

        elif reduction_type in GenomeData._REDUCTION_TYPE_OPTIONS[ReductionType.MDS]:
            random_state = np.random.randint(0, 10 ** 9)
            self.position_data = reduce_using_mds(
                genes_matrix=self.genome_data_mat, reduced_size=2, random_state=random_state)

            self.reduction_type_used = ReductionType.MDS

        elif reduction_type in GenomeData._REDUCTION_TYPE_OPTIONS[ReductionType.PCA]:
            self.position_data = reduce_using_pca(
                genes_matrix=self.genome_data_mat, reduced_size=2)

            self.reduction_type_used = ReductionType.PCA

        elif reduction_type in GenomeData._REDUCTION_TYPE_OPTIONS[ReductionType.SVD]:
            self.position_data = reduce_using_svd(
                genes_matrix=self.genome_data_mat, reduced_size=2)

            self.reduction_type_used = ReductionType.SVD

    def get_reduced_data(self):
        return {gid: self.position_data[idx] for idx, gid in enumerate(self.index_to_id)}

    def set_genome_colors_by_fitness(self, fitness_values, col_low, col_high):
        """
        Set the colors of the genomes based on fitness values.

        Args:
            fitness_values: A dict mapping genome ID to fitness.
            col_low: The color indicating low fitness.
            col_high: The color indicating high fitness.
        """
        self.visual_data_container.set_colors_by_fitness(
            fitness_values=fitness_values, col_low=col_low, col_high=col_high)

    def set_genome_colors_by_group(self, genome_groups):
        """
        Set the colors of the genomes based on what group they belong to.

        Args:
            genome_groups: The group number for each genome.
        """
        self.visual_data_container.set_colors_by_group(genome_groups=genome_groups)

    def set_genome_colors(self, genome_colors: dict):
        """
        Set the colors of all the genomes.

        Args:
            genome_colors: The color for each genome.
        """
        self.visual_data_container.set_colors(genome_colors=genome_colors)

    def visualize_genomes2D(
            self,
            save_fpath: str, args):

        if self.reduction_type_used is not None:
            self.visual_data_container.visualize_genomes2D(
                save_fpath=save_fpath,
                positions=self.get_reduced_data(),
                args=args)
        else:
            print("You must reduce the genomes to positions before displaying.")

    def visualize_genomes3D(
            self,
            save_fpath: str, args):

        if self.reduction_type_used is not None:

            if self.position_data.shape[1] == 2:
                position_data = [(self.index_to_id[i], x, y) for i, (x, y) in enumerate(self.position_data)]
                self.position_data = position_data

            self.visual_data_container.visualize_genomes3D(
                save_fpath=save_fpath,
                positions=self.get_reduced_data(),
                args=args)
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

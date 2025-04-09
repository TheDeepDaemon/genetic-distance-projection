from .dim_reduction.models import GraphingModel
from .genome_data_collector import GenomeDataCollector
from .dim_reduction import (
    reduce_using_neural_net,
    reduce_using_simple_neural_net,
    reduce_using_pca,
    reduce_using_svd,
    reduce_using_mds,
    reduce_using_t_sne)
from ..program_arguments import ProgramArguments
from enum import Enum
import zipfile
import io
import os
import datetime
import json
import numpy as np


class IdentifierMismatchError(Exception):
    def __init__(self, expected, received):
        self.expected = expected
        self.received = received
        message = f"Identifying argument mismatch:\nExpected: {expected}\nReceived: {received}"
        super().__init__(message)


class ReductionType(Enum):
    NEURAL_NET = 1
    SIMPLE_NEURAL_NET = 2
    MDS = 3
    PCA = 4
    SVD = 5
    T_SNE = 6


class GenomeMatrix:

    # the acceptable string arguments for each reduction type
    _REDUCTION_TYPE_OPTIONS = {
        ReductionType.NEURAL_NET: ['nn', 'neural_network'],
        ReductionType.SIMPLE_NEURAL_NET: ['snn', 'simple_neural_network'],
        ReductionType.MDS: ['mds', 'multi_dimensional_scaling'],
        ReductionType.PCA: ['pca', 'principal_component_analysis'],
        ReductionType.SVD: ['svd', 'singular_value_decomposition'],
        ReductionType.T_SNE: ['t-sne', 'tsne', 't-stochastic_neighbor_embedding']}

    # the function used to perform each type of reduction
    _REDUCTION_TYPE_FUNCTIONS = {
        ReductionType.NEURAL_NET: lambda self, args: reduce_using_neural_net(
            genome_data_mat=self.genome_data_mat, args=args, model_save_fname=self.generate_model_fname(args)),
        ReductionType.SIMPLE_NEURAL_NET: lambda self, args: reduce_using_simple_neural_net(
            genome_data_mat=self.genome_data_mat, args=args, model_save_fname=self.generate_model_fname(args)),
        ReductionType.MDS: lambda self, args: reduce_using_mds(
            genes_matrix=self.genome_data_mat, reduced_size=2),
        ReductionType.PCA: lambda self, args: reduce_using_pca(
            genes_matrix=self.genome_data_mat, reduced_size=2),
        ReductionType.SVD: lambda self, args: reduce_using_svd(
            genes_matrix=self.genome_data_mat, reduced_size=2),
        ReductionType.T_SNE: lambda self, args: reduce_using_t_sne(
            genes_matrix=self.genome_data_mat, reduced_size=2)}

    def __init__(self):
        self.index_to_id = None
        self.genome_data_mat = None
        self.position_data = None
        self.reduction_type_used = None
        self.model_save_fname = None

    def init_data(self, genome_data_collector: GenomeDataCollector):
        """
        Load the genome data_storage from a dictionary.

        Args:
            genome_data_collector: The object that has been used to collect all genome data.
        """

        genes_matrix, genome_ids, gene_keys = genome_data_collector.convert_to_matrix()

        self.genome_data_mat = genes_matrix.astype(dtype=np.float32)
        self.index_to_id = np.array(genome_ids, dtype=np.int64)
        self.position_data = None

    def get_id_to_index(self):
        return {k: i for i, k in enumerate(self.index_to_id)}

    def save_data(self, zip_fpath: str, identifying_args: dict=None):
        """
        Save all data_storage to files in the specified directory.

        Args:
            zip_fpath: The zip file to save to.
            identifying_args: Keywords and values used to identify the options used for the genome data.
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

            dt = datetime.datetime.now()

            info = {
                "reduction_type": rt,
                "date_time": dt.isoformat(),
                "identifying_args": identifying_args,
                "model_save_fname": self.model_save_fname
            }

            zipf.writestr("info.json", json.dumps(info))

            print(f"Genome data saved to {zip_fpath}")

    @staticmethod
    def _get_info(zipf):
        info_fname = "info.json"
        if info_fname in zipf.namelist():
            with zipf.open(info_fname) as f:
                return json.loads(f.read().decode('utf-8'))

    def load_data(self, zip_fpath: str, identifying_args: dict):
        """
        Load all data_storage to files from the specified directory.

        Args:
            zip_fpath: The directory to load from.
            identifying_args: Keywords and values used to identify the options used for the genome data.
        """

        with zipfile.ZipFile(zip_fpath, "r") as zipf:

            info = GenomeMatrix._get_info(zipf)

            loaded_id_args = info["identifying_args"]

            if loaded_id_args != identifying_args:
                raise IdentifierMismatchError(identifying_args, loaded_id_args)

            rt = info["reduction_type"]
            if rt is not None:
                self.reduction_type_used = ReductionType(int(rt))

            if "model_save_fname" in info:
                self.model_save_fname = info["model_save_fname"]

            # iterate through files in the zip file
            for fname in zipf.namelist():

                # get the name and the extension
                field_name, ext = os.path.splitext(fname)

                # only look at stored numpy arrays
                if ext.lower() == ".npy":

                    # load the array
                    with zipf.open(fname) as f:
                        arr = np.load(f, allow_pickle=False)

                        # set the attribute based on the filename
                        setattr(self, field_name, arr)

            print(f"Genome data loaded from {zip_fpath}")

    @staticmethod
    def get_save_datetime(zip_fpath: str):
        """
        Get the date and time that the genome data object was saved.

        Args:
            zip_fpath: The path of the save data zip file.

        Returns:
            The datetime object with the date and time it was saved at.
        """
        with zipfile.ZipFile(zip_fpath, "r") as zipf:
            info = GenomeMatrix._get_info(zipf)
            return datetime.datetime.fromisoformat(info["date_time"])

    @staticmethod
    def get_info(zip_fpath: str):
        """
        Get the date and time that the genome data object was saved.

        Args:
            zip_fpath: The path of the save data zip file.

        Returns:
            The datetime object with the date and time it was saved at.
        """
        with zipfile.ZipFile(zip_fpath, "r") as zipf:
            return GenomeMatrix._get_info(zipf)

    @staticmethod
    def info_matches(zip_fpath: str, identifying_args: dict):
        info = GenomeMatrix.get_info(zip_fpath)
        return info["identifying_args"] == identifying_args

    @staticmethod
    def find_latest_genome_data(data_dir, identifying_args: dict=None):
        """
        Find which genome data was created last.

        Args:
            data_dir: The data directory.
            identifying_args: Keywords and values used to identify the options used for the genome data.

        Returns:
            The filename of the latest genome data in that directory.
        """

        datetimes = dict()

        for fname in os.listdir(data_dir):
            _, f_ext = os.path.splitext(fname)

            # if it is a zip file
            if f_ext == ".zip":
                zip_fpath = os.path.join(data_dir, fname)

                # make sure the info matches, then consider it
                if GenomeMatrix.info_matches(zip_fpath=zip_fpath, identifying_args=identifying_args):
                    datetimes[fname] = GenomeMatrix.get_save_datetime(zip_fpath)

        if len(datetimes) > 0:
            return max(datetimes, key=datetimes.get)
        else:
            raise FileNotFoundError(f"The \'{data_dir}\' directory doesn't contain available genome data saves "
                                    f"with matching keyword arguments.")

    def reduce_genome(self, reduction_type: str, args: ProgramArguments):
        """
        Perform dimensionality reduction on the genome data_storage.

        Args:
            reduction_type: The type of dimensionality reduction to use.
            args: Any program arguments.
        """

        # iterate through all reduction types, check if each has been selected
        for rt, str_labels in GenomeMatrix._REDUCTION_TYPE_OPTIONS.items():

            # if this reduction type argument is in the list of possible reduction types
            if reduction_type.lower() in str_labels:

                # perform dimensionality reduction with the selected reduction type
                self.position_data = GenomeMatrix._REDUCTION_TYPE_FUNCTIONS[rt](self, args)
                self.reduction_type_used = rt # keep track of which reduction type was used
                return # return when done so only one can be used

        raise ValueError(f"Reduction type not recognized: {reduction_type}")

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

    def get_positions(self):
        """
        Constructs a dictionary containing the position for each genome ID.

        Returns:
            The dictionary of positions.
        """
        return {gid: self.position_data[idx] for idx, gid in enumerate(self.index_to_id)}

    def get_unreduced_gene_vectors(self):
        """
        Constructs a dictionary containing the position for each genome ID.

        Returns:
            The dictionary of positions.
        """
        return {gid: self.genome_data_mat[idx] for idx, gid in enumerate(self.index_to_id)}

    def generate_model_fname(self, args: ProgramArguments):
        details = f"{args.run_type}_{args.reduction_type}_{args.epochs}_{args.batch_size}_{args.learning_rate}"
        dt = datetime.datetime.now()
        fname = f"model_save_{details}_{dt.strftime("%Y-%m-%d_%H-%M-%S")}.pth"
        self.model_save_fname = fname
        return fname

    def get_model(self, args: ProgramArguments):
        return GraphingModel.load(os.path.join(args.model_save_dir, self.model_save_fname))

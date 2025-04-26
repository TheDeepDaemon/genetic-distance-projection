import json
import os
import zipfile
from datetime import datetime
from typing import Union
import copy


class GenomeData:

    def __init__(self, source: Union['GenomeData', str, os.PathLike]=None, *args, **kwargs):
        if isinstance(source, GenomeData):
            genome_data: GenomeData = source
            self._population = copy.deepcopy(genome_data._population)
            self._population_info = copy.deepcopy(genome_data._population_info)
        elif isinstance(source, (str, os.PathLike)):
            self._load_from_path(zip_fpath=source)
        else:
            self._population = dict()
            self._population_info = dict()

        if "minimizing_fitness" in kwargs:
            self._minimizing_fitness = kwargs["minimizing_fitness"]
        else:
            self._minimizing_fitness = True

    def save(self, zip_fpath: Union[str, os.PathLike], **kwargs):
        """
        Save the genome data to a zip file containing a set of JSONs.

        Args:
            zip_fpath: The filepath of the zip file.
        """
        with zipfile.ZipFile(zip_fpath, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            self._save_contents(zip_file=zipf, **kwargs)

        print(f"Genome data saved to {zip_fpath}")

    def _save_contents(self, zip_file, **kwargs):
        """
        Save the contents of this object to an opened zip file.

        Args:
            zip_file: The file (not path) we are writing to.
        """
        kwargs.setdefault('indent', 4)
        zip_file.writestr("population.json", json.dumps(self._population, **kwargs))
        zip_file.writestr("population_info.json", json.dumps(self._population_info, **kwargs))

    @staticmethod
    def load(zip_fpath: Union[str, os.PathLike]):
        """
        Load a genome data object from a zip file.

        Args:
            zip_fpath: The filepath to load from.

        Returns:
            A genome data object.
        """
        genome_data: GenomeData = GenomeData()
        genome_data._load_from_path(zip_fpath=zip_fpath)
        return genome_data

    def _load_from_path(self, zip_fpath: Union[str, os.PathLike]):
        """
        Given a path, load a zip file to this object.

        Args:
            zip_fpath: The path to load from.
        """
        with zipfile.ZipFile(zip_fpath, "r") as zipf:
            self._load_contents(zip_file=zipf)

    def _load_contents(self, zip_file):
        """
        Load the contents of the zip file to this object.

        Args:
            zip_file: The zip file (not path) we are reading from.
        """

        with zip_file.open("population.json") as f:
            population = json.loads(f.read().decode('utf-8'))
            self._population = {int(k): v for k, v in population.items()}

        with zip_file.open("population_info.json") as f:
            population_info = json.loads(f.read().decode('utf-8'))
            self._population_info = {int(k): v for k, v in population_info.items()}

    @staticmethod
    def _get_info(zip_file):
        """
        Get the info from this zip file.

        Args:
            zip_file: The actual file.

        Returns:
            A dictionary.
        """
        info_fname = "info.json"
        if info_fname in zip_file.namelist():
            with zip_file.open(info_fname) as f:
                return json.loads(f.read().decode('utf-8'))

    @staticmethod
    def get_info(zip_fpath: str):
        """
        Get the info from this zip file.

        Args:
            zip_fpath: The zip file path.

        Returns:
            A dictionary.
        """
        with zipfile.ZipFile(zip_fpath, "r") as zipf:
            return GenomeData._get_info(zipf)

    @staticmethod
    def get_save_datetime(zip_fpath: str):
        """
        Get the datatime of this save.

        Args:
            zip_fpath: The zip file path.

        Returns:
            The datetime object.
        """
        with zipfile.ZipFile(zip_fpath, "r") as zipf:
            info = GenomeData._get_info(zipf)
            return datetime.fromisoformat(info["date_time"])

    @staticmethod
    def info_matches(zip_fpath: str, identifying_args: dict):
        """
        Check if all info matches.

        Args:
            zip_fpath: The zip file path.
            identifying_args: The identifiers.

        Returns:
            Whether it matches.
        """
        info = GenomeData.get_info(zip_fpath)
        return info["identifying_args"] == identifying_args

    @staticmethod
    def get_all_matching_saves(dir_path, identifying_args):
        """
        Get a list of the saves with matching identifiers.

        Args:
            dir_path: The directory to search.
            identifying_args: The identifiers.

        Returns:
            A list of file paths.
        """
        # get all valid files
        save_paths = [
            os.path.join(dir_path, fname) for fname in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, fname)) and (os.path.splitext(fname)[1].lower() == ".zip")]

        # return the ones that have matching identifying arguments
        return [
            fpath for fpath in save_paths
            if GenomeData.info_matches(fpath, identifying_args)]

    @staticmethod
    def find_latest_genome_data(dir_path, identifying_args: dict):
        """
        Find the latest genome data save that matches.

        Args:
            dir_path: The directory to search.
            identifying_args: The identifiers.

        Returns:
            The file path of the save.
        """

        datetimes = dict()

        matching_save_paths = GenomeData.get_all_matching_saves(dir_path, identifying_args)

        for fpath in matching_save_paths:
            datetimes[fpath] = GenomeData.get_save_datetime(fpath)

        if len(datetimes) > 0:
            return max(datetimes, key=datetimes.get)
        else:
            raise FileNotFoundError(f"The \'{dir_path}\' directory doesn't contain available genome data saves "
                                    f"with matching keyword arguments.")

    def get_unique_genome_id_list(self):
        pop_keys = self._population.keys()
        assert (pop_keys == self._population_info.keys())
        return list(pop_keys)

    def get_unique_gene_key_list(self):
        unique_gene_keys = set()
        for genome in self._population.values():
            assert(isinstance(genome, dict))
            unique_gene_keys.update(set(genome.keys()))
        return list(unique_gene_keys)

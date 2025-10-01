"""
 ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___
This code block is just here to make is easy to import the GDP module from this particular file.
It is not a part of the example, so you can ignore it.
Note: I know this is a hacky way to do it, but if you are just trying to run an example script,
it requires minimal input on your part.
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
"""
 ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___ ___
"""
import os
from gdp import GenomeDataCollector
import json


def convert_format(data_directory: str, output_zip_fpath: str):
    """
    Convert the format used for EXAMM outputs, a directory containing json files, to a single json.

    Args:
        data_directory: The directory containing the json files.
    """

    data_collector = GenomeDataCollector()

    for fname in os.listdir(data_directory):

        fpath = os.path.join(data_directory, fname)
        _, f_ext = os.path.splitext(fname)

        if os.path.isfile(fpath) and (f_ext == ".json"):

            with open(fpath, 'r', encoding='utf-8') as f:

                json_info = json.load(f)

                genome_id = json_info["generation_number"]

                data_collector.add_population_member(genome_id)
                data_collector.set_population_member_info(genome_id=genome_id, info=json_info)

    fname_, ext_ = os.path.splitext(output_zip_fpath)

    if ext_.lower() != ".zip":
        output_zip_fpath = f"{output_zip_fpath}.zip"

    data_collector.save(output_zip_fpath)
    print(f"Saved zip to: {output_zip_fpath}")


if __name__=="__main__":

    input_dir = "examm_neat_data"
    output_dir = "formatted_data"

    for parent_dname in os.listdir(input_dir):

        parent_dpath = os.path.join(input_dir, parent_dname)

        if os.path.isdir(parent_dpath):

            for dname in os.listdir(parent_dpath):

                dpath = os.path.join(parent_dpath, dname)

                if os.path.isdir(dpath):

                    output_name = f"{parent_dname}_{dname}"
                    output_path = os.path.join(output_dir, output_name)

                    convert_format(data_directory=dpath, output_zip_fpath=output_path)

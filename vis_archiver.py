import os
import shutil

if __name__=="__main__":

    # create an archive directory if it doesn't exist
    os.makedirs("vis_archive", exist_ok=True)

    # move each file in the "vis_output" directory over to the archive
    for fname in os.listdir("vis_output"):
        source = os.path.join("vis_output", fname)
        destination = os.path.join("vis_archive", fname)
        shutil.move(source, destination)

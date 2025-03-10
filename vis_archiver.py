import os
import shutil

if __name__=="__main__":

    os.makedirs("vis_archive", exist_ok=True)

    for fname in os.listdir("vis_output"):
        source = os.path.join("vis_output", fname)
        destination = os.path.join("vis_archive", fname)
        shutil.move(source, destination)

import time
from sys import argv

from src.datasets.utils import get_dataset

if __name__ == "__main__":
    name = argv[1]
    print(f"Generate and save dataset {name}")
    start = time.time()
    _ = get_dataset(name)
    end = time.time()
    print(f"Finished ! Time elapsed: {end - start}")

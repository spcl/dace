import argparse
import numpy as np
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, nargs="?", default="kronecker")
    args = vars(parser.parse_args())

    print(f"Numpifying folder {args['folder']}", flush=True)

    for filename in os.listdir(args['folder']):
        if filename.endswith('.el'):
            input_file = os.path.join(args['folder'], filename)
            data = np.genfromtxt(input_file, dtype=np.int32)
            output_file = os.path.join(args['folder'], f'{filename[:-3]}.npy')
            np.save(output_file, data)

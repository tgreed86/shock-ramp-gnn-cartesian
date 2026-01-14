#!/usr/bin/env python3
import h5py
import numpy as np
import argparse
import pprint

def dump_dataset(name, obj, out_file):
    """Write dataset contents to a text file."""
    if isinstance(obj, h5py.Dataset):
        out_file.write("=" * 80 + "\n")
        out_file.write(f"DATASET: {name}\n")
        out_file.write(f"  Shape: {obj.shape}\n")
        out_file.write(f"  Dtype: {obj.dtype}\n")
        out_file.write("-" * 80 + "\n")

        # Load dataset into memory
        data = obj[()]

        # Format depends on dimensionality
        if np.isscalar(data):
            out_file.write(f"{data}\n")
        else:
            # Pretty-print small arrays
            if data.size <= 5000:
                out_file.write(pprint.pformat(data) + "\n")
            else:
                # If too large, summarize first few entries
                flat = data.reshape(-1)
                out_file.write("Array too large, showing first 200 entries only:\n")
                out_file.write(pprint.pformat(flat[:200]) + "\n")

        out_file.write("\n")

def hdf5_to_txt(input_path, output_path):
    print(f"Reading HDF5 file: {input_path}")
    with h5py.File(input_path, "r") as f, open(output_path, "w") as out_file:
        out_file.write(f"HDF5 FILE DUMP: {input_path}\n\n")
        f.visititems(lambda name, obj: dump_dataset(name, obj, out_file))

    print(f"Finished. Output written to:\n  {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, required=True,
                        help="Path to the .hdf5 file")
    parser.add_argument("--outfile", type=str, default="hdf5_dump.txt",
                        help="Path to output .txt file")
    args = parser.parse_args()

    hdf5_to_txt(args.infile, args.outfile)

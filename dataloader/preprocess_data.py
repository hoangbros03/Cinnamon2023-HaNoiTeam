# Preprocessing data file, SHOULDN'T RUN ON PERSONAL COMPUTER
import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# CONSTANT VARIABLES
TOTAL_INPUT_CHARACTER = 22
MAX_CHARACTERS = 500
DIMS = 5

# Configure
np.set_printoptions(threshold=sys.maxsize)


def get_characters_embedding(dim: int = 5) -> dict:
    """
    Get embedding of characters. Use PCA
    to reduce the dim
    Parameters
    ----------
    dim: Dimension of each character embedding
    Returns
    -------
    A dictionary containing character embedding
    """
    command = "cat word2vec_vi_syllables_100dims.txt |\
          grep -e '^[abcdefghijklmnopqrstuvwxyz] '"

    # Run the command and capture the output
    completed_process = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, text=True
    )

    # Get the output from the completed process
    output = completed_process.stdout

    # Process to 2D list
    output = [i.split(" ") for i in output.split("\n")]
    output.pop()

    # Get number embedding only
    number_only_arr = np.array(output)[:, 1:].astype(np.float)

    # Create dataframe
    df = pd.DataFrame(number_only_arr)
    df["Character"] = np.array(output)[:, 0]
    df.set_index("Character", inplace=True)

    # Apply PCA
    pca = PCA(n_components=5)
    pca.fit(df)
    pca_df = pca.transform(df)
    pca_df = pd.DataFrame(pca_df)
    pca_df["Character"] = np.array(output)[:, 0]
    pca_df.set_index("Character", inplace=True)

    # Convert to dict
    input_characters_dict = {}
    for i in range(len(output)):
        input_characters_dict[output[i][0]] = np.array(pca_df.loc[output[i][0]])
    return input_characters_dict


def get_classes_for_characters_dict() -> tuple:
    """
    Function that return dicts for mapping characters to classes
    Parameters
    ----------
    Nothing
    Returns
    -------
    Some dictionaries
    """
    output_characters = "A, Á, À, Ả, Ã, Ạ, Ă, Ắ, Ằ, Ẳ, Ẵ, Ặ, Â, Ấ, Ầ, Ẩ, Ẫ, Ậ, D, Đ,\
          E, É, È, Ẻ, Ẽ, Ẹ, Ê, Ế, Ề, Ể, Ễ, Ệ, I, Í, Ì, Ỉ, Ĩ, Ị, O, Ó, Ò, Ỏ, Õ, Ọ, Ô,\
              Ố, Ồ, Ổ, Ỗ, Ộ, Ơ, Ớ, Ờ, Ở, Ỡ, Ợ, U, Ú, Ù, Ủ, Ũ, Ụ, Ư, Ứ, Ừ, Ử, Ữ, Ự"
    output_characters = output_characters.lower().split(", ")
    a_to_output = output_characters[0:18]
    d_to_output = output_characters[18:20]
    e_to_output = output_characters[20:32]
    i_to_output = output_characters[32:38]
    o_to_output = output_characters[38:56]
    u_to_output = output_characters[56:]
    return a_to_output, d_to_output, e_to_output, i_to_output, o_to_output, u_to_output


def convert_to_array(
    string: str,
    max_characters: int = MAX_CHARACTERS,
    dtype: np.dtype = np.float16,
    dims: int = DIMS,
) -> np.ndarray:
    """
    Convert a string to embbed array
    input_characters_dict must be existed before calling this function
    Parameters
    ---------
    string: Input string
    max_characters: Max characters
    dtype: Dtype of output np array
    dims: Dimension of output np array
    Returns
    -------
    A np array
    """
    arr = np.array(
        list(string[:max_characters].lower())
    )  # Convert to lowercase and convert to numpy array
    arr[arr == " "] = "space"  # Handle spaces

    default_value = np.zeros(dims)

    result = np.array(
        [input_characters_dict.get(char, default_value) for char in arr],
        dtype=dtype,
    )

    padding_length = max(0, max_characters - len(arr))
    padding = np.zeros((padding_length, dims), dtype=dtype)

    return np.concatenate((result, padding), axis=0).reshape((1, max_characters, dims))


def get_output_idx(c: str) -> int:
    """
    Function to get output index
    Parameters
    ----------
    c: The character
    Returns
    -------
    The index of character
    """
    (
        a_to_output,
        d_to_output,
        e_to_output,
        i_to_output,
        o_to_output,
        u_to_output,
    ) = tupleOutput
    if c in a_to_output:
        return a_to_output.index(c)
    if c in d_to_output:
        return d_to_output.index(c)
    if c in e_to_output:
        return e_to_output.index(c)
    if c in i_to_output:
        return i_to_output.index(c)
    if c in o_to_output:
        return o_to_output.index(c)
    if c in u_to_output:
        return u_to_output.index(c)
    else:
        return 0


def convert_to_output(string: str) -> np.ndarray:
    """
    Legacy. Added to use if needed
    """
    arr = [*string]
    result = np.zeros((MAX_CHARACTERS, 18))
    for i in range(MAX_CHARACTERS):
        if i < len(arr):
            result[i, get_output_idx(arr[i].lower())] = 1
        else:
            result[i, 0] = 1
    result = np.expand_dims(result, 0)
    # result = np.reshape(result,(1,-1,18))
    return result


def convert_to_label(string: str) -> np.ndarray:
    """
    Get label for output string
    Parameters
    ----------
    String: Input string
    Returns
    -------
    A np array
    """
    arr = [*string]
    result = np.zeros((MAX_CHARACTERS))
    for i in range(MAX_CHARACTERS):
        if i < len(arr):
            result[i] = get_output_idx(arr[i].lower())
        else:
            result[i] = 0
    result = np.expand_dims(result, 0)
    # result = np.reshape(result,(1,-1,18))
    return result


def restore_to_string(
    raw_string: str,
    arr: np.ndarray,
) -> str:
    """
    Get correct string using raw string and output of the model
    Parameters
    ----------
    raw_string: No tone string
    arr: Output np array
    Returns
    -------
    A output string
    """
    raw_characters = ["A", "D", "E", "I", "O", "U"]
    (
        a_to_output,
        d_to_output,
        e_to_output,
        i_to_output,
        o_to_output,
        u_to_output,
    ) = tupleOutput
    raw_characters_dicts = [
        a_to_output,
        d_to_output,
        e_to_output,
        i_to_output,
        o_to_output,
        u_to_output,
    ]
    raw_arr = [*raw_string]
    to_upper = False
    for i in range(len(raw_arr)):
        if raw_arr[i].upper() in raw_characters:
            if raw_arr[i].lower() != raw_arr[i]:
                to_upper = True
            result = raw_characters_dicts[raw_characters.index(raw_arr[i].upper())][
                np.int8(arr[0][i])
            ]
        else:
            continue
        if to_upper:
            raw_arr[i] = result.upper()
        else:
            raw_arr[i] = result
        # finally
        to_upper = False
    return "".join(raw_arr)


def read_file(path: str) -> list:
    """
    Function to read raw dataset
    Parameters
    ----------
    Path: path to raw dataset location
    Returns
    -------
    A list containing strings
    """
    # Using readlines()
    file1 = open(Path(path), "r")
    Lines = file1.readlines()
    file1.close()
    print(f"Total lines: {len(Lines)}")
    return Lines


def batch_process(
    type: str,
    input_path: str,
    output_path: str,
    limit: int = None,
    batch_size: int = 500,
) -> None:
    """
    Process the embedding process and write to a specified file
    Parameters
    ----------
    type: Either input of output
    input_path: Path to the raw dataset
    output_path: Path to the output file
    limit: Number of string needed
    batch_size: Choose how many string process concurrently.
    If batch_size too large, ram won't hold enough and your system can crash.
    Returns
    -------
    Nothing, but a file created
    """

    def get_nparray(func, arr: list, output_dim: int = 2500) -> np.ndarray:
        """
        Get np array from strings
        Internal function of batch_process
        Parameters
        ----------
        func: Function to process
        arr: Arr of raw string
        output_dim: Dimension of output array
        Returns
        -------
        A np array
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = list(executor.map(func, arr[:threshold]))

        # Converting the list to a NumPy array and reshaping
        result_array = np.reshape(np.array(result), (threshold, output_dim))
        return result_array

    # Get all raw string
    string_list = read_file(input_path)

    # Set limit
    if limit is not None:
        threshold = limit
    else:
        threshold = len(string_list)

    # Check type
    if type == "input":
        OUTPUT_DIM = MAX_CHARACTERS * DIMS
        fmt = "%.4e"
        func = convert_to_array
    elif type == "output":
        OUTPUT_DIM = MAX_CHARACTERS
        fmt = "%.4d"
        func = convert_to_label
    else:
        raise ValueError("Type passing to this function must be either input or output")

    # Process and write to file
    with open(output_path, "ab") as f:
        for i in range(0, min(len(string_list), threshold), batch_size):
            start = time.time()
            batch = string_list[i : i + batch_size]
            np.savetxt(
                f,
                np.reshape(
                    get_nparray(func, batch, OUTPUT_DIM), (batch_size, OUTPUT_DIM)
                ),
                fmt=fmt,
            )
            print(f"Time taken in each loop: {time.time() - start}")
    print("Process done!")


if __name__ == "__main__":
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Get preprocess data")
    parser.add_argument(
        "-i", "--input-path", type=str, required=True, help="Path of raw strings"
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Output path of output file",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        required=True,
        help="type of data. Input if raw string, output if normal Vietnamese string",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="Number of string needed to be processed",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=500,
        help="batch size when processing, \
        choose too large number can cause crashing",
    )

    args = parser.parse_args()
    input_characters_dict = get_characters_embedding()
    tupleOutput = get_classes_for_characters_dict()
    batch_process(
        args.type, args.input_path, args.output_path, args.limit, args.batch_size
    )

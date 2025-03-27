from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
import zarr
import zarr.storage
from omegaconf import OmegaConf
from pyprojroot import here
from plum import dispatch
from numpy.typing import NDArray

# Register the "here" resolver
OmegaConf.register_new_resolver("here", lambda: here())


@dataclass
class RQASettings:
    threshold: Optional[float]
    recurrence_rate: Optional[float]
    joint_pair_rec: List[List]
    indiv_joints: List[str]


@dataclass
class EMDSettings:
    cross_person: bool = False


@dataclass
class SDTWSettings:
    gamma: float = 0.1
    cross_person: bool = False


@dataclass
class Config:
    base_data_path: Path
    bvh_audio_folder_paths: Dict[str, Dict[str, Path]]  # {"person<a>": {"bvh": Path, "audio": Path}}
    rqa_settings: RQASettings
    emd_settings: EMDSettings
    sdtw_settings: SDTWSettings


@dispatch
def read_zarr_into_dict(zarr_paths: Dict[str, Path], chunk: str) -> Dict[str, Dict[str, NDArray]]:
    """
    Read multiple zarr files into a nested dictionary for a specific chunk.

    Args:
        zarr_paths: Dictionary mapping person identifiers to their zarr file paths
                    e.g., {"person1": Path("/path/to/person1.zarr.zip"), ...}
        chunk: Name of the chunk to extract data for.

    Returns:
        Dict[str, Dict[str, NDArray]]: Nested dictionary containing the zarr data:
        {
            "person1": {
                "joint1": array(...),
                "joint2": array(...),
                ...
            },
            "person2": {
                ...
            },
            ...
        }
    """
    result = {}

    for person, zarr_path in zarr_paths.items():
        store = zarr.storage.ZipStore(zarr_path, read_only=True)
        root = zarr.open_group(store=store, mode="r")

        result[person] = {}
        for joint in root[chunk].keys():
            result[person][joint] = root[chunk][joint][:]

        store.close()

    return result


@dispatch
def read_zarr_into_dict(zarr_path: Path, chunk: str):
    """
    Read a zarr file into a dictionary.
    Args:
        zarr_path: Path to the zarr file in zip
        person: Name of the person to extract data for.
    Returns:
        Dict: Dictionary containing the zarr data for the specified person.
    """
    store = zarr.storage.ZipStore(zarr_path, read_only=True)
    root = zarr.open_group(store=store, mode="r")
    res = {}
    for joint in root[chunk].keys():
        res[joint] = root[chunk][joint][:]
    store.close()
    return res


@dispatch
def read_zarr_into_dict(zarr_path: Path, chunk: str, joint: str):
    """
    Read a zarr file into a dictionary.
    Args:
        zarr_path: Path to the zarr file in zip
        person: Name of the person to extract data for.
        joint: Name of the joint to extract data for.
    Returns:
        Dict: Dictionary containing the zarr data for the specified person and joint.
    """
    store = zarr.storage.ZipStore(zarr_path, read_only=True)
    root = zarr.open_group(store=store, mode="r")

    res = root[chunk][joint][:]
    store.close()
    return res


@dispatch
def read_zarr_into_dict(zarr_path: Path):
    """
    Read a zarr file into a dictionary.
    Args:
        zarr_path: Path to the zarr file in zip format.
    Returns:
        Dict: Dictionary containing the zarr data.
    """
    store = zarr.storage.ZipStore(zarr_path, read_only=True)
    root = zarr.open_group(store=store, mode="r")
    res = {}
    for person in root.keys():
        res[person] = {}
        for joint in root[person].keys():
            res[person][joint] = root[person][joint][:]
    store.close()
    return res


def load_file_paths(mapping: Dict) -> Tuple[pl.DataFrame, Dict[str, Path]]:
    """
    Load file paths from a mapping and organize them into a polars DataFrame.

    This function processes a dictionary mapping people to their file types and paths,
    extracts BVH motion files, audio files, and chunk keys, and returns them as a structured DataFrame.

    Parameters
    ----------
    mapping : Dict
        A nested dictionary with the following structure:
        {
            person_name: {
                "bvh": Path_to_bvh_directory,
                "zarr": Path_to_zarr_file,
                "audio": Path_to_audio_directory
            },
            ...
        }

    Returns
    -------
    pl.DataFrame
        A polars DataFrame with the following columns:
        - person: The person identifier from the mapping
        - chunk: Sequential chunk number (1-indexed)
        - chunk_name: The chunk key from the zarr store
        - bvh: Path to the BVH motion file
        - audio: Path to the audio WAV file

    Notes
    -----
    - BVH files are expected to have the '.bvh' extension
    - Audio files are expected to have the '.wav' extension
    - The function assumes the number of BVH files, audio files, and chunk keys match
    """

    all_paths = []
    zarr_paths = {}
    for person, ftype_map in mapping.items():
        bvh_files = []
        audio_files = []
        chunk_keys = []
        for ftype, path in ftype_map.items():
            match ftype:
                case "bvh":
                    bvh_files = list(path.glob("*.bvh"))
                    bvh_files.sort()
                case "zarr":
                    zarr_root = zarr.open_group(
                        store=zarr.storage.ZipStore(path.expanduser(), read_only=True), mode="r"
                    )
                    chunk_keys = list(zarr_root.keys())
                    chunk_keys.sort()
                    zarr_paths[person] = path
                case "audio":
                    audio_files = list(path.glob("*.wav"))
                    audio_files.sort()
        all_paths.extend(
            [
                (person, idx + 1, chunk_key, bvh_file, audio_file)
                for idx, (bvh_file, audio_file, chunk_key) in enumerate(zip(bvh_files, audio_files, chunk_keys))
            ]
        )

    return pl.DataFrame(all_paths, schema=["person", "chunk", "chunk_name", "bvh", "audio"], orient="row"), zarr_paths

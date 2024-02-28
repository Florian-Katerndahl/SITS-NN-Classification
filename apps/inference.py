#! /usr/bin/env python

# TODO fix GPU inference: https://stackoverflow.com/questions/71278607/pytorch-expected-all-tensors-on-same-device
import argparse
from datetime import datetime
from pathlib import Path
from re import search
from typing import List, Union, Dict, Optional
import numpy as np
import rasterio
import rioxarray as rxr
import torch
import xarray
import logging
from time import time
from sits_classifier import models
from sits_classifier.utils.inference import ModelType, fp_to_doy, predict_lstm, predict_transformer, pad_doy_sequence, pad_datacube

parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description="Run inference with already trained LSTM classifier on a remote-sensing time series represented as "
                "FORCE ARD datacube.")
parser.add_argument("-w", "--weights", dest="weights", required=True, type=Path,
                    help="Path to pre-trained classifier to be loaded via `torch.load`. Can be either a relative or "
                         "absolute file path.")
parser.add_argument("--input-tiles", dest="input", required=True, type=Path,
                    help="List of FORCE tiles which should be used for inference. Each line should contain one FORCE "
                         "tile specifier (Xdddd_Ydddd).")
parser.add_argument("--input-dir", dest="base", required=False, type=Path, default=Path("."),
                    help="Path to FORCE datacube. By default, use the current PWD.")
parser.add_argument("--input-glob", dest="iglob", required=False, type=str, default="*",
                    help="Optional glob pattern to restricted files used from `input-dir`.")
parser.add_argument("--output-dir", dest="out", required=False, type=Path, default=Path("."),
                    help="Path to directory into which predictions should be saved. By default, use the "
                        "current PWD.")
parser.add_argument("--date-cutoff", dest="date", required=True, type=int,
                    help="Cutoff date for time series which should be included in datacube for inference.")
parser.add_argument("--mask-dir", dest="masks", required=False, type=Path, default=None,
                    help="Path to directory containing folders in FORCE tile structure storing "
                         "binary masks with a value of 1 representing pixels to predict. Others can be nodata "
                         "or 0. Masking is done on a row-by-row basis. I.e., the entire unmasked datacube "
                         "is constructed from the files found in `input-dir`. Only when handing a row of "
                         "pixels to the DL-model for inference are data removed. Thus, this does not reduce "
                         "the memory footprint, but can speed up inference significantly under certain "
                         "conditions.")
parser.add_argument("--mask-glob", dest="mglob", required=False, type=str, default="mask.tif",
                    help="Optional glob pattern to restricted file used from `mask-dir`.")
parser.add_argument("--row-size", dest="row-block", required=False, type=int, default=None,
                    help="Row-wise size to read in at once. If not specified, query dataset for block size and assume "
                         "constant block sizes across all raster bands in case of multilayer files. Contrary to "
                         "what GDAL allows, if the entire raster extent is not evenly divisible by the block size, "
                         "an error will be raised and the process aborted. If only `row-size` is given, read the "
                         "specified amount of rows and however many columns are given by the datasets block size. "
                         "If both `row-size` and `col-size` are given, read tiles of specified size.")
parser.add_argument("--col-size", dest="col-block", required=False, type=int, default=None,
                    help="Column-wise size to read in at once. If not specified, query dataset for block size and "
                         "assume constant block sizes across all raster bands in case of multilayer files. Contrary to "
                         "what GDAL allows, if the entire raster extent is not evenly divisible by the block size, "
                         "an error will be raised and the process aborted. If only `col-size` is given, read the "
                         "specified amount of columns and however many rows are given by the datasets block size. "
                         "If both `col-size` and `row-size` are given, read tiles of specified size.")
parser.add_argument("--log", dest="log", required=False, action="store_true",
                    help="Emit logs?")
parser.add_argument("--log-file", dest="log-file", required=False, type=str,
                    help="If logging is enabled, write to this file. If omitted, logs are written to stdout.")
parser.add_argument("--cpus", dest="cpus", required=False, default=None, type=int,
                    help="Number of CPUS for Inter-OP and Intra-OP parallelization of pytorch.")

cli_args: Dict[str, Union[Path, int, bool, str]] = vars(parser.parse_args())

if cli_args.get("log"):
    if cli_args.get("log-file"):
        logging.basicConfig(level=logging.INFO, filename=cli_args.get("log-file"))
    else:
        logging.basicConfig(level=logging.INFO)

if (cli_args.get("cpus")):
    torch.set_num_threads(cli_args.get("cpus"))
    torch.set_num_interop_threads(cli_args.get("cpus"))

device: str = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

inference_model: Union[torch.nn.LSTM, torch.nn.Transformer] = torch.load(cli_args.get("weights"), map_location=device).eval()

if "lstm" in inference_model.__module__.lower():
    inference_type: ModelType = ModelType.LSTM
elif "transformer" in inference_model.__module__.lower():
    inference_type: ModelType = ModelType.TRANSFORMER
else:
    raise RuntimeError("Unknown model type supplied")

with open(cli_args.get("input"), "rt") as f:
    FORCE_tiles: List[str] = [tile.replace("\n", "") for tile in f.readlines()]

TRANSFORMER_TARGET_LENGTH: int = 329

for tile in FORCE_tiles:
    start: float = time()
    logging.info(f"Processing FORCE tile {tile}")
    s2_tile_dir: Path = cli_args.get("base") / tile
    tile_paths: List[str] = [str(p) for p in s2_tile_dir.glob(cli_args.get("iglob"))]
    cube_inputs: List[str] = [
        tile_path for tile_path in tile_paths if int(search(r"\d{8}", tile_path).group(0)) <= cli_args.get("date")
    ]
    cube_inputs.sort()

    with rasterio.open(cube_inputs[0]) as f:
        metadata = f.meta
        metadata["count"] = 1
        metadata["dtype"] = rasterio.uint8
        metadata["nodata"] = 0
        row_block, col_block = f.block_shapes[0]

    tile_rows: int = metadata["height"]
    tile_cols: int = metadata["width"]
    output_torch: torch.tensor = torch.zeros([tile_rows, tile_cols], dtype=torch.long)

    row_step: int = cli_args.get("row-block") or row_block
    col_step: int = cli_args.get("col-block") or col_block

    if tile_rows % row_step != 0 or tile_cols % col_step != 0:
        raise AssertionError("Rows and columns must be divisible by their respective step sizes without remainder.")

    logging.info(f"Processing tile {tile = } in chunks of {row_step = } and {col_step = }")

    for row in range(0, tile_rows, row_step):
        for col in range(0, tile_cols, col_step):
            start_chunked: float = time()
            logging.info(f"Creating chunked data cube")
            s2_cube: Union[xarray.Dataset, xarray.DataArray, list[xarray.Dataset]] = []
            for cube_input in cube_inputs:
                ds: Union[xarray.Dataset, xarray.DataArray] = rxr.open_rasterio(cube_input)
                clipped_ds = ds.isel(y=slice(row, row + row_step),
                                     x=slice(col, col + col_step))
                s2_cube.append(clipped_ds)
                ds.close()

            logging.info(f"Converting chunked data cube to numpy array")
            s2_cube_np: np.ndarray = np.array(s2_cube, ndmin=4, dtype=np.float32)
                        
            if inference_type ==  ModelType.TRANSFORMER:
                logging.info("Adding DOY information to data cube")
                # observations: int = len(cube_inputs)
                sensing_doys: List[Union[datetime, float]] = pad_doy_sequence(TRANSFORMER_TARGET_LENGTH, [fp_to_doy(it) for it in cube_inputs])
                sensing_doys_np: np.ndarray = np.array(sensing_doys)
                sensing_doys_np = sensing_doys_np.reshape((TRANSFORMER_TARGET_LENGTH, 1, 1, 1))
                sensing_doys_np = np.repeat(sensing_doys_np, row_step, axis=2)  # match actual data cube
                sensing_doys_np = np.repeat(sensing_doys_np, col_step, axis=3)  # match actual data cube
                s2_cube_np = pad_datacube(TRANSFORMER_TARGET_LENGTH, s2_cube_np)
                s2_cube_np = np.concatenate((s2_cube_np, sensing_doys_np), axis=1)

            logging.info("Transposing Numpy array")
            s2_cube_npt: np.ndarray = np.transpose(s2_cube_np, (2, 3, 0, 1))

            del s2_cube
            del s2_cube_np
            del sensing_doys

            if inference_type == ModelType.TRANSFORMER:
                logging.info("Normalizing data cube")
                # x = col_step
                # y = row_step
                observations: int = s2_cube_npt.shape[2]
                bands: int = s2_cube_npt.shape[3]
                bn_layer: torch.nn.BatchNorm1d = torch.nn.BatchNorm1d(bands)
                #s2_cube_npt_norm = np.zeros(*s2_cube_npt.shape)
                for row in range(row_step):
                    for col in range(col_step):
                        pixel: np.ndarray = s2_cube_npt[row, col, :, :]
                        pixel[pixel == -9999.0] = 0
                        pixel_torch: torch.tensor = torch.from_numpy(pixel).float()
                        pixel_normalized: np.ndarray = bn_layer(pixel_torch).detach().numpy()
                        # FIXME Why the offsets and on dimension less than above where pixel object was created?
                        #s2_cube_npt_norm[row:row + 329, col:col + 11, :] = pixel_normalized
                        s2_cube_npt[row:row + observations, col:col + bands, :] = pixel_normalized


            mask: Optional[np.ndarray] = None

            if cli_args.get("masks"):
                try:
                    mask_path: str = [str(p) for p in (cli_args.get("masks") / tile).glob(cli_args.get("mglob"))][0]
                except IndexError:
                    mask_path: str = [str(p) for p in cli_args.get("masks").glob(cli_args.get("mglob"))][0]
                
                with rxr.open_rasterio(mask_path) as ds:
                    mask_ds: xarray.Dataset = ds.isel(y=slice(row, row + row_step),
                                                      x=slice(col, col + col_step))
                    mask: np.ndarray = np.squeeze(np.array(mask_ds, ndmin=2, dtype=np.bool_), axis=0)
                    del mask_ds

            logging.info(f"Converting chunked numpy array to torch tensor, moving tensor to '{device}'.")
            s2_cube_torch: Union[torch.Tensor, torch.masked.masked_tensor] = torch.from_numpy(s2_cube_npt)
            s2_cube_torch = s2_cube_torch.to(device)

            if inference_type == ModelType.LSTM:
                output_torch[row:row + row_step, col:col + col_step] = predict_lstm(inference_model, s2_cube_torch, mask,
                                                                                    col, col_step, row, row_step)
            else:
                output_torch[row:row + row_step, col:col + col_step] = predict_transformer(inference_model, s2_cube_torch, mask,
                                                                                           col, col_step, row, row_step, device)
                        
            logging.info(f"Processed chunk in {time() - start_chunked:.2f} seconds")

            del s2_cube_torch

    output_numpy: np.array = output_torch.numpy()

    with rasterio.open(cli_args.get("out") / (tile + ".tif"), 'w', **metadata) as dst:
        dst.write_band(1, output_numpy.astype(rasterio.uint8))

    logging.info(f"Processed tile {tile} in {time() - start:.2f} seconds")

    del metadata
    del output_torch
    del output_numpy

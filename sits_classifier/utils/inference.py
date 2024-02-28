from typing import Optional, Any, List, Union
import torch
from datetime import datetime, date
from re import compile, Pattern
import numpy as np
from enum import Enum

class ModelType(Enum):
    LSTM        = 1
    TRANSFORMER = 2
    UNDEFINED   = 3


def pad_doy_sequence(target: int, observations: List[datetime]) -> List[Union[datetime, float]]:
    diff: int = target - len(observations)
    if diff < 0:
        raise NotImplementedError("Support for time series longer than orignial training data not implemented.")
    elif diff > 0:
        observations = observations + ([0.0] * diff)

    # TODO remove assertion for "production"
    assert(target == len(observations))

    return observations
    

def pad_datacube(target: int, datacube: np.ndarray) -> np.ndarray:
    diff: int = target - datacube.shape[0]
    if diff < 0:
        raise NotImplementedError("Support for time series longer than orignial training data not implemented.")
    elif diff > 0:
        datacube = np.pad(datacube, ((0, diff), (0,0), (0,0), (0,0)))
    
    # TODO remove assertion for "production"
    assert(target == datacube.shape[0])

    return datacube


def fp_to_doy(file_path: str) -> datetime:
    date_in_fp: Pattern = compile(r"(?<=/)\d{8}(?=_)")
    sensing_date: str = date_in_fp.findall(file_path)[0]
    d: datetime = datetime.strptime(sensing_date,"%Y%m%d")
    doy: int = d.toordinal() - date(d.year, 1, 1).toordinal() + 1  # https://docs.python.org/3/library/datetime.html#datetime.datetime.timetuple
    return doy


def predict(model, data: torch.tensor, it: ModelType) -> Any:
    """
    Apply previously trained LSTM to new data
    :param model: previously trained model
    :param torch.tensor data: new input data
    :return Any: Array of predictions
    """
    with torch.no_grad():
        outputs = model(data if it == ModelType.LSTM else data.unsqueeze(0))
        _, predicted = torch.max(outputs.data if it == ModelType.LSTM else outputs, 1)
    return predicted


def predict_lstm(lstm: torch.nn.LSTM, dc: torch.tensor, mask: Optional[np.ndarray], c: int, c_step: int, r: int, r_step: int) -> torch.tensor:
    prediction: torch.Tensor = torch.zeros((r_step, c_step), dtype=torch.long)
    prediction.zero_()
    if mask:
        merged_row: torch.Tensor = torch.zeros(c_step, dtype=torch.long)
        for chunk_rows in range(0, r_step):
            merged_row.zero_()
            squeezed_row: torch.Tensor = predict(
                lstm,
                dc[chunk_rows, mask[chunk_rows]],
                ModelType.LSTM)
            merged_row[mask[chunk_rows]] = squeezed_row
            prediction[chunk_rows, 0:c_step] = merged_row
    else:
        for chunk_rows in range(0, r_step):
            prediction[chunk_rows, 0:c_step] = predict(lstm, dc[chunk_rows], ModelType.LSTM)
    
    return prediction


def predict_transformer(transformer: torch.nn.Transformer, dc: torch.tensor, mask: Optional[np.ndarray], c: int, c_step: int, r: int, r_step: int, device: str) -> torch.tensor:
    prediction: torch.Tensor = torch.zeros((r_step, c_step), dtype=torch.long)
    prediction.zero_()
    if mask:
        raise NotImplementedError("Masked datacubes when using transformer models is not implemented.")
    else:
        for row in range(r_step):
            for col in range(c_step):
                pixel: torch.tensor = dc[row, col, :, :]
                pixel.to(device)
                prediction[row, col] = predict(transformer, pixel, ModelType.TRANSFORMER).cpu()  # always move to cpu

    return prediction


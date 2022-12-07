# major part of code sourced from aws sagemaker example:
# https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/scikit_bring_your_own/container/decision_trees/predictor.py

import io
import os
import sys
import traceback
import warnings
from tempfile import NamedTemporaryFile
from typing import Union

import pandas as pd
from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.responses import FileResponse

warnings.filterwarnings("ignore")

import algorithm.utils as utils
from algorithm.model.classifier import MODEL_NAME
from algorithm.model_server import ModelServer

prefix = "/opt/ml_vol/"
data_schema_path = os.path.join(prefix, "inputs", "data_config")
model_path = os.path.join(prefix, "model", "artifacts")
failure_path = os.path.join(prefix, "outputs", "errors", "serve_failure")


# get data schema - its needed to set the prediction field name
# and to filter df to only return the id and pred columns
data_schema = utils.get_data_schema(data_schema_path)


# initialize your model here before the app can handle requests
model_server = ModelServer(model_path=model_path, data_schema=data_schema)


# The FastAPI app for serving predictions
app = FastAPI()


async def gen_temp_file(ext: str = ".csv"):
    """Generate a temporary file with a given extension"""
    with NamedTemporaryFile(suffix=ext, delete=True) as temp_file:
        yield temp_file.name


@app.get("/ping", tags=["ping", "healthcheck"])
async def ping() -> dict:
    """Determine if the container is working and healthy."""
    response = f"Hello, I am {MODEL_NAME} model and I am at you service!"
    return {
        "success": True,
        "message": response,
    }


@app.post("/infer", tags=["inference"], response_class=FileResponse)
async def infer(
    input: UploadFile = File(...), temp=Depends(gen_temp_file)
) -> Union[FileResponse, dict]:
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if input.content_type == "text/csv":
        data = await input.read()
        temp_io = io.StringIO(data.decode("utf-8"))
        data = pd.read_csv(temp_io)
    else:
        return {
            "success": False,
            "message": f"Content type {input.content_type} not supported (only CSV data allowed)",
        }

    print(f"Invoked with {data.shape[0]} records")

    # Do the prediction
    try:
        predictions = model_server.predict(data, data_schema)
        # Convert from dataframe to CSV
        predictions.to_csv(temp, index=False)
        return FileResponse(temp, media_type="text/csv")
    except Exception as err:
        # Write out an error file. This will be returned as the failureReason to the client.
        trc = traceback.format_exc()
        with open(failure_path, "w") as s:
            s.write("Exception during inference: " + str(err) + "\n" + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print("Exception during inference: " + str(err) + "\n" + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        return {
            "success": False,
            "message": f"Exception during inference: {str(err)} (check failure file for more details)",
        }

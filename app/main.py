import joblib, uvicorn, logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from pathlib import Path
from .data_models import ModelInfo, ModelInput, ModelOutPut
from constants import USER_COL
from typing import List

# set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def load_model(model_path: Path = Path("./model/model.joblib.gz")):
    if model_path.exists():
        logging.info(f"Loading model...")
        model = joblib.load(model_path)
        logging.info("Done with loading model..")
        return model, ModelInfo(version=model.version, timestamp=model.timestamp)
    else:
        raise FileNotFoundError(f"Model doesn't exists at this path: {str(model_path)}")


app = FastAPI()
MODEL, MODEL_INFO = load_model()


@app.get("/", response_model=ModelInfo)
async def model_info():
    """Gets model info as version & timestamp"""
    logging.info(f"Getting model version..")
    return {"version": MODEL_INFO.version, "timestamp": MODEL.timestamp}


@app.post("/v1/predict", response_model=ModelOutPut, status_code=200)
async def get_model_prediction(input: ModelInput):
    input_df = pd.json_normalize(input.__dict__)
    logging.info(f"{len(input_df)} observations to predict on for user: {input.uuid}")
    input_df = input_df.drop([USER_COL], axis=1)
    prob_default = MODEL.predict_proba(input_df)[:, 1]
    return {"uuid": input.uuid, "probability_default": prob_default}


@app.post("/v1/predict/multiple", response_model=List[ModelOutPut], status_code=200)
async def get_multiple_model_predictions(inputs: List[ModelInput]):
    responses = []
    prev_id = ""
    for model_input in inputs:
        if prev_id == model_input.uuid:
            raise HTTPException(
                status_code=400,
                detail=f"User {model_input.uuid} has already been predicted one...",
            )
        input_df = pd.json_normalize(model_input.__dict__)
        logging.info(
            f"{len(input_df)} observations to predict on for user: {model_input.uuid}"
        )
        input_df = input_df.drop([USER_COL], axis=1)
        prob_default = MODEL.predict_proba(input_df)[:, 1]
        responses.append({"uuid": model_input.uuid, "probability_default": prob_default})
        prev_id = model_input.uuid
    return responses


if __name__ == "__main__":
    uvicorn.run("main:app")

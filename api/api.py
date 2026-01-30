import json
import os
from typing import List, Optional

import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

from catboost_train import load_and_preprocess_data, train_catboost_model, predict_with_model
from rule.agent import create_agent
from rule.logger import setup_logger
from rule.processor import process_row

app = FastAPI(title="Script Analysis API", description="API for analyzing and predicting script content compliance")
logger = setup_logger()
logger.info("Starting Script Analysis API")


class ScriptInput(BaseModel):
    title: str
    content: str
    human_result: Optional[str] = None
    predicted_result: Optional[str] = None

    class Config:
        extra = "forbid"  # Reject unknown fields


class BatchScriptInput(BaseModel):
    scripts: List[ScriptInput]


class TrainInput(BaseModel):
    train_data: str  # CSV content as a string
    test_data: Optional[str] = None  # Optional test CSV content as a string


class PredictInput(BaseModel):
    test_data: str  # CSV content as a string


# Middleware to log all incoming requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    try:
        payload = await request.json()
        logger.info(
            f"Incoming request: {request.method} {request.url}, payload: {json.dumps(payload, ensure_ascii=False)}")
    except Exception:
        logger.info(f"Incoming request: {request.method} {request.url}, no JSON payload")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code} for {request.method} {request.url}")
    return response


@app.post("/analyze", summary="Analyze a single script")
async def analyze_script(script: ScriptInput):
    logger.info(f"Received request to analyze script: {script.title}")
    try:
        row = pd.Series({
            '标题': script.title,
            '正文': script.content,
            '人工结果': script.human_result,
            '预测结果': script.predicted_result
        })
        if not script.title.strip() or not script.content.strip():
            logger.error(f"Invalid input for script {script.title}: Title or content is empty")
            raise ValueError("Title or content cannot be empty")
        agent = create_agent()
        result = process_row(row, agent, logger)
        logger.debug(f"Analysis completed for script: {script.title}")
        return result
    except ValueError as e:
        logger.error(f"Validation error for script {script.title}: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing script {script.title}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/batch", summary="Analyze multiple scripts")
async def analyze_scripts_batch(batch: BatchScriptInput):
    logger.info(f"Received batch analysis request for {len(batch.scripts)} scripts")
    try:
        df = pd.DataFrame([{
            '标题': script.title,
            '正文': script.content,
            '人工结果': script.human_result,
            '预测结果': script.predicted_result
        } for script in batch.scripts])

        agent = create_agent()
        results = []
        for _, row in df.iterrows():
            if not row['标题'].strip() or not row['正文'].strip():
                logger.error(f"Invalid row, title: {row['标题']}, reason: Title or content is empty")
                results.append({
                    'title': row['标题'],
                    'verdict': row['预测结果'] if pd.notna(row['预测结果']) else '未知',
                    'analysis': "Invalid row: Title or content is empty"
                })
                continue
            result = process_row(row, agent, logger)
            results.append(result)

        logger.info(f"Batch analysis completed for {len(results)} scripts")
        return results
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@app.post("/train", summary="Train CatBoost model")
async def train_model(input: TrainInput):
    logger.info("Received request to train model")
    try:
        from io import StringIO
        train_df = pd.read_csv(StringIO(input.train_data))
        logger.debug(f"Loaded training data with {len(train_df)} rows")

        # Load and preprocess data
        if input.test_data:
            test_df = pd.read_csv(StringIO(input.test_data))
            x,y, x_test, _, _, _ = load_and_preprocess_data(train_df, test_df)
        else:
            x, y = load_and_preprocess_data(train_df)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

        # Train model
        model, accuracy = train_catboost_model(x_train, y_train, x_val, y_val)

        model_path = './mnt/trained_catboost_model_api.cbm'
        model.save_model(model_path)
        logger.info(f"Model trained and saved to {model_path}, accuracy: {accuracy:.4f}")

        return {"message": "Model trained successfully", "accuracy": accuracy}
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")


@app.post("/predict", summary="Predict script outcomes")
async def predict_scripts(input: PredictInput):
    logger.info("Received request to predict scripts")
    try:
        # Load model
        model_path = './mnt/trained_catboost_model_api.cbm'
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise HTTPException(status_code=404, detail="Model file not found")

        model = CatBoostClassifier()
        model.load_model(model_path)
        # Parse CSV string to DataFrame
        from io import StringIO
        test_df = pd.read_csv(StringIO(input.test_data))
        logger.debug(f"Loaded test data with {len(test_df)} rows")

        x, y, x_test, x_real, title, content = load_and_preprocess_data(test_df, test_df)
        predictions, probabilities = predict_with_model(model, x_test)
        results = [
            {
                'title': t,
                'content': c,
                'human_result': hr,
                'predicted_result': pr,
                'predicted_probability': float(prob)  # Convert to float for JSON serialization
            }
            for t, c, hr, pr, prob in zip(title, content, x_real, predictions, probabilities)
        ]

        logger.info(f"Predictions completed for {len(results)} scripts")
        return {"message": "Predictions completed", "results": results}
    except Exception as e:
        logger.error(f"Error predicting scripts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/health", summary="Check API health")
async def health_check():
    logger.debug("Health check requested")
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

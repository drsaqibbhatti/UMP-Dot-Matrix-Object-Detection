import typer
import json
import os
import logging
import sys
from datetime import datetime
from mainV4 import run_train
from Inference_auc_v2_1 import run_inference
from typing import Optional

app = typer.Typer()

# Logging setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Prevent duplicate handlers
if not logger.hasHandlers():
    # Instead of the default StreamHandler, set to output to stdout explicitly
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    logger.addHandler(ch)

def flush_stdout():
    sys.stdout.flush()

@app.command()
def train(
    train_paths: str = typer.Option(..., help="Training dataset directories separated by semicolons ';'"),
    val_paths: Optional[str] = typer.Option(None, help="Validation dataset directories separated by semicolons ';'"),
    result_model_dir_path: str = typer.Option(..., help="Directory path where the trained model will be saved"),
    data_type : str = typer.Option(..., help="Type of model to train, i.e., Inner_LT, Inner_RT, Outer_1, Outer_2"),
    hyper_param_path : str = typer.Option(..., help="hyper param json path"),
):
    """
    Start training a new model with the specified training and validation datasets.
    """
    train_list = train_paths.split(";")
    val_list = val_paths.split(";") if val_paths else []

    logging.info(f"training process start")
    flush_stdout()

    if data_type == "Inner_LT":
        imageHeight, imageWidth = 320, 480
    elif data_type == "Inner_RT":
        imageHeight, imageWidth = 640, 320
    elif data_type == "Outer_1":
        imageHeight, imageWidth = 320, 320
    elif data_type == "Outer_2":
        imageHeight, imageWidth = 800, 320
    else:
        logging.error(f"Unsupported data_type: {data_type}")
        raise ValueError(f"Unsupported data_type: {data_type}")

    logging.info(f"Training {data_type} model with image size: {imageHeight}x{imageWidth}")
    flush_stdout()

    run_train(
        train_path=train_list,
        val_path=val_list,
        result_model_dir_path=result_model_dir_path,
        imageHeight=imageHeight,
        imageWidth=imageWidth,
        data_type=data_type,
        hyperparam_path = hyper_param_path
    )
    

    
@app.command()
def resume(
    train_model_path: str = typer.Option(..., help="Path to the existing model file to resume training from"),
    train_paths: str = typer.Option(..., help="Training dataset directories separated by semicolons ';'"),
    val_paths: Optional[str] = typer.Option(None, help="Validation dataset directories separated by semicolons ';'"),
    result_model_dir_path: str = typer.Option(..., help="Directory path where the trained model will be saved"),
    data_type : str = typer.Option(..., help="Type of model to train, i.e., Inner_LT, Inner_RT, Outer_1, Outer_2"),
    hyper_param_path: str = typer.Option(..., help="hyper param json path"),
):
    """
    Resume training from an existing model with new datasets.
    """
    train_list = train_paths.split(";")
    val_list = val_paths.split(";") if val_paths else []

    logging.info(f"resume process start")
    flush_stdout()

    if data_type == "Inner_LT":
        imageHeight, imageWidth = 320, 480
    elif data_type == "Inner_RT":
        imageHeight, imageWidth = 640, 320
    elif data_type == "Outer_1":
        imageHeight, imageWidth = 320, 320
    elif data_type == "Outer_2":
        imageHeight, imageWidth = 800, 320
    else:
        logging.error(f"Unsupported data_type: {data_type}")
        raise ValueError(f"Unsupported data_type: {data_type}")

    logging.info(f"Training {data_type} model with image size: {imageHeight}x{imageWidth}")
    flush_stdout()

    run_train(
        train_path=train_list,
        val_path=val_list,
        result_model_dir_path=result_model_dir_path,
        imageHeight=imageHeight,
        imageWidth=imageWidth,
        data_type=data_type,
        resume_training_path=train_model_path,
        hyperparam_path=hyper_param_path
    )

@app.command()
def inference(
        inference_model_path: str = typer.Option(..., help="Path to the existing model file to resume training from"),
        val_paths: str = typer.Option(..., help="Validation dataset directories separated by semicolons ';'"),
        result_auc_path: str = typer.Option(..., help="Path to which the AUC graph will be stored (including the name),';'"),
        data_type : str = typer.Option(..., help="Type of model to train, i.e., Inner_LT, Inner_RT, Outer_1, Outer_2"),
):
    """
        Resume training from an existing model with new datasets.
    """

    val_list = val_paths.split(";")
    logging.info(f"inference process start")
    flush_stdout()


    if data_type == "Inner_LT":
        imageHeight, imageWidth = 320, 480
    elif data_type == "Inner_RT":
        imageHeight, imageWidth = 640, 320
    elif data_type == "Outer_1":
        imageHeight, imageWidth = 320, 320
    elif data_type == "Outer_2":
        imageHeight, imageWidth = 800, 320
    else:
        logging.error(f"Unsupported data_type: {data_type}")
        raise ValueError(f"Unsupported data_type: {data_type}")


    run_inference(
        val_path=val_list,
        model_dir_path=inference_model_path,
        imageHeight=imageHeight,
        imageWidth=imageWidth,
        data_type=data_type,
        auc_path=result_auc_path
    )


if __name__ == "__main__":
    app()

from fastapi import FastAPI
from fastapi.responses import FileResponse
from train import main as train_main
import sys
import argparse

app = FastAPI()

class Args:
    def __init__(self, experiment, gpu, opts):
        self.experiment = experiment
        self.gpu = gpu
        self.opts = opts

@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.post("/train/{experiment_name}")
async def train_model(experiment_name: str):
    # This is a placeholder for the actual training logic.
    # We will need to adapt the train.py script to be callable from here.
    return {"message": f"Training started for experiment: {experiment_name}"}

@app.post("/train_phase2")
async def train_phase2():
    try:
        # Create a mock sys.argv
        sys.argv = ['train.py', '-exp', 'custom_finetune_ph2']
        train_main()
        return {"message": "Phase 2 training started successfully."}
    except Exception as e:
        return {"message": f"An error occurred during phase 2 training: {e}"}

@app.post("/train_phase3")
async def train_phase3():
    try:
        # Create a mock sys.argv
        sys.argv = ['train.py', '-exp', 'custom_finetune_ph3']
        train_main()
        return {"message": "Phase 3 training started successfully."}
    except Exception as e:
        return {"message": f"An error occurred during phase 3 training: {e}"}

# modal_app.py
import time
from hive.ml.supervised_training.pre_run_to_pt import create_webdataset
from modal import App, Image, Secret, Volume
from pathlib import Path
from hive.ml.model.models import create_hive_gatv2_gnn
from hive.ml.supervised_training.train_L import train_hive_model
import wandb

app = App("hive-training-pipeline")

image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "lightning",
        "torch-geometric",
        "webdataset",
        "tqdm",
        "pyrsistent",
        "wandb"
    )
    .add_local_dir("./hive", remote_path="/root/hive")
    .add_local_file('./game_strings/combined.txt', "/root/combined.txt")
)

# Use Volume instead of NetworkFileSystem
dataset_volume = Volume.from_name("hive-dataset-storage", create_if_missing=True)
REMOTE_DATASET_DIR = "/data/hive_webdataset"

@app.function(
    image=image,
    secrets=[Secret.from_name("wandb-secret")],
    cpu=64,
    volumes={"/data": dataset_volume},
    timeout=3600,
)
def preprocess():
    """Runs the imported data preparation script."""
    print("Starting data preprocessing...")
    
    create_webdataset(
        filepath="/root/combined.txt",
        max_games=None,
        output_dir=Path(REMOTE_DATASET_DIR),
        num_processes=None,  # Automatically use all available cores
        total_chunks=250,
    )
    print("Preprocessing complete.")


@app.function(
    image=image,
    secrets=[Secret.from_name("wandb-secret")],
    gpu="T4",
    volumes={"/data": dataset_volume},
    timeout=21600,
)
def train():
    """Runs the training script using imported components."""
    print("Starting model training...")
    checkpoint_dir = "/data/lightning_checkpoints"
    experiment_name = None
    
    model = create_hive_gatv2_gnn(
        hidden_dim=128,
        num_layers=3,
        heads=4,
        dropout=0.05,
        residual=False,
        batch_norm=False,
        pool_method='add',
        task_heads=["value"]
    )
    
    train_hive_model(
        model=model,
        data_directory=REMOTE_DATASET_DIR,
        experiment_name=experiment_name,
        checkpoint_dir=checkpoint_dir,
        total_epochs=5,
        batch_size=512,
        num_workers=6,
        learning_rate=0.05,
        shuffle_buffer_size=10000,
        project_name="hive_value_only_validation",
        task_weights={"value": 1},
    )

    wandb.finish()


@app.local_entrypoint()
def main(task: str = "all"):
    """Main entrypoint for the Modal app."""
    if task == "preprocess":
        preprocess.remote()
    elif task == "train":
        train.remote()
    elif task == "all":
        preprocess.remote()
        print(f"Processing complete. Starting training in 10 seconds...")
        time.sleep(10)
        train.remote()
    else:
        print(f"Unknown task: {task}. Available tasks: preprocess, train, all")
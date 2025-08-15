import json
from pathlib import Path
from datetime import datetime
import uuid
import joblib
import shutil

from kcm.koopman_category_model import KoopmanCategoryModel

from matplotlib import pyplot as plt



def load_koopman_model(run_dir):
    model_path = Path(run_dir) / "koopman_model.pkl"
    params_path = Path(run_dir) / "params.json"
    
    model = KoopmanCategoryModel.load(model_path)
    with open(params_path, "r") as f:
        params = json.load(f)
    
    return model, params




def create_discovery_run_dir(base_dir="experiments", tag="discovery_run"):

    repo_root = Path(__file__).resolve().parents[2]
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    run_dir = Path(repo_root / base_dir) / f"{tag}_{ts}_{uid}"
    (run_dir / "plots").mkdir(parents=True)
    # (run_dir / "results").mkdir()
    # (run_dir / "logs").mkdir()
    print(f"Created discovery run directory: {run_dir}")
    return run_dir


def save_discovery_params(params, run_dir):
    path = Path(run_dir) / "discovery_params.json"
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved discovery parameters to {path}")



def copy_koopman_params_to_discovery(koopman_dir, discovery_dir):
    koopman_dir = Path(koopman_dir)
    discovery_dir = Path(discovery_dir)

    src = koopman_dir / "params.json"
    dst = discovery_dir / "koopman_params.json"  # Rename to avoid overwriting discovery params

    if src.exists():
        shutil.copy(src, dst)
        print(f"Copied Koopman params from {src} → {dst}")
    else:
        print(f"Warning: {src} does not exist — nothing copied.")
        


def save_plot(run_dir, filename="plot.png", subfolder="plots", dpi=300, close=True):
    run_dir = Path(run_dir)
    plot_path = run_dir / subfolder / filename
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close()
    print(f"Saved plot to {plot_path}")




def save_artifact(obj, run_dir, name):
    path = Path(run_dir) / f"{name}.pkl"
    joblib.dump(obj, path)
    print(f"Saved trainer object to {path}")






def fix_colab_path(colab_path: str, experiments_root: Path) -> Path:
    if "experiments" in colab_path:
        # Strip everything before and including 'experiments'
        tail = Path(colab_path).parts
        idx = tail.index("experiments")
        relative_path = Path(*tail[idx:])
        return experiments_root / relative_path.relative_to("experiments")
    else:
        # Fallback: treat as already correct or skip
        return Path(colab_path)
    



def load_koopman_and_discoveries(kcm_folder: str, experiments_root: str = Path.cwd().parent / "experiments"):
    
    kcm_path = Path(experiments_root) / kcm_folder
    koopman_model_path = kcm_path / "koopman_model.pkl"
    koopman_params_path = kcm_path / "params.json"

    # Load Koopman model and params
    # koopman_model = joblib.load(koopman_model_path)
    with open(koopman_params_path, "r") as f:
        koopman_params = json.load(f)

    # Find associated discovery runs
    discoveries = []
    for path in Path(experiments_root).glob("discovery_run_*"):
        params_path = path / "discovery_params.json"
        if not params_path.exists():
            continue
            
        with open(params_path, "r") as f:
            disc_params = json.load(f)

        # Match based on Koopman path if available
        koopman_path_raw = disc_params.get("koopman_path", "")
        fixed_koopman_path = fix_colab_path(koopman_path_raw, experiments_root)
        
        if kcm_folder in str(fixed_koopman_path):
            entry = {"path": path, "params": disc_params}


            # Load trainer objects
            kcm_trainer_path = path / "kcm_trainer.pkl"
            basic_trainer_path = path / "basic_trainer.pkl"
            if kcm_trainer_path.exists():
                # print('Loading in kcm_trainer...')
                entry["kcm_trainer"] = joblib.load(kcm_trainer_path)
            if basic_trainer_path.exists():
                # print('Loading in basic_trainer...')
                entry["basic_trainer"] = joblib.load(basic_trainer_path)

            
            discoveries.append(entry)

        

    return {
        # "model": koopman_model,
        "params": koopman_params,
        "discoveries": discoveries
    }
import yaml, os

class Config:
    def __init__(self, cfg_path="configs/default.yaml"):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.paths = cfg["paths"]
        self.params = cfg["params"]

        os.makedirs(self.paths["save_dir"], exist_ok=True)
        os.makedirs(os.path.join(self.paths["save_dir"], "pred_label_png"), exist_ok=True)

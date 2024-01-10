import argparse
from omegaconf import OmegaConf

def call_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="base_config")

    args, _ = parser.parse_known_args()
    conf = OmegaConf.load(f"./{args.config}.yaml")
    
    return conf
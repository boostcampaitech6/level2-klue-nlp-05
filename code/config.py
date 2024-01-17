import argparse
from omegaconf import OmegaConf

def call_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="1.2.14_config1")

    args, _ = parser.parse_known_args()
    conf = OmegaConf.load(f"./config/{args.config}.yaml")
    
    return conf
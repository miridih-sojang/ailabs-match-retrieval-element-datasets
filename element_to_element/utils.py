import argparse
import yaml


def get_args():
    parser = argparse.ArgumentParser(description='MIRIDIH MATCHING ELEMENTS PROJECTS')
    parser.add_argument('--config_path', type=str, help='Use Config Directory')
    args = parser.parse_args()
    return args


def read_yaml(path):
    with open(path, 'r') as stream:
        return yaml.safe_load(stream)
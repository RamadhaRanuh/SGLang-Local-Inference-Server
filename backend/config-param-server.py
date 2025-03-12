"""
SGLang Model Profile Selector
-----------------------------
A utility script to select and apply model profiles to your SGLang configuration
"""

import argparse # For parsing command line arguments
import yaml # For parsing YAML files
import os # For interacting with the operating system
import sys


def parse_args():
    """
    Parse command line arguments for generating Docker Compose File.
    """
    parser = argparse.ArgumentParser(description='Generate Docker Compose File for SGLang deployment')
    parser.add_argument('--config', type=str, default = 'deployment-config.yaml', help = 'Path to configuration YAML file')
    parser.add_argument('--output', type=str, default = 'docker-compose.yaml', help = 'Docker Compose file path')

    return parser.parse_args()

def validate_config(config):
    """
    Validate configuration file and set defaults for missing values.
    """

    # Ensure main sections exist
    required_sections = ['model', 'server', 'docker']
    for section in required_sections:
        if section not in config:
            config[section] = {}

    # Set defaults for model section
    if 'path' not in config['model']:
        config['model']['path'] = "meta-llama/Llama-3-8B-Instruct"


    # Set defaults for server section
    if 'host' not in config['server']:
        config['server']['host'] = "0.0.0.0"
    if 'port' not in config['server']:
        config['server']['port'] = 30000


    # Set defaults for docker section
    if 'image' not in config['docker']:
        config['docker']['image'] = "lmsysorg/sglang:latest"
    if 'container_name' not in config['docker']:
        config['docker']['container_name'] = "sglang"
    if 'restart_policy' not in config['docker']:
        config['docker']['restart_policy'] = "always"


    return config

def build_command_args(config):
    """
    Build command line arguments for sglang.launch_server.
    """

    cmd_args = []

    # Add model path
    cmd_args.append(f"--model-path {config['model']['path']}")

    # Add server settings
    if 'server' in config:
        if 'host' in config['server']:
            cmd_args.append(f"--host ")
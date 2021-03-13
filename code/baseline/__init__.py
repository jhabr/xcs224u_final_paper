import os

current_path = os.path.dirname(os.path.abspath(__file__))
tokens = current_path.split("/")
current_dir_index = tokens.index('code')
ROOT = "/".join(tokens[:current_dir_index + 1])

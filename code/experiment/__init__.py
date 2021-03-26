import os


def set_python_path():
    if os.getenv('PYTHONPATH') is None:
        head, _ = os.path.split(os.path.abspath(os.getcwd()))
        os.environ['PYTHONPATH'] = head


set_python_path()

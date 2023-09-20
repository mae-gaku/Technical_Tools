from os.path import dirname, abspath

def get_project_root_path():
    project_root = dirname(dirname(dirname(abspath(__file__)))) + '/'
    return project_root

def get_config_path():
    project_root = get_project_root_path()
    config_path = project_root + 'config.json'
    return config_path
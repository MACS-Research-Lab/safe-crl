import os

ENV_ASSET_DIR_V1 = os.path.join(os.path.dirname(__file__), "assets_v1")
ENV_ASSET_DIR_V2 = os.path.join(os.path.dirname(__file__), "assets_v2")
ENV_ASSET_DIR_SAFE = os.path.join(os.path.dirname(__file__), "assets_v2/sawyer_xyz")


def full_v1_path_for(file_name):
    return os.path.join(ENV_ASSET_DIR_V1, file_name)


def full_v2_path_for(file_name):
    return os.path.join(ENV_ASSET_DIR_V2, file_name)


def full_safe_path_for(file_name: str) -> str:
    return os.path.join(ENV_ASSET_DIR_SAFE, file_name)

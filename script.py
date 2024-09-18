from huggingface_hub import snapshot_download

local_dir = "./3d_icon" #@param
dataset_to_download = "LinoyTsaban/3d_icon" #@param
snapshot_download(
    dataset_to_download,
    local_dir=local_dir, repo_type="dataset",
    ignore_patterns=".gitattributes",
)

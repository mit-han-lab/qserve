import glob
import json
import os

# for dataset_name in ["cc12m", "cc3m"]:
for dataset_name in ['cc3m']:
    # data_path = f"/home/haotiant/dataset/{dataset_name}-wds"
    data_path = f"/dataset/{dataset_name}-wds"
    with open(os.path.join(data_path, "_info.json"), "r") as f:
        info = json.load(f)

    for split in ["train", "validation"]:
        if split == 'validation' and dataset_name == 'cc12m':
            continue
        meta_dict = []
        split_info = info["splits"][split if split == "train" else "validation"]
        filenames = split_info["filenames"]
        shard_lengths = split_info["shard_lengths"]
        for fn, sl in zip(filenames, shard_lengths):
            cur_dict = dict(url=fn, nsamples=sl)
            file_size = os.path.getsize(os.path.join(data_path, fn))
            cur_dict.update(filesize=file_size)
            meta_dict.append(cur_dict)

        final_dict = dict()
        final_dict["shardlist"] = meta_dict
        final_dict["wids_version"] = 1
        with open(os.path.join(data_path, f"cc_meta_{split}.json"), "w") as f:
            f.write(json.dumps(final_dict, indent=2))

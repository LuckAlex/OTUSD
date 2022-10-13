from cleanfid import fid
import torch
fdir1 = 'F:\\fid_data'
score = fid.compute_fid(fdir1, mode = 'legacy_pytorch', dataset_name="FFHQ", dataset_res=1024, dataset_split="trainval", num_workers=0, device=torch.device("cuda"))
print(score)
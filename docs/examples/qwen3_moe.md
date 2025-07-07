Qwen3 MOE training guide

1. Download qwen3 moe model

```shell
python3 scripts/download_hf_model.py \
  --repo_id Qwen/Qwen3-30B-A3B \
  --local_dir .
```

2. Merge qwen3 moe model to support GroupGemm optimize
``` shell
python3 scripts/moe_ckpt_merge/moe_merge.py --raw_hf_path Qwen3-30B-A3B  --merge_hf_path Qwen3-30B-A3B-merge
```

3. Train qwen3 moe model
```
bash train.sh tasks/train_torch.py configs/pretrain/qwen3-moe.yaml 
```
# Deployment Guide: Hugging Face Spaces

This project is deployed as two separate Hugging Face repositories:
- A model repo for fine-tuned weights
- A Space repo for the Streamlit app

The Space app downloads the weights at runtime from the model repo. This keeps the Space lightweight and avoids large Git LFS uploads to the Space repo.

## Prerequisites

1. A Hugging Face account with write access to the target org.
2. A Hugging Face token with write permissions.
3. Local files present in the training repo:
   - `outputs/` (fine-tuned checkpoints)
   - `shakespeare_transformer.pt`
4. A Python environment with `huggingface_hub` installed.

## Deployment (recommended)

Use the deployment helper script in the Space repo. It uploads the weights to the model repo and the app code to the Space repo.

1. Set your token and cache paths.

```bash
export HF_TOKEN="hf_..."
export HF_HOME=/tmp/hf
export HF_HUB_CACHE=/tmp/hf/hub
```

2. Run the deploy script.

```bash
python3 deploy_hf.py \
  --weights-source <path_to_training_repo> \
  --weights-repo <org>/<weights_repo> \
  --space-repo <org>/<space_repo> \
  --space-dir <path_to_space_repo>
```
# if u want to know more about the script deploy_hf.py , u can ask me in the discussion section of the repo 

What it does:
- Uploads `outputs/`, `input.txt`, and `shakespeare_transformer.pt` to the model repo.
- Uploads the Streamlit app code to the Space repo.
- Skips uploading `outputs/` and large `.pt` files to the Space repo.

## Space runtime configuration

The app reads weights from the model repo at startup using environment variables:
- `WEIGHTS_REPO_ID` (model repo to download)
- `WEIGHTS_CACHE_DIR` (local cache directory inside the Space)

If you change the model repo name, set `WEIGHTS_REPO_ID` in the Space Settings.

## Troubleshooting

1. HTTP 400: Invalid option at sdk
- The org may restrict SDKs. If this happens, rerun with:

```bash
python3 deploy_hf.py --space-sdk gradio
```

- Or create the Space manually in the UI as Streamlit, then run:

```bash
python3 deploy_hf.py --skip-weights --space-repo <org>/<space_repo>
```

2. HTTP 401/403
- Token is missing or lacks write access.
- Use a token with write permission for the org.

3. Permission errors under `~/.local` or cache
- Always set `HF_HOME` and `HF_HUB_CACHE` as shown above.

4. Weights not found at runtime
- Confirm weights are uploaded to the model repo.
- Confirm `WEIGHTS_REPO_ID` matches the model repo.

## Optional flags

- `--skip-weights` to only upload the Space code
- `--skip-space` to only upload weights
- `--space-sdk <sdk>` to override the requested Space SDK

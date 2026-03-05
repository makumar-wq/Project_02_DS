# 🚀 How to Deploy VLM Caption Lab to Hugging Face Spaces

Since this project requires heavy Machine Learning models (BLIP, ViT-GPT2), the best way to share it with your mentor or reviewers is by deploying it for **free** on **Hugging Face Spaces**. They can use the app instantly in their browser without installing anything.

Here are the step-by-step instructions to deploy it right now.

---

### Step 1: Create a Hugging Face Space
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and create a free account (or log in).
2. Click **Create new Space**.
3. Fill out the form:
   - **Space name**: `vlm-caption-lab` (or whatever you like)
   - **License**: Choose `MIT` or `Creative Commons`
   - **Select the Space SDK**: Click **Streamlit**
   - **Space hardware**: Choose the **Free (CPU basic)** option.
4. Click **Create Space**.

### Step 2: Upload Your Code using the Web UI
The easiest way is to drag and drop your files.
1. In your new Space, click on the **Files** tab.
2. Click **Add file > Upload files**.
3. Select and upload the following files from your local `project_02` folder:
   - `app.py`
   - `config.py`
   - `data_prep.py`
   - `eval.py`
   - `requirements.txt`
   - `input.txt`
   - `shakespeare_transformer.pt`
4. Also, recreate the `configs/`, `models/`, and `experiments/` folders in the Hugging Face UI and upload the python files inside them. *(Or, if you know Git, just `git push` your whole repository to the Space!)*

### Step 3: Handle the Large `outputs/` Folder (Fine-tuned Weights)
Your `outputs/` folder is 2.4 GB. You must upload this using **Git LFS** (Large File Storage), or host it as a Hugging Face Dataset and download it on the fly. 

To keep it simple under a time crunch:
1. Go to **Settings** in your Space.
2. Scroll to **Variables and secrets**.
3. Your app will run using base weights automatically. The mentor will be able to test the *architectures* immediately.
4. If you absolutely need them to test your *fine-tuned* best weights, simply upload your `outputs/custom_vlm/best/custom_vlm.pt` file manually via the **Files** tab (it's small enough!). You can skip the massive ViT-GPT2 weights.

### Step 4: Watch it Build
Once your files (especially `app.py` and `requirements.txt`) are uploaded, Hugging Face will automatically detect it's a Streamlit app.
1. Click the **App** tab.
2. You will see a "Building" log. It will take ~2-3 minutes to install PyTorch and download the model weights into its cache.
3. Once the status turns green to **Running**, your app is live!

### Step 5: Share the Link!
Just copy the URL from your browser (e.g., `https://huggingface.co/spaces/your-username/vlm-caption-lab`) and send it to your mentor. You're done!

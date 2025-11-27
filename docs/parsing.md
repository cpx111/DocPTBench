# DocPTBench

In this walkthrough, we will demonstrate how to parse images using **PaddleOCR-VL** and evaluate the results within the DocPTBench framework.

## 1. Project Initialization & Installation

To avoid dependency conflicts, we will use two separate environments: one for the evaluation framework and one for the OCR model inference.

### Step 1.1: Clone & Setup OmniDocBench (Evaluation Env)

First, clone the repository and **ensure you are on the `v1_0` branch.** 

```bash
# 1. Clone the repository
git clone https://github.com/OmniDocBench/OmniDocBench.git
cd OmniDocBench

# 2. Switch to v1.0 branch (CRITICAL)
git checkout v1_0

# 3. Create Conda environment for Evaluation
conda create -n omnidocbench python=3.10
conda activate omnidocbench

# 4. Install dependencies
pip install -r requirements.txt
```

### Step 1.2: Setup PaddleOCR-VL (Inference Env)

Create a separate virtual environment for running the OCR model.

```bash
# 1. Create virtual environment
python -m venv .venv_paddleocr
source .venv_paddleocr/bin/activate

# 2. Install PaddlePaddle (CUDA 12.6 example)
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# 3. Install PaddleOCR with doc-parser
python -m pip install -U "paddleocr[doc-parser]"

# 4. Install specific safetensors (Linux example)
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
```

## 2. Data Preparation

```bash
cd OmniDocBench
# from huggingface download DocPTBench dataset
huggingface-cli download topdu/DocPTBench --local-dir ./DocPTBench --repo-type dataset
# from modelscope download DocPTBench datasets
# modelscope download --dataset topdktu/DocPTBench --local_dir ./DocPTBench
```

**Directory Structure:**

```
DocPTBench/                   <-- Root of the cloned repo
├── DocPTBench_combined.json  <-- Ground Truth
├── images/                      <-- Image folder
│   ├── images_synreal/         # 981 synthesized images (imitating real-world scenarios)
│   ├── images_synreal_dewarp/  # 981 unwarped synthesized images
│   ├── images_pic_synreal/     # 400 photos shot in the real world
│   └── images_pic_dewarp/      # 400 unwarped photos
```

## 3. Inference (Generating .md Files)

We need to generate Markdown files for all 4 folders. We will use a Python script to automate this.

**Create a file named `run_inference.py` in the `OmniDocBench` root:**

> **Note:** You must run this script **four times**, changing the `input_folder` and `output_folder` variables each time to process all four directories (e.g., run once for `images_synreal`, once for `images_synreal_dewarp`, etc.).

```python
import os
from pathlib import Path
from paddleocr import PaddleOCRVL

# Initialize the pipeline
pipeline = PaddleOCRVL()

sub_dirs = [
    "images_synreal",
    "images_synreal_dewarp",
    "images_pic_synreal",
    "images_pic_dewarp",
]

for sub_dir in sub_dirs:

    # --- Configuration ---
    # CHANGE THESE PATHS FOR EACH RUN
    input_folder = Path(f"./DocPTBench/images/{sub_dir}")   
    output_folder = Path(f"./output/PaddleOCR-VL-{sub_dir}") 

    # Ensure output directory exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Supported extensions
    valid_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    # Iterate over all files in the input folder
    for input_file in input_folder.iterdir():
        if input_file.suffix.lower() not in valid_extensions:
            continue
            
        print(f"Processing: {input_file.name}")

        try:
            # 1. Run Prediction
            output = pipeline.predict(input=str(input_file))

            # 2. Collect Markdown and Images from all pages
            markdown_list = []
            markdown_images = []
            
            for res in output:
                md_info = res.markdown
                markdown_list.append(md_info)
                markdown_images.append(md_info.get("markdown_images", {}))

            # 3. Concatenate text
            markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)

            # 4. Save Markdown File
            # Saved directly to: output_folder/filename.md
            mkd_file_path = output_folder / f"{input_file.stem}.md"
            
            with open(mkd_file_path, "w", encoding="utf-8") as f:
                f.write(markdown_texts)
            
            print(f"  -> Saved MD to: {mkd_file_path}")

            # 5. Save Extracted Images (Figures/Tables from the document)
            # These are saved into subfolders within the output directory to avoid clutter
            for item in markdown_images:
                if item:
                    for rel_path, image_obj in item.items():
                        # Combine output folder with the relative path provided by Paddle
                        # e.g., output_folder/markdown_images/img_0.jpg
                        save_path = output_folder / rel_path
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        image_obj.save(save_path)
        except Exception as e:
            print(f"Error processing {input_file.name}: {e}")
    print(f"Processing {sub_dir} complete.")
```

**Run the Inference:**

```bash
# Ensure you are in the PaddleOCR environment
source .venv_paddleocr/bin/activate

python run_inference.py
# Remember to edit the script and re-run for all 4 folders!
```

## 4. Result Organization (The Merge Step)

Per the benchmark requirements, we must copy the results from the "pic" (photographed) directories into the "synreal" (source) directories. This allows us to verify how well the model parses the photographed versions against the original ground truth structure.

**Expected Output Structure:**

```
OmniDocBench/                   
├── ...
├── output/                     
│   ├── PaddleOCR-VL-images_synreal/
│   ├── PaddleOCR-VL-images_synreal_dewarp/
│   ├── PaddleOCR-VL-images_pic_synreal/
│   └── PaddleOCR-VL-images_pic_dewarp/
└── ...
```

**Execute the Merge:**

```bash
# Copy photo results into md results
mkdir -p ./output/PaddleOCR-VL-photo_md/
cp -r ./output/PaddleOCR-VL-images_synreal/* ./output/PaddleOCR-VL-photo_md/
cp -r ./output/PaddleOCR-VL-images_pic_synreal/* ./output/PaddleOCR-VL-photo_md/
# Copy dewarp results into md results
mkdir -p ./output/PaddleOCR-VL-dewarp_md/
cp -r ./output/PaddleOCR-VL-images_synreal_dewarp/* ./output/PaddleOCR-VL-dewarp_md/
cp -r ./output/PaddleOCR-VL-images_pic_dewarp/* ./output/PaddleOCR-VL-dewarp_md/
```

## 5. Evaluation

### Step 5.1: Create/Modify Configuration File

Modify the config file `configs/end2end.yaml`.

**For Photographed Document Evaluation:**

```yaml
# ...
dataset:
  dataset_name: end2end_dataset
  ground_truth:
    data_path: ./DocPTBench/DocPTBench_combined.json
  prediction:
    # Point to the consolidated folder
    data_path: ./output/PaddleOCR-VL-photo_md
  match_method: quick_match
```

*(Note: To evaluate the dewarped set later, change `prediction.data_path` to `./output/PaddleOCR-VL-dewarp_md`)*

### Step 5.2: Run Validation

Now, switch back to the **OmniDocBench** environment to run the evaluation. You will likely need to run the evaluation twice: once for the photographed dataset and once for the dewarped dataset.

```bash
# Create a new bash script for evaluation
# Activate the benchmark environment
conda activate omnidocbench

# Run evaluation
python pdf_validation.py --config configs/end2end.yaml
```

The results will be printed to the console and saved in the `./result/` directory.
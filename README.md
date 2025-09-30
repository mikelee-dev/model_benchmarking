# Simple YOLO Model Benchmarking

A clean, minimal tool for benchmarking YOLO models using Ultralytics' built-in `model.val()` method. No complex framework - just simple, reliable model testing and comparison.

## 🌟 Features

- **Simple**: Single Python file, no complex dependencies
- **Accurate**: Uses Ultralytics' built-in `model.val()` for reliable metrics
- **Fast**: Direct model testing without framework overhead
- **Clean**: JSON output with all essential metrics
- **Flexible**: Test single models or compare multiple models
- **Auto-download**: Models and datasets downloaded automatically on first run

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create conda environment
conda create -n model_benchmarking python=3.9
conda activate model_benchmarking

# Install dependencies
pip install -r requirements.txt
```

### 2. Test a Single Model
```bash
# Uses COCO8 dataset (downloaded automatically)
python simple_benchmark.py --model yolov8l --dataset coco8.yaml
```

### 3. Compare Multiple Models
```bash
# Models and dataset downloaded automatically on first run
python simple_benchmark.py --models yolov8l yolo11l yolov8s-world --dataset coco8.yaml
```

## 📊 Example Usage

### Single Model Test
```bash
$ python simple_benchmark.py --model yolov8l --dataset coco8.yaml

Testing yolov8l on coco8.yaml
Results saved to results/yolov8l_results.json
```

**Output JSON:**
```json
{
  "model": "yolov8l",
  "dataset": "coco8.yaml",
  "mAP50": 0.976,
  "mAP50_95": 0.760,
  "precision": 0.914,
  "recall": 0.839,
  "f1": 0.875,
  "eval_time": 23.3
}
```

### Model Comparison
```bash
$ python simple_benchmark.py --models yolov8l yolo11l yolov8s-world --dataset coco8.yaml

=== COMPARISON RESULTS ===
Model           mAP@0.5    Precision  Recall     Time(s) 
------------------------------------------------------------
yolov8l             97.6%     91.4%     83.9%   23.3
yolo11l             96.7%     90.9%     86.2%    8.0
yolov8s-world       84.9%     63.6%     76.7%    4.7
```

## 🔧 Command Line Options

```bash
python simple_benchmark.py [OPTIONS]

Options:
  --model MODEL        Single model to test (e.g., yolov8l)
  --models MODEL ...   Multiple models to compare
  --dataset PATH       Dataset YAML file path (default: coco8.yaml)
  --output DIR         Output directory (default: results)
  --help               Show help message
```

## 📋 Supported Models

All Ultralytics-compatible YOLO models:
- **YOLOv8**: `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`
- **YOLOv11**: `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x`
- **YOLO-World**: `yolov8s-world`, `yolov8m-world`, `yolov8l-world`, `yolov8x-world`
- **Custom**: Any `.pt` model file compatible with Ultralytics

## 📊 Metrics Explained

- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds 0.5-0.95
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1**: Harmonic mean of Precision and Recall
- **Eval Time**: Total evaluation time in seconds

## 📁 Output Files

- `results/{model}_results.json` - Individual model results
- `results/comparison_results.json` - Comparison results (when using `--models`)

## 🎯 Use Cases

### 1. Model Selection
```bash
# Compare different YOLO variants
python simple_benchmark.py --models yolov8n yolov8s yolov8m yolov8l yolov8x --dataset your_dataset.yaml
```

### 2. Performance Analysis
```bash
# Test specific model performance
python simple_benchmark.py --model yolov8l --dataset datasets/coco8.yaml --output analysis_results
```

### 3. Quick Benchmarking
```bash
# Fast comparison of two models
python simple_benchmark.py --models yolov8l yolo11l --dataset datasets/coco8.yaml
```

## 🔍 Dataset Format

The tool expects datasets in YOLO format with a YAML configuration file:

```yaml
# dataset.yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 80  # number of classes
names: ['person', 'bicycle', 'car', ...]  # class names
```

## 🛠️ Adding Custom Models and Datasets

### Adding Local Models

**1. Place your model file in the project directory:**
```bash
# Copy your custom model
cp /path/to/your/custom_model.pt ./custom_model.pt
```

**2. Test your custom model:**
```bash
# Test single custom model
python simple_benchmark.py --model custom_model --dataset coco8.yaml

# Compare custom model with others
python simple_benchmark.py --models yolov8l custom_model --dataset coco8.yaml
```

**3. Supported model formats:**
- `.pt` files (PyTorch models)
- Any model compatible with `ultralytics.YOLO()`
- Custom trained YOLO models
- Converted models from other frameworks

### Adding Local Datasets

**1. Create dataset directory structure:**
```bash
mkdir -p datasets/my_dataset/images/{train,val}
mkdir -p datasets/my_dataset/labels/{train,val}
```

**2. Organize your data:**
```
datasets/my_dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── val/
│       ├── img3.jpg
│       └── img4.jpg
└── labels/
    ├── train/
    │   ├── img1.txt
    │   └── img2.txt
    └── val/
        ├── img3.txt
        └── img4.txt
```

**3. Create dataset YAML file:**
```yaml
# datasets/my_dataset.yaml
path: datasets/my_dataset
train: images/train
val: images/val
nc: 5  # number of classes
names: ['class1', 'class2', 'class3', 'class4', 'class5']
```

**4. Test with your dataset:**
```bash
# Test model on custom dataset
python simple_benchmark.py --model yolov8l --dataset datasets/my_dataset.yaml

# Compare models on custom dataset
python simple_benchmark.py --models yolov8l yolo11l --dataset datasets/my_dataset.yaml
```

### Label Format

Labels should be in YOLO format (one `.txt` file per image):
```
# img1.txt
0 0.5 0.5 0.3 0.4  # class_id center_x center_y width height
1 0.2 0.3 0.1 0.2
```

Where:
- `class_id`: Integer class ID (0, 1, 2, ...)
- `center_x, center_y`: Normalized center coordinates (0-1)
- `width, height`: Normalized dimensions (0-1)

### Example: Custom Dataset Setup

```bash
# 1. Create dataset structure
mkdir -p datasets/vehicles/images/{train,val}
mkdir -p datasets/vehicles/labels/{train,val}

# 2. Copy your images
cp /path/to/train/images/* datasets/vehicles/images/train/
cp /path/to/val/images/* datasets/vehicles/images/val/

# 3. Copy your labels
cp /path/to/train/labels/* datasets/vehicles/labels/train/
cp /path/to/val/labels/* datasets/vehicles/labels/val/

# 4. Create YAML config
cat > datasets/vehicles.yaml << EOF
path: datasets/vehicles
train: images/train
val: images/val
nc: 3
names: ['car', 'truck', 'bus']
EOF

# 5. Test your dataset
python simple_benchmark.py --model yolov8l --dataset datasets/vehicles.yaml
```

## ⚡ Performance Notes

- **Evaluation Time**: Includes model loading, validation, and metric calculation
- **Memory**: Models are loaded fresh for each test to avoid state interference
- **Accuracy**: Uses Ultralytics' built-in validation for reliable metrics
- **YOLO-World**: Tested with full COCO vocabulary (standard benchmarking practice)

## 🛠️ Requirements

- Python 3.9+
- Ultralytics
- PyTorch (installed with Ultralytics)

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

This is a minimal tool designed for simplicity. If you need additional features, consider:
- Adding visualization plots
- Supporting more output formats
- Adding batch processing capabilities
- Integrating with other ML frameworks
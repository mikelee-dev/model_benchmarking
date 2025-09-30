#!/usr/bin/env python3
"""
Simple YOLO Model Benchmarking Tool
Tests YOLO models and reports evaluation metrics using model.val()
"""

import argparse
import json
import time
from pathlib import Path
from ultralytics import YOLO
import yaml

def test_model(model_name, dataset_path, output_dir="results"):
    """Test a single model and return metrics."""
    print(f"Testing {model_name} on {dataset_path}")
    
    # Load model (will download automatically if not present)
    model = YOLO(f'{model_name}.pt')
    
    # Run validation
    start_time = time.time()
    results = model.val(data=dataset_path, verbose=False)
    eval_time = time.time() - start_time
    
    
    # Extract metrics
    metrics = {
        'model': model_name,
        'dataset': dataset_path,
        'mAP50': float(results.box.map50),
        'mAP50_95': float(results.box.map),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr),
        'f1': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr) if (results.box.mp + results.box.mr) > 0 else 0,
        'eval_time': eval_time
    }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results_file = output_path / f"{model_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to {results_file}")
    return metrics

def compare_models(models, dataset_path, output_dir="results"):
    """Compare multiple models."""
    print(f"Comparing models: {models}")
    
    all_results = []
    for model in models:
        try:
            metrics = test_model(model, dataset_path, output_dir)
            all_results.append(metrics)
        except Exception as e:
            print(f"Error testing {model}: {e}")
            continue
    
    # Save comparison
    comparison_file = Path(output_dir) / "comparison_results.json"
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n=== COMPARISON RESULTS ===")
    print(f"{'Model':<15} {'mAP@0.5':<10} {'Precision':<10} {'Recall':<10} {'Time(s)':<8}")
    print("-" * 60)
    
    for result in all_results:
        print(f"{result['model']:<15} {result['mAP50']*100:>8.1f}% {result['precision']*100:>8.1f}% {result['recall']*100:>8.1f}% {result['eval_time']:>6.1f}")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Simple YOLO Model Benchmarking')
    parser.add_argument('--model', type=str, help='Single model to test (e.g., yolov8l)')
    parser.add_argument('--models', nargs='+', help='Multiple models to compare')
    parser.add_argument('--dataset', type=str, default='coco8.yaml', help='Dataset path (will download automatically)')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    if args.model:
        test_model(args.model, args.dataset, args.output)
    elif args.models:
        compare_models(args.models, args.dataset, args.output)
    else:
        print("Please specify either --model or --models")

if __name__ == "__main__":
    main()

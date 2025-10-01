#!/usr/bin/env python3
"""
Simple YOLO Model Training Script
Trains YOLO models using Ultralytics on COCO8 dataset
"""
import argparse
import time
from pathlib import Path
from ultralytics import YOLO

def train_model(model_name, epochs=10, imgsz=640, batch=16):
    """
    Train a YOLO model on COCO8 dataset
    
    Args:
        model_name (str): Model name (e.g., 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt')
        epochs (int): Number of training epochs
        imgsz (int): Image size for training
        batch (int): Batch size
    """
    print(f"Training {model_name} on COCO8 dataset...")
    print(f"Epochs: {epochs}, Image size: {imgsz}, Batch size: {batch}")
    
    # Load model
    model = YOLO(model_name)
    
    # Train the model
    start_time = time.time()
    results = model.train(
        data='coco8.yaml',  # COCO8 dataset
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        verbose=True
    )
    train_time = time.time() - start_time
    
    print(f"\nTraining completed in {train_time:.2f} seconds")
    print(f"Model saved to: {results.save_dir}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train YOLO models on COCO8 dataset')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                       help='Model name (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)')
    parser.add_argument('--epochs', type=int, default=10, 
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--imgsz', type=int, default=640, 
                       help='Image size for training (default: 640)')
    parser.add_argument('--batch', type=int, default=16, 
                       help='Batch size (default: 16)')
    
    args = parser.parse_args()
    
    # Validate model name
    valid_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
    if args.model not in valid_models:
        print(f"Warning: {args.model} not in recommended list: {valid_models}")
    
    # Train the model
    train_model(args.model, args.epochs, args.imgsz, args.batch)

if __name__ == "__main__":
    main()

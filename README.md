## Script for Downstream model training

To evaluate model performance using Advex powered data:
1. Clone this repository and cd into `model-training`
2. Run `python standalone_yolo.py --real-dir /path/to/customer_data --syn-dir /path/to/advex_data --epochs 200 --seed 1`

This script trains a model on customer data and another model on customer data and advex data. It will generate a report comparing
metrics such as F1, Recall, and Average Precision across the two models.

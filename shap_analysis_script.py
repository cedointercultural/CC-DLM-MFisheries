#!/usr/bin/env python3

import os
import sys
import warnings
import argparse
import pandas as pd
import tensorflow as tf

# Suppress TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Desactiva los mensajes de INFO y WARNING

# Ensure the script can import the shap_analysis module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Suppress Python warnings
warnings.filterwarnings("ignore")

from scripts.shap_analysis import prepare_future_data, run_shap_analysis, evaluate_all_models

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Perform SHAP Analysis on Machine Learning Models')
    
    # Add arguments
    parser.add_argument('--models-folder', 
                        default='modelos_moe', 
                        help='Path to the folder containing model files')
    parser.add_argument('--future-data', 
                        default='future_data.csv', 
                        help='Path to the future data CSV file')
    parser.add_argument('--output', 
                        default='shap_analysis_results.csv', 
                        help='Path to the output CSV file')
    parser.add_argument('--verbose', 
                        action='store_true', 
                        help='Enable verbose output')

    # Parse arguments
    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.future_data):
        print(f"Error: Future data file '{args.future_data}' does not exist.")
        sys.exit(1)

    if not os.path.exists(args.models_folder):
        print(f"Error: Models folder '{args.models_folder}' does not exist.")
        sys.exit(1)

    # Perform SHAP analysis
    try:
        evaluate_all_models(
            models_folder=args.models_folder,
            future_data_file=args.future_data,
            shap_analysis_function=run_shap_analysis,
            output_csv=args.output,
            verbose=args.verbose
        )

        # Load and display results
        if os.path.exists(args.output):
            shap_df = pd.read_csv(args.output)
            print("\nSHAP Analysis Results:")
            print(shap_df.head())
            print(f"\nTotal results saved to: {args.output}")
        else:
            print("No results were generated.")

    except Exception as e:
        print(f"An error occurred during SHAP analysis: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
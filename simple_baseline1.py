import argparse
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import pandas as pd

def compare_binary_vectors(binary_vector_true, binary_vector_predicted):
    binary_vector_true = np.array(binary_vector_true)
    binary_vector_predicted = np.array(binary_vector_predicted)
    
    correctly_predicted = np.sum(binary_vector_true * binary_vector_predicted)
    total_true_orfs = np.sum(binary_vector_true)
    percentage_correctly_predicted = (correctly_predicted / total_true_orfs) * 100 if total_true_orfs > 0 else 0

    prediction_length = len(binary_vector_predicted)
    prediction_ones = np.sum(binary_vector_predicted)
    percentage_prediction_ones = (prediction_ones / prediction_length) * 100

    return percentage_correctly_predicted, percentage_prediction_ones

def compare_metrics(true_orfs, predicted_orfs):
    accuracy = accuracy_score(true_orfs, predicted_orfs)
    precision = precision_score(true_orfs, predicted_orfs)
    recall = recall_score(true_orfs, predicted_orfs)
    f1 = f1_score(true_orfs, predicted_orfs)
    
    return accuracy, precision, recall, f1

def count_ones_zeros(binary_vector):
    ones_count = np.sum(binary_vector)
    zeros_count = len(binary_vector) - ones_count
    return ones_count, zeros_count

def predict_orfs(sequence):
    orfs = []
    start_codon = "ATG"
    stop_codons = ["TAA", "TAG", "TGA"]
    
    in_orf = False
    orf_start = 0
    
    for i in range(len(sequence) - len(start_codon) + 1):
        codon = sequence[i:i+len(start_codon)]
        
        if codon == start_codon:
            if not in_orf:
                in_orf = True
                orf_start = i
        elif codon in stop_codons:
            if in_orf:
                orfs.append((orf_start, i+len(start_codon)))  # Include the stop codon in the ORF
                in_orf = False
            
    return orfs

def orf_positions_to_binary_vector(sequence, orf_positions):
    binary_vector = [0] * len(sequence)
    for start, end in orf_positions:
        adjusted_end = min(end + 1, len(sequence))
        binary_vector[start:adjusted_end] = [1] * (adjusted_end - start)
    return binary_vector

def filter_small_orfs(sequence, orf_positions, max_size=150):
    small_orfs = []
    for start, end in orf_positions:
        orf_size = end - start  # Assuming end is exclusive
        if orf_size <= max_size:
            small_orfs.append((start, end))
    return small_orfs


def main(parquet_folder):
    train_parquet_file = os.path.join(parquet_folder, 'train_data.parquet')
    inference_parquet_file = os.path.join(parquet_folder, 'inference_data.parquet')
    train_df = pd.read_parquet(train_parquet_file)
    inference_df = pd.read_parquet(inference_parquet_file)
    
    combined_df = pd.concat([train_df, inference_df], ignore_index=True)
    
    accuracies_all = []
    precisions_all = []
    recalls_all = []
    f1_scores_all = []
    
    accuracies_small = []
    precisions_small = []
    recalls_small = []
    f1_scores_small = []
    
    for index, row in combined_df.iterrows():
        sequence = row['sequence']
        true_orfs = row['target'] if 'target' in row else None
        orf_lengths = row['orf_lengths'] if 'orf_lengths' in row else None
        
        start_time = time.time()
        predicted_orfs_positions = predict_orfs(sequence)
        end_time = time.time()
        
        print(f"ORF prediction for sequence {index} took {end_time - start_time} seconds.")

        # Convert positions to binary vectors for all ORFs
        predicted_orfs_all = orf_positions_to_binary_vector(sequence, predicted_orfs_positions)
        
        if true_orfs is not None:
            if isinstance(true_orfs, list):  # If true_orfs is a list of positions, convert it
                true_orfs_all = orf_positions_to_binary_vector(sequence, true_orfs)
            else:
                true_orfs_all = true_orfs  # Assume it's already a binary vector

            # Calculate metrics for all ORFs
            accuracy_all, precision_all, recall_all, f1_all = compare_metrics(true_orfs_all, predicted_orfs_all)
            
            accuracies_all.append(accuracy_all)
            precisions_all.append(precision_all)
            recalls_all.append(recall_all)
            f1_scores_all.append(f1_all)
            
            print(f"All ORFs - Sequence Index: {index}, Accuracy: {accuracy_all}, Precision: {precision_all}, Recall: {recall_all}, F1 Score: {f1_all}")
            
            # Check for sequences with small ORFs and evaluate
            if orf_lengths is not None and np.any(np.array(orf_lengths) <= 150):
                # Since we cannot directly filter predicted_orfs_positions for small ORFs, we evaluate the sequence if it contains any small ORF
                accuracies_small.append(accuracy_all)
                precisions_small.append(precision_all)
                recalls_small.append(recall_all)
                f1_scores_small.append(f1_all)
                
                print(f"Small ORFs - Sequence Index: {index}, Accuracy: {accuracy_all}, Precision: {precision_all}, Recall: {recall_all}, F1 Score: {f1_all}")

    # Calculate and print average metrics for all ORFs
    avg_accuracy_all = np.mean(accuracies_all) if accuracies_all else "No data"
    avg_precision_all = np.mean(precisions_all) if precisions_all else "No data"
    avg_recall_all = np.mean(recalls_all) if recalls_all else "No data"
    avg_f1_all = np.mean(f1_scores_all) if f1_scores_all else "No data"
    
    print("\nAverage Metrics for All ORFs:")
    print(f"Accuracy: {avg_accuracy_all}, Precision: {avg_precision_all}, Recall: {avg_recall_all}, F1 Score: {avg_f1_all}")
    
    # Calculate and print average metrics for sequences with small ORFs, if there are any
    avg_accuracy_small = np.mean(accuracies_small) if accuracies_small else "No data"
    avg_precision_small = np.mean(precisions_small) if precisions_small else "No data"
    avg_recall_small = np.mean(recalls_small) if recalls_small else "No data"
    avg_f1_small = np.mean(f1_scores_small) if f1_scores_small else "No data"
    
    print("\nAverage Metrics for Sequences with Small ORFs:")
    print(f"Accuracy: {avg_accuracy_small}, Precision: {avg_precision_small}, Recall: {avg_recall_small}, F1 Score: {avg_f1_small}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict ORFs and evaluate predictions.')
    parser.add_argument('--parquet_folder', type=str, required=True, help='Path to the folder containing parquet files.')
    args = parser.parse_args()
    main(args.parquet_folder)

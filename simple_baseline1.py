import argparse
import os
import glob
import re
from Bio import SeqIO
import gffpandas.gffpandas as gp
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import psutil
import time

def load_fasta_files(fasta_folder_path):
    # Memory usage before loading FASTA files
    mem_before = psutil.virtual_memory().used
    fasta_paths = glob.glob(os.path.join(fasta_folder_path, '*.fna'))
    fasta_records = {}
    for path in fasta_paths:
        for record in SeqIO.parse(path, "fasta"):
            fasta_records[record.id] = str(record.seq)
    # Memory usage after loading FASTA files
    mem_after = psutil.virtual_memory().used
    print(f"Memory used for loading FASTA files: {mem_after - mem_before} bytes")
    return fasta_records

def load_gff_files_and_extract_positions(gff_folder_path):
    gff_paths = glob.glob(os.path.join(gff_folder_path, '*.gff'))
    gff_records = []

    for path in gff_paths:
        annotation = gp.read_gff3(path)
        gff_df = annotation.df
        gff_df = gff_df[gff_df['type'] == 'CDS']
        positions = gff_df[['start', 'end']].values.tolist()
        sequence_id = gff_df['seq_id'].iloc[0]
        gff_records.append({'ID': sequence_id, 'Positions': positions})
    return gff_records

def extract_orf_positions(sequence, start_codon='ATG', stop_codons=['TAA', 'TAG', 'TGA']):
    orf_positions = []
    start_indices = [m.start() for m in re.finditer(start_codon, sequence)]
    for start_index in start_indices:
        for stop_codon in stop_codons:
            stop_index = sequence.find(stop_codon, start_index)
            if stop_index!= -1:
                orf_positions.append((start_index, stop_index + len(stop_codon)))
                break
    return orf_positions

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


def true_positions_to_binary_vector(sequence, true_orf_positions):
    binary_vector = [0] * len(sequence)
    for start, end in true_orf_positions:
        binary_vector[start:end+1] = [1] * (end - start + 1)
    return binary_vector

def predicted_positions_to_binary_vector(sequence, predicted_orf_positions):
    binary_vector = [0] * len(sequence)
    for start, end in predicted_orf_positions:
        binary_vector[start:end+1] = [1] * (end - start + 1)
    return binary_vector

def compare_binary_vectors(binary_vector_true, binary_vector_predicted):
    # Ensure both vectors are NumPy arrays
    binary_vector_true = np.array(binary_vector_true)
    binary_vector_predicted = np.array(binary_vector_predicted)
    
    # Calculate the percentage of correctly predicted ORFs
    correctly_predicted = np.sum(binary_vector_true * binary_vector_predicted)
    total_true_orfs = np.sum(binary_vector_true)
    percentage_correctly_predicted = (correctly_predicted / total_true_orfs) * 100 if total_true_orfs > 0 else 0

    # Calculate the percentage of the prediction that is 1 for each prediction
    prediction_length = len(binary_vector_predicted)
    prediction_ones = np.sum(binary_vector_predicted)
    percentage_prediction_ones = (prediction_ones / prediction_length) * 100

    return percentage_correctly_predicted, percentage_prediction_ones

def compare_metrics(true_orfs, predicted_orfs):
    true_orfs_bin = np.array(true_orfs)
    predicted_orfs_bin = np.array(predicted_orfs)
    
    accuracy = accuracy_score(true_orfs_bin, predicted_orfs_bin)
    precision = precision_score(true_orfs_bin, predicted_orfs_bin)
    recall = recall_score(true_orfs_bin, predicted_orfs_bin)
    f1 = f1_score(true_orfs_bin, predicted_orfs_bin)
    
    return accuracy, precision, recall, f1

def count_ones_zeros(binary_vector):
    ones_count = binary_vector.count(1)
    zeros_count = binary_vector.count(0)
    return ones_count, zeros_count

def main(fasta_dir, gff_dir, output_dir, inference_ratio=0.2, random_state=42):
    fasta_files = [os.path.join(fasta_dir, f) for f in os.listdir(fasta_dir) if f.endswith(".fna")]
    gff_files = [os.path.join(gff_dir, f) for f in os.listdir(gff_dir) if f.endswith(".gff")]

    # Reserve inference set
    fasta_files, inference_fasta_files = train_test_split(fasta_files, test_size=inference_ratio, random_state=random_state)
    gff_files, inference_gff_files = train_test_split(gff_files, test_size=inference_ratio, random_state=random_state)

    for fasta_file, gff_file in zip(fasta_files, gff_files):
        sequence = SeqIO.read(fasta_file, "fasta").seq
        gff_records = gp.read_gff3(gff_file)
        gff_df = gff_records.df
        gff_df = gff_df[gff_df['type'] == 'CDS']
        positions = gff_df[['start', 'end']].values.tolist()
        sequence_id = gff_df['seq_id'].iloc[0]

        start_time = time.time()
        true_orf_positions = extract_orf_positions(sequence)
        true_orfs = true_positions_to_binary_vector(sequence, true_orf_positions)

        predicted_orf_positions = predict_orfs(sequence)
        predicted_orfs = predicted_positions_to_binary_vector(sequence, predicted_orf_positions)

        end_time = time.time()
        print(f"ORF prediction for {os.path.basename(fasta_file)} took {end_time - start_time} seconds.")

        # Calculate and print the counts of 1s and 0s for the predicted ORFs
        ones_count, zeros_count = count_ones_zeros(predicted_orfs)
        print(f"Sequence ID: {sequence_id}, Ones: {ones_count}, Zeros: {zeros_count}")

        # Call compare_metrics to compute accuracy, precision, recall, and F1 score
        accuracy, precision, recall, f1 = compare_metrics(true_orfs, predicted_orfs)
        
        print(f"Sequence ID: {sequence_id}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict ORFs and evaluate predictions.')
    parser.add_argument('--fasta', type=str, required=True, help='Path to the folder containing FASTA files.')
    parser.add_argument('--gff', type=str, required=True, help='Path to the folder containing GFF files.')
    parser.add_argument('--output', type=str, required=True, help='Output directory for Prodigal outputs.')
    args = parser.parse_args()
    main(args.fasta, args.gff, args.output)
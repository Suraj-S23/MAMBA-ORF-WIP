'''import os
import subprocess
import gffpandas.gffpandas as gp
import argparse
from Bio import SeqIO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_prodigal(fasta_path, output_dir):
    base_name = os.path.splitext(os.path.basename(fasta_path))[0]
    output_gff = os.path.join(output_dir, f"{base_name}.gff")
    
    # Specify the full path to the prodigal executable
    prodigal_path = "D:\Suraj\ALUF\SEM 3\MASTER PROJECT\prodigal.exe"  # Adjust the path as necessary
    
    subprocess.run([prodigal_path, "-i", fasta_path, "-f", "gff", "-o", output_gff, "-p", "meta"])
    return output_gff

def parse_gff(gff_path):
    true_orfs = []
    with open(gff_path, 'r') as gff_file:
        for line in gff_file:
            if not line.startswith('#'):
                fields = line.strip().split('\t')
                if fields[2] == 'CDS':
                    start = int(fields[3])
                    end = int(fields[4])
                    true_orfs.append([start, end])
    return true_orfs

def convert_orfs_to_binary_vector(sequence, orfs):
    binary_vector = [0] * len(sequence)
    for start, end in orfs:
        binary_vector[start:end+1] = [1] * (end - start + 1)
    return binary_vector

def compare_orfs(predicted_orfs, true_orfs, sequence):
    # Convert ORF positions to binary vectors for comparison
    true_orfs_bin = convert_orfs_to_binary_vector(sequence, true_orfs)
    predicted_orfs_bin = convert_orfs_to_binary_vector(sequence, predicted_orfs)
    
    # Calculate metrics
    accuracy = accuracy_score(true_orfs_bin, predicted_orfs_bin)
    precision = precision_score(true_orfs_bin, predicted_orfs_bin)
    recall = recall_score(true_orfs_bin, predicted_orfs_bin)
    f1 = f1_score(true_orfs_bin, predicted_orfs_bin)
    
    return accuracy, precision, recall, f1

def main(fasta_dir, gff_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for fasta_file in os.listdir(fasta_dir):
        if fasta_file.endswith(".fna"):
            fasta_path = os.path.join(fasta_dir, fasta_file)
            gff_file = os.path.splitext(fasta_file)[0] + ".gff"
            gff_path = os.path.join(gff_dir, gff_file)
            
            predicted_gff_path = run_prodigal(fasta_path, output_dir)
            true_orfs = parse_gff(gff_path)
            predicted_orfs = parse_gff(predicted_gff_path)
            
            # Assuming sequence is available, you need to pass it to compare_orfs
            # For simplicity, let's assume it's the first sequence in the FASTA file
            sequence = next(SeqIO.parse(fasta_path, "fasta")).seq
            accuracy, precision, recall, f1 = compare_orfs(predicted_orfs, true_orfs, sequence)
            
            print(f"Accuracy for {fasta_file}: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare Prodigal predictions with true ORF positions from GFF files.')
    parser.add_argument('--fasta_dir', type=str, required=True, help='Path to the directory containing FASTA files.')
    parser.add_argument('--gff_dir', type=str, required=True, help='Path to the directory containing GFF files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory for Prodigal results.')
    
    args = parser.parse_args()
    
    main(args.fasta_dir, args.gff_dir, args.output_dir)
'''

import os
import subprocess
import argparse
from Bio import SeqIO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_prodigal(fasta_path, output_dir):
    base_name = os.path.splitext(os.path.basename(fasta_path)[0])
    output_gff = os.path.join(output_dir, f"{base_name}.gff")
    
    # Specify the full path to the prodigal executable with raw string
    prodigal_path = r"D:\Suraj\ALUF\SEM 3\MASTER PROJECT\prodigal.exe"  # Adjust the path as necessary
    
    subprocess.run([prodigal_path, "-i", fasta_path, "-f", "gff", "-o", output_gff, "-p", "meta"])
    return output_gff

def parse_gff(gff_path):
    true_orfs = []
    with open(gff_path, 'r') as gff_file:
        for line in gff_file:
            if not line.startswith('#'):
                fields = line.strip().split('\t')
                if fields[2] == 'CDS':
                    start = int(fields[3]) - 1  # Adjust for 0-based indexing
                    end = int(fields[4])  # Adjust for inclusive end
                    true_orfs.append([start, end])
    return true_orfs

def convert_orfs_to_binary_vector(sequence, orfs):
    binary_vector = [0] * len(sequence)
    for start, end in orfs:
        # Ensure start and end are within sequence bounds
        adjusted_start = max(0, min(start, len(sequence)))
        adjusted_end = max(0, min(end, len(sequence)))
        
        # Adjust for 0-based indexing and exclusive end
        binary_vector[adjusted_start:adjusted_end] = [1] * (adjusted_end - adjusted_start)
    return binary_vector


def compare_orfs(predicted_orfs, true_orfs, sequence):
    # Convert ORF positions to binary vectors for comparison
    true_orfs_bin = convert_orfs_to_binary_vector(sequence, true_orfs)
    predicted_orfs_bin = convert_orfs_to_binary_vector(sequence, predicted_orfs)
    
    # Calculate metrics for all ORFs
    accuracy = accuracy_score(true_orfs_bin, predicted_orfs_bin)
    precision = precision_score(true_orfs_bin, predicted_orfs_bin)
    recall = recall_score(true_orfs_bin, predicted_orfs_bin)
    f1 = f1_score(true_orfs_bin, predicted_orfs_bin)
    
    # Filter small ORFs (smaller than 150 bases)
    small_true_orfs = [orf for orf in true_orfs if orf[1] - orf[0] < 150]
    small_predicted_orfs = [orf for orf in predicted_orfs if orf[1] - orf[0] < 150]
    
    # Convert small ORF positions to binary vectors
    small_true_orfs_bin = convert_orfs_to_binary_vector(sequence, small_true_orfs)
    small_predicted_orfs_bin = convert_orfs_to_binary_vector(sequence, small_predicted_orfs)
    
    # Calculate metrics for small ORFs
    small_accuracy = accuracy_score(small_true_orfs_bin, small_predicted_orfs_bin)
    small_precision = precision_score(small_true_orfs_bin, small_predicted_orfs_bin)
    small_recall = recall_score(small_true_orfs_bin, small_predicted_orfs_bin)
    small_f1 = f1_score(small_true_orfs_bin, small_predicted_orfs_bin)
    
    return accuracy, precision, recall, f1, small_accuracy, small_precision, small_recall, small_f1

def main(fasta_dir, gff_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for fasta_file in os.listdir(fasta_dir):
        if fasta_file.endswith(".fna"):
            fasta_path = os.path.join(fasta_dir, fasta_file)
            gff_file = os.path.splitext(fasta_file)[0] + ".gff"
            gff_path = os.path.join(gff_dir, gff_file)
            
            predicted_gff_path = run_prodigal(fasta_path, output_dir)
            true_orfs = parse_gff(gff_path)
            predicted_orfs = parse_gff(predicted_gff_path)
            
            # Assuming sequence is available, you need to pass it to compare_orfs
            sequence = next(SeqIO.parse(fasta_path, "fasta")).seq
            
            # Convert ORF positions to binary vectors for comparison
            true_orfs_bin = convert_orfs_to_binary_vector(sequence, true_orfs)
            predicted_orfs_bin = convert_orfs_to_binary_vector(sequence, predicted_orfs)
            
            # Ensure sequence length matches for conversion
            if len(sequence) != len(true_orfs_bin) or len(sequence) != len(predicted_orfs_bin):
                print(f"Warning: Length mismatch for {fasta_file}. Skipping.")
                continue
            
            accuracy, precision, recall, f1, small_accuracy, small_precision, small_recall, small_f1 = compare_orfs(predicted_orfs, true_orfs, sequence)
            
            print(f"Accuracy for {fasta_file}: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
            print(f"Small ORFs - Accuracy for {fasta_file}: {small_accuracy}, Precision: {small_precision}, Recall: {small_recall}, F1 Score: {small_f1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare Prodigal predictions with true ORF positions from GFF files.')
    parser.add_argument('--fasta_dir', type=str, required=True, help='Path to the directory containing FASTA files.')
    parser.add_argument('--gff_dir', type=str, required=True, help='Path to the directory containing GFF files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory for Prodigal results.')
    
    args = parser.parse_args()
    
    main(args.fasta_dir, args.gff_dir, args.output_dir)

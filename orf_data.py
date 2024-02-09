import os
import glob
import pandas as pd
from Bio import SeqIO
import gffpandas.gffpandas as gp
import numpy as np
import argparse

class DataProcessor:
    def __init__(self, fasta_folder_path, gff_folder_path, output_file):
        self.fasta_folder_path = fasta_folder_path
        self.gff_folder_path = gff_folder_path
        self.output_file = output_file

    def extract_positions(self, gff_df):
        positions = list(zip(gff_df['start'].astype(int), gff_df['end'].astype(int)))
        return positions

    def load_fasta_files(self):
        fasta_paths = glob.glob(os.path.join(self.fasta_folder_path, '*.fna'))
        records = []
        for fasta_path in fasta_paths:
            for seq_record in SeqIO.parse(fasta_path, "fasta"):
                seq_data = {'ID': seq_record.id, 'Sequence': str(seq_record.seq)}
                records.append(seq_data)
        return pd.DataFrame(records)

    def load_gff_files_and_extract_positions(self):
        gff_paths = glob.glob(os.path.join(self.gff_folder_path, '*.gff'))
        records = []
        for gff_path in gff_paths:
            annotation = gp.read_gff3(gff_path)
            gff_df = annotation.df
            gff_df = gff_df[gff_df['type'] == 'CDS']
            positions = self.extract_positions(gff_df)
            sequence_id = gff_df['seq_id'].iloc[0]
            records.append({'ID': sequence_id, 'Positions': positions})
        return pd.DataFrame(records)

    def convert_to_numeric_classes(self, sequence, positions):
        labels = np.zeros(len(sequence), dtype=int)
        for start, end in positions:
            start_idx = int(start)
            end_idx = int(end)
            labels[start_idx-1] = 1
            labels[end_idx-1] = 2
            labels[start_idx:end_idx] = 3
        return labels.tolist()

    def process_data(self):
        fasta_df = self.load_fasta_files()
        positions_df = self.load_gff_files_and_extract_positions()
        merged_df = pd.merge(fasta_df, positions_df, on='ID', how='left').dropna().reset_index(drop=True)

        merged_df['Labels'] = merged_df.apply(lambda row: self.convert_to_numeric_classes(row['Sequence'], row['Positions']), axis=1)

        merged_df.to_parquet(self.output_file)
        print(f"Merged DataFrame saved as Parquet file at: {self.output_file}")
        
        self.verify_output()

    def verify_output(self):
        print("Reading the saved Parquet file to verify its contents:")
        saved_df = pd.read_parquet(self.output_file)
        print(saved_df.head())  # Print only the first few rows for verification

def main():
    parser = argparse.ArgumentParser(description="Process fasta and gff files and save the merged data to a Parquet file.")
    parser.add_argument("fasta_folder", help="Path to the folder containing fasta files")
    parser.add_argument("gff_folder", help="Path to the folder containing gff files")
    parser.add_argument("output_file", help="Path to the output Parquet file")

    args = parser.parse_args()

    processor = DataProcessor(args.fasta_folder, args.gff_folder, args.output_file)
    processor.process_data()

if __name__ == "__main__":
    main()

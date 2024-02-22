import os
import glob
import pandas as pd
from Bio import SeqIO
import gffpandas.gffpandas as gp
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import tracemalloc
import cProfile
import pstats

class DataProcessor:
    def __init__(self, fasta_folder_path, gff_folder_path, output_file, batch_size=100, chunk_size=1000):
        self.fasta_folder_path = fasta_folder_path
        self.gff_folder_path = gff_folder_path
        self.output_file = output_file
        self.batch_size = batch_size
        self.chunk_size = chunk_size

    def extract_positions(self, gff_df):
        positions = list(zip(gff_df['start'].astype(int), gff_df['end'].astype(int)))
        return positions

    def load_fasta_file(self, fasta_path):
        # Stream fasta file and yield records to reduce memory footprint
        for seq_record in SeqIO.parse(fasta_path, "fasta"):
            yield {'ID': seq_record.id, 'Sequence': str(seq_record.seq)}

    def load_gff_file_and_extract_positions(self, gff_path):
        annotation = gp.read_gff3(gff_path)
        gff_df = annotation.df
        gff_df = gff_df[gff_df['type'] == 'CDS']
        positions = self.extract_positions(gff_df)
        sequence_id = gff_df['seq_id'].iloc[0]
        return {'ID': sequence_id, 'Positions': positions}

    def convert_to_numeric_classes(self, row):
        sequence_id, sequence, positions = row
        labels = np.zeros(len(sequence), dtype=int)  
        for start, end in positions:
            if start <= 0 or end > len(sequence):
                continue
            labels[start - 1] = 1  # Start codon
            labels[start:end - 1] = 3  # Internal positions
            labels[end - 1] = 2  # Stop codon

        # Efficient chunking
        for i in range(0, len(sequence), self.chunk_size):
            yield {
                'ID': f"{sequence_id}_{i//self.chunk_size + 1}",
                'Sequence': sequence[i:i+self.chunk_size],
                'Labels': labels[i:i+self.chunk_size].tolist()
            }

    def process_data(self):
        tracemalloc.start()  # Start tracking memory
        fasta_paths = glob.glob(os.path.join(self.fasta_folder_path, '*.fna'))
        gff_paths = glob.glob(os.path.join(self.gff_folder_path, '*.gff'))

        # Use generators to minimize memory footprint
        fasta_records = (record for path in fasta_paths for record in self.load_fasta_file(path))
        gff_records = (self.load_gff_file_and_extract_positions(path) for path in gff_paths)

        # Convert generators to DataFrames with minimal memory overhead
        fasta_df = pd.DataFrame(fasta_records)
        gff_df = pd.DataFrame(gff_records)

        merged_df = pd.merge(fasta_df, gff_df, on='ID', how='inner')

        processed_chunks = []
        for _, row in merged_df.iterrows():
            processed_chunks.extend(list(self.convert_to_numeric_classes(row[['ID', 'Sequence', 'Positions']].values)))

        # Convert processed chunks directly to DataFrame and save
        processed_df = pd.DataFrame(processed_chunks)
        print(processed_df)
        processed_df.to_parquet(self.output_file)

        del fasta_df, gff_df, processed_chunks  # Explicitly free up memory
        gc.collect()  # Trigger garbage collection

        current, peak = tracemalloc.get_traced_memory()
        print(f"After saving data and cleaning up - Memory used: {current / 10**6:.2f} MB, Peak: {peak / 10**6:.2f} MB")
        tracemalloc.stop()

def main():
    parser = argparse.ArgumentParser(description="Process fasta and gff files and save the merged data to a Parquet file.")
    parser.add_argument("fasta_folder", help="Path to the folder containing fasta files")
    parser.add_argument("gff_folder", help="Path to the folder containing gff files")
    parser.add_argument("output_file", help="Path to the output Parquet file")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Size of chunks to cut sequences and labels")

    args = parser.parse_args()

    processor = DataProcessor(args.fasta_folder, args.gff_folder, args.output_file, args.batch_size, args.chunk_size)
    
    # Profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    processor.process_data()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(10)  # Print top 10 functions by cumulative time

if __name__ == "__main__":
    main()
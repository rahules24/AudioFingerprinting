import argparse
from shazam import shazam

samplesperseg = 1024

# The number of samples to overlap between adjacent segments
samplesoverlap = samplesperseg // 8

NFFT_n = samplesperseg

# Define the neighborhood size:
window_size = 50

# target zone offset in seconds
targetzone_offset = 0.1

# target zone width in seconds
targetzone_width = 10

# target zone height in frequency
targetzone_height = 1000

#create Shazam object
shazam_obj = shazam(samplesperseg, samplesoverlap, NFFT_n, window_size, targetzone_width, targetzone_height, targetzone_offset)


songs_folders = [
    # r'C:\Users\Kuldeep\OneDrive\Desktop\Studies\Semester 2\MS\library1',
    # r'C:\Users\Kuldeep\OneDrive\Desktop\Studies\Semester 2\MS\library2'
    'library1',
    'library2',
    # 'library3'
]

# Create the argument parser
parser = argparse.ArgumentParser(description="Build a database from song files.")

# Add the arguments
parser.add_argument("-i", "--input", required=True, help="Path to the folder containing song files.")
parser.add_argument("-o", "--output", required=True, help="Path to the output database file.")

# Parse the arguments
args = parser.parse_args()

# Access the parsed arguments
input_folder = args.input
output_file = args.output

# Use the parsed arguments in your script
print(f"Input folder: {[input_folder]}")
print(f"Output file: {output_file}")

#dbfile_path = 'fingerprint_database.db'

shazam_obj.build_database([input_folder], output_file)
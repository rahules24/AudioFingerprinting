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
shazam_obj = shazam(samplesperseg, samplesoverlap, NFFT_n, window_size, targetzone_width, targetzone_height, targetzone_offset, plot = True)

parser = argparse.ArgumentParser(description='Identify a song from a sample using a fingerprint database.')
parser.add_argument('-i', '--input', required=True, help='Input audio sample to identify')
parser.add_argument('-d', '--db', required=True, help='SQLite database file containing fingerprints')
args = parser.parse_args()

# Access the parsed arguments
input_sample = args.input
database_file = args.db

# Use the parsed arguments in your script
print(f"Input audio sample to identify: {input_sample}")
print(f"Database file: {database_file}")

print("Identified Track:", shazam_obj.identify_sample(args.input, args.db))

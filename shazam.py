import sqlite3
import os
from scipy.io import wavfile
from scipy import signal
from scipy.ndimage import maximum_filter
import numpy as np
import hashlib
from models import Song, Fingerprints
import time
import warnings
import matplotlib.pyplot as plt

class shazam:

    def __init__(self, samplesperseg, samplesoverlap, NFFT, window_size, targetzone_width, targetzone_height, targetzone_offset = 0.1, plot = False):
        self.samplesperseg = samplesperseg
        self.samplesoverlap = samplesoverlap
        self.NFFT = NFFT
        self.window_size = window_size
        self.targetzone_width = targetzone_width
        self.targetzone_height = targetzone_height
        self.targetzone_offset = targetzone_offset

        self.plot = plot

    # read .wav file track
    def read_song(self, song_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", wavfile.WavFileWarning) # type: ignore
            sample_rate, audio_signal = wavfile.read(song_path)

            # Print the sample rate
            # print("Sample Rate: ", sample_rate)   

            channels = audio_signal.shape[1]
            # print(f"number of channels = {channels}")
            
            length = audio_signal.shape[0] / sample_rate
            # print(f"length = {length}s")

            # print("length of a channel of audio signal in samples: ", len(audio_signal[:, 0]))

            return sample_rate, audio_signal
    
    # compute the spectrogram
    def get_spectrogram(self, audio_channel, sample_rate):
        
        f, t, Sxx = signal.spectrogram(audio_channel,
                                   window='hann',
                                   nperseg=int(self.samplesperseg),
                                   noverlap=int(self.samplesoverlap),
                                   nfft=int(self.NFFT),
                                   fs=sample_rate)
        return f, t, Sxx

    # compute constellation points
    def get_constellationpoints(self, f, t, Sxx):
        # Apply the maximum filter to identify local maxima in the spectrogram.
        local_maxima = maximum_filter(Sxx, size=self.window_size) == Sxx
        #print(local_maxima)

        # Extract the coordinates of the local maxima
        maxima_coordinates = np.where(local_maxima == True)

        t_constellationvalues = t[np.array(maxima_coordinates[1])]
        f_constellationvalues = f[np.array(maxima_coordinates[0])]

        constellation_points = list(zip(t_constellationvalues, f_constellationvalues))

        return constellation_points

    # compute combinatorial hashes
    def generate_combinatorialhashes(self, constellation_points):
        anchor_points = constellation_points
        # Create an empty list
        hash_list = []

        anchortime_list = []

        # loop over every point in constellation as anchor point
        for anchor_point in anchor_points:
            # print("anchor point : ", anchor_point)
            t_anchor, f_anchor = anchor_point  # Unpack the point into individual variables
            targetzone_left = t_anchor + self.targetzone_offset
            targetzone_right = t_anchor + self.targetzone_offset + self.targetzone_width

            targetzone_lowerlimit = f_anchor - self.targetzone_height / 2.0
            targetzone_upperlimit = f_anchor + self.targetzone_height / 2.0

            # loop over every point in the constellation in the target zone of the current anchor point 
            for constellation_point in constellation_points:
                t_constellation, f_constellation = constellation_point  # Unpack the point into individual variables

                # check if constellation point in the target zone of the current anchor point 
                if(targetzone_left <= t_constellation <= targetzone_right) and (targetzone_lowerlimit <= f_constellation <= targetzone_upperlimit) and (anchor_point != constellation_point):
                    t1 = t_anchor
                    f1 = f_anchor
                    t2 = t_constellation
                    f2 = f_constellation

                    # print("\t constellation point : ", constellation_point)
                    delta_t = np.around(t2 - t1, 2)
                    message = str(np.around(f1, 2)) + str(np.around(f2, 2)) + str(delta_t)
                    md5_hash = hashlib.md5(message.encode()).hexdigest()
                    hash_list.append(md5_hash)
                    anchortime_list.append(t1)
                    #print("\t md5 Hash of ",message,  " is: ", md5_hash, "  ,t1: " ,t1)

        return hash_list, anchortime_list 

    # build the databse
    def build_database(self, songs_folders, db_file):
        conn = sqlite3.connect(db_file)
        print(f"Database connection open.")

        
        c = conn.cursor()
        # Delete all records from the 'song' table
        c.execute('''DELETE FROM song''')
        # Delete all records from the 'Fingerprints' table
        c.execute('''DELETE FROM Fingerprints''')
        conn.commit()
        
        #c.execute('''CREATE TABLE IF NOT EXISTS fingerprints (song_id INTEGER, hash TEXT)''')
        for folder in songs_folders:

            start_time_folder = time.time()
            print(f"\t Processing folder : {folder}")

            for song in os.listdir(folder):
                if song.endswith('.wav'):

                    start_time_song = time.time()

                    print(f"\t\t \n Processing {song} in {folder}")
                    song_path = os.path.join(folder, song)

                    print(f"\t\t Reading song in {song_path}")
                    sample_rate, audio_signal = self.read_song(song_path)

                    print(f"\t\t creating spectrogram")
                    f, t, Sxx = self.get_spectrogram(audio_signal[:, 0], sample_rate)
                    
                    print(f"\t\t creating constellations")
                    constellation_points = self.get_constellationpoints( f, t, Sxx)

                    
                    hash_list, anchortime_list = self.generate_combinatorialhashes(constellation_points)
                    print(f"\t\t generate ",len(hash_list) ," combinatorialhashes")

                    # Add song to database if not already present
                    #print(f"\t\t adding song to database")
                    song_obj = Song.get_or_create(title=song)
                    #print(song_obj.id)
                    # Add fingerprint(hashes + offsets) to database
                    print(f"\t\t adding fingerprints to database")
                    #Song.add_fingerprints(song_obj, fingerprint_data_list = hash_list, anchortime_list = anchortime_list)
                    fingerprint_values = [(h, a, song_obj.id) for h, a in zip(hash_list, anchortime_list)]
                    #print(fingerprint_values)
                    c.executemany('''INSERT INTO fingerprints (fingerprint, anchortime, song_id) VALUES (?, ?, ?)''', fingerprint_values)
                    conn.commit()
                    end_time_song = time.time()
                    print(f"\t\t time taken for song {song_obj.title} is : {(end_time_song - start_time_song) / 60.0 :.6f} minutes.")

            end_time_folder = time.time()
            print(f"\t time taken for folder {folder} is : {(end_time_folder - start_time_folder) / 60.0 :.6f} minutes.") 

        # Display the structure of the fingerprints table
        # c.execute("SELECT sql FROM sqlite_master WHERE name='fingerprints'")
        # print(c.fetchone()[0])

        conn.commit()
        print(f"Data commited.")

        conn.close()
        print(f"Database connection closed.")
    
    # identify a sample
    def identify_sample(self, sample_path, db_path):
        start_time_identify = time.time()

        #print(f"Reading song in {sample_path}")
        sample_rate, sample_audio_signal = self.read_song(sample_path)

        #print(f"\t\t creating spectrogram of audio sample")
        f, t, Sxx = self.get_spectrogram(sample_audio_signal[:, 0], sample_rate)
        
        #print(f"\t\t creating constellations of audio sample")
        sample_constellation_points = self.get_constellationpoints(f, t, Sxx)

        #print(f"generate combinatorialhashes of audio sample")
        sample_hash_list, sample_anchortime_list = self.generate_combinatorialhashes(sample_constellation_points)  # Same params as in builddb.py
        sample_fingerprints = list(zip(sample_hash_list, sample_anchortime_list))

        # data of identified track
        identified_hash_maxcount = 0
        identified_track = ""
        identified_hash_table = {}
        identified_songtime_list = []
        identified_sampletime_list = []

        db = sqlite3.connect(db_path)
        cursor = db.cursor()
        
        # select unique song ids and titles from song table
        cursor.execute('''SELECT id, title FROM song''')
        songs = cursor.fetchall()

        for song in songs:
            cursor.execute('''SELECT fingerprint, anchortime FROM fingerprints WHERE song_id=?''',(song[0],))
            song_fingerprints = cursor.fetchall()

            # Creating a hash table (dictionary) for every song
            song_hash_table = {}

            # matching hash table
            songtime_list = []
            sampletime_list = []

            for sample_fingerprint in sample_fingerprints:
                for song_fingerprint in song_fingerprints:

                    # for equal hashes store time offset as key and number of occurences as value
                    if sample_fingerprint[0] == song_fingerprint[0]:
                        
                        songtime_list.append(float(song_fingerprint[1]))
                        sampletime_list.append(sample_fingerprint[1])

                        key_offset = np.around(float(song_fingerprint[1]) - sample_fingerprint[1], 2)

                        if key_offset not in song_hash_table:
                            song_hash_table[key_offset] = 1
                        else:
                            song_hash_table[key_offset] = song_hash_table[key_offset] + 1


            if len(song_hash_table) != 0:
                if max(song_hash_table.values()) > identified_hash_maxcount:
                    identified_hash_maxcount = max(song_hash_table.values())
                    identified_track = song[1]
                    identified_hash_table = song_hash_table
                    identified_songtime_list = songtime_list
                    identified_sampletime_list = sampletime_list
        db.close()

        if self.plot == True:
            # optionally plot scatterplot of matching hash locations
            self.plot_scatter(identified_songtime_list, identified_sampletime_list)

            # optionally plot histogram of time offsets
            self.plot_histogram(identified_hash_table)
        end_time_identify = time.time()

        print(f"\t time taken to identify sample is : {(end_time_identify - start_time_identify) / 60.0 :.6f} minutes.")
        return identified_track
    
    # plot histogram of time offsets
    def plot_histogram(self, hash_table):
        keys = hash_table.keys()
        values = hash_table.values()

        plt.bar(keys, values)
        plt.xlabel('Keys')
        plt.ylabel('Values')
        plt.title('Hash Table Histogram')
        plt.show()

    # plot scatterplot of matching hash locations
    def plot_scatter(self, songtime_list, sampletime_list):

        # Create the scatter plot
        plt.scatter(songtime_list, sampletime_list)

        # Set labels and title
        plt.xlabel('time in full song')
        plt.ylabel('time in sample')
        plt.title('Scatter Plot')

        # Increase the range of the plot
        plt.xlim(0, 208)  # Set the x-axis limits from 0 to 208
        plt.ylim(0, 12)  # Set the y-axis limits from 0 to 12
        #plt.subplots_adjust(left=0, right=5, top=5, bottom=0)

        # Set equal aspect ratio
        plt.gca().set_aspect('equal')

        # Increase figure size
        #plt.figure(figsize=(20, 20))  # Set the figure size to 8x6 inches

        # Display the plot
        plt.show()

    # classify all samples in the base dir based on database
    def classify(self, base_dir, dbfile_path):
        
        true_tracks = []
        predicted_tracks = []
        # Loop over all files in the base directory and its subfolders
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                true_track = file[:2]
                true_tracks.append(true_track)
                
                
                samplefile_path = os.path.join(root, file)
                print("identifying sample : ", samplefile_path)
                predicted_track = self.identify_sample(samplefile_path, dbfile_path)
                predicted_tracks.append(predicted_track[:2])
                
                print("true_track : ", true_track, " predicted_track : ", predicted_track)

        return true_tracks, predicted_tracks
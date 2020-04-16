#EGN4060 Spr2020 Final Project
#Julia Warner and Victoria Guise

import numpy as np
import scipy
import sklearn.cluster

import time
from threading import Thread

#librosa module used to analyze audio
import librosa

#mixer from pygame module used to play music
import pygame
from pygame import mixer


#DJ-R3X class
class DJR3X:
    #constructor
    def __init__(self):
        self.visualiser = DJ_Visualiser()

    #input: valid filepath to an mp3 file
    #DJ-R3X will play the song, dance, and flash his lights
    def dj(self, song_path):
        #load song
        song, sample_rate = librosa.load(song_path)
        mixer.music.load(song_path)

        #extract array of timestamps that mark the beats
        song_harmonic, song_percussive = librosa.effects.hpss(song)
        tempo, beat_frames = librosa.beat.beat_track(y=song_percussive, sr=sample_rate, trim=False)
        beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
        beat_index = 0
        beat_array_len = len(beat_times)
		
		#initialize BPM for visualizer
        self.visualiser.init_bpm(tempo)

        #extract song segment information
        segment_times, segment_labels = self.segment_song(song, sample_rate, tempo, beat_frames)
        segment_index = 0

        print("It's time to party!")
        mixer.music.play()

        #initial time to base meaurements off of
        t0 = time.time()
        prev_time = time.time()

        #loop until song is over
        while(beat_index < beat_array_len):
            #check if current time is a beat
            time_elapsed = time.time() - t0
            if(time_elapsed >= beat_times[beat_index]):
                #bop head, move on to next beat
                self.bop()
                beat_index += 1

            #check if current time is segment change
            if(time_elapsed >= segment_times[segment_index]):
                #switch segment according to label
                self.new_segment(segment_labels[segment_index])
                segment_index += 1
				
			#update visualiser
            current_time = time.time()
            deltaT = current_time - prev_time
            self.visualiser.update(deltaT)
            prev_time = current_time

    #splits song into distinct musical segments
    #code taken from http://librosa.github.io/librosa_gallery/auto_examples/plot_segmentation.html
    #returns an array of times indicating segment changes and array of segment labels
    def segment_song(self, y, sr, tempo, beats):
        #compute a log-power CQT to extract music note information
        BINS_PER_OCTAVE = 12 * 3
        N_OCTAVES = 7
        C = librosa.amplitude_to_db(librosa.cqt(y=y, sr=sr,
                                    bins_per_octave=BINS_PER_OCTAVE,
                                    n_bins=N_OCTAVES * BINS_PER_OCTAVE),
                                    ref=np.max)

        #beat-synchronus the CQT to reduce dimensionality
        Csync = librosa.util.sync(C, beats, aggregate=np.median)

        #build a recurrance matrix to represent recurring note patterns
        R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity',
                                              sym=True)
        # Enhance diagonals with a median filter (Equation 2)
        df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
        Rf = df(R, size=(1, 7))

        #get median distance between beats
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        Msync = librosa.util.sync(mfcc, beats)

        path_distance = np.sum(np.diff(Msync, axis=1)**2, axis=0)
        sigma = np.median(path_distance)
        path_sim = np.exp(-path_distance / sigma)

        R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

        #compute balanced combinations
        deg_path = np.sum(R_path, axis=1)
        deg_rec = np.sum(Rf, axis=1)

        mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)

        A = mu * Rf + (1 - mu) * R_path

        #compute the normalized laplacian
        L = scipy.sparse.csgraph.laplacian(A, normed=True)
        # and its spectral decomposition
        evals, evecs = scipy.linalg.eigh(L)


        # We can clean this up further with a median filter.
        # This can help smooth over small discontinuities
        evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))


        # cumulative normalization is needed for symmetric normalize laplacian eigenvectors
        Cnorm = np.cumsum(evecs**2, axis=1)**0.5

        # If we want k clusters, use the first k normalized eigenvectors.
        # Fun exercise: see how the segmentation changes as you vary k
        k = 9
        X = evecs[:, :k] / Cnorm[:, k-1:k]

        #use these k components to cluster beats in segmentation
        KM = sklearn.cluster.KMeans(n_clusters=k)
        seg_ids = KM.fit_predict(X)

        #locate segment boundaries
        bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

        # Count beat 0 as a boundary
        bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)

        # Compute the segment label for each boundary
        bound_segs = list(seg_ids[bound_beats])

        # Convert beat indices to frames
        bound_frames = beats[bound_beats]

        # Make sure we cover to the end of the track
        bound_frames = librosa.util.fix_frames(bound_frames,
                                                x_min=None,
                                                x_max=C.shape[1]-1)

        #calculate time boundaries for different segments
        bound_times = librosa.frames_to_time(bound_frames)

        #return boundary times and segment label for each boundary
        return bound_times, bound_segs;

    def new_segment(self, segment_label):
        print('\n\n')
        self.visualiser.set_lighting(segment_label)

    #this function causes DJ-R3X's head to bop
    def bop(self):
        print('Head bop!')
#end DJ-R3X class

#Class for pygame visualisation code
class DJ_Visualiser():
	def __init__(self):
		#define lighting sequences variables
		self.lighting_array = [(1, 255, 0, 0), (1, 0, 130, 255), (1, 255, 130, 0), (0, 130, 255, 0), (0, 0, 0, 255), (0, 130, 0, 255), (2, 255, 0, 0), (2, 0, 255, 0), (2, 0, 255, 130)]
		self.l_changing_i = self.lighting_array[0][0]
		self.l_current_rgb = [self.lighting_array[0][1], self.lighting_array[0][2], self.lighting_array[0][3]]
		self.l_increasing = True
		self.l_index = 0
		self.l_time = 0
		
		#load DJ-R3X images
		self.head_image = pygame.image.load('head.png')
		self.body_image = pygame.image.load('body.png')
		
		#initialize pygame window
		self.screen = pygame.display.set_mode((500, 500))
		pygame.display.set_caption("Welcome to Oga's Cantina!")
		
		#initialize lighting color
		self.screen.fill((0,0,0))
		
		#initialize DJ image
		self.body_pos = (150, 210)
		self.screen.blit(self.body_image, self.body_pos)
		self.head_x = 200
		self.head_y = 100
		self.screen.blit(self.head_image, (self.head_x, self.head_y))
		
		#head bop variables
		self.h_time = 0
		self.h_lowering = True
		self.bpm = 60
		self.spb = 1 #seconds per beat
		
		pygame.display.update()
	
	#BPM used in head bop calculations
	#if init_bpm is not called, default to 90BPM as set in constructor
	def init_bpm(self, BPM):
		self.bpm = BPM
		self.spb = (1 / BPM) * 60
	
	#calls methods that update the lighting and head effects with new time
	def update(self, deltaT):
	    #update lighting 
		self.update_lighting(deltaT)
		
		#update dj
		self.update_head(deltaT)
		
		pygame.display.update()
		
	#expects delta time variable in seconds
	#updates the color value of the screen background and adds that to the screen
	def update_lighting(self, deltaT):
		#calculate amount of change
		self.l_time += deltaT
		if(self.l_time >= 2.0):
			self.l_time = 0
			self.l_increasing = not self.l_increasing
		change = self.l_time * 60 #change 120 color val every 2 seconds, 120/2 = 60
		if(not self.l_increasing):
		   change *= -1
			
		#determine new value for changing rgb value
		if(self.l_increasing):
			base = self.lighting_array[self.l_index][self.l_changing_i+1]
		else:
			base = self.lighting_array[self.l_index][self.l_changing_i+1] + 120 #light shifts are 120 apart
		new = base + change
		if(new > 255):
			new = 255
		elif(new < 0):
			new = 0
		
		#display new screen color
		self.l_current_rgb[self.l_changing_i] = new
		self.screen.fill((self.l_current_rgb[0], self.l_current_rgb[1], self.l_current_rgb[2]))
	
	
	def update_head(self, deltaT):
		#calculate amount of change
		self.h_time += deltaT
		if(self.h_time >= self.spb):
			self.h_time = 0
			self.h_lowering = not self.h_lowering
		change = self.h_time * 30 / self.spb #change 30 pixels every beay
		
		#determine new head y value
		if(self.h_lowering):
			new_y = 100 + change
		else:
			new_y = 130 - change
		
		#set new head y
		self.head_y = new_y
		
		#display the DJ
		self.screen.blit(self.body_image, self.body_pos)
		self.screen.blit(self.head_image, (self.head_x, self.head_y))
	
	#expects an integer value 0-8, used to change over to new lighting sequence
	def set_lighting(self, segment_label):
		self.l_index = segment_label
		self.l_changing_i = self.lighting_array[segment_label][0]
		self.l_current_rgb = [self.lighting_array[segment_label][1], self.lighting_array[segment_label][2], self.lighting_array[segment_label][3]]
	

#main function
def main():
    #initialize pygame
    pygame.init()
    mixer.init()

    rex = DJR3X()

    rex.dj('./Songs/Mad.mp3')

    print('\nThat was fun!\n')

    try:
        while(1):
            #do something
            continue;
    except KeyboardInterrupt:
        pass
#end main function


if __name__ == '__main__':
    main()
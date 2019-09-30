# implementing k-means clustering algorithm for random distributions, in 2D 

# implemented from Lecture Notes on Data Science: K-Means Clustering, by Christian Bauckhage (researchgate)
# The algorithm, as given in pseudo code from figure 2, was coded from scratch

# Author: Ljubisa Sehovac
# github: sehovaclj


# importing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import numpy.random as random
from random import uniform
import math
from math import sqrt




# main function
def main(seed, num_dists, num_samples, add_noise, num_clusters, eps, max_iter, pause_length, keep_inits):

	# preserving random state
	np.random.seed(seed)

	###########################################################################################33

	# create random distributions to use as clusters

	# empty lists to store dists
	x_dists = []
	y_dists = []

	# creating random distributions, with a slight random shift
	for i in range(num_dists):
		x = random.randn(num_samples) + (random.randint(2,6)*random.randn())
		y = random.randn(num_samples) + (random.randint(2,6)*random.randn())
		x_dists.append(x)
		y_dists.append(y)

	# if a larger noise distribution is desired, set add_noise=True
	if add_noise:
		x = random.randn(num_samples) * (random.randint(4,8)*random.randn())
		y = random.randn(num_samples) * (random.randint(4,8)*random.randn())
		x_dists.append(x)
		y_dists.append(y)

		num_dists = num_dists+1


	# plot the initial distributions
	plt.figure(1)
	
	for i in range(num_dists):
		if add_noise:
			if i != num_dists-1:
				plt.scatter(x_dists[i], y_dists[i], label='Dist '+str(i+1))
			elif i == num_dists-1:
				plt.scatter(x_dists[i], y_dists[i], label='Noise Dist')
		elif not add_noise:
			plt.scatter(x_dists[i], y_dists[i], label='Dist '+str(i+1))

	plt.title('Initial distributions/samples in colour')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend()

	# plot distributions without colour. Keep track of this figure for remaining K-means clustering
	plt.figure(2)

	plots=[]

	for i in range(num_dists):
		plots.append({'plot_'+str(i+1) : plt.scatter(x_dists[i], y_dists[i], c='k')})

	plt.title('K-means Clustering')
	plt.xlabel('x')
	plt.ylabel('y')

	plt.pause(pause_length)	
	
	# convert lists to Matrices
	A = []
	for i in range(num_dists):
		A.append(np.array([x_dists[i], y_dists[i]]).transpose())

	A = np.array(A) # shape = num_dists x num_samples x 2 (the 2 here represents the (x,y) coordinate pairing)
	# A now contains all distributions in (x,y) coordinate form

	# find the minimum and maximum values of the X and Y vectors of our distributions. This will be used as our interval for our initial random centroids
	min_x = np.array(x_dists).min(); min_y = np.array(y_dists).min()
	max_x = np.array(x_dists).max(); max_y = np.array(y_dists).max()

	# obtain first random (uniform) centroids
	centroids_prev = []
	# if keep_inits=True, it will run the same simulation. If keep_inits false, it will keep dists the same, but have different initial centroids
	for i in range(num_clusters):
		if keep_inits:
			centroids_prev.append([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])	
		if not keep_inits:
			centroids_prev.append([uniform(min_x, max_x), uniform(min_y, max_y)])

	centroids_prev = np.array(centroids_prev)

	# add centroids to colourless plot
	plot_cens = plt.scatter(centroids_prev[:,0], centroids_prev[:,1], c='r', marker='*', s=200, label='Initial Centroids')
	leg_cens = plt.legend()
	plt.xlim(left=min_x-1.0, right=max_x+1.0)
	plt.ylim(bottom=min_y-1.0, top=max_y+1.0)
	plt.pause(pause_length) # show plot for desired time frame, in seconds

	# remove plots from figure
	for i in range(num_dists):
		plots[i]['plot_'+str(i+1)].remove()

	# colours for clusters. Randomly choosen. This is for plotting
	# hence, the max number of desired clusters is  
	colours = np.zeros([num_clusters, 1, 3]) # use only 3 rgb values
	for i in range(num_clusters):
		colours[i] = [ random.random(), random.random(), random.random() ]

	###################################################################################################################

	# K-means algorithm

	# create empty lists of clusters
	Clusters = []
	for i in range(num_clusters):
		Clusters.append( [] )

	
	# Run algorithm until convergence
	counter = 0
	convergence = False
	while not convergence:

		for N in range(num_dists):
			for n in range(num_samples):
				distances = []
				# for each sample, compute distance to each centroid
				for c in range(num_clusters):
					distances.append( sqrt( (A[N][n][0]-centroids_prev[c][0])**2 + (A[N][n][1]-centroids_prev[c][1])**2 ) )  
				# find smallest distance
				C_idx = np.array(distances).argmin()
				# append sample, or point, to that cluster
				Clusters[C_idx].append( A[N][n] ) 


		# plot clusters
		plots=None
		plots = []
		for i in range(num_clusters):
			plots.append( {'plot_'+str(i+1) : plt.scatter( np.array(Clusters[i])[:,0] , np.array(Clusters[i])[:,1], c=colours[i], label='Cluster '+str(i+1) ) } )
		leg_clusters = plt.legend()
		# show plot, and hold for desired timeframe 	
		plt.pause(pause_length)

		# compute mean of cluster -- and make this our new centroid for that cluster
		centroids_new = centroids_prev.copy() # creating new variable here, need it for convergence tests
		for i in range(num_clusters):
			centroids_new[i] = [np.array(Clusters[i])[:,0].mean(), np.array(Clusters[i])[:,1].mean()]

		# remove old centroids and legend
		plot_cens.remove()
		leg_cens.remove()

		# update iter number
		counter += 1

		# plot new centroids
		plot_cens = plt.scatter(centroids_new[:,0], centroids_new[:,1], c='r', marker='*', s=200, label='Centroids Iter '+str(counter) )
		leg_cens = plt.legend()

		plt.pause(pause_length)


		# Now need to test for convergence. Easiest way, also saves most memory, is to test if the centroids distance from the
		# previous centroid is less than epsilon (usually a small value). Hence, if the centroid has not moved or barely moved, the desired number of clusters has been obtained
		distances_bool = []

		for i in range(num_clusters):
			dist_cen = sqrt( (centroids_new[i][0]-centroids_prev[i][0])**2 + (centroids_new[i][1]-centroids_prev[i][1])**2 )
			if dist_cen <= eps:
				distances_bool.append(True)
			elif dist_cen > eps:
				distances_bool.append(False)	 			
		
		# if all distances are under a certain eps threshold, end algorithm;
		# if not, assign new centroids as previous, and clear clusters from plots 
		if all(distances_bool):
			print('\nConvergence achieved in {} iterations'.format(counter))
			convergence=True
		elif not all(distances_bool):
			centroids_prev = centroids_new
			# clear old clusters from plot (figure 2)
			for i in range(num_clusters):
				plots[i]['plot_'+str(i+1)].remove()		
			# Empty Clusters and create new empty ones
			Clusters=None
			Clusters = []
			for i in range(num_clusters):
				Clusters.append( [] )
			leg_clusters.remove()

		# also end the algorithm if the maximum number of iterations is reached
		if counter == max_iter:
			print('Convergence timed out by Maximum number of iterations')
			convergence=True



	# return desired variables

	return A, Clusters, centroids_new




#############################################################################################################################################

if __name__ == "__main__":
	
	A, clusters, centroids = main(seed=11, num_dists=5, num_samples=200, add_noise=True, num_clusters=3,
				eps=0.01, max_iter=15, pause_length=1.0, keep_inits=False)	







# end of script

# Using Principal Component Analysis (PCA) on an NBA dataset.
# 

# Author: Ljubisa Sehovac
# github: sehovaclj


# importing
import numpy as np
import pandas as pd

import numpy.random as random

import numpy.linalg as la

import math



##############################################################################################################################

# main function 

def main( covariance_params, iterative_params ):

	# assign parameters according to method
	if method=='covariance':
		year = covariance_params['year']
		how_pca = covariance_params['how_pca']
		num_pcs = covariance_params['NUM_PCS']
		percent_energy = covariance_params['percent_energy']

	if method=='iterative':
		seed = iterative_params['seed']
		year = iterative_params['year']
		tol = iterative_params['tol']
		max_iter = iterative_params['max_iter']
		num_pcs = iterative_params['num_pcs']

	# import dataset
	d = pd.read_csv('Seasons_stats_complete.csv')

	# Take data since desired start year 
	d = d.loc[d['Year'] >= year]

	# reset index and drop columns
	del d['Unnamed: 0']
	del d['Pos'] # having played basketball my entire life, the position column is not needed. (Lebron interchanges with PF and SF but plays PG sometimes)

	# remove all rows that have a player on team TOT (Total). A player for one team can be a completely different player for another team.
	# hence, will treat this almost like two different players
	d = d[d.Tm != 'TOT']
	
	del d['Tm'] # get rid of team column entirely. For fantasy, the team is not as important as indiviual stats.
			# sometimes a good player will play on a bad team, and vice versa

	# drop player names column since they are strings. Could create player IDs by enumerating strings,
	# but assigning a numerica value based on alphabetic order for each year might not be the best idea..
	del d['Player']

	d = d.reset_index()
	del d['index']

	#################################################################################################################

	# PCA

	# both methods will have the first step in common, which is:
	# step 1: zero mean the columns, or features. Here, if matrix is n x p, p are our features 
	for i in d.columns:
		d[i] = d[i] - d[i].mean()
 
	X = np.array(d.copy())
	
	# method 1) 'coveriance'
	# 	this is the non-iterative method, and is recommended if the dataset is not large. (not large number of features)

	if method=='covariance':

		# step 2: compute covariance of desired matrix
		cov_X = 1/(len(X)-1) * np.matmul(X.transpose(), X) # if the matrix was p x n dimensions, formula would use np. matmul(X, X.transpose())
	
		# step 3: compute eigenvectors and eigenvalues of covariance matrix obtained above
		eigvals, V = la.eig(cov_X) # here, V is our matrix as eigenvectors as columns

		# step 4: sort the eigvals in descending order (greatest to least). Keep track of order to be able to change up eigenvectors
		evals_sorted = np.sort(eigvals.copy())[::-1]

		eig_order = []
		for i in range(len(evals_sorted)):
			eig_order.append( np.where(evals_sorted==eigvals[i]) )

		eig_order = np.array(eig_order).reshape(len(eigvals))

		# step 4b: sort eigenvectors corresponding to largest to least eigenvalues
		evecs_sorted = []
		for i in range(len(eig_order)):
			evecs_sorted.append( V[:,eig_order[i]] )

		evecs_sorted = np.array(evecs_sorted).transpose() # transpose to get original shape of V	

		# to make sure the proper eigenvectors have been switched, if year=2000.0, evecs_sorted[:,10]==V[:,11] returns true
		if year==2000.0:
			print("\nEigenvectors have been switched: {}".format(all(evecs_sorted[:,10]==V[:,11])))
		# aka the 10th and 11 eigenvalues have been switched, so we need to switch the corresponding eigenvectors as well
	
		# step 5: taking the principal components (PCs)
		# here, it is left to the individual to decide how they want to choose their PCs.
		# option 1) By the number of PCs ('num_pcs') 
		# option 2) By the percentage of "energy" ('percent_energy') of PCs

		if how_pca=='num_pcs':
			# take the first number of desired PCs (first number of desired sorted eigenvectors)
			S = evecs_sorted[:,:num_pcs]

		# compute eigenvalue "energy" / also can be thought of as percentage of sum
		elif how_pca=='percent_energy':
			# find the eigenvalue that uses desired cumulative energy
			for i in range(len(eigvals)):		
				r = sum(eigvals[:i+1]) / sum(eigvals)
				if r >= percent_energy:
					eigval_stop = i+1
					break

			# obtain the respective number of eigenvectors 
			S = evecs_sorted[:,:eigval_stop]
		
		# raise error if something other than 'num_pcs' or 'percent_energy' is given	
		elif how_pca not in ['num_pcs', 'percent_energy']:
			raise ValueError(how_pca, " is not an available option. Please select between num_pca and percent_energy.")	

		# step 6: once we have obtained our principal eigenvectors (S), we can finally compute our new dataset Y
		# since our X has dimensions n x p, and S has dimensions p x num_pcs (or p x eigval_stop), we need to tranpose both
		Y = S.transpose() @ X.transpose() # hence num_pcs x p @ p x n, output matrix is num_pcs x n

		# this is your final new dataset
		Y = Y.transpose() # transpose Y to have features along the columns, hence now n x num_pcs

		return d, X, cov_X, eigvals, V, eig_order, evecs_sorted, S, Y







	#################################################################################################################################################

	# method 2) 'iterative'
	# this shows how to do PCA using the iterative approach. This is the preferred method when you have a large number of features
	# refer to Covariance-free computation section on the PCA wikipedia page
	elif method=='iterative':
		# first, preserve random state
		np.random.seed(seed)

		# function to find the principal eigenvector of the matrix passed to it
		def iter_pca( matrix, tolerance, max_num_iter ):

			# assign r as a random vector of dimensions p x 1
			r = random.randn(matrix.shape[-1]).reshape(matrix.shape[-1], 1)
			# normalize r
			r = r / la.norm(r)

			counter = 0

			# do while convergence is not True
			convergence=False
			while not convergence:

				# assign zero vector of dimensions p x 1	
				s = np.zeros([matrix.shape[-1], 1])

				# for each row in matrix passed to function
				for i in range(matrix.shape[0]):
					# reshape row to have dimensions 1 x p
					xi = matrix[i].reshape(1, matrix.shape[-1])
						
					s = s + xi.T @ (xi @ r) # main iter part

				# compute the eigenvalues
				eigval = r.T @ s

				# compute the error
				err = la.norm( r - s )

				# assign r as the normalized vector s
				r = s / la.norm(s)

				counter += 1

				# exit while if error is less than some desirable tolerance or the max number of iterations have been reached
				if err < tolerance:
					print("\nError is less than tolerance, convergence achieved after {} iterations".format( counter ))
					convergence=True
				
				if counter == max_num_iter:
					print("\nMaximum number of iterations ({}) reached".format( counter ))
					convergence=True

			# return variables
			return s, r, eigval, err



		########################################################################################################################

		# computing the desired number of principal components, while storing outputs after each iter_pca function run
		S = []
		R = []
		Evals = []
		Errs = []
	
		X_og = X.copy()

		for i in range(num_pcs):
			print("\nFinding principal component number: {}".format(i+1))

			# run iterative function		
			s, r, eigval, err = iter_pca( X, tol, max_iter ) # we start with X, our original data matrix 
			# store values
			S.append(s.reshape(-1)); R.append(r.reshape(-1)); Evals.append(eigval); Errs.append(err)

			# deflate our matrix X using our computed eigenvector (r)
			X = X - X @ ( r @ r.T ) # use this new matrix X to compute next PC and corresponding eigenvector


		# convert lists back to arrays
		S = np.array(S); R = np.array(R); Evals = np.array(Evals); Errs = np.array(Errs)

		# transform original data to new dataset
		Y = R @ X_og.T

		Y = Y.T

		# return variables
		return S, R, Evals, Errs, Y




################################################################################################################################

if __name__ == "__main__":

	######## choose method here. Choice will use different parameters, as well as have different outputs #############
	method='iterative'

	if method=='covariance':

		# assign dict parameters
		cov_params = {'year': 2000.0,
				'how_pca': 'num_pcs',
				'NUM_PCS': 4,
				'percent_energy': 0.99}

		iter_params = 0 

		# calling main function
		d, X, cov_X, evals, V, eig_order, evecs_sorted, S, Y = main(
			covariance_params=cov_params, iterative_params=iter_params)
	
	if method=='iterative':

		# assign dict parameters
		cov_params = 0

		iter_params = {'seed': 1,
				'year': 2000.0,
				'tol': 0.1,
				'max_iter': 100,
				'num_pcs': 4}

		# note that in outputs above, S is the eigenvector matrix, where here, R are our principal eigenvectors
		# calling main function
		S_iter, R, Evals, Errs, Y2 = main( covariance_params=cov_params,
					iterative_params=iter_params )





######################################################################################################

# FEW THINGS TO NOTE:

	# WHEN USING THE ITERATIVE METHOD:

	# 	"imprecisions in already computed approximate principal components additively affect the 
	# 	accuracy of the subsequently computed principal components, thus increasing the 
	# 	error with every new computation"

	# Hence, if you run the code using method='covariance', then change method to
	# method='iterative' and run exec(open('pca_nba.py').read()) in the python terminal -- this
	# will keep your output variables from the covariance method and you can compare with the
	# output variables from the iterative method. Note that the first 3 eigenvectors are
	# almost identical (except for rounding errors), but the iterative method starts to lose
	# accuracy for the 4th and subsequent eigenvectors


# ALSO NOTE:

	# The dataset used, regarding nba player stats over the years, is not the best dataset for PCA.
	# You would ideally like to use PCA on a dataset with a much larger number of features, and
	# this specific dataset was used to show PCA on an arbitrary dataset (both covariance and
	# the iterative approach).

	# lastly, note that PCA does NOT tell you the "most" important features of a dataset, rather
	# PCA is used as a powerful feature reduction mechanism that helps us handle high-dimensional
	# data with too many features.
	# In particular, PCA is a method ********THAT CAPTURES THE ESSENSE OF THE ORIGINAL DATA*******





# end of script




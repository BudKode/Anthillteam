#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy( self,time_allowance=60.0 ):
		pass
	
	
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
	def branchAndBound( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		count = 0
		bssf = None
		start_time = time.time()
		max_queue_size = 1
		total_states = 1
		pruned_states = 0

		#Initialze the priority queue
		queueball = HeapQueue()
		#Find a starting BSSF
		randresults = self.defaultRandomTour(time_allowance=30.0)
		bssf = randresults['soln']
		#Pick a starting city
		starting_city = cities[0]
		starting_route = [starting_city]
		#Initialize it and put it in the priority queue
		starting_matrix = []
		for city in cities:
			city_costs = []
			for to_city in cities:
				city_costs.append(city.costTo(to_city))
			starting_matrix.append(city_costs)
		starting_matrix = np.array(starting_matrix)
		starting_node = Node(starting_city, 0, starting_route, starting_matrix, 0)
		queueball.insert(starting_node)

		#Start the search!
		while not queueball.checkEmpty() and time.time()-start_time < time_allowance:  #<---------------O(n+(Summation from n-1 to 1 of (ni-1*n-1)))
			checkNode = queueball.delete_min()   #<------------O(logn)
			#Error check
			if checkNode.lowerbound == np.inf:
				print("Error: infinite lowerbound")
				break
			if len(checkNode.route) > ncities:
				print("Error: gone too far!")
				break
			#check for obsoletion (prunability post queue-insertion)
			if checkNode.lowerbound >= bssf.cost:
				pruned_states += 1
				continue
			#check for solution
			if len(checkNode.route) == ncities:
				checkCost = checkNode.lowerbound + checkNode.city.costTo(starting_city)
				if checkCost < bssf.cost:
					bssf = TSPSolution(checkNode.route)
					count += 1
			#Add acceptable child states to queue
			to_cities = checkNode.matrix[checkNode.city_index]
			for index in range(ncities):
				if to_cities[index] != np.inf:
					#populate node state
					child_city = cities[index]
					child_route = checkNode.route + [child_city]
					child_matrix = np.copy(checkNode.matrix)
					#set parent row and child col to np.inf
					for i in range(ncities):								#<------------O(n)
						child_matrix[checkNode.city_index][i] = np.inf
						child_matrix[i][index] = np.inf
					child_matrix[index][checkNode.city_index] = np.inf
					child_matrix[index][0] = np.inf
					child_node = Node(child_city, index, child_route, child_matrix, checkNode.lowerbound+to_cities[index])	#<------------O(4n^2)
					total_states += 1
					#check for prunability
					if child_node.lowerbound >= bssf.cost:
						pruned_states += 1
					else:
						queueball.insert(child_node)						#<------------O(logn)
			if len(queueball.queue) > max_queue_size:
				max_queue_size = len(queueball.queue)


		end_time = time.time()

		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = max_queue_size
		results['total'] = total_states
		results['pruned'] = pruned_states + len(queueball.queue)
		return results


	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy( self,time_allowance=60.0 ):
		pass
		




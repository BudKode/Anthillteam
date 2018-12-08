#!/usr/bin/python3


import math
import numpy as np
import random
import time


#---------------------------- AntHill Classes --------------------------------------

class Colony:
	def __init__(  self, cities, pherimone_map, start_city ):
		self.num_ants = 1
		self.cities = cities
		self.ants = []
		self.max_pherimone_per_ant = 3
		self.pherimone_map = pherimone_map
		self.start_city = start_city
		print("Start City = " + str(start_city))
		#Play with this. This is what we give each ant to compare with to see if their route is worth marking
		self.decent_route_average = len(self.pherimone_map) * 100
		#Play with this. This is what we give each ant as a flag for the optimal route to quit iterations and return a final solution
		self.optimal_route_pherimone_count = (self.num_ants / 2) * self.max_pherimone_per_ant 
		self.final_tour = []
		self.bssf = None

		cities = list(cities)
		cities.remove(self.start_city)
		self.cities_to_visit = cities

	def release_ants( self ):
		ants = []
		for i in range(self.num_ants):
			ants.append(Ant(self.start_city, self.cities, self.decent_route_average, self.optimal_route_pherimone_count, self.pherimone_map))
		iterrations = 0
		while ants:
			# if iterrations > 4 * len(self.pherimone_map):
			# 	print("Error! Too many ant iterrations!")
			# 	self.ants = []
			# 	break
			# iterrations += 1
			for ant in ants:
				if (ant.action != ant.finish):
					ant.action()
				else:
					ants.remove(ant)
					self.ants.append(ant)
		return

	def findBSSF(self):
		# print("entering findBSSF")
		for ant in self.ants:
			if (self.bssf == None or self.bssf.cost > ant.solution.cost):
				print("Updating BSSF to " + str(ant.solution.cost))
				self.bssf = ant.solution


class Ant:
	def __init__( self, city, cities_to_visit, route_average, optimal_pherimone_count, pherimone_map ):
		self.route = []
		self.pherimone_map = pherimone_map
		self.cities_to_visit = list(cities_to_visit)
		self.current_city = city
		self.route_average = route_average
		self.report_optimal_count = optimal_pherimone_count
		self.action = self.pick_route

	def pick_route( self ):
		total_pherimone_chance = 0
		for city in self.cities_to_visit:
			total_pherimone_chance += self.pherimone_map[self.current_city._index][city._index].path_weight

		random_num = random.uniform(0, total_pherimone_chance)

		chosen_route_index = 0
		for city in self.cities_to_visit:
			random_num -= self.pherimone_map[self.current_city._index][city._index].path_weight

			if (random_num <= 0):
				chosen_route_index = city._index
				break

		self.route.append(self.current_city)

		if not self.cities_to_visit:
			# The ant has found a full circuit
			self.action = self.check_solution
		else:
			# There is still a city to find
			for city in self.cities_to_visit:
				if city._index == chosen_route_index:
					print("Next City is " + str(city))
					self.current_city = city
					self.cities_to_visit.remove(city)
					break

	def check_solution( self ):
		self.solution = TSPSolution(self.route)

		if self.solution.cost == np.inf:
			# Didn't find a valid route
			self.action = self.finish
		else:
			# print("Found valid solution with cost " + str(self.solution.cost))
			self.action = self.calculate_pherimone
		

	def calculate_pherimone( self ):
		# We can definately play with this
		self.pherimone = self.route_average / self.solution.cost

		self.backtrack_route = list(self.route)
		self.action = self.backtrack

	def backtrack( self ):
		for city in self.backtrack_route:
			self.drop_pherimone(city)

		
		self.action = self.finish

	def drop_pherimone( self, to_city ):
		self.pherimone_map[self.current_city._index][to_city._index].path_weight += self.pherimone
		self.backtrack_route.remove(self.current_city)
		self.current_city = to_city

	def finish( self ):
		# signals the ant is done
		pass

class Edge:
	def __init__( self, path_length, path_weight ):
		self.path_length = path_length
		self.path_weight = path_weight


#---------------------------- End AntHill Classes --------------------------------------

#---------------------------- Branch and Bound Classes --------------------------------------
class Node:
	def __init__( self, city, city_index, route, parent_matrix, parent_bound):
		self.city = city
		self.city_index = city_index
		self.route = route
		self.matrix = np.array(parent_matrix)
		self.lowerbound = parent_bound
		self.calculate_matrix()			#<------------O(4n^2)
	
	def calculate_matrix( self ):
		#Big-O: O(4n^2). Has to run a constant time check/change for each row and column in each row and column
		#Get dimensions of array
		rows = len(self.matrix)
		cols = len(self.matrix[0])
		#Reduce array
		for row in range(rows):															#<------------O(n) - totals O(2n^2)
			minvalue = min(self.matrix[row])											#<------------O(n)
			if minvalue < np.inf:
				#Add minvalue to self.lowerbound
				self.lowerbound += minvalue
				#Subtract minvalue from all (non-infinite) values in row
				for col in range(cols):													#<------------O(n)
					if self.matrix[row][col] < np.inf:
						self.matrix[row][col] = self.matrix[row][col] - minvalue
		for col in range(cols):															#<------------O(n) - totals O(2n^2)
			minvalue = min([row[col] for row in self.matrix])							#<------------O(n)
			if minvalue < np.inf:
				#Add minvalue to self.lowerbound
				self.lowerbound += minvalue
				#Subtract minvalue from all (non-infinite) values in col
				for row in range(rows):													#<------------O(n)
					if self.matrix[row][col] < np.inf:
						self.matrix[row][col] = self.matrix[row][col] - minvalue
		return

	def get_priority( self ):
		#Big O: O(1). Constant time operation
		if self.lowerbound - len(self.route) > 0:
			return self.lowerbound - (5000 * len(self.route))
		return 0


class HeapQueue:
    def __init__( self ):
        #Initialize the queue
        #Constant time O(1)
        self.queue = []

    def sort_heap( self, node_index ):
        #Takes the given index and checks that it is still organized correctly
        #If out of order, it will sort the whole heap back to correct
        #Logarithmic time O(logn)
        check_val = self.queue[node_index].get_priority()
        #BubbleUp
        if node_index != 0:
            parent_index = math.ceil(node_index/2) -1
            if self.queue[parent_index].get_priority() > check_val:
                up_node = self.queue[node_index]
                down_node = self.queue[parent_index]
                self.queue[parent_index] = up_node
                self.queue[node_index] = down_node
                self.sort_heap(parent_index)
        #FilterDown
        child_index_left = ((node_index + 1) * 2) - 1
        child_index_right = child_index_left + 1
        if child_index_left < len(self.queue):                                                  #If left child exists
            if self.queue[child_index_left].get_priority() < check_val:                             #If left child less than parent
                if child_index_right < len(self.queue):                                                 #If right child exists
                    if self.queue[child_index_right].get_priority() < self.queue[child_index_left].get_priority():      #If right child less than left
                        up_node = self.queue[child_index_right]                                                 #swap right child up
                        down_node = self.queue[node_index]
                        self.queue[node_index] = up_node
                        self.queue[child_index_right] = down_node
                        self.sort_heap(child_index_right)
                    else:                                                                                   #Else left is less or equal
                        up_node = self.queue[child_index_left]                                                  #swap up left child
                        down_node = self.queue[node_index]
                        self.queue[node_index] = up_node
                        self.queue[child_index_left] = down_node
                        self.sort_heap(child_index_left)
                else:                                                                                   #Else left is only child
                    up_node = self.queue[child_index_left]                                                  #swap up left child
                    down_node = self.queue[node_index]
                    self.queue[node_index] = up_node
                    self.queue[child_index_left] = down_node
                    self.sort_heap(child_index_left)
            elif child_index_right < len(self.queue):                                               #Elif right child exists
                if self.queue[child_index_right].get_priority() < check_val:                            #If right child less than parent
                    up_node = self.queue[child_index_right]                                                 #swap right child up
                    down_node = self.queue[node_index]
                    self.queue[node_index] = up_node
                    self.queue[child_index_right] = down_node
                    self.sort_heap(child_index_right)

    def insert( self, element ):
        #Adds a new element to the heap complete with shortest path and parent node
        #Logarithmic time O(logn)
        new_index = len(self.queue)
        self.queue.append(element)
        self.sort_heap(new_index)       #<------------O(logn)

    def delete_min( self ):
        #Return the element with the smallest key and "remove" it from the set
        #Logarithmic time O(log n)
        smallest = self.queue.pop(0)
        if len(self.queue) > 2:
            largest = self.queue.pop()
            self.queue.insert(0, largest)
            self.sort_heap(0)           #<--------------O(logn)
        return smallest               

    def checkEmpty( self ):
        #check if empty, meaning end
        #Constant time O(1)
        if not self.queue:
            return True
        return False

#---------------------------- End Branch and Bound Classes --------------------------------------

class TSPSolution:
	def __init__( self, listOfCities):
		self.route = listOfCities
		self.cost = self._costOfRoute()
		#print( [c._index for c in listOfCities] )

	def _costOfRoute( self ):
		cost = 0
		#print('cost = ',cost)
		last = self.route[0]
		for city in self.route[1:]:
			#print('cost increasing by {} for leg {} to {}'.format(last.costTo(city),last._name,city._name))
			cost += last.costTo(city)
			last = city
		#print('cost increasing by {} for leg {} to {}'.format(self.route[-1].costTo(self.route[0]),self.route[-1]._name,self.route[0]._name))
		cost += self.route[-1].costTo( self.route[0] )
		#print('cost = ',cost)
		return cost

	def enumerateEdges( self ):
		elist = []
		c1 = self.route[0]
		for c2 in self.route[1:]:
			dist = c1.costTo( c2 )
			if dist == np.inf:
				return None
			elist.append( (c1, c2, int(math.ceil(dist))) )
			c1 = c2
		dist = self.route[-1].costTo( self.route[0] )
		if dist == np.inf:
			return None
		elist.append( (self.route[-1], self.route[0], int(math.ceil(dist))) )
		return elist


def nameForInt( num ):
	if num == 0:
		return ''
	elif num <= 26:
		return chr( ord('A')+num-1 )
	else:
		return nameForInt((num-1) // 26 ) + nameForInt((num-1)%26+1)






class Scenario:

	HARD_MODE_FRACTION_TO_REMOVE = 0.20 # Remove 20% of the edges

	def __init__( self, city_locations, difficulty, rand_seed ):
		self._difficulty = difficulty

		if difficulty == "Normal" or difficulty == "Hard":
			self._cities = [City( pt.x(), pt.y(), \
								  random.uniform(0.0,1.0) \
								) for pt in city_locations]
		elif difficulty == "Hard (Deterministic)":
			random.seed( rand_seed )
			self._cities = [City( pt.x(), pt.y(), \
								  random.uniform(0.0,1.0) \
								) for pt in city_locations]
		else:
			self._cities = [City( pt.x(), pt.y() ) for pt in city_locations]


		num = 0
		for city in self._cities:
			#if difficulty == "Hard":
			city.setScenario(self)
			city.setIndexAndName( num, nameForInt( num+1 ) )
			num += 1

		# Assume all edges exists except self-edges
		ncities = len(self._cities)
		self._edge_exists = ( np.ones((ncities,ncities)) - np.diag( np.ones((ncities)) ) ) > 0

		#print( self._edge_exists )
		if difficulty == "Hard":
			self.thinEdges()
		elif difficulty == "Hard (Deterministic)":
			self.thinEdges(deterministic=True)

	def getCities( self ):
		return self._cities


	def randperm( self, n ):				#isn't there a numpy function that does this and even gets called in Solver?
		perm = np.arange(n)
		for i in range(n):
			randind = random.randint(i,n-1)
			save = perm[i]
			perm[i] = perm[randind]
			perm[randind] = save
		return perm

	def thinEdges( self, deterministic=False ):
		ncities = len(self._cities)
		edge_count = ncities*(ncities-1) # can't have self-edge
		num_to_remove = np.floor(self.HARD_MODE_FRACTION_TO_REMOVE*edge_count)

		#edge_exists = ( np.ones((ncities,ncities)) - np.diag( np.ones((ncities)) ) ) > 0
		can_delete	= self._edge_exists.copy()

		# Set aside a route to ensure at least one tour exists
		route_keep = np.random.permutation( ncities )
		if deterministic:
			route_keep = self.randperm( ncities )
		for i in range(ncities):
			can_delete[route_keep[i],route_keep[(i+1)%ncities]] = False

		# Now remove edges until 
		while num_to_remove > 0:
			if deterministic:
				src = random.randint(0,ncities-1)
				dst = random.randint(0,ncities-1)
			else:
				src = np.random.randint(ncities)
				dst = np.random.randint(ncities)
			if self._edge_exists[src,dst] and can_delete[src,dst]:
				self._edge_exists[src,dst] = False
				num_to_remove -= 1

		#print( self._edge_exists )




class City:
	def __init__( self, x, y, elevation=0.0 ):
		self._x = x
		self._y = y
		self._elevation = elevation
		self._scenario	= None
		self._index = -1
		self._name	= None

	def setIndexAndName( self, index, name ):
		self._index = index
		self._name = name

	def setScenario( self, scenario ):
		self._scenario = scenario

	''' <summary>
		How much does it cost to get from this city to the destination?
		Note that this is an asymmetric cost function.
		 
		In advanced mode, it returns infinity when there is no connection.
		</summary> '''
	MAP_SCALE = 1000.0
	def costTo( self, other_city ):

		assert( type(other_city) == City )

		# In hard mode, remove edges; this slows down the calculation...
		# Use this in all difficulties, it ensures INF for self-edge
		if not self._scenario._edge_exists[self._index, other_city._index]:
			#print( 'Edge ({},{}) doesn\'t exist'.format(self._index,other_city._index) )
			return np.inf

		# Euclidean Distance
		cost = math.sqrt( (other_city._x - self._x)**2 +
						  (other_city._y - self._y)**2 )

		# For Medium and Hard modes, add in an asymmetric cost (in easy mode it is zero).
		if not self._scenario._difficulty == 'Easy':
			cost += (other_city._elevation - self._elevation)
			if cost < 0.0:
				cost = 0.0
		#cost *= SCALE_FACTOR


		return int(math.ceil(cost * self.MAP_SCALE))


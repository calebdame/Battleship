import numpy as np
import random
from itertools import permutations
from collections import Counter
import string
from time import time
from matplotlib import pyplot as plt
from tqdm import tqdm

class Battleship:
    """
    Main Battleship class where rules of the game are defined (dimentions, ships, etc.) 
    and the initial board is built.
    
    dim   : int  , size of the square board (dim , dim)
    ships : list , list of ship lengths that fit on the board
    """
    
    def __init__(self, dim=10, ships=[2,3,3,4,5]):
        self.dim = dim
        self.numShips = len(ships)
        self.ships = tuple(ships)
        self.names = tuple(string.ascii_lowercase[:len(ships)]) # letter names to track in the case of duplicate lengths
        self.sunkDict = dict(zip(self.names,[0 for i in ships])) # keep track of which are fixed in place
        self.shipLengths = dict(zip(self.names,ships)) # put lengths to names
        self.generateComponentLayouts() 
        self.generateRandomOrders()
    
    def generateRandomOrders(self):
        """
        Generate dictionary of random orders so they don't have to be made on the fly (it saves slightly more time)
        """
        perms = list(permutations(self.names))
        self.numPerms = len(perms)
        self.randomOrderDict = dict(zip(range(self.numPerms), [list(i) for i in perms]))
        
    def generateComponentLayouts(self):
        """
        Load all legal moves for each ship to define possible moves
        """

        self.possibleShipsDict = dict()
        self.possibleShipsNumDict = dict()
        
        for name in self.names:
            orientations1 = [{(i+temp,j) for temp in range(self.shipLengths[name])} for i in range(self.dim - self.shipLengths[name] + 1) for j in range(self.dim)]
            orientations2 = [{(j,i+temp) for temp in range(self.shipLengths[name])} for i in range(self.dim - self.shipLengths[name] + 1) for j in range(self.dim)]
            self.possibleShipsDict[name] = orientations1 + orientations2
            self.possibleShipsNumDict[name] = len(orientations1) + len(orientations2)
        
    def randomBoard(self):
        """
        Build completely random board picking and placing ships in a random order 
        """
        order = self.randomOrderDict[np.random.randint(0, self.numPerms)]
        final_set, self.boats = set(), []
        
        for name in order:
            while 1:
                ship_coords = self.possibleShipsDict[name][np.random.randint(self.possibleShipsNumDict[name])]
                if ship_coords.isdisjoint(final_set):
                    final_set.update(ship_coords)
                    self.boats.append((name, ship_coords))
                    break
        self.boats = tuple(self.boats)
        return final_set
                
class BattleshipPlayer(Battleship):
    """
    Class that allows the user to play battleship with the calculated probabilities from the sampler.
    Only plays by hand.
    
    randomOrder : (bool) , whether boards built in the sampler will have the boats placed in a 
        random order or from largest to smallest (the former being less bias, but the latter being faster)
    batchSize   : (int)  , the number of samples to take at each iteration
    printTime   : (bool) , print the time it takes for each iteration
    
    dim, ships inherited from Battleship class
    """
    
    def __init__(self, dim=10, ships=[2,3,3,4,5], randomOrder=True, batchSize=1000, printTime=False):
        super().__init__(dim, ships)
        self.board = self.randomBoard()
        self.hits, self.hitsSunk, self.misses = set(), set(), set()
        self.possibleShipsDictCond = self.possibleShipsDict.copy()
        self.possibleShipsNumDictCond = self.possibleShipsNumDict.copy()
        self.lastTurnBoards = list()
        self.nextInx = (-1,-1)
        self.randomOrder = randomOrder
        self.batchSize = batchSize
        self.printTime = printTime
        self.numMaxHits = np.sum(self.ships)
        self.order = [x for _, x in sorted(zip(ships, self.names))] 
        # the default order of placing ships when making random boards (largest -> smallest, most efficient, but most biased)
        
        
    def updateOrientations(self):
        """
        updates each ship's legal moves to the best of our ability
          - removes locations that have misses
          - when a boat is sunk, its location is fixed and other boats cannot overlap with the sunken hit
          - Finds places where only a single boat can be, and limits a boat's choices to that subset

        """  
        
        for i in self.boats:
            boat, location = i
            if location.issubset(self.hits):
                self.hitsSunk.update(location)
                self.sunkDict[boat] = 1
                self.possibleShipsDictCond[boat] = list(location)
                self.possibleShipsNumDictCond[boat] = 1
        sunkNum = len(self.hitsSunk)
        
        for name in self.names:
            if 1 - self.sunkDict[name]:
                new_orient = [config for config in self.possibleShipsDictCond[name] if self.misses.isdisjoint(config) and config.isdisjoint(self.hitsSunk)]
                self.possibleShipsDictCond[name] = new_orient
                self.possibleShipsNumDictCond[name] = len(new_orient)
                
        for hit in self.hits:
            temp = []
            for i in self.possibleShipsDictCond.items():
                name, locations = i
                if hit in set().union(*locations):
                    temp.append(name)
            if len(temp) == 1:
                self.possibleShipsDictCond[temp[0]] = [k for k in self.possibleShipsDictCond[temp[0]] if hit in k]
                self.possibleShipsNumDictCond[temp[0]] = len(self.possibleShipsDictCond[temp[0]])
                
                
        self.order = [k for k, _ in sorted(self.possibleShipsNumDictCond.items(), key=lambda item: item[1])]
                
    def randomSelection(self, order, mustHappen):
        """
        Select a random board, conditioned on the the current misses, hits, and sinks
          - Initially, naively finds a legal board and checks if all the hits are accounted for
          - If that fails, we iterate through each of the ships 3 times and check if there are 
            changes that could have been made to increase our hit coverage.
          - If that too fails, we restart and reselect 
          - if self.randomOrder = True, we shuffle the order each time, otherwise we default to
            selecting ships in ascending order of available legal moves in order to maximize the
            probability of there being possible boards. 
        """

        what_where = dict()
        while 1:
            final_set = mustHappen
            for i in order:
                alternatives = [k for k in self.possibleShipsDictCond[i] if k.isdisjoint(final_set)]
                if alternatives == []:
                    return set()
                ship_coords = np.random.choice(alternatives)
                what_where[i] = ship_coords
                final_set = final_set.union(ship_coords)
            if self.hits.issubset(final_set):
                return list(final_set)

            count = 0
            what_where["other"] = mustHappen
            while count < 3:
                count += 1
                for entry in what_where.items():
                    needed = self.hits.difference(final_set)
                    num_intersected_already = len(entry[1].intersection(self.hits))
                    if  entry[0] != "other" and num_intersected_already < self.shipLengths[entry[0]]:                    
                        temp = set().union(*[k[1] for k in what_where.items() if k[0] != entry[0]])
                        alternatives = [k for k in self.possibleShipsDictCond[entry[0]] if k.isdisjoint(temp) and len(needed.intersection(k)) > num_intersected_already]
                        if alternatives == []:
                            continue
                        what_where[entry[0]] = np.random.choice(alternatives)
                        final_set = set().union(*[k[1] for k in what_where.items()])
                        if self.hits.issubset(final_set):
                            return list(final_set)

            random.shuffle(order)
            
            
    def randomConditionalBoard(self, orderIncoming):
        """
        Initialize the order:
          - remove non-choices
          - shuffle boards between tries if necessary
        """

        order = orderIncoming.copy()
        mustHappen = set()
        for i in self.sunkDict.items():
            if i[1]:
                order.remove(i[0])
                mustHappen.update(dict(self.boats)[i[0]]) 
        
        final_set = set()
        while len(final_set) == 0:
            final_set = self.randomSelection(order, mustHappen)
            if self.randomOrder:
                random.shuffle(order)

        return final_set

            
    
    def buildAggBoard(self):
        """
        Main function to build boards and gereate probabilites
          - recall all the boards generrated last turn which are still valid (major time saver)
          - get the rest of the necessary boards from randomConditionalBoard
          - order the collection of possible hits to get the maximal value

        """


        t = time()
        order = [k for k, v in sorted(self.possibleShipsNumDictCond.items(), key=lambda item: item[1])] if not self.randomOrder else self.randomOrderDict[np.random.randint(0, self.numPerms)]
        self.updateOrientations()
        lboards, boards = [], []
        
        if self.nextInx in self.hits:
            lboards = [i for i in self.lastTurnBoards if self.nextInx in i]
            boards = [j for i in lboards for j in list(i) if self.nextInx in i]
            
        if self.nextInx in self.misses:
            lboards = [i for i in self.lastTurnBoards if self.nextInx not in i]
            boards = [j for i in lboards for j in list(i) if self.nextInx not in i]
        numIter = len(lboards)    
        while numIter < self.batchSize:
            if self.randomOrder:
                random.shuffle(order)
            numIter += 1
            temp = self.randomConditionalBoard(order)
            boards += temp
            lboards.append(set(temp))
        self.lastTurnBoards = lboards
        
        self.aggDict = dict(Counter(boards).most_common())
        
        self.numIter = numIter
        if self.printTime:
            print(time()-t)

    def view(self, graph=False,notext=False):
        """
        prints the board hits and plots the board and generates the matrix of 
        probabilities when playing 'by hand'
        """

        self.buildAggBoard()
        matrix = np.zeros((self.dim,self.dim))
        for location in self.aggDict.keys():
            matrix[location[0], location[1]] = self.aggDict[location]
        matrix = matrix / self.numIter
        if graph:
            self._print(matrix,notext)
        print(self.numIter, "Iterations")
        self.maxInx = ((np.argmax(matrix) % self.dim), int(np.floor((np.argmax(matrix) / self.dim))))
        
    def _print(self, matrix, notext):
        """
        Does the `AcTuAl` plotting of the matrix
        """
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_axes([0,0,1,1])
        ax.matshow(matrix, cmap='copper_r')
        if not notext:
            for (i, j), z in np.ndenumerate(matrix):
                ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center', size=15, color='blue')

        ax.set_xticks(range(0,self.dim))
        ax.set_yticks(range(0,self.dim)) 
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.set_xlabel("Y")
        ax.set_ylabel("X")
        plt.show()

        
    def guess(self, guessInx, showLoc = False):
        """
        Function to guess while playing the game 'by hand'
        """
        self.nextInx = guessInx
        if guessInx in self.board:
            print("HIT")
            self.hits.add(guessInx)
        else:
            print("MISS")
            self.misses.add(guessInx)
 
    def refreshGame(self):
        self.board = self.randomBoard()
        self.hits, self.misses, self.hitsSunk = set(), set(), set()
        self.sunkDict = dict(zip(self.names,[0 for i in self.ships]))
        self.possibleShipsDictCond = self.possibleShipsDict.copy()
        self.possibleShipsNumDictCond = self.possibleShipsNumDict.copy()
        self.lastTurnBoards = list()
        self.nextInx = (-1,-1)
            
class BattleshipAutoplay(BattleshipPlayer):
    """
    Supplemental Class the facilitate Automatic runthroughs of the game, returning the number of 
    turns a game takes with the option to print the time taken at each step.
        
    dim, ships, randomOrder, batchSize, and printTime 
    .
    .
    .
    all inherited from BattleshipPlayer class
    """

    
    def __init__(self, dim=10, ships=[2,3,3,4,5], randomOrder=False, batchSize=1000, printTime=False):
        super().__init__(dim, ships, randomOrder, batchSize, printTime)
        
    def play(self, refresh=False):
        """
        Play the game automatically, returning the number of turns the game took
        """
        while True:
            if len(self.hits) == self.numMaxHits:
                n = len(self.hits) + len(self.misses)
                if refresh:
                    self.refreshGame()
                return n
            self.buildAggBoard()
            for i in self.hits:
                del self.aggDict[i]
            
            self.nextInx = max(self.aggDict, key=self.aggDict.get)
            if self.nextInx in self.board:
                self.hits.add(self.nextInx)
            else:
                self.misses.add(self.nextInx)

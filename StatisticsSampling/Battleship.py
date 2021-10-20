import numpy as np
import random
from itertools import permutations
from collections import Counter
import string
from time import time

class Battleship:
    
    def __init__(self, dim=10, ships=[2,3,3,4,5]):
        self.dim = dim
        self.numShips = len(ships)
        self.ships = ships
        self.names = list(string.ascii_lowercase[:len(ships)])
        self.sunkDict = dict(zip(self.names,[0 for i in ships]))
        self.shipLengths = dict(zip(self.names,ships))
        self.generateComponentLayouts()
        self.order = [x for _, x in sorted(zip(ships, self.names))]
        
        
    def generateComponentLayouts(self):
        
        self.possibleShipsDict = dict()
        self.possibleShipsNumDict = dict()
        
        for name in set(self.names):
            orientations = []
            
            for i in range(self.dim - self.shipLengths[name] + 1):
                for j in range(self.dim):
                    orientations += [set([(i+temp,j) for temp in range(self.shipLengths[name])]), set([(j,i+temp) for temp in range(self.shipLengths[name])])]
            
            self.possibleShipsDict[name] = orientations
            self.possibleShipsNumDict[name] = len(orientations)
        
    def randomBoard(self):
        order = self.order.copy()
        final_set = set()
        self.boats = []
        for name in order:
            while 1:
                ship_coords = self.possibleShipsDict[name][np.random.randint(self.possibleShipsNumDict[name])]
                if len(final_set) + len(ship_coords) == len(final_set.union(ship_coords)):
                    final_set = final_set.union(ship_coords)
                    self.boats += [(name, ship_coords)]
                    break
        return final_set
                
class BattleshipEnv(Battleship):
    
    def __init__(self, dim=10, ships=[2,3,3,4,5], lag=2):
        super().__init__(dim, ships)
        self.board = self.randomBoard()
        self.lag = lag
        self.hits = set()
        self.hitsSunk = set()
        self.misses = set()
        self.possibleShipsDictCond = self.possibleShipsDict.copy()
        self.possibleShipsNumDictCond = self.possibleShipsNumDict.copy()
        
        
    def updateOrientations(self):
        
        for i in self.boats:
            boat, location = i
            if len(location.union(self.hits)) == len(self.hits):
                self.hitsSunk= self.hitsSunk.union(location)
                self.sunkDict[boat] = 1
                self.possibleShipsDictCond[boat] = [location]
                self.possibleShipsNumDictCond[boat] = 1
                
        sunkNum = len(self.hitsSunk)
        
        for name in set(self.names):
            if self.sunkDict[name] != 1:
                orientations = self.possibleShipsDictCond[name]
                new_orient = []
                for config in orientations:
                    if (len(self.misses.union(config)) == len(self.misses) + self.shipLengths[name]) and (len(config.union(self.hitsSunk)) ==  self.shipLengths[name] + sunkNum):
                        new_orient += [config]
                self.possibleShipsDictCond[name] = new_orient
                self.possibleShipsNumDictCond[name] = len(new_orient)

                
    def randomSelection(self, order, mustHappen):
        while 1:
            final_set = mustHappen
            for i in order:
                t = time()
                while 1:
                    ship_coords = self.possibleShipsDictCond[i][np.random.randint(self.possibleShipsNumDictCond[i])]

                    if len(final_set) + len(ship_coords) == len(final_set.union(ship_coords)):
                        final_set = final_set.union(ship_coords)
                        break

                    if time() - t > self.lag/10:
                        return set()

            if len(final_set.difference(self.hits)) == len(final_set) - len(self.hits):
                return list(final_set)
            
    def randomConditionalBoard(self, orderIncoming):
        order = orderIncoming.copy()
        mustHappen = set()
        for i in self.sunkDict.items():
            if i[1]:
                order.remove(i[0])
                mustHappen = mustHappen.union(dict(self.boats)[i[0]]) 
        
        final_set = set()
        while len(final_set) == 0:
            final_set = self.randomSelection(order, mustHappen)
        return final_set
            
    
    def buildAggBoard(self):
        
        order = [k for k, v in sorted(self.possibleShipsNumDictCond.items(), key=lambda item: item[1])]
        start_time = time()
        
        self.updateOrientations()
        numIter = 0
        boards = []
        while time() - start_time < self.lag:
            numIter += 1
            boards += self.randomConditionalBoard(order)
        
        self.aggDict = dict(Counter(boards).most_common(self.dim**2))
        self.numIter = numIter

    def view(self, graph=False):
        
        self.buildAggBoard()
        matrix = np.zeros((self.dim,self.dim))
        for location in self.aggDict.keys():
            matrix[location[0], location[1]] = self.aggDict[location]
        matrix = matrix / self.numIter
        if graph:
            self._print(matrix)
        print(self.numIter, "Iterations")
        self.maxInx = ((np.argmax(matrix) % self.dim), int(np.floor((np.argmax(matrix) / self.dim))))
        
    def _print(self, matrix):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_axes([0,0,1,1])
        ax.matshow(matrix, cmap='copper_r')

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
        
        if guessInx in self.board:
            print("HIT")
            self.hits = self.hits.union({guessInx})
        else:
            print("MISS")
            self.misses = self.misses.union({guessInx})

            
class BattleshipAutoplay(BattleshipEnv):
    
    def __init__(self, dim=10, ships=[2,3,3,4,5], lag=2):
        super().__init__(dim, ships, lag)
        
    def play(self, verbose=False, refresh=False):
        if verbose:
            print(self.board)
        while True:
            if len(self.hits) == np.sum(self.ships):
                n = len(self.hits) + len(self.misses)
                if refresh:
                    self.refreshGame()
                return n
            self.buildAggBoard()
            
            for i in self.hits:
                del self.aggDict[i]
            
            self.nextInx = max(self.aggDict, key=self.aggDict.get)
            if verbose:
                print(self.nextInx, self.numIter, self.possibleShipsNumDictCond)
            if self.nextInx in self.board:
                self.hits = self.hits.union({self.nextInx})
                if verbose:
                    print("HIT")
            else:
                self.misses = self.misses.union({self.nextInx})
                if verbose:
                    print("MISS")
    
    def refreshGame(self):
        self.board = self.randomBoard()
        self.hits = set()
        self.misses = set()
        self.sunkDict = dict(zip(self.names,[0 for i in self.ships]))
        self.hitsSunk = set()
        self.possibleShipsDictCond = self.possibleShipsDict.copy()
        self.possibleShipsNumDictCond = self.possibleShipsNumDict.copy()

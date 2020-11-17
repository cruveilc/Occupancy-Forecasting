__author__ = 'quocdung.ngo'

import sys
sys.path.append("../")
import json
from src.ligpgm3.nodedata import NodeData
from src.ligpgm3.graphskeleton import GraphSkeleton
from src.ligpgm3.discretebayesiannetwork import DiscreteBayesianNetwork
from src.ligpgm3.tablecpdfactorization import TableCPDFactorization
from numpy import random
import copy

def getRandomNumber():
    return random.random_sample()

class bayesianOccupantSimulator:
    def __init__(self,occupantBusyState:dict(),weather,target,PastElement):
        structure = "../../../data/StructureJournal1.json"
        # structure = "../../../data/Structure2.json"
        self.skel = GraphSkeleton()
        self.skel.load(structure)
        self.skel.toporder()
        self.nd = NodeData()
        self.nd.load(structure)
        self.evidence = dict(CO2='Medium')
        self.evidence.update(Weather=weather)
        for key,value in occupantBusyState.items():
            if str(value) == 'outOfWorkingTime':
                self.outOfWorkingTime = 1
            else:
                self.outOfWorkingTime = 0
                if str(key) == str('stephane'):
                    # print('Stephane ',value)
                    self.evidence.update(ProfessorWorking=str(value))
                    self.evidence.update(ProfessorBusy=str("ThreeQuart"))
                elif str(key) == str('khadija'):
                    # print('Khadija ',value)
                    self.evidence.update(PermanentWorking=str(value))
                elif str(key) ==str('audrey'):
                    # print('Audrey ',value)
                    self.evidence.update(IntermittentWorking=str(value))
                else:
                    # print('Guest ',value)
                    self.evidence.update(GuestWorking=str(value))
        self.target = target
        if PastElement is not None:
            self.doorStatus = 2
            self.PastElement = PastElement
        else:
            self.PastElement = None
        # print("Target : ",self.target, "evidence : ",self.evidence)
    def inferenceComputation(self):
        # load evidence
        # print("evidence",self.evidence,"target",self.target)
        evidence = self.evidence
        query = self.target
        # load bayesian network
        skelInput = copy.deepcopy(self.skel)
        ndInput = self.nd
        bn = DiscreteBayesianNetwork(skelInput, ndInput)
        # load factorization
        fn = TableCPDFactorization(bn)
        # calculate probability distribution
        self.result = fn.condprobve(query, evidence)
        # output - toggle comment to see
        # print ("Result:",json.dumps(self.result.vals, indent=2))
    def perform(self):
        if self.outOfWorkingTime == 1 and self.PastElement is not None:
            return 'close'
        elif self.outOfWorkingTime == 1 and self.PastElement is None:
            return 0
        elif self.outOfWorkingTime != 1 and self.PastElement is not None:
            if self.PastElement is not None:
                self.inferenceComputation()
                return self.result.vals[0]
                if self.doorStatus == 0:
                    self.evidence.update({self.PastElement:'Open'})
                elif self.doorStatus == 1:
                    self.evidence.update({self.PastElement:'Move'})
                else:
                    self.evidence.update({self.PastElement:'Close'})
                if(randomNumber<=inferenceResult[0]):
                    # self.doorMouvement = 'Open'
                    self.doorStatus = 0;
                elif(randomNumber>inferenceResult[0] and randomNumber<=(inferenceResult[0]+inferenceResult[1])):
                    # self.doorMouvement = 'Move'
                    self.doorStatus = 1;
                else:
                    # self.doorMouvement = 'Close'
                    self.doorStatus = 2;
                return self.result.vals[0]
        else:
            self.inferenceComputation()
            return self.result.vals[2]
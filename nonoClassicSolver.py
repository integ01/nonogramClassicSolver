###########################################################\
#
#  NonoGram solver based on classic puzzle solving: iteratively 
#  applying 'rules' over the nono-grid to find locations that 
#  can be marked. The are three possible states to each grid 
#  point: marked ('X'), Non mark ('.'), Empty ('_')
#  The end result of a solved puzzle should include only marked or
#  non-marked points signifying the color blocks of the nonoGram
#  image.   
#  

import nonoLineMatchesLib as NonoLib
from nonoLineMatchesLib import printd
import numpy as np
import sys
from enum import Enum   
import argparse
import pdb


QFILE = "nonoQuiz-big-bug.csv" #Input Quiz file

DELTA = 0.01
change = -1 #Number of changes to grid pixels.
count = 0 #Number of iterations


Debug = False

class RowCol(Enum):
        ROW = 1
        COL = 4


      
################### class puzzleState ##########################

class puzzleState():
    def __init__(self, hlist=[], vlist=[], waitOnStep=True):
        self.hlist = hlist
        self.vlist = vlist
        self.waitOnStep = waitOnStep
        self.c_size = len(vlist)
        self.r_size = len(hlist)
        self.c_match = {}
        self.r_match = {}
        self.state = {}
        for i in range (self.c_size):
            self.state[i] = [9]*self.r_size

    def getState(self,c,r):
        return self.state[c][r]

    def printState(self):

        vlens = [len(col) for col in self.vlist]
        maxVl = max(vlens) 
        hlens = [len(r) for r in self.hlist]
        maxHl = max(hlens) 


        print (self.c_size, self.r_size)
        tabSize = (maxHl*3+3)
        linespace = ' '*tabSize+' ' 
        vlist = []
        for item in  self.vlist:
            vlist.append(item[:])

        for i in range(maxVl,0,-1):
            linestr = linespace
            for vidx in range(self.c_size):
                vitem = vlist[vidx]
                if len(vitem) >= i:
                    field = str(vitem[0])
                    if len(field)>1:
                        linestr += field+","
                    else:
                        linestr += field+" ,"
                    vlist[vidx].pop(0)
                else:
                    linestr += " , "
            print(linestr)
        print (linespace + "-----------------------------")


        for j in range(self.r_size):
            linestr = " , "*(maxHl-len(self.hlist[j]))
            for h in self.hlist[j]:
                linestr += "{0}, ".format(h)
            tabSpace = (tabSize-len(linestr))*' '          
            linestr += tabSpace + '|'
            #pdb.set_trace()
            for c in range(self.c_size):
                if self.state[c][j] == 1:
                    nchr = "X"
                elif self.state[c][j] == 0:                
                    nchr = "."
                else:
                    nchr = "_"
                linestr += nchr+"  "
            print(linestr)


    def setUp(self):
        for i in range (self.c_size):
            qParam = self.vlist[i]
            lineState = [ self.state[i][r] for r in range(self.r_size)]
            self.c_match[i] = [[NonoLib.Match(idx = j, size = qParam[j], start=-1, end= -1, mask="", full=False) \
                            for j in range(len(qParam)) ]]
            printd (qParam)

        for i in range (self.r_size):
            qParam = self.hlist[i]
            lineState = [ self.state[c][i] for c in range(self.c_size)]
            self.r_match[i] = [[NonoLib.Match(idx = j, size = qParam[j], start=-1, end= -1, mask="", full=False) \
                            for j in range(len(qParam)) ]]

        printd (qParam)

    def done(self):
        for c in range(self.c_size):
            for r in range(self.r_size):
                if self.state[c][r] == 9:
                    return False
        return True                
 
    def updateState(self,c,r, val):
        global change
        if self.state[c][r] == 9:
            self.state[c][r] = val
            if val != 9:
                change += 1
        elif self.state[c][r] != val:
            printd("Dist Error: trying to change Row:{0} Col:{1} from {2} to {3} ".format(r,c, 1-val,val))                   
            return -1
        return 0

    
    def getLineState(self, idx, rowOrCol):
        if rowOrCol == RowCol.ROW:
            return [ self.state[c][idx] for c in range(self.c_size)]
        else:
            return [ self.state[idx][r] for r in range(self.r_size)]            

    def iterateStep(self):
        global change
        global count

        #count +=1
        print ("iteration#{0}: changes:{1}".format(count,change))
        if count %1 == 0 and self.waitOnStep:
            input("Press any key")
        if change == 0 and self.waitOnStep:
            e = input("edit?")
            if e in "Yy":
                try: 
                    raw = input("row,col:")
                    fields = raw.split(",")
                    r1 = int(fields[0])
                    c1 = int(fields[1])
                    self.updateState(c1,r1,1)
                    change =1
                except:
                    printd("wrong input")

###################End of class puzzleState ##########################



def findHiLowRange(lineState,low , hi):
    plow = low+1
    phi = hi
    while (low >0 and lineState[low]==9):
        low -= 1
    low = max(0, low)        
    if  lineState[low] == 0:
        low +=1 
    elif lineState[low] == 1 :
        low +=2
    low = min(plow,low)


    while (hi < len(lineState) and lineState[hi]==9):
        hi += 1

    if hi < len(lineState) and lineState[hi] == 1:
        hi = max(phi,hi-1)
    
    hi = min(hi,len(lineState))

    return low, hi

############################################################################
# function getLineDistMatch(): Basic calculation for a quiz line's placement 
#  distribution.
#
# Input: ‘qParams’ - Is the quiz line parameters (l_1..l_k) list .
#        ‘mRanges’ - are a list of tuples  (startIdx ,move) - which correspond to each 
#                    qParams item (l_1..l_k) and define line block start-index location and 
#                    its possible maximal move (or shift) in the grid.
#        ‘lsize’ - Is the size of the grid line.
# Return: 
#        dist - list of lists of line distributions.
############################################################################
def getLineDistMatch(qParams, mRanges, lsize):
    if len(qParams) == 0:
        return np.array([])
    sumT = sum(qParams) + (len(qParams) -1)
    if (sumT == 0):
        return np.zeros([lsize])
    distM = np.zeros([len(qParams),lsize])
    for x in range(lsize):
        for idx, (qparam, qrange) in enumerate(zip(qParams,mRanges)):
            move = qrange[1]
            start = qrange[0]
            prob = 1./(move+1.)
            if (start <= x) and (x < start+qparam+move+1):              
                #fact=min(min(x-start+1,qparam) , start+qparam+move- x)
                #print("x={}, fact:{}".format(x,fact))
                distM[idx,x] += prob * min(min(min(x-start+1,qparam),move+1) , start+qparam+move- x)
        #print("x={},dist:{}".format(x,distM[:,x]))
    gidx = 0
    gstart = mRanges[0][0]
    gmove =mRanges[0][1]
    lastParam = qParams[0]
    
    for idx, (qparam, qrange) in enumerate(zip(qParams[1:],mRanges[1:])):
      move = qrange[1]
      start = qrange[0] 
      if gmove == move and start == gstart+  lastParam+1:
            distM[idx,:] += distM[idx-1,:]
            distM[idx-1,:] = 0
    dist = np.max(distM,axis=0)
    return dist

##############################################################################
# function getLineDistMatch2(): Another calculation for a quiz line's placement 
# distribution (for reference purposes).
#
# Input: ‘qParams’ - Is the quiz line parameters (l_1..l_k) list .
#        ‘mRanges’ - are a list of tuples  (startIdx ,move) - which correspond to each 
#                    qParams item (l_1..l_k) and define line block start-index location and 
#                    its possible maximal move (or shift) in the grid.
#        ‘lsize’ - Is the size of the grid line.
# Return: 
#        dist - list of lists of line distributions.
##############################################################################   
def getLineDistMatch2( qParams, mRanges, lsize):
       if len(qParams) == 0:
             return np.array([])
       #  SumT = number of  '1' + number of spaces between
       sumT = sum(qParams) + (len(qParams) -1)
       if (sumT == 0):
         return np.zeros([lsize])
       distM = np.zeros([len(qParams),lsize])
       basic_loc=[0]
       for idx, (qparam, qrange) in enumerate(zip(qParams,mRanges)):
            move = qrange[1]
            start = qrange[0]
            prob = 1./(move+1.)
            for shift in range(move+1):  
                 assert(start+qparam+shift <= lsize)
                 distM[idx,start+shift:start+qparam+shift] += np.full(qparam,prob)
       gidx = 0
       gstart = mRanges[0][0]
       gmove =mRanges[0][1]
       lastParam = qParams[0]
    
       for idx, (qparam, qrange) in enumerate(zip(qParams[1:],mRanges[1:])):
         move = qrange[1]
         start = qrange[0] 
         if gmove == move and start == gstart+  lastParam+1:
            distM[idx,:] += distM[idx-1,:]
            distM[idx-1,:] = 0
       dist = np.max(distM,axis=0)
         
       return dist



##############################################################################
#  function 'getLineDistRanges':   
#   finds the sub-ranges of `uncertain' locations of blocks, and prepares list 
#   for each of the quiz blocks (line parameters), where it can be placed. 
#  Inputs: 
#    ‘matchl’ - is the list of list of matches found for the line.
#    ‘lineState’ - is puzzle line state with existing marking.
#    ‘gQParam’ - is quiz line parameters.
#  Returns:
#   A list of tuples (startIdx ,move) which correspond to each of the quiz 
#   parameter items (l_1..l_k) .
##############################################################################
def getLineDistRanges(matchl, lineState,gQParam):
    
    matchRanges = []
    matchBlock = []
    qParaml = []
    low = 0 
    hi = 0
    nextLow = 0
    move = -1
    i=0
    while i < len(matchl):
        lasti = i
        match = matchl[i]
        if match.end != -1:
            blkLen = match.end - match.start
        else: blkLen = 0

        if match.mask != '': # had a mask
            low,hi = findHiLowRange(lineState, match.start-1, match.end)
            low = max (low, nextLow)
            low = max(low , match.start-(match.size-blkLen))
            hi = min(hi, match.end + (match.size-blkLen))
            move = hi-low - match.size
            if (hi-low <0 or move < 0):
                printd ("Error: hi:{0}, low:{1}, move:{2}".format(hi,low, move)             )
            matchRanges.append((low,move))
            qParaml.append(match.size)
            nextLow = match.end + 1
            i+=1
        else:
            paramSum = 0
            low,hi = findHiLowRange(lineState, matchl[lasti].start-1, matchl[lasti].end)
            while i<len(matchl) and matchl[i].mask== '':
                if (matchl[i].start < 0):
                    matchRanges.append((paramSum,-1))
                elif (matchl[i].start >= 0) and matchl[i].start + matchl[i].size <= hi:
                    matchRanges.append((matchl[i].start,-1))
                else:
                    break
                qParaml.append(matchl[i].size)
                paramSum += (matchl[i].size +1)
                i+=1
            
            move = hi-low - paramSum +1
            printd("hi,low:{},{}".format(hi,low))
            printd("paramSum:" + str(paramSum))
            printd("move:"+ str(move))
            
            for ii in range(len(matchRanges)):
                if matchRanges[ii][1] == -1:
                    matchRanges[ii] = (matchRanges[ii][0],move)
            #matchRanges =  map(lambda x: (x[0],move) if x[1]==-1 else x, matchRanges)  
            nextLow = low +paramSum
         
        if (hi-low <0):
            printd ("Error")      
        if (move<0):
            printd ("Case Over fit: ")
            if Debug: NonoLib.printMatchList([matchl])
            printd ("Q param {0} in between: {1}--{2} ".format(qParaml,low,hi))
            return [], []
    
    printd (qParaml)
    printd (matchRanges)
    return qParaml, matchRanges




##########################################################################
# function matchStateToQuizParams() - Match lines (columns or rows) markings 
# to quiz line parametes and generate match combinations.
# Input: pzl - class puzzleState, 
#        rowOrCol, 'class RowCol(Enum)' - match all rows or columns lines.
# Return: 
#        matchl - list of list of line matches, data type:'class Match'.
##########################################################################
def matchStateToQuizParams(pzl, rowOrCol):
        
        size = pzl.r_size if rowOrCol == RowCol.ROW else pzl.c_size
        qParamL = pzl.hlist if rowOrCol == RowCol.ROW else pzl.vlist
        matchsL = {}
        for i in range (size):
            lineState = pzl.getLineState(i,rowOrCol)
            printd ("{0}-#:{1}".format(rowOrCol,i))
            matchs = NonoLib.genMatches(qParamL[i], lineState)
            #NonoLib.printMatchList(matchs)
            matchsL[i] = checkMatch(pzl,matchs, lineState)
        return matchsL



def checkMatch(pzl,matchDict, lineState):
    res = []
    for matchl in matchDict:
        cond = True
        for midx, mitm in enumerate(matchl):
            blank_indxs = NonoLib.find_blanks(mitm)

            for bidx in  blank_indxs:
                if lineState[bidx] == 0:
                    printd ("Found bad match - removing match, idx:{0}".format(bidx))
                    cond = False
                    continue 
                else: 
                    #TODO - add setting line here
                    pass
        if cond:
            res.append( matchl)
    return res

####################################################################
# function matchMissingBlocksWithFreeZones() - 
#  Fill in non-matched (empty) quiz parameters with 
#  possible free zone placements and add to the match 
#  combinations list.
# Input: pzl - class puzzleState, 
#        rowOrCol, 'class RowCol(Enum)' - match all rows or columns lines.
#        matchsL - Current list of line matches placements, data type:'class Match'.
# Return: 
#        matchl - Updated list of line placements, data type:'class Match'.
####################################################################
def matchMissingBlocksWithFreeZones (pzl, rowOrCol, matchsL):
        
    size = pzl.r_size if rowOrCol == RowCol.ROW else pzl.c_size
    qParamL = pzl.hlist if rowOrCol == RowCol.ROW else pzl.vlist
    resMatchs = {}
    for lineIdx in range (size):
        printd ("=================")
        printd ("{0}-#:{1}".format(rowOrCol,lineIdx))
        printd ("Qparam:{0}".format(qParamL[lineIdx]))
        lineState = pzl.getLineState(lineIdx,rowOrCol)
        resMatchGrp = []
        for matchl in matchsL[lineIdx]:
          freeZonesIdxs = NonoLib.getFreeZones(lineState)
          resMatchGrp += NonoLib.genCombEmptyZonesRec(matchl,0,freeZonesIdxs,0, [])
        if Debug: NonoLib.printMatchList(resMatchGrp)
        resMatchs[lineIdx] = resMatchGrp
    return  resMatchs



##########################################################################
#  function scanAxisDist:  Generate distributions rows/cols lines according
#  to match placements,
#    pzl - class puzzleState.
#    rowOrCol - 'class RowCol(Enum)' - distributions for all rows or columns lines.
#    matchsL - list of list of line matches, data type:'class Match'.
#  Return:
#    resDistL - Dictionary of line distributions, data type:Dictionary
##########################################################################
def scanAxisDist(pzl, rowOrCol, matchsL):        
    size = pzl.r_size if rowOrCol == RowCol.ROW else pzl.c_size
    qParamL = pzl.hlist if rowOrCol == RowCol.ROW else pzl.vlist
    matchTrimL = {}
    resDistL = {}
    for lineIdx in range (size):
        printd ("=================")
        printd ("{0}-#:{1}".format(rowOrCol,lineIdx))
        printd("Qparam:{0}".format(qParamL[lineIdx]))
        lineState = pzl.getLineState(lineIdx,rowOrCol)
        resMatchGrp = matchsL[lineIdx]
        zones =[]
        matchTrim = []
        for match in resMatchGrp:
            qParaml, matchRanges = getLineDistRanges(match, lineState, qParamL[lineIdx])
            if len(qParaml) ==0:
              continue
            assert(NonoLib.compareList(qParaml,qParamL[lineIdx]))
            lineDist = getLineDistMatch(qParaml,matchRanges, len(lineState))
            if Debug:
               lineDist2 = getLineDistMatch2(qParaml,matchRanges, len(lineState))
               NonoLib.compareLineDist(lineDist, lineDist2)

            matchTrim.append(match)
            printd (lineDist)
            printd ("-------")
            zones.append(lineDist)
        resDistL[lineIdx] = zones
        matchTrimL[lineIdx] = matchTrim
    return resDistL


def combDist(distl):
    global DELTA
    
    distArr = np.array(distl)
    dcount = distArr.shape[0]
    dGridLen = distArr.shape[1]

    sumLines = np.sum(distArr,axis=0)

    combArr = np.ones(dGridLen)*0.5
    combArr[sumLines < DELTA] = 0.
    combArr[sumLines > dcount - dcount*DELTA] = 1.
    
    return combArr


##########################################################################
#  function fillByDistribution(): Start filling the puzzle grid, 
#    according to line distributions. 
#  Each location with probability ‘1.0’ is a candidate to be ‘marked-full’.
#  Each location with probability ‘0.0’ zero is a candidate to be ‘marked-blank’.
#  Input: pzl - class puzzleState, 
#         rowOrCol, data type:class RowCol(Enum)-  fill row or column lines.
#         distD, data type:Dictionary of lists - list of grid probabilities.
#  Return: 0 - Success, -1 - puzzle conflict fail.
##########################################################################
def fillByDistribution(pzl, rowOrCol, distD):
        
    size = pzl.r_size if rowOrCol == RowCol.ROW else pzl.c_size
    res = 0  
    for l in distD: 
        distl = distD[l]
        if len(distl) == 0: 
            printd("Line:{0} #{1} distribution empty".format(rowOrCol,l)) 
            continue
        lineComb = list(combDist(distl))
        for j in range (len(lineComb)):
            stateUpdate = -1
            if lineComb[j] + DELTA > 1.:
                stateUpdate = 1
            if lineComb[j] - DELTA < 0.:
                stateUpdate = 0
            if (stateUpdate != -1):
                if rowOrCol == RowCol.COL:
                    res = pzl.updateState(l,j,stateUpdate)
                else:
                    res = pzl.updateState(j,l,stateUpdate)
            if res != 0:   break
                      
    return res

##########################################################################
#  function fillByMatches(): fill block markings witch have blank gaps 
#  according to the found matches.
#  Input: pzl - class puzzleState, 
#         rowOrCol, data type:class RowCol(Enum)-  scan row or column lines.
#         matchsL, data type:class Match - list of line matches.
#  Output - Update puzzle state with markings.
#  Return: 0 - Success, -1 - puzzle conflict fail.
##########################################################################
def fillByMatches(pzl, rowOrCol, matchsL):               
    ## fill in Matched Blocks 
    size = pzl.r_size if rowOrCol == RowCol.ROW else pzl.c_size
    for lidx in range(size):
            match =  matchsL[lidx]
            if len(match) == 1:
                for midx, mitm in enumerate(match[0]):
                    blank_indxs = NonoLib.find_blanks(mitm)
                    for bidx in blank_indxs:
                        if rowOrCol == RowCol.COL:
                            res = pzl.updateState(lidx,bidx,1)
                        else:
                            res = pzl.updateState(bidx,lidx,1)
                        if res != 0: return -1
    return 0


    ##########################################################################
    #
    #  solve(pzl) - Iterative algorithm for filling Nonogram puzzle
    #  Input: pzl - class puzzleState.
    #
    ##########################################################################
def solve(pzl):
    global change
    global count
    res = 0 
    count = 0
    while (not pzl.done() and change != 0):
        change = 0
        ########## Match grid line markings to the quiz line parameters (columns/rows)#########
        #         and generate possible match combinations            
        printd ("========Fit existing line blocks to the quiz line paramters ===========")
        printd ("           and generate possible match combinations.")
        c_match = matchStateToQuizParams( pzl, RowCol.COL)
        r_match = matchStateToQuizParams( pzl, RowCol.ROW)

        ######### Fill in non-matched (empty) quiz parameters with  #########
        #          possible free zone placements in the match 
        #          combinations list.
        printd ("========Fill in non-matched (empty) quiz parameters with ")
        printd ("======== possible free zone placements combinations. ===========")
        c_match = matchMissingBlocksWithFreeZones (pzl, RowCol.COL, c_match)
        r_match = matchMissingBlocksWithFreeZones (pzl, RowCol.ROW, r_match)


        ########### Scan Distributions rows/cols ###################
        printd ("========Scan Distributions rows/cols ===========")                
        cdist = scanAxisDist(pzl, RowCol.COL, c_match)
        rdist = scanAxisDist(pzl, RowCol.ROW, r_match)

        ############# Start Filling According to Distributions ##############
        printd ("===== Start Filling According to Distributions ====") 
        res = fillByDistribution(pzl,RowCol.COL,cdist)
        if res != 0: break
        res = fillByDistribution(pzl,RowCol.ROW,rdist)
        if res != 0: break
        ############# Start Filling According to Matches##############
        ## fill in Matched blocks .
        printd ("===== Start Filling in Matched Blocks ====") 
        res = fillByMatches(pzl,RowCol.COL, c_match )
        if res != 0: break
        res = fillByMatches(pzl,RowCol.ROW, r_match )
        if res != 0: break

        count +=1
        if pzl.waitOnStep: 
          pzl.printState()
          pzl.iterateStep()
    return res, pzl.done(), change
#########################        
# britStartHints - an example of setting start marking hints
#
def britStartHints(pzl):
    listStates = [ (3,3), (4,3), (12,3), (13,3), (21,3), \
                 (6,8), (7,8), (10,8), (14,8), (15,8),(18,8),\
                  (6,16) , (11,16), (11,16), (16,16), (20,16),\
                   (3,21), (4,21), (9,21), (10,21), (15,21), (20,21), (21,21)]
    for c,r in listStates:
        pzl.updateState(c,r,1)

############################################################
#                            main
# argument - input file (csv), see example file for format
#
############################################################
def main(argv):
    global QFILE

    inputfile = QFILE
    if len(argv) > 0:
        inputfile = argv[0]
    print ('Input file is '+ inputfile)

    hlist, vlist = NonoLib.readQuizFile(inputfile)
    print (hlist)
    print (vlist)
    y_size = len(hlist)
    x_size = len(vlist)

    pzl = puzzleState(hlist, vlist)  # initialize the puzzle state
    #britStartHints(pzl)   #an example of setting start markings.
    pzl.printState()

    res, done, numChanges = solve(pzl)
    pzl.printState()
    print ("Puzzle res:{}, Done:{}, last number of changes:{}".format(res, done, numChanges))
    if res == 0:
       if  done:
         print ("Puzzle Succesfully Solved!")
       elif numChanges == 0:
         print ("Puzzle Not Completed, (possibly more than one solution")
       else:
          print ("Puzzle Not Completed, unexpected result")
    else:
       print ("Puzzle Not Completed, conflict in definition")
    

if __name__ == "__main__":

    main(sys.argv[1:])





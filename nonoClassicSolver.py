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

import nonoLineMatchesLib as NonoMatch
import numpy as np
import sys
from enum import Enum   
import pdb
fqz = None


QFILE = "nonoQuiz-big-bug.csv"

DELTA = 0.01
change = -1
count = 0

interFlag = False


class RowCol(Enum):
        ROW = 1
        COL = 4

################### class puzzleState ##########################

class puzzleState():
    def __init__(self, hlist=[], vlist=[]):
        self.hlist = hlist
        self.vlist = vlist
        self.c_size = len(vlist)
        self.r_size = len(hlist)
        self.c_match = {}
        self.r_match = {}
        #'_', '_', '_', '_', '_', '_', '_', '_','_', '_']
        #column_empty = dict{ i: '_' for i in range self.c_size}
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
#                linestr += str("{0}, ".format(vitem[0]))
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
                #linestr += "{0}, ".format(self.state[c][j])
                linestr += nchr+"  "
            print(linestr)


    def setUp(self):
        for i in range (self.c_size):
            qParam = self.vlist[i]
            lineState = [ self.state[i][r] for r in range(self.r_size)]
            self.c_match[i] = [[NonoMatch.Match(idx = j, size = qParam[j], start=-1, end= -1, mask="", full=False) \
                            for j in range(len(qParam)) ]]
            print (qParam)

        for i in range (self.r_size):
            qParam = self.hlist[i]
            lineState = [ self.state[c][i] for c in range(self.c_size)]
            self.r_match[i] = [[NonoMatch.Match(idx = j, size = qParam[j], start=-1, end= -1, mask="", full=False) \
                            for j in range(len(qParam)) ]]

        print (qParam)

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
            print("Dist Error: trying to change Row:{0} Col:{1} from {2} to {3} ".format(r,c, 1-val,val))                   

    


    def iterateStep(self):
        global interFlag
        global change
        global count

        count +=1
        print ("iteration#{0}: changes:{1}".format(count,change))
        if count %1 == 0 and interFlag:
            input("Press any key")
        if change == 0 and interFlag:
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
                    print("wrong input")

###################End of class puzzleState ##########################






def genCombEmptyZonesRec(matchl, midx, freeZoneIdxs, minIdx, res):

    if midx >= len(matchl):
        return [res]
    elif matchl[midx].mask != "":#(midx <= len(matchl)) and 
        if matchl[midx].start >= minIdx :
            next = res[:]
            next.append(matchl[midx])
            minIdx = matchl[midx].end +1
            ret = genCombEmptyZonesRec(matchl, midx+1, freeZoneIdxs, minIdx, next)
            return ret
        else:
            print ("jumping case, free zone space {0} > block matchs:{1}".format(\
          minIdx, matchl[midx].start ))
            return []
    elif freeZoneIdxs == []: #if match.mask == "":
        if len(matchl) >0:
            print ("jumping case, no free zone space left for empty block" + matchl[midx].print_fields() )
        else:
            print ("jumping case, no free zone space left for empty block and empty matchs" )
        return []
    else:
        match = matchl[midx] 
        retList = []

        fidx =0
  
        for zone in freeZoneIdxs:
            blkLen = zone[1]
            blkStartIdx = zone[0]
            match = matchl[midx]
            if True:
                if minIdx > blkStartIdx+blkLen:
                    print ("Skip: Min index {0} >  end of freezone: {1}".format(minIdx,blkStartIdx+ blkLen))
                    continue
                startIdx  = max (minIdx, blkStartIdx)
                if startIdx + match.size > blkLen + blkStartIdx:
                    print ("Skip: free len: {0} < match size: {1}".format(blkLen,match.size))
                    continue

                newMatch = NonoMatch.Match(idx = midx, size=match.size, start=startIdx, \
                                   end= startIdx+ match.size, mask="", full=False)
                next = res[:]
                next.append(newMatch)
                minIdxNext = startIdx + match.size +1
                if (blkStartIdx+blkLen > minIdxNext):
                    ret = genCombEmptyZonesRec(matchl, midx+1, freeZoneIdxs, minIdxNext, next)
                else:
                    ret = genCombEmptyZonesRec(matchl, midx+1, freeZoneIdxs[1:], minIdxNext, next)          
                retList += ret  
        return retList


def findMinIdZones(matchl, lineState):

    freeZonesIdxs = NonoMatch.getFreeZones(lineState)
    zoneIdx = 0
    emptyMatch =[]

    selcomb = genCombEmptyZonesRec(matchl,0,freeZonesIdxs,0, [])
    #for combItm in selcomb:
        # print ("==============")
        #for item in combItm:
        #  print(item.print_fields())
    return selcomb




def findMinIdx(matchl, lineState):
    idx = 0
    startIdx = 0
    minRes = []
    while (len(matchl) > idx):
        match = matchl[idx]
        if match.mask == "":
            startIdx = max(match.start, startIdx)
            minIdx = startIdx
            while minIdx < len(lineState) and minIdx-startIdx < match.size and \
            lineState[minIdx] == 9:   minIdx += 1
            if (minIdx-startIdx != match.size):
                minIdx +=1
                startIdx = minIdx
            else:
                if ( lineState[minIdx] != 1):
                    match.start = startIdx
                    match.end = minIdx
                    minRes.append(minIdx)
                    startIdx = minIdx+1
                    idx +=1
        else:
            minRes.append( match.end)
            startIdx = match.end+1
            idx += 1
    return minRes, matchl



def getSubLineDist(match, lineState): #, minIdx):
    global fqz
    low = match[0].start-1
    while (low >0 and lineState[low]==9):
        low -= 1
    if lineState[low] == 1 or lineState[low] == 0:
        low +=1


    blkLen = match[0].end - match[0].start

    low = max(0, low)
    if (match[0].mask!= ''): 
        low = max(low, match[0].start-(match[0].size-blkLen))

    #low = max(minIdx, low)

    blkLen2 = match[-1].end - match[-1].start
    hi = match[-1].end
    while (hi < len(lineState) and lineState[hi]==9):
        hi += 1
    if hi < len(lineState) and lineState[hi] == 1:
        hi -=1
    hi = min(hi,len(lineState))
    if (match[-1].mask!= ''): 
        hi = min(hi, match[-1].end + (match[-1].size-blkLen2))

    qParam = [match[i].size for i in range(len(match))]
    #subLineState = lineState[low:hi]
    if (hi-low <0):
        print ("Error")
    if sum(qParam) + len(qParam) -1 > hi -low:
        print ("Case Over fit: ")
        NonoMatch.printMatchList([match])
        print (" {0} in between: {1}--{2} ".format(qParam,low,hi))
        return (low, hi, [])  
    lineDist = fqz.getLineDist(qParam, hi-low)
    return (low, hi, lineDist)

def findHiLowRange(lineState,low , hi):
    while (low >0 and lineState[low]==9):
        low -= 1
    if lineState[low] == 1 or lineState[low] == 0:
        low +=1 
    low = max(0, low)        
    while (hi < len(lineState) and lineState[hi]==9):
        hi += 1
    if hi < len(lineState) and lineState[hi] == 1:
        hi -=1
    hi = min(hi,len(lineState))
    return low, hi
    
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
            if (start <= x) and (x < start+qparam+move):              
                distM[idx,x] += prob * min(min(x-start+1,move+1) , start+qparam+move - x)
    dist = np.max(distM,axis=0)

    return dist
    
def getSubLineDist3(matchl, lineState,gQParam): #, minIdx):
    global fqz

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
            #if (hi-low <0 or move < 0):
            #    print ("Error: hi:{0}, low:{1}, move:{2}".format(hi,low, move)             )
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
            for ii in range(len(matchRanges)):
                if matchRanges[ii][1] == -1:
                    matchRanges[ii] = (matchRanges[ii][0],move)
            #matchRanges =  map(lambda x: (x[0],move) if x[1]==-1 else x, matchRanges)  
            nextLow = low +paramSum
         
        if (hi-low <0):
            print ("Error")      
        if (move<0):
            print ("Case Over fit: ")
            NonoMatch.printMatchList([matchl])
            print ("Q param {0} in between: {1}--{2} ".format(qParaml,low,hi))
            return []
    

        #lineDist = fqz.getLineDist(qParam, hi-low)
    print (qParaml)
    print (matchRanges)
    lineDist = getLineDistMatch(qParaml,matchRanges, len(lineState))

    return lineDist
 

def emptyMarkFullEdgesLines(match, listState):
    resState = listState[:]
    fullLine = True
    semiFull = True
    #if len(match) != 1:
    #  return listState, False, False
    if True:
        ml = match
        for j, m in enumerate(ml):
#            if m.size == len(m.mask):
            if (m.mask == ""): 
                fullLine = False
                semiFull = False
            elif m.full:
                if (m.start>0):
                    resState[m.start-1] =0
                if (m.end < len(listState)):
                    resState[m.end] =0            
            else : fullLine = False
        if fullLine :
            print( "Line full match")
            for j in range (len(listState)):
                if resState[j] != 1: 
                    resState[j] = 0 
    return resState, fullLine, semiFull



def getLineState(pzl, idx, rowOrCol):
        if rowOrCol == RowCol.ROW:
            return [ pzl.state[c][idx] for c in range(pzl.c_size)]
        else:
            return [ pzl.state[idx][r] for r in range(pzl.r_size)]            

##################################################
## Match Row/columns To Quiz Parameters.
###################################################
def matchStateToQuizParams(pzl, rowOrCol):
        
        size = pzl.r_size if rowOrCol == RowCol.ROW else pzl.c_size
        qParamL = pzl.hlist if rowOrCol == RowCol.ROW else pzl.vlist
        matchsL = {}
        for i in range (size):
            lineState = getLineState(pzl,i,rowOrCol)
            print ("{0}-#:{1}".format(rowOrCol,i))
            matchs = NonoMatch.genMatches(qParamL[i], lineState)
            NonoMatch.printMatchList(matchs)
            matchsL[i] = checkMatch(pzl,matchs, lineState)
            #NonoMatch.printMatchList(matchsL[i])
        return matchsL



def checkMatch(pzl,matchDict, lineState):
    res = []
    for matchl in matchDict:
        cond = True
        for midx, mitm in enumerate(matchl):
            blank_indxs = NonoMatch.find_blanks(mitm)

            for bidx in  blank_indxs:
                if lineState[bidx] == 0:
                    print ("Found bad match - removing match, idx:{0}".format(bidx))
                    cond = False
                    continue #TODO break??
                else: 
                    #TODO - add setting line here
                    pass
        if cond:
            res.append( matchl)
    return res


def scanAxisDist(pzl, rowOrCol):        
    size = pzl.r_size if rowOrCol == RowCol.ROW else pzl.c_size
    qParamL = pzl.hlist if rowOrCol == RowCol.ROW else pzl.vlist
    matchsL = pzl.r_match if rowOrCol == RowCol.ROW else pzl.c_match
    matchTrimL = {}
    resDistL = {}
    for lineIdx in range (size):
        print ("=================")
        print ("{0}-#:{1}".format(rowOrCol,lineIdx))
        print("Qparam:{0}".format(qParamL[lineIdx]))
        lineState = getLineState(pzl,lineIdx,rowOrCol)
        resMatchGrp = []
        for matchl in matchsL[lineIdx]:
            resMatchGrp += findMinIdZones(matchl, lineState)
        #self.c_match[c] = res
        NonoMatch.printMatchList(resMatchGrp)
        zones =[]
        matchTrim = []
        for match in resMatchGrp:
            res = getSubLineDist3(match, lineState, qParamL[lineIdx])
            if len(res) == 0:
                continue
            else:
                matchTrim.append(match)
                print (res)
                print ("-------")
                zones.append(res)
        resDistL[lineIdx] = zones
        matchTrimL[lineIdx] = matchTrim
    return (matchTrimL, resDistL)

def combDist(distl):
    global DELTA
    siz = distl[0].shape
    one = np.ones(siz)
    zero = np.zeros(siz)
    for dist in distl:
        zero += dist
        dist[dist< 1.0- DELTA] = 0.
        one *= dist
    one[one < 1. - DELTA] = 0.5
    one[zero < 0.+ DELTA ] = 0.
    return one


def getCommonConjMatch( matchDict):
    if len(matchDict) <= 1:
        return matchDict
    #else
    baseMatch = matchDict[0]
    for mlIdx in range(1,len(matchDict)):
        nextMatch = matchDict[mlIdx]
        baseMatch = NonoMatch.Conj(baseMatch, nextMatch)
    #Complement with empty list
    if len(baseMatch) > 0:
        res = matchDict[0]
        bidx =0
        for midx, mitm in enumerate(res):
            if bidx < len(baseMatch):
                if mitm.start == baseMatch[bidx].start:
                    res[midx] = baseMatch[bidx]
                    res[midx].idx = midx
                    bidx += 1
                else:
                    res[midx].full = False
            else:
                res[midx].full = False
        return [res]
    else :
        return matchDict

 ######## Start Filling According to Distributions ########
def fillByDistribution(pzl, rowOrCol, distD):
        
    size = pzl.r_size if rowOrCol == RowCol.ROW else pzl.c_size
        
    #for i in range (size):
    for l in distD: 
        distl = distD[l]
        if len(distl) >0: #TODO - check why empty
            lineComb = list(combDist(distl))
            for j in range (len(lineComb)):
                stateUpdate = -1
                if lineComb[j] + DELTA > 1.:
                    stateUpdate = 1
                if lineComb[j] - DELTA < 0.:
                    stateUpdate = 0
                if (stateUpdate != -1):
                    if rowOrCol == RowCol.COL:
                        pzl.updateState(l,j,stateUpdate)
                    else:
                        pzl.updateState(j,l,stateUpdate)
        else:
            print("Line:{0} #{1} distribution empty".format(rowOrCol,l))           



############# Start Filling According to Matches##############
def fillByMatches(pzl, rowOrCol):               
    ## fill in Matched Blocks 
    # for col, match in enumerate(self.c_match):
    size = pzl.r_size if rowOrCol == RowCol.ROW else pzl.c_size
    matchsL = pzl.r_match if rowOrCol == RowCol.ROW else pzl.c_match
    for lidx in range(size):
            match =  matchsL[lidx]
            if len(match) == 1:
                for midx, mitm in enumerate(match[0]):
                    blank_indxs = NonoMatch.find_blanks(mitm)
                    for bidx in  blank_indxs:
                        if rowOrCol == RowCol.COL:
                            pzl.updateState(lidx,bidx,1)
                        else:
                            pzl.updateState(bidx,lidx,1)                        


def markBlanksInFullLines(pzl, rowOrCol):
    print ("========Mark Blanks in Full rows/cols===========")
            ## Search full rows/cols and mark fulls
    size = pzl.r_size if rowOrCol == RowCol.ROW else pzl.c_size
    sizeCross = pzl.c_size if rowOrCol == RowCol.ROW else pzl.r_size
    matchsL = pzl.r_match if rowOrCol == RowCol.ROW else pzl.c_match
    for lidx in range (size):
            lineState = getLineState(pzl,lidx,rowOrCol)
            ###Optimization - Get Common intersection of matches
            if lidx == 16:
                print ("")
            #  matchs2 = self.getCommonConjMatch(self.c_match[c])
            matchs = matchsL[lidx]
            #matchs = self.getCommonConjMatch(self.c_match[c])
            print("Col:{0}".format(lidx))
            if len(matchs) > 1 :
                NonoMatch.printMatchList(matchs)        
                print ("Vs.")
                matchs = getCommonConjMatch(matchsL[lidx])
                #if c == 16:          
                NonoMatch.printMatchList(matchs)
                if lidx == 16:
                    if len(matchs) == 1 :
#                    if matchs[0][3].mask != "":
                        print ("")
                if len(matchs[0]) != len(matchsL[lidx][0]):
                    print ("Assert Error - match len")

            if len(matchs) == 1:
                resState, fullLine, semi = emptyMarkFullEdgesLines(matchs[0],lineState)
                if fullLine or semi:
                    print("full match:{0}, semiFull:{1}".format(lidx, fullLine, semi))    
                for i in range(sizeCross):
                    if rowOrCol == RowCol.COL:
                            pzl.updateState(lidx,i,resState[i])
                    else:
                            pzl.updateState(i,lidx,resState[i])       
                    

    ##########################################################################
    #
    #    Solve - Iterative algorithm for filling Nonogram puzzle
    #
    ##########################################################################
def solve(pzl):
    global change
    global count

    count = 0
    while (not pzl.done() and change != 0):
        if (count == 4):
            print("")
        change = 0
        ########## Match columns/rows to quiz line parametes #########
        #         and generate possible match ombinations            
        print ("========Matched Marked rows/cols to Params (All combinations) ===========")
        pzl.c_match = matchStateToQuizParams( pzl, RowCol.COL)
        pzl.r_match = matchStateToQuizParams( pzl, RowCol.ROW)

        ########### Scan Distributions rows/cols ###################
        print ("========Scan Distributions rows/cols ===========")                
        pzl.c_match, cdist = scanAxisDist(pzl, RowCol.COL)
        pzl.r_match, rdist = scanAxisDist(pzl, RowCol.ROW)
        
        ############# Start Filling According to Distributions ##############
        print ("===== Start Filling According to Distributions ====") 
        fillByDistribution(pzl,RowCol.COL,cdist)
        fillByDistribution(pzl,RowCol.ROW,rdist)

        ############# Start Filling According to Matches##############
        ## fill in Matched Blanks in col
        print ("===== Start Filling in incomplete Matched Blocks ====") 
        fillByMatches(pzl,RowCol.COL )
        fillByMatches(pzl,RowCol.ROW )

        ############ Search full rows/cols and mark fulls  #############
        print ("========Mark Full rows/cols===========")        
        markBlanksInFullLines(pzl, RowCol.COL)
        markBlanksInFullLines(pzl, RowCol.ROW)


        pzl.printState()
        pzl.iterateStep()

        


def britStart(pzl):
    listStates = [ (3,3), (4,3), (12,3), (13,3), (21,3), \
                 (6,8), (7,8), (10,8), (14,8), (15,8),(18,8),\
                  (6,16) , (11,16), (11,16), (16,16), (20,16),\
                   (3,21), (4,21), (9,21), (10,21), (15,21), (20,21), (21,21)]
    for c,r in listStates:
        pzl.updateState(c,r,1)

def main(argv):
    global fqz
    global QFILE
    inputfile = None
    if len(argv) > 0:
        inputfile = argv[0]
    if inputfile is None:
        inputfile = QFILE
    print ('Input file is '+ inputfile)

    #outputfile = argv[1]
    # path = '/home/integ/Desktop/Data/nonoGram/'
    #print ('Path is '+ path)


    #fqz = QuizLoad2Dist()
    #hlist, vlist = fqz.readQuizFile(inputfile )
    hlist, vlist = NonoMatch.readQuizFile(inputfile)
    print (hlist)
    print (vlist)
    y_size = len(hlist)
    x_size = len(vlist)

    pzl = puzzleState(hlist, vlist)
    #britStart(pzl)
    pzl.printState()

    solve(pzl)
    pzl.printState()


if __name__ == "__main__":
    main(sys.argv[1:])

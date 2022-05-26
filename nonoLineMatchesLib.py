import pdb
import csv

PUZ_DIMY = 0
Debug = False

def readQuizFile(name):     
    flag = 0
    hlist =[]
    vlist =[]
    fo = open(name, "rt")
    with fo as csvfile:
        qzread = csv.reader(csvfile, delimiter=',')
        for row in qzread:
            if (len(row) == 0):
                continue
            if (row[0] == 'H'): 
                flag =1
                clist = hlist
            elif (row[0] == 'V'): 
                flag = 1
                clist = vlist
            elif (flag == 1): 
                clist.append([int(x) for x in row if x !=''])
    return hlist, vlist


class Match:
    def __init__(self, idx = -1, size = -1, start = -1, end =-1, mask = [], full = False):
        self.idx = idx
        self.size = size
        self.start = start
        self.end = end
        self.mask = mask
        self.full = full
    def print_fields(self):
        if self.full:
            of="C"
        else:
            of="?"
        if  self.end - self.start == 1:
            return "{0}_qblk:{1}@{2}:{3}({4})".format(self.idx, self.size, self.start, ''.join(self.mask),of)
        else:
            return "{0}_qblk:{1}@{2}:{3}({4})".format(self.idx, self.size, self.start, ''.join(self.mask),of)




def find_blanks(m):
    return [ i+m.start for i in range(len(m.mask)) if m.mask[i] == '_']

def printd(s):
  global Debug
  if Debug:
     print(s)

def printMatchList(retlist):
    if retlist == []:
        print("[[]]")
    for matches in retlist:
        res = ">"
        for match in matches:
            res += match.print_fields()
            res += ", "
        print (res)

def Conj(ml1, ml2):
    res = []
    midx2 =0
    midx1 = 0
    while midx1 < len(ml1) and  (midx2 < len(ml2)):
        mitm1 = ml1[midx1]
        mitm2 = ml2[midx2]
        if (mitm1.start > mitm2.start) :
            midx2 += 1
        elif (mitm2.start > mitm1.start) :
            midx1 += 1
        else:
            if mitm1.full and mitm2.full and mitm1.size == mitm2.size:
                if midx1 <= midx2:
                    res.append(mitm1)
                else:
                    res.append(mitm2)
            midx1 += 1
            midx2 += 1
    return res

def getBlocks(line):
    found = False;
    res = []
    cur_bidx = 0
    for i in range(len(line)):
        if not found:
            if line[i] == 1:
                res.append((i,1))
                cur_bidx = len(res) -1
                found = True
        else:
            if line[i] == 1:
                res[cur_bidx] = (res[cur_bidx][0],res[cur_bidx][1]+1)
            else:
                found = False
    return res



def mergeMasks(blockMasks):
    mask = []
    startIdx = blockMasks[0][0]
    endIdx = blockMasks[-1][0] + blockMasks[-1][1]
    prevIdx =0
    for i, blk in enumerate(blockMasks):
        if i == 0:
            mask.append(blk[2])
        else: 
            emptySize = blk[0] - prevIdx 
            if (emptySize < 0):
                print ("Error mergeMasks: emptySize:{0}".format(emptySize))
            mask.append('_'*emptySize )
            mask.append(blk[2])
        prevIdx = blk[0]+blk[1]
    mask = ''.join(mask)
    return (startIdx,endIdx-startIdx, mask)

def compareList(l1,l2):
   if(len(l1)==len(l2) and len(l1)==sum([1 for i,j in zip(l1,l2) if i==j])):
      return True
   else:
      return False


def compareLineDist(dist1, dist2):
    lsize = 30
    diff = []
    for i in range(lsize):
       if abs (dist1[i] - dist2[i])> 1e-4:
           diff += [i]
    if len(diff)>0:
       print ("<<Unequal distributions:>>")
       print ("unequal idexes:"+str(diff))
       print (dist1)
       print ("----<><>---")
       print(dist2)
       print ("<</Unequal distributions:/>>")


##########################################################################
#  function getFreeZones: Find locations of non-marked grid spaces.
#  Inputs:
#    line - puzzle line pixels.
#  Return:
#    list of tuples with free locations [(startIdx,endIdx)]
##########################################################################
def getFreeZones(line):
    found = False;
    res = []
    cur_bidx = -1
    prev_i = 9
    for i in range(len(line)):
        if not found:
            if line[i] == 9 and prev_i != 1:
                res.append((i,1))
                cur_bidx += 1 # = len(res) -1
                
                found = True
        else:
            if line[i] == 9:
                res[cur_bidx] = (res[cur_bidx][0],res[cur_bidx][1]+1)
            else:
                found = False
        prev_i = line[i]
    return res

##########################################################################
#  function genCombEmptyZonesRec: Using match list to generate possible 
#  placements for all 'empty' matches using the empty zones in the grid.
#  Inputs:
#    matchl - list of list of line matches, data type:'class Match'.
#    imatch   - current match list idx in the match list.
#    freeZoneIdxs, tuples (startIdx, endIdx) - list of empty zones of the line, 
#    minPos - current line position index to check.
#    res - intermediate recursion result.
#  Return:
#    matchl - Update list of list of line matches, data type:'class Match'.  
##########################################################################
def genCombEmptyZonesRec(matchl, imatch, freeZoneIdxs, minPos, res):

    if imatch >= len(matchl):
        assert(len(res)==len(matchl))
        return [res]
    elif matchl[imatch].mask != "":#(imatch <= len(matchl)) and 
        if matchl[imatch].start >= minPos :
            next = res[:]
            next.append(matchl[imatch])
            minPos = matchl[imatch].end +1
            ret = genCombEmptyZonesRec(matchl, imatch+1, freeZoneIdxs, minPos, next)
            if (imatch == 0): 
               for r in ret:
                 assert(len(r)==len(matchl))

            return ret
        else:
            printd ("jumping case, free zone space {0} > block matchs:{1}".format(\
          minPos, matchl[imatch].start ))
            return []
    elif freeZoneIdxs == []: #if match.mask == "":
        if len(matchl) >0:
            printd ("jumping case, no free zone space left for empty block" + matchl[imatch].print_fields() )
        else:
            printd ("jumping case, no free zone space left for empty block and empty matchs" )
        return []
    else:
        match = matchl[imatch] 
        retList = []

        fidx =0
  
        for zone in freeZoneIdxs:
            zoneLen = zone[1]
            zoneStartPos = zone[0]
            match = matchl[imatch]
            if True:
                if minPos > zoneStartPos+zoneLen:
                    printd ("Skip: Min index {0} >  end of freezone: {1}".format(minPos,zoneStartPos+ zoneLen))
                    continue
                startPos  = max (minPos, zoneStartPos)
                if startPos + match.size > zoneStartPos + zoneLen:
                    printd ("Skip: free len: {0} < match size: {1}".format(zoneLen,match.size))
                    continue

                newMatch = Match(idx = imatch, size=match.size, start=startPos, \
                                   end= startPos+ match.size, mask="", full=False)
                next = res[:]
                next.append(newMatch)
                minPosNext = startPos + match.size +1
                if (zoneStartPos+zoneLen > minPosNext):
                    ret = genCombEmptyZonesRec(matchl, imatch+1, freeZoneIdxs, minPosNext, next)
                else:
                    ret = genCombEmptyZonesRec(matchl, imatch+1, freeZoneIdxs[1:], minPosNext, next)          
                retList += ret  
                if (imatch == 0) and len(ret)>0:
                 for r in ret:
                   assert(len(r)==len(matchl))

        return retList

def genSubGroupsRec(blockMasks, l,res):
    if l == 1:
        res.append(mergeMasks(blockMasks))
        return [res]
    n = len(blockMasks)
    if (n <l):
        print ("Error : genSubGroupsRec: n{0}<l{1} ".format(n,l))

    combList = []
    for i in range(1,n-l+2):
        next = res[:]
        next.append(mergeMasks(blockMasks[:i]))
        comb = genSubGroupsRec(blockMasks[i:],l-1, next)
        combList += comb
    return combList

            

def genSubGroups(blockIdxs, leng):   
    base = [ (  (blk[0], blk[1], 'X'*blk[1]) ) for blk in blockIdxs]
    
    if (len(blockIdxs) == leng):
        return [base]
    res = genSubGroupsRec(base,leng,[])

    return res




##########################################################################
#  function genCombRec5: Using parameter list to generate possible 
#  matches with grid 'markings'.
#  Inputs:
#    qParam -list to quiz input
#    pidx - index of sub list to find combinations.
#    blockIdxs - list of tuples of discovered blocks - (startIdx,size,MaskOfBlock)
#    minIdx - current minimum line pixel index.
#    res - intermediate recursion result.
#  Return:
#    matchl - Update list of list of line matches, data type:'class Match'.  
##########################################################################
def genCombRec5(qParam,pidx,blockIdxs,minIdx, res):
    global PUZ_DIMY
    #pdb.set_trace()
    if pidx == len(qParam) or blockIdxs ==[]:
        #print (len(res), minIdx)
        for id in range (pidx,len(qParam)):
            minIdx += qParam[id] +1
        minIdx -= 1

        if (minIdx > PUZ_DIMY):
            printd ("jumping case minimum end index:({0}) > max index:{1}".format(minIdx,PUZ_DIMY ))
            return []
        out = ""
        return [res]
    else:
        retList = []
        blkLen = blockIdxs[0][1]
        blkStartIdx = blockIdxs[0][0]
            
        for pidx in range(pidx,len(qParam)-len(blockIdxs)+1):  
            if qParam[pidx] < blkLen:
                #print ("jumping case blkLen:{0}, > param len:{1}".format(qParam[pidx],blkLen))
                continue
            if (minIdx > blkStartIdx):
                #print ("jumping case blk start index:({0},{1}) < minimum index:{2}".format(blkStartIdx, blkLen,minIdx ))
                continue
            next = res[:]
            next.append(Match(idx = pidx, size = qParam[pidx], start=blkStartIdx, end= blkStartIdx+ blkLen, mask=blockIdxs[0][2], full=(qParam[pidx]==blkLen)))
            minIdx += qParam[pidx] +1
            minIdxNext = max(minIdx, (blkStartIdx+ blkLen+1))
            ret = genCombRec5(qParam,pidx+1, blockIdxs[1:],minIdxNext,next)
            
            retList += ret
        return retList

def genMatches(qParam,lineState):
    global PUZ_DIMY
    PUZ_DIMY = len(lineState)

    printd ("Print In:")
    printd (qParam)
    printd (lineState)
    
    blockIdxs = getBlocks(lineState)
    maxSubLen = min( len(qParam),len(blockIdxs))
    blockMasks = []
    for leng in range(maxSubLen,0,-1):
	    blockMasks += genSubGroups(blockIdxs, leng)
    if maxSubLen == 0:
        finalComb = []
        for i in range(len(qParam)):
            finalComb.append(Match(idx = i, size = qParam[i], start=-1, end= -1, mask="", full=False))
        return [finalComb]

    retlist = []
    for blocks in blockMasks:
        selcomb = genCombRec5(qParam, 0, blocks, 0,[])
        for combItm in selcomb:           
            finalComb = []
            for i in range(len(qParam)):
                finalComb.append(Match(idx = i, size = qParam[i], start=-1, end= -1,\
                 mask="", full=False))          
            for item in combItm:
                finalComb[item.idx] = item
            retlist += [finalComb]
    return retlist


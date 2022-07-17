

import argparse
import sys, getopt
import nonoLineMatchesLib as NonoLib
import copy
from os import walk
import shutil
from nonoClassicSolver import *


BASE_PATH= "../"
QFILE = "../Q-Multi.csv"
FLAGS = None
MAX_GUESS = 2

def scanGuessFreeLocs(pzl):
    locs = []
    for c in range(pzl.c_size):
            for r in range(pzl.r_size):
                if pzl.state[c][r] == 9:
                   locs.append((r,c))
    return locs

def testValidLoc(r,c,sol):
   for pzl in sol:
       if pzl.state[c][r] == 1:
           return False
   return True

##########################################################################
#  function recSolve:  Find one or more possible solutions for nonoGrams 
#   Using recursive search and using the nonoClassicSolver.solve() function
#   The recursive search is limited to a maximum depth:
#     (input param-FLAGS.depth=10)
# Input:
#  qfile - quiz file.
#  verbose - print out solution process and options.
# Return: 
#        list of puzzle solutions (class puzzleState)
##########################################################################
def recSolve(pzl, depth=0, verbose=True):
   global FLAGS

   if depth > FLAGS.depth:
       return []
   res, done, change = solve(pzl)
   if verbose:
     print ("----------------------------")
     pzl.printState()
     print ("Depth:{}, Res:{}, Done:{}, last changes:{}".format(depth, res,done, change))
   if res == -1:
     if verbose:
       print(">>Branch Contradiction - aborting path")
       print("==============")
     return []
   elif done:
     if verbose:
       print(">>Found legal solution")
       print("==============")
     return [pzl]
   elif change == 0:
     sol = []
     locs = scanGuessFreeLocs(pzl)
     if len(locs) > MAX_GUESS:
          locs = locs[:MAX_GUESS]
     tries = 0
     for (r,c) in locs:

       if not testValidLoc(r,c,sol):
          if verbose:
            print ("Depth:{}, Skipping location: {},{}".format(depth, r,c))
          continue
       tries+=1
       newpzl = copy.deepcopy(pzl)
       newpzl.updateState(c,r,1)      
       if verbose:
         print ("Depth:{}, Guess#{}, Update loc (row,col): ({},{}) - Recursive".format(depth,tries,r,c))
       ressol = recSolve(newpzl, depth+1,verbose)
       if len(ressol)>0 :
          sol += ressol
       if len(sol)>2: break
     return sol

##########################################################################
#  function solveQuiz:  Find one or more possible solutions for nonoGrams 
#   The search process is limited to a maximum depth (input param-FLAGS.depth)
# Input:
#  qfile - quiz file.
#  verbose - print out solution process and options.
# Return: 
#        list of puzzle solutions (class puzzleState)
##########################################################################
def solveQuiz(qfile, verbose=True):
  hlist, vlist = NonoLib.readQuizFile(qfile)
  #print (hlist)
  #print (vlist)
  y_size = len(hlist)
  x_size = len(vlist)

  pzl = puzzleState(hlist, vlist,  waitOnStep=False)  # initialize the puzzle state
  
  if verbose:
    print ('Q file: '+ qfile + " --Start solving")  
    pzl.printState()
  sol =recSolve(pzl, depth=0, verbose=verbose)
  if verbose:
    print ("=======  Recursive Results =========")
    print ("Total # solutions found:{}".format(len(sol)))
    for i,pz in enumerate(sol):
      print ("#{}:".format(i))
      pz.printState()
  return sol

##########################################################################
#  function solveQuizSet:  Find nonoGrams with a single solution, 
#    from the input directory. Copy them to a destination path.
#  qpath - quiz directory path for quiz files.
#  dstPath - destination path for the nonoGrams with a single 
#              solution.
##########################################################################
def solveQuizSet(qpath,dstPath):
    f = []
    for (dirpath, dirnames, filenames) in walk(qpath):
         f.extend(filenames)
         break
    solSingle =[]
    solMulti = []
    solNone = []
    for fl in f:
      if "Q-" in fl:
           sol = solveQuiz(qpath+'/'+fl ,  verbose=False)
           if len(sol)>1:
              print(fl + " has: {} solutions".format(len(sol)))
              solMulti.append((fl,sol))
           elif len(sol)==0:
              print(fl + "?? Could not find solutions")
              solNone.append((fl,sol))
           else:
              print(fl + " has single solution")
              solSingle.append((fl,sol))
    
    pdb.set_trace()
    def insertPic(name, index): return name[:index+1] + 'Pic' + name[index+1:]
    
    for it in solSingle: 
          shutil.copyfile(qpath+'/'+it[0],dstPath+ '/'+it[0])
    for it in solSingle: 
          shutil.copyfile(qpath+'/'+insertPic(it[0],it[0].find('Q-')),dstPath+ '/' +insertPic(it[0],it[0].find('Q-')) )

############################################################
#                            main
# usage: nonoSolverRecursive.py [-h] [--quiz QFile] [--depth N]
#                              [--basepath BASEPATH] [--qdir QDIR]
#                              [--ddir DDIR]
# uses global FLAGS for parameters.
############################################################
def main():
  global QFILE
  global FLAGS
  global BASE_PATH

  basepath= BASE_PATH 
  if FLAGS.basepath:
     basepath = FLAGS.basepath

  print ('Max search depth:'+str(FLAGS.depth)) 

  if FLAGS.qdir:
    qpath = basepath+FLAGS.qdir
    if FLAGS.ddir:
      dstPath=basepath+FLAGS.ddir
    else:
      dstPath = basepath+'nonoGramTrainSingleSol'
    print ('Input Quiz Path: '+ qpath)
    print ('Dest Quiz Path: '+ dstPath)
    solveQuizSet(qpath,dstPath)
  elif len(FLAGS.quiz)>0:
    inputfile = FLAGS.quiz
    solveQuiz(inputfile)
  else:
     inputfile = QFILE
     solveQuiz(inputfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Recursive Nono Classic Sovler') 
    parser.add_argument('--quiz', type=str, default='', metavar='QFile',  help='file name for the quiz file.')
    parser.add_argument('--depth', type=int, default=10, metavar='N', help='max recursive depth of search')
    parser.add_argument('--basepath', type=str, default='', help='base path for quiz directories')

    parser.add_argument('--qdir', type=str, default='', help='quiz directory Q- files.')
    parser.add_argument('--ddir', type=str, default='', help='destination directory for filtered quizes with a single solution.')

    FLAGS, unparsed = parser.parse_known_args()

    main()


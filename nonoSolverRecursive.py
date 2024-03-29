#
#     Copyright (C) 2022 Benny Shimony
#
#     This file is part of nonogramClassicSolver.
#
#    nonogramClassicSolver is free software: you can redistribute it and/or modify it under
#    the terms of the GNU General Public License as published by the Free Software Foundation,
#    either version 3 of the License, or (at your option) any later version.
#
#    nonogramClassicSolver is distributed in the hope that it will be useful, but 
#    WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
#    FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along with Foobar. 
#    If not, see <https://www.gnu.org/licenses/>. 
#
import argparse
import sys, getopt
import nonoLineMatchesLib as NonoLib
import copy
import os
from os import walk
import shutil
from nonoClassicSolver import *
import nonoGenLib as nonoGen


BASE_PATH= ""
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
    

    def insertPic(name, index): return name[:index+1] + 'Pic' + name[index+1:]
    
    for it in solSingle: 
          shutil.copyfile(qpath+'/'+it[0],dstPath+ '/'+it[0])
    for it in solSingle: 
          shutil.copyfile(qpath+'/'+insertPic(it[0],it[0].find('Q-')),dstPath+ '/' +insertPic(it[0],it[0].find('Q-')) )

############################################################
#                            main
# usage: nonoSolverRecursive.py [-h] [--qfile QFile] [--depth DEPTH]
#                              [--basepath BASEPATH] [--qdir QDIR]
#                              [--ddir DDIR] [--gen N] [--gen-dim DIM]
#
# Generate and Sort Solution using a Recursive Nono Classic Solver
#
# optional arguments:
#  -h, --help           show this help message and exit
#  --qfile QFile        file name for the quiz file.
#  --depth DEPTH        max recursive depth of search
#  --basepath BASEPATH  base path for quiz directories
#  --qdir QDIR          quiz directory for solving multi quiz set of files.
#  --ddir DDIR          destination directory for storing result quizes with a
#                       single solution.
#  --gen N              generate N random nonogram quizes and store them in the
#                       `qdir` directory path.
#  --gen-dim DIM        dimension (DIMxDIM) of nonogram quizes to generate.
#
# uses global FLAGS for parameters.
############################################################

def main():
  global QFILE
  global FLAGS
  global BASE_PATH
  N = 0
  dim = 10
  basepath= BASE_PATH 
  if FLAGS.basepath:
     basepath = FLAGS.basepath

  print ('Max search depth:'+str(FLAGS.depth)) 

  if FLAGS.qdir:
    qpath = os.path.join(basepath, FLAGS.qdir)
    if FLAGS.ddir:
      dstPath= os.path.join(basepath, FLAGS.ddir)
    else:
      dstPath = os.path.join(basepath, 'nonoGramTrainSingleSol')
    print ('Input Quiz Path: '+ qpath)
    print ('Dest Quiz Path: '+ dstPath)

    if not os.path.exists(qpath):
        print ("path:'{}' does not exist, please create it first.".format(qpath))
        return
    if not os.path.exists(dstPath):
        print ("destination path:'{}' does not exist, please create it first.".format(dstPath))
        return

    if FLAGS.gen:
      N = FLAGS.gen
      if FLAGS.gen_dim:
        dim = FLAGS.gen_dim
      print ("Generating {} puzzles of dimension:{}x{} in the input path:{}".format(N,dim,dim,qpath))
      nonoGen.genQuizSet(qpath,N,dim)

    print ("Solving and Sorting Solutions of puzzles to desination path:'{}'".format(dstPath))
    solveQuizSet(qpath,dstPath)
  elif len(FLAGS.qfile)>0:
    inputfile = FLAGS.qfile
    solveQuiz(inputfile)
  else:
     inputfile = QFILE
     solveQuiz(inputfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and Sort Solution using a Recursive Nono Classic Solver') 
    parser.add_argument('--qfile', type=str, default='', metavar='QFile',  help='file name for the quiz file.')
    parser.add_argument('--depth', type=int, default=10, help='max recursive depth of search')
    parser.add_argument('--basepath', type=str, default='', help='base path for quiz directories')

    parser.add_argument('--qdir', type=str, default='', help='quiz directory for solving multi quiz set of files.')
    parser.add_argument('--ddir', type=str, default='', help='destination directory for storing result quizes with a single solution.')

    parser.add_argument('--gen', type=int, default=100, metavar='N', help='generate N random nonogram quizes and store them in the `qdir` directory path.')

    parser.add_argument('--gen-dim', type=int, default=10, metavar='DIM', help='dimension (DIMxDIM) of nonogram quizes to generate.')
    FLAGS, unparsed = parser.parse_known_args()

    main()


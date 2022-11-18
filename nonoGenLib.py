#
#     Copyright (C) 2022 Benny Shimony
#
#    nonoGenLib.py : This file is part of nonogramClassicSolver.
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
import numpy as np
import csv
import sys
import os


##########################################################################
# GenNonoRndQuiz class - 
#
# Supports generating random nono-gram quizes.
# There are several paramters that can be 'tweaked' in the random generation:
# See the function genImage().
#
##########################################################################
class GenNonoRndQuiz:

   ##########################################################################
   #  function genImage:  generate random pixel image on a grid xSize,ySize 
   #   The function selects a random position and random 'square' block size
   #   and fills that location with 'marked' pixels.
   # Input:
   #  ySize, xSize - pixel image size.
   #  Iter - Number of iterations of random square selections.
   #  maxBlock - the maximum size of chosen 'block' square to mark.
   # Return: 
   #       random pixel image on a grid (xSize,ySize )
   ##########################################################################
   def genImage(self, ySize, xSize, Iter, maxBlock):
    	picOut = np.zeros((ySize,xSize), dtype = float)
    	sampley =  np.random.choice(range(ySize),Iter)
    	samplex =  np.random.choice(range(xSize),Iter)
    	size =  np.random.choice(range(1,maxBlock),Iter)
    	while (np.sum(picOut) == 0):
    		for i in range (Iter):
    			if (sampley[i]+size[i] <= ySize) and (samplex[i]+size[i] <= xSize):
    			  picOut[sampley[i]:sampley[i]+size[i], samplex[i]:samplex[i]+size[i]] = 1.0
    	return picOut


   def genLineList(self, linearr):
        linelist=[]
        x = 0 
        lena = linearr.shape[0]
#        print ("lena:",  lena)
        x =0 
        while (x< lena):
          while (x< lena) and (linearr[x]!= 1):
      	    x+= 1
          count =0
          while (x< lena) and (linearr[x]!= 0):
            count+=1
            x+=1
          if (count > 0):
             linelist.append(count)
        if linelist == []:
        	linelist.append(0)
        return linelist

	
	
   def genQuiz(self, picOut):
        hlist =[]
        vlist =[]
        ySize = picOut.shape[0]
        xSize = picOut.shape[1]

        for i in range(ySize):
          hsub = self.genLineList(picOut[i,:])
          hlist.append(hsub)
        for i in range( xSize):
          vsub = self.genLineList(picOut[:,i])
          vlist.append(vsub)
        return hlist, vlist


   def writeQuizFile(self, hlist, vlist, name):     
        with open(name, "wt") as fo:
          out = csv.writer(fo, delimiter=',')
          out.writerow('H')
          for hitem in hlist:
          	out.writerow(hitem)
          out.writerow('V')
          for vitem in vlist:
          	out.writerow(vitem)

   def writeQuizTarget(self, pic, name):
        with open(name, "wb") as fo:
          np.savetxt(fo, pic, '%s', ',')

##########################################################################
#  function genQuizSet: 
# generate number of random nono-gram quiz's and store them in the selected
# directory path.
#
# Input:
#  path - quiz directory to save to.
#  nsamples - Number of quizs to generate.
#  dim -  the dimension of of the nono-gram quiz (dim X dim).
# Return:
#  Output directory with quiz & their solutions.
#  
##########################################################################
def genQuizSet(path, nsamples, dim=10):

    gri = GenNonoRndQuiz()
    picB = np.array((dim,dim), dtype=float)
    if (nsamples < 0 or nsamples > 10000):
        print ("illegal number argument ")
        return 
    if not os.path.exists(path):
        print ("path:'{}' does not exist, please create it first.".format(path))
        return
    path = os.path.join(path+"/")
    print ("Generating {} quizes in path:'{}'.".format(nsamples,path))
    for i in range(nsamples):
        numIter = np.random.randint(10,high=40)
        maxBlockSize = np.random.randint(low = int(0.2 * dim),high=int(0.7*dim))
        picB = gri.genImage(dim,dim, numIter, maxBlockSize)
        hlist, vlist = gri.genQuiz(picB)
        #print (picB)
        #print (hlist)
        #print (vlist)
        gri.writeQuizFile(hlist, vlist, path+'Q-%d.csv' % i)
        gri.writeQuizTarget( picB, path+'QPic-%d.csv' % i)







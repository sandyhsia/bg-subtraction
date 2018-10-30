import numpy as np
import random

class ViBe:
    '''
    classdocs
    '''
    __defaultNbSamples = 20        
    __defaultReqMatches = 2        
    __defaultRadius = 20;          
    __defaultSubsamplingFactor = 16
    __BG = 0                       
    __FG = 255                     
    __c_xoff=[-1,0,1,-1,1,-1,0,1,0]  
    __c_yoff=[-1,0,1,-1,1,-1,0,1,0]  
    
    __samples=[]              	   
    __Height = 0
    __Width = 0

    def __init__(self, grayFrame):
        '''
        Constructor
        '''
        self.__Height = grayFrame.shape[0]
        self.__Width = grayFrame.shape[1]
        

        for i in range(self.__defaultNbSamples+1):
            self.__samples.insert(i,np.zeros((grayFrame.shape[0],grayFrame.shape[1]),dtype=grayFrame.dtype));
            
        self.__init_params(grayFrame)
      
    def __init_params(self,grayFrame):
        #recod random row and col from generation
        rand=0
        r=0
        c=0

        #initialize w.r.t to each sample pixel
        for y in range(self.__Height):
            for x in range(self.__Width):
                for k in range(self.__defaultNbSamples):
                    #randomly get sampled pixel value
                    rand=random.randint(0,8)
                    r=y+self.__c_yoff[rand]
                    if r<0:
                        r=0
                    if r>=self.__Height: 
                        r=self.__Height-1    #row
                    c=x+self.__c_xoff[rand]
                    if c<0:
                        c=0 
                    if c>=self.__Width:
                        c=self.__Width-1     #col
                    #store sampled pixel value
                    self.__samples[k][y,x] = grayFrame[r,c]
            self.__samples[self.__defaultNbSamples][y,x] = 0
            
    def update(self,grayFrame,frameNo):
        foreground = np.zeros((self.__Height,self.__Width),dtype=np.uint8)
        for y in range(self.__Height):        #Height
            for x in range(self.__Width):     #Width
                #for the need to judge whether a pixel is bg, 
                #index: counter for compared pixel
                #count: counter for matched samples 
                count=0;index=0;
                dist=0.0;
                while (count<self.__defaultReqMatches) and (index<self.__defaultNbSamples):
                    dist= float(grayFrame[y,x]) - float(self.__samples[index][y,x]);
                    if dist<0: dist=-dist
                    if dist<self.__defaultRadius: count = count+1
                    index = index+1

                if count>=self.__defaultReqMatches:
                    #judge to be bg pixel
		    #only bg pixel can be broadcast and update sampled pixel value
                    self.__samples[self.__defaultNbSamples][y,x]=0
    
                    foreground[y,x] = self.__BG
    
                    rand=random.randint(0,self.__defaultSubsamplingFactor)
                    if rand==0:
                        rand=random.randint(0,self.__defaultNbSamples)
                        self.__samples[rand][y,x]=grayFrame[y,x]
                    rand=random.randint(0,self.__defaultSubsamplingFactor)
                    if rand==0:
                        rand=random.randint(0,8)
                        yN=y+self.__c_yoff[rand]
                        if yN<0: yN=0
                        if yN>=self.__Height: yN=self.__Height-1
                        rand=random.randint(0,8)
                        xN=x+self.__c_xoff[rand]
                        if xN<0: xN=0
                        if xN>=self.__Width: xN=self.__Width-1
                        rand=random.randint(0,self.__defaultNbSamples)
                        self.__samples[rand][yN,xN]=grayFrame[y,x]
                else:
                    #judge to be fg pixel
                    foreground[y,x] = self.__FG;
                    self.__samples[self.__defaultNbSamples][y,x] += 1
                    if self.__samples[self.__defaultNbSamples][y,x]>50:
                        rand=random.randint(0,self.__defaultNbSamples)
                        if rand==0:
                            rand=random.randint(0,self.__defaultNbSamples)
                            self.__samples[rand][y,x]=grayFrame[y,x]
        return foreground

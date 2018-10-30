import cv2
import numpy as np
import os
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

import util as util

class frameProcessor:
    """
        Process video item
    """

    def __init__(self, videoItem=None, setting=None):
        
        self.videoItem = videoItem
        self.setting = setting
        self.initRunningDir = os.getcwd()

    def setSetting(self, setting):
        self.setting = setting

    def process(self):
        
        # chose methods
        if self.setting["SVD"]["isUsed"]:
            ret, fgMasks = self.SVD()
            return ret, fgMasks
        
        elif self.setting["RobustPCA"]["isUsed"]:
            print("Will be implemented soon.")
            return 0, None
        
        elif self.setting["ViBe"]["isUsed"]:
            print("Will be implemented soon.")
            return 0, None
        
        else:
            print("Not supported.")
            return -1, None


    def SVD(self):
        
        print("Processing with SVD method...")
        print("When processing, frame is resized as: ", self.setting["scale"]/100)
        print("Creating image matrix...")
        init, M = util.create_data_matrix_from_video(self.videoItem, self.setting["fps"], self.setting["scale"])
        self.initFrames = init.reshape(self.setting["initDims"] + (-1,))

        # Begin the low rank approx and get residual
        M = M.astype(float)
        U, Sigma, Vt = svds(M, k=self.setting["SVD"]["nRankApprox"])
        lowRank = U @ np.diag(Sigma) @ Vt # low_rank is background

        residual = M - lowRank
        residualFrames = residual.reshape(self.setting["procDims"] + (-1,))

        # Get fgMask from residual frames
        print("Processing fg Masks...")
        maskFrames = np.zeros((self.setting["initDims"][0], self.setting["initDims"][1], np.shape(residualFrames)[2]))
        for i in range(np.shape(residualFrames)[2]):
            frame = residualFrames[:, :, i]
            frame = cv2.resize(frame, (self.setting["initDims"][1], self.setting["initDims"][0]), interpolation=cv2.INTER_CUBIC)
            (thresh, mask) = cv2.threshold(np.abs(frame), 25, 255, cv2.THRESH_BINARY)
            maskFrames[:, :, i] = mask

        return 1, maskFrames

    def postProcessing(self, fgFrames, semanticHelp=False):
        print("try to Postprocessing...")
        refinedFrames = np.zeros(np.shape(fgFrames))
        
        if semanticHelp:
            from semanticHelper import segmentOne

        for i in range(np.shape(fgFrames)[2]):
            frame = fgFrames[:, :, i]
            
            # fill the holes in the mask, tiny mask such as noise can be filtered by area later.
            kernel = np.ones((5,5) ,np.uint8)
            closing = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel).astype(np.uint8)

            refined = np.zeros(np.shape(frame)) 
            if np.sum(closing) == 0:
                refinedFrames[:, :, i] = refined
                continue
            
            if semanticHelp:
                minScore = self.setting["semanticHelp"]["minScore"]
                idList = self.setting["semanticHelp"]["idList"]
                one_image_np = self.initFrames[:, :, :, i]
                segmanticMasks_dict = segmentOne(one_image_np, minScore, idList)

            img, contours, hir = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            j = 0
            for cnt in contours:
                (x,y),r = cv2.minEnclosingCircle(cnt)
                area = cv2.contourArea(cnt)
                if area > 500: #and area/(3.1415926*r*r) > 0.1:
                    cv2.fillPoly(refined, pts =[cnt], color=max(255-j, 1))
                    
                    # if needs help from semantic, go through maskrcnn inference model
                    # draw 
                    if semanticHelp:
                        intersection = 0
                        for tmp_mask in segmanticMasks_dict['detection_masks']:
                            intersection += np.sum(tmp_mask*refine > 0)

                        if len(segmanticMasks_dict['detection_masks']) > 0 and intersection == 0:
                            cv2.fillPoly(refined, pts =[cnt], color=0)
                    
                    j += 1

            refinedFrames[:, :, i] = refined 

        if semanticHelp:
            os.chdir(self.initRunningDir)

        return 1, refinedFrames


    def mix(self, Masks, startCounter=0, combineFrameNames=[], debug=True):

        print("Combining the color mask...")
        if os.path.isdir('../debug') == False:
            os.mkdir('../debug')
        if os.path.isdir('../debug/combine') == False:
            os.mkdir('../debug/combine')
        if debug:
            if os.path.isdir('../debug/mask') == False:
                os.mkdir('../debug/mask')
            if os.path.isdir('../debug/init')== False:
                os.mkdir('../debug/init')

        for i in range(np.shape(Masks)[2]):
            init = self.initFrames[:, :, :, i]
            mask = Masks[:, :, i]
            frameID = util.getFrameID(startCounter+i)
            
            if debug:
                initFrameName = "../debug/init/"+frameID+".png"
                cv2.imwrite(initFrameName, init)

            # check each connected-component, and draw color on the init_frame 
            # for visualization
            if np.sum(mask) > 0:
                for j in range(255):
                    init[mask == 255-j, :] = init[mask == 255-j, :]*0.7
                    init[mask == 255-j, 2] += int(max(255-j*60, 0)*0.3)
                    init[mask == 255-j, 1] += int(min(j*30, 255)*0.3)
                    init[mask == 255-j, 0] += int(min(j*60, 255)*0.3)
                    if(np.sum(mask == 255-i-1) == 0):
                        break
            
            frameName = "../debug/combine/"+frameID+".png"
            cv2.imwrite(frameName, init)

            if debug:
                maskName = "../debug/mask/"+frameID+".png"
                cv2.imwrite(maskName, mask)
            
            combineFrameNames.append(frameName)
        
        return 1, combineFrameNames

    def cleanup(self, debug=True):
        # means remaining tmp files to debug
        if debug:
            return False
        
        else:    
            import shutil
            shutil.rmtree('../debug')
            return os.path.isdir('../debug')

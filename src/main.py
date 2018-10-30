import os, sys
import numpy as np
import cv2
import moviepy.editor as mpe
from moviepy.video.io.bindings import mplfig_to_npimage
import matplotlib.pyplot as plt
from frameProcessor import *

# later this Setting can be written into xml/json
Setting = {         # video storage info
                    "videoPath": "/home/dongqxia/projects/bgsubtraction/submission/video_dataset/D.mp4",
                    "storePath": "../video_output/mixed_D_10s20s.mp4",
                    "subClipInterval": 10, # second, means your method processes a subclip with this time interval
                    "startingSeconds": 10, # second, means your method starts to process video from here till end
                    "endingSeconds": 20,

                    # methods
                    "SVD": {"isUsed": True, "nRankApprox": 2},
                    "RobustPCA": {"isUsed": False},
                    "Vibe": {"isUsed": False},

                    # video setting
                    "scale": 50, # if 100, will not be changed.
                    "fps": 15,
                    "initDims": (0, 0),
                    "procDims": (0, 0),
                    "channel": 3, # rgb

                    # others
                    "postProcessing": False,
                    "semanticHelp":{"isUsed":False, "minScore":0.6, "idList":[1]},
                    "debug": True}

if __name__ == '__main__':

    print("video to be process:", Setting["videoPath"])
    video = mpe.VideoFileClip(Setting["videoPath"])

    # resolution of image
    # change resolution might accelerate processing
    originalWidth = video.size[1]
    originalHeight = video.size[0]
    dims = (int(originalWidth * Setting["scale"] / 100), int(originalHeight * Setting["scale"] / 100))
    print("When processing, will scale every frame to {d[0]} * {d[1]}".format(d=dims))
    Setting["initDims"] = (originalWidth, originalHeight, Setting["channel"])
    Setting["procDims"] = dims

    subClipInterval = Setting["subClipInterval"] # second
    startingSeconds = Setting["startingSeconds"]
    endingSeconds = Setting["endingSeconds"]
    if endingSeconds == -1:
        endingSeconds = video.duration

    combineFrameNames = []
    for i in range(int(startingSeconds/subClipInterval), int(np.ceil(endingSeconds/subClipInterval))):
        start = i*subClipInterval
        end = start + subClipInterval
        if end > video.duration:
            end = video.duration
        print("processing", start, "to", end, "...")
        frameProc = frameProcessor(video.subclip(start, end), Setting)
        ret, fgMasks = frameProc.process()

        if Setting["postProcessing"]:
            ret, fgMasks = frameProc.postProcessing(fgMasks, Setting["semanticHelp"]["isUsed"])

        ret, combineFrameNames = frameProc.mix(fgMasks, start*Setting["fps"], combineFrameNames, Setting["debug"])

    
    # output the frames to video
    clip = mpe.ImageSequenceClip(combineFrameNames, fps=Setting["fps"])
    clip.write_videofile(Setting["storePath"])

    # and cleanup tmp files
    frameProc.cleanup(Setting["debug"])




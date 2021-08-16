import cv2
import sys
import numpy as np
from skimage.measure import compare_ssim as ssim
import time
from multiprocessing.pool import ThreadPool
from collections import deque
from sklearn.decomposition import PCA
from common import clock, draw_str, StatValue,FindFile
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def checkTermination(dis):
    # print (dis)
    # if dis[-1-:-1] > dis[-2] and :
    peaks, _ = find_peaks(dis, width=10)
    # print (dis)
    if len(peaks)>0:
        print (peaks)
        print (dis)
        return peaks[0]
    else:
        return 999
    
def extractMxyContour(c):
    M = cv2.moments(c)
    return int(M["m10"] / M["m00"]),int(M["m01"] / M["m00"])

def distance(x,y):
    return np.sqrt(x*x+y*y)

def normalize(data)->list:
    return [i/max(data) for i in data]

def pyrDown(currentframe,pyrDownNum):
     for i in range(pyrDownNum):
         currentframe = cv2.pyrDown(currentframe)
     return currentframe

class CushionTracking():
    def __init__(self,*args,**kws):
        print ("*****%s start*****" % (time.strftime("%X", time.localtime())))
        self.VideoDir = args[0] 
        self.plottarget = args[1]
        self.threaded_mode = args[2]
        self.ROI = args[3]
        self.delta_t = args[4]
        self.time_offset = args[5]
        self.extend = args[6]
        self.pyrDownNum = 2
        self.scale = np.power(2,self.pyrDownNum)
        self.FlagRecord= True
        self.Video = FindFile(self.VideoDir, '.avi')[0]
        self.outfile = self.VideoDir.split("\\")[-1]+"_"
        self.offframe = int(abs(self.time_offset)/self.delta_t)
        self.endframe = int(self.ROI - self.time_offset)/self.delta_t
        self.t0 = 0
        
    def run(self):
        for file in self.Video:
            self.outVideo = self.VideoDir + "\\_"+file.split("\\")[-1]
            self.outImage = self.VideoDir + "\\"+file.split("\\")[-1]
            self.target = self.VideoDir + "\\"+file.split("\\")[-1].replace("avi","txt")
            if self.plottarget == True:
                self.getTarget
            self.area_list = []
            self.pca_contour = []
            self.pca_contour_id = []
            self.multiProccessing(file)
            self.end = self.time_offset + self.frames_num*self.delta_t
            self.frametime = np.linspace(self.time_offset,self.end,int(self.frames_num)+1)
            print ("START:",self.time_offset,"END:",self.end,"NUM:",int(self.frames_num),"DELTA_T:",self.delta_t)
            # print ("tIME:",self.time_offset,"END:",self.end,"NUM:",int(self.frames_num),"DELT_T:",self.delta_t)
            self.curveplot()
            
            
    @property
    def getTarget(self):
        ftarget = open(self.target)
        line = ftarget.readline()
        data = line.split(",")
        self.time_offset = float(data[0])
        self.delta_t = float(data[1])
        # self.cutframe = int(self.time_offset/self.delta_t)
        self.ref_axis = [float(i) for i in data[2:]] 
        self.ROI = max(self.ref_axis)*1.2
        self.offframe = int(abs(self.time_offset)/self.delta_t)
        self.endframe = int(self.ROI - self.time_offset)/self.delta_t
        print ("ROI:",self.ROI,"ENDFRAME:",self.endframe)
 
        
    def curveplot(self):
        self.dis =[]
        pca_contour_len = len(self.pca_contour)
        cX0,cY0 = extractMxyContour(self.pca_contour[0])
        print ('Before deployment frame:',self.frame0id - self.offframe,"After deployment frame:",pca_contour_len)
        frameNumBefore = int(self.frame0id - self.offframe)
        for i in range(frameNumBefore):
            self.dis.append(0)
        for i in range(pca_contour_len):
            pca_contour_frame = self.pca_contour[i] 
            cX,cY = extractMxyContour(pca_contour_frame)
            pca_contour_frame =[j[0] for j in pca_contour_frame]
            self.dis.append(distance(abs(cX-cX0),abs(cX-cX0)))
        self.dis = normalize(self.dis)
        self.frametime = self.frametime[self.offframe:len(self.dis)+self.offframe]
        plt.plot(self.frametime,self.dis,label="Dis")
        plt.scatter(self.frametime[-self.extend],self.dis[-self.extend],label="Full time")
        # plt.plot(self.frametime,self.area_list,label="Area")
        if self.plottarget :
            for i in self.ref_axis:
                plt.plot([i,i],[0,2])
        figure_title = "Cost Time:%s   Full time:%s" %(str(self.timecost),str(self.frametime[-self.extend]))
        plt.title(figure_title)
        plt.xlabel("Time(ms)")
        plt.ylabel("Value")
        plt.legend()
        plt.grid('on')
        plt.savefig(self.outImage+".png")
        plt.cla()                           
    
    def PCAdeco(self,data):
        pca = PCA(n_components=2)
        pca.fit(data)
        return pca.singular_values_
    
    def frame0(self,frame):
        fshape = frame.shape
        self.fheight = fshape[0]
        self.fwidth = fshape[1]
        self.frameArea = self.fheight*self.fwidth/np.power(2,2*self.pyrDownNum)
        
        self.videoWriter = cv2.VideoWriter(self.outVideo,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),30,(self.fwidth,self.fheight))
        self.bg  = pyrDown(frame, self.pyrDownNum)
        self.bg = cv2.bilateralFilter(self.bg, d=0, sigmaColor=100, sigmaSpace=15)
        
    def multiProccessing(self,video_file):
        t_start = time.time()
        self.discheck = np.array([])
        self.cap = cv2.VideoCapture(video_file)
        self.frames_num = int(self.cap.get(7))-1
        print ('Frame number of movie:',self.frames_num)
        first_frame = 0
    
        # Multi_thread
        threadn = cv2.getNumberOfCPUs()
        pool = ThreadPool(processes = threadn)
        # print ("deque",deque())
        pending = deque()
        print("CPU",threadn)
        # latency = StatValue()
        res_list = []
        frame_id = 0
        latency = StatValue()
        # frame_interval = StatValue()
        while(True):
            frame_id += 1

            if (frame_id > self.endframe+self.extend):
                
                break
            ret, frame = self.cap.read()
            if (frame_id < abs(self.offframe)):
                print (frame_id , abs(self.offframe))
                continue
            # print ("frame_id",frame_id)
            if first_frame == 0:
               self.frame0(frame)
               first_frame = 1
           
            while len(pending) > 0 and pending[0].ready():
                res= pending.popleft().get()
                # latency.update(clock() - t0)
                # cv2.imshow('threaded video', res)
                res_list.append(res)
            if len(pending) < threadn:
                
                if self.threaded_mode:
                    try:
                        task = pool.apply_async(self.frameprocess, (frame.copy(),frame_id))
                    except:
                        break
                    pending.append(task)
                else:
                    # ret, frame = self.cap.read()
                    res = self.frameprocess(frame.copy(),frame_id)
                    
                    cv2.imshow('threaded video', res)
                    res_list.append(res)
            if cv2.waitKey(150) & 0xff == 27:
                break
        for image in res_list:
            self.videoWriter.write(image)
        self.videoWriter.release()
        self.cap.release()
        cv2.destroyAllWindows()
        self.timecost = round(time.time() - t_start,1)
        print ("*****%s END*****" % (time.strftime("%X", time.localtime())))
        print ("*****%.3fs *****" % (self.timecost))

    
    def frameprocess(self,frame,frame_id):

        frame_copy =pyrDown(frame, self.pyrDownNum)
        
        frame_blur = cv2.bilateralFilter(frame_copy, d=0, sigmaColor=100, sigmaSpace=15)
        # print(imageA_blur.shape,frame_blur.shape)
        score,gra, diff = ssim(self.bg, frame_blur, full=True,gradient=True,gaussian_weights=True,win_size=3)
        # print("SSIM: {}".format(score))
        diff = (diff*255).astype('uint8')
        # cv2.imshow('diff',diff)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(diff,0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
        # thresh_dilate = cv2.dilate(thresh.copy(), None, iterations=1)
        # thresh_erode = cv2.erode(thresh, None, iterations=1)
        # thresh_dilate = cv2.dilate(thresh_erode, None, iterations=1)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        index_list =[]

        for index,c in enumerate(contours):
            # x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
    
           
            # 找出图像矩
            try:
                cX,cY = extractMxyContour(c)

                contourArea =cv2.contourArea(contours[index])
                if contourArea > self.frameArea/4:
                    break
                # print (contourArea)
                if cY < self.fheight/5. :
                    
                    # print (cY ,fheight/5)
                    areas.append(contourArea)
                    index_list.append(index)
            except:
                pass
            # 在图像上绘制轮廓及中心
        if len(areas)>0:
            if self.FlagRecord:
                self.frame0id = frame_id
                print ("Deployment start frame:",self.frame0id)
                self.FlagRecord = False
            # print (max(areas))
            max_id = index_list[areas.index(max(areas))]
            c = contours[max_id]
            cX,cY = extractMxyContour(c)
            self.discheck = np.append(self.discheck,distance(cX,cY))
            # self.endframe = checkTermination(self.discheck)
            # self.dis.append(distance(cX,cY))
            self.endframe =  checkTermination(self.discheck)
            self.drawPoint(frame,cX,cY)
            self.pca_contour.append(c)
            self.pca_contour_id.append(frame_id)
            self.drawContour(frame,c,0.001)
            self.drawRect(frame,c)
            self.drawMinareBox(frame,c)

        else:
            time.sleep(0)

        return frame
    
    def drawContour(self,frame,c,ratio):
        epsilon = ratio*cv2.arcLength(c,True) 
        approx = cv2.approxPolyDP(c,epsilon,True)
        cv2.drawContours(frame, [approx*self.scale],-1, (0, 255, 0), 1)
    
    def drawMinareBox(self,frame,c):
        # 得到最小矩形的坐标
        rect = cv2.minAreaRect(c)
        #
        box = cv2.boxPoints(rect)  
        box = np.int0(box)
        cv2.drawContours(frame, [box*self.scale],-1, (0, 255, 0), 1)
        return box
       
    def drawRect(self,frame,c):
        x, y, w, h = cv2.boundingRect(c*self.scale)
        
        cv2.rectangle(frame,pt1=(x, y), pt2=(x+w, y+h),color=(255, 255, 255), thickness=3)
        
    def drawPoint(self,frame,x,y):
        cv2.circle(frame, (x*self.scale, y*self.scale), 10,(0, 0, 255), 0)

if __name__ == "__main__":

    try:
        wkdir = sys.argv[1]
        dirs = FindFile(wkdir, '.avi')[0]
        print(wkdir)
    except:
        print('No file')
        wkdir = 'C:\\Users\\yujin.wang\\Desktop\\Codie\\Opencv\\CushionfromCTCTEST\\DAB-TEST'

    a = CushionTracking(wkdir,False,False,35,0.25,-10,8)
    a.run()
    
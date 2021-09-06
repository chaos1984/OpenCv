import cv2
import sys,os
import numpy as np
from skimage.measure import compare_ssim as ssim
import time
from multiprocessing.pool import ThreadPool
from collections import deque
from sklearn.decomposition import PCA
from common import FindFile
import matplotlib.pyplot as plt
from scipy.signal import find_peaks,peak_prominences,medfilt
import pytesseract


# def runmultifile(VideoDir):
#    a = CushionTracking(VideoDir,target=False,mp=True,resolution=0.025)
#    a.run()
   

def cv_show(name,img):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def OCR(img,h,w):
    img = img[int(3*h/4):h,0:w]
    # cv_show("img",img)
    ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ref = cv2.threshold(ref, 210, 255, cv2.THRESH_BINARY_INV)[1]
    text = pytesseract.image_to_string(ref,lang='eng')
    # print (text)
    text = text.replace("—","-")
    return text

def checkTermination(dis,resolution):
    dis = -1*dis/np.max(dis)+1
    peaks, _ = find_peaks(dis,prominence=resolution)
    resolution = dis[peaks] - 0.35
    if len(resolution)>0:
        if resolution[0]>0.2:
            print (peaks,dis[peaks])
            resolution[0] = 0.2
        elif resolution[0]<0.12:
            print (peaks,dis[peaks])
            resolution[0] = 0.12
        peaks, _ = find_peaks(dis,prominence=resolution[0])
        
    if len(peaks)>0:
        
        # prominences = peak_prominences(dis, peaks)[0]
        
        # print ("~~~~~~")
        # print (resolution[0])
        # print (peaks,len(dis))
        # print (_)
        # print (peaks)
        # print (dis)
        return len(dis)
    else:
        return 999
    
def checkTermination1(dis,resolution):
    dis = -1*dis/np.max(dis)+1
    peaks, _ = find_peaks(dis,prominence=0.2)
    if len(peaks)>0:
        print (peaks)
        if np.median(dis[-5:]) < np.median(dis[-7:-2]):
            print ('OK',len(dis)-100)
            return len(dis)-100
    else:
        return 999
    
def checkTermination2(dis,resolution):
    dis = -1*dis/np.max(dis)+1
    dis = medfilt(dis,3)
    peaks, _ = find_peaks(dis,prominence=resolution)
    # resolution = dis[peaks] - 0.35
    # if len(resolution)>0:
    #     if resolution[0]>0.2:
    #         print (peaks,dis[peaks])
    #         resolution[0] = 0.2
    #     elif resolution[0]<0.12:
    #         print (peaks,dis[peaks])
    #         resolution[0] = 0.12
    #     peaks, _ = find_peaks(dis,prominence=resolution[0])
        
    if len(peaks)>0:
        
        # prominences = peak_prominences(dis, peaks)[0]
        
        return True
    else:
        return False

def checkTerminate(data):
    # data = medfilt(data,3)
    fullframe = len(data)-1
    for i in range(len(data)):
        if data[-1-i]>data[-2-i] or data[-2-i]>data[-3-i] or data[-3-i]>data[-4-i]:
            fullframe = len(data)-1-i
        else:
            return fullframe
    
    
def extractMxyContour(c):
    M = cv2.moments(c)
    return int(M["m10"] / M["m00"]),int(M["m01"] / M["m00"])

def distance(x,y):
    return np.sqrt(x*x+y*y)

def normalize(data):
    try:
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) /_range
    except:
        print (data)

def pyrDown(currentframe,pyrDownNum):
     for i in range(pyrDownNum):
         currentframe = cv2.pyrDown(currentframe)
     return currentframe

class CushionTracking():
    def __init__(self,filedir,target=False,mp=True,resolution=0.03):
        self.filedir = filedir
        self.plottarget = target
        self.threaded_mode = mp
        # self.delta_t = delta_t
        # self.time_offset = offset
        self.resolution = resolution
        self.pyrDownNum = 2
        self.scale = np.power(2,self.pyrDownNum)
        self.FlagRecord= True
        # self.offframe = int(abs(self.time_offset)/self.delta_t)
        
        self.t0 = 0
        
    def run(self):
        for file in self.filedir:
            self.filename = os.path.basename(file)
            print ("\n*****%s start*****" % (time.strftime("%X", time.localtime())))
            print (self.filename)
            t_start = time.time()
            # print (file)
            self.outVideo = file.replace(self.filename,"_"+self.filename)
            self.outImage = file.replace(self.filename,self.filename[:-4]+".jpg")
    
            # self.target = self.VideoDir + "\\"+file.split("\\")[-1].replace("avi","txt")
            if self.plottarget == True:
                self.getTarget
            self.area_list = []
            self.contour = []
            self.contour_id = []
            self.multiProccessing(file)
            self.end = self.time_offset + self.frames_num*self.delta_t
            self.frametime = np.linspace(self.time_offset,self.end,int(self.frames_num)+1)
            print ("START:",self.time_offset,"END:",self.end,"NUM:",int(self.frames_num),"DELTA_T:",self.delta_t)
            # print ("tIME:",self.time_offset,"END:",self.end,"NUM:",int(self.frames_num),"DELT_T:",self.delta_t)
            self.timecost = round(time.time() - t_start,1)
            self.curveplot()
            print ("*****%s END*****" % (time.strftime("%X", time.localtime())))
            print ("*****%.3fs *****\n" % (self.timecost))  
            time.sleep(0.3)
            
            
        
    def curveplot(self):
        self.dis =[]
        contour_len = len(self.contour)
        print ('Before deployment frame:',self.frame0id - self.offframe,"After deployment frame:",contour_len)
        frameNumBefore = int(self.frame0id - self.offframe-1)
        for i in range(frameNumBefore):
            self.dis.append(0)
        for i in range(contour_len):
            contour_frame = self.contour[i] 
            cX,cY = extractMxyContour(contour_frame)
            contour_frame =[j[0] for j in contour_frame]
            self.dis.append(distance(abs(cX-self.cX0),abs(cX-self.cX0)))

        self.dis = normalize(self.dis)
        self.frametime = self.frametime[self.offframe:len(self.dis)+self.offframe]
        fullframe = checkTerminate(self.dis)
        print ("**Full time**: %3.2f" %(self.frametime[fullframe]))
        figure_title = "Time:%s   Full time  %3.2f " %(str(self.timecost),self.frametime[fullframe])
        plt.title(figure_title)
        plt.plot(self.frametime,self.dis,label="Dis")
        plt.scatter(self.frametime[fullframe],self.dis[fullframe],label="Full time")
        plt.xlabel("Time(ms)")
        plt.ylabel("Value")
        plt.legend()
        plt.grid('on')
        plt.savefig(self.outImage)
        plt.cla()                        
        # return (self.frametime,self.dis,self.frametime[-self.extend],self.dis[-self.extend])                    
    
    def PCAdeco(self,data):
        pca = PCA(n_components=2)
        pca.fit(data)
        return pca.singular_values_
    
    def getVideoInfo(self,frame):
        fshape = frame.shape
        self.fheight = fshape[0]
        self.fwidth = fshape[1]
        self.text = OCR(frame,self.fheight,self.fwidth)
        self.text = self.text.split("\n")
        self.text = self.text[1].split(" ")
        self.delta_t = float(self.text[3])/float(self.text[2])
        self.offframe = abs(int(self.text[2]))
        self.time_offset =-1* self.delta_t * self.offframe 
        print ("**Figure data**")
        print ("Delta_t: %f Offset frame: %d" %(self.delta_t,self.offframe) )
    
    def frame0(self,frame):
        self.getVideoInfo(frame)
        self.videoWriter = cv2.VideoWriter(self.outVideo,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),30,(self.fwidth,self.fheight))
        self.frameArea = self.fheight*self.fwidth/np.power(2,2*self.pyrDownNum)
        self.bg  = pyrDown(frame, self.pyrDownNum)
        self.bg = cv2.bilateralFilter(self.bg, d=0, sigmaColor=100, sigmaSpace=15)
        
    def multiProccessing(self,video_file):
       
        self.discheck = np.array([])
        self.flagterminat =False
        self.cap = cv2.VideoCapture(video_file)
        self.frames_num = int(self.cap.get(7))-1
        print ('Frame number of movie:',self.frames_num)
        ret, frame = self.cap.read()
        self.frame0(frame)
        
    
        # Multi_thread
        threadn = cv2.getNumberOfCPUs()
        pool = ThreadPool(processes = threadn)
        pending = deque()
        print("CPU",threadn)
        res_list = []
        frame_id = 0

        while(True):
            frame_id += 1
            # print ("-1")
            # print (-1)
            if (self.flagterminat):
                
                break
            
            ret, frame = self.cap.read()
            if (frame_id < abs(self.offframe)+1):
                continue
           
            while len(pending) > 0 and pending[0].ready():
                res= pending.popleft().get()
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
                    
                    # cv2.imshow('threaded video', res)
                    res_list.append(res)
            if cv2.waitKey(150) & 0xff == 27:
                break
        for image in res_list:
            self.videoWriter.write(image)
        self.videoWriter.release()
        self.cap.release()
        cv2.destroyAllWindows()


    
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
            self.contour.append(c)
            cX,cY = extractMxyContour(c)
            if len(self.contour)==1:
                self.cX0,self.cY0 = extractMxyContour(self.contour[0])
            self.discheck = np.append(self.discheck,distance(cX-self.cX0,cY-self.cY0))
            # -1 = checkTermination(self.discheck)
            # self.dis.append(distance(cX,cY))
            self.flagterminat = checkTermination2(self.discheck,self.resolution)
            self.drawPoint(frame,cX,cY)
            # self.contour_id.append(frame_id)
            self.drawContour(frame,c,0.001)
            # self.drawRect(frame,c)
            # self.drawMinareBox(frame,c)

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
        wkdir = 'C:\\Users\\yujin.wang\\Desktop\\Codie\\Opencv\\CushionfromCTCTEST\\DAB_TEST3'
        
    if wkdir[-3:] != "avi":
            VideoDir = FindFile(wkdir , '.avi')[0]
            for file in VideoDir:
                print (file)
                a = CushionTracking([file],target=False,mp=True,resolution=0.02)
                a.run()
    else:
            VideoDir = [wkdir] 
            a = CushionTracking(VideoDir,target=False,mp=True,resolution=0.025)
            a.run()
    # a = CushionTracking(wkdir,target=False,mp=True,resolution=0.025)
    # a.run()
    # curve_x = a.frametime
    # curve_y = a.dis
    # scatter_x = a.frametime[-1]
    # scatter_y = a.dis[-1]
    
    
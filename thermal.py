import cv2
import numpy as np

class NormImage():
    def __init__(self, data):
        self.data = data
    
    def show_median_filtered(self, name):
        img = 255 * (self.data - self.data.min()) / (self.data.max() - self.data.min())
        img = np.uint8(img)
        img = cv2.medianBlur(img, 3)
        #ret, img = cv2.threshold(img, 35, 200, cv2.THRESH_BINARY)
        cv2.imshow(name, img)
        print('{} distribution'.format(name))
        print('mean', np.mean(img))
        print('std', np.std(img))
    
    def show(self, name):
        #image_norm = 255 * (self.data - self.data.min()) / (self.data.max() - self.data.min())
        #cv2.imshow(name, image_norm/255)
        cv2.imshow(name, cv2.normalize(self.data, 0, 255, cv2.NORM_MINMAX))
    
    def save(self, filename):
        ''' save min-max normalized image '''
        image_norm = 255 * (self.data - self.data.min()) / (self.data.max() - self.data.min())
        np.array(image_norm, np.int)
        cv2.imwrite('dump/{}'.format(filename), image_norm)
    
    def stats(self, name):
        print('{} distribution'.format(name))
        print('mean', np.mean(self.data))
        print('std', np.std(self.data))
        print('min/max', self.data.min(), self.data.max())
        

class Video():
    def __init__(self, file, cx, cy, cw):
        self.file = file
        self.cx = cx
        self.cy = cy
        self.cw = cw
        self.cw2 = int(cw/2)
    
    def x(self):
        return self.cx-self.cw2
    
    def y(self):
        return self.cy-self.cw2
    
    def w(self):
        return self.cw

class VideoCrop():
    def __init__(self, video, start, duration=20):
        self.video = video
        self.start = start
        self.end = start+duration

class VideoCropProcessor():
    def __init__(self, video_crop):
        self.video = video_crop.video
        self.video_crop = video_crop
    
    def run(self):
        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name
        cap = cv2.VideoCapture('videos/{}.mp4'.format(self.video.file))
         
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        
        # set starting point:
        cap.set(cv2.CAP_PROP_POS_MSEC, self.video_crop.start*1000)    
        
        # Read until video is completed
        w = self.video.w()
        counter = 0
        self.previous = None
        self.integrated = np.zeros((w, w), dtype='float64')
        img_integrated = NormImage(self.integrated)
        gradients = []
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_msec <= self.video_crop.start*1000:
                continue
            if pos_msec > self.video_crop.end*1000:
                print('end of {} @ {}s'.format(self.video.file, self.video_crop.start))
                print('gradient distributions')
                print('mean', np.mean(gradients))
                print('std', np.std(gradients))
                img_integrated.stats('integrated')
                print('---')
                img_integrated.save('integrate-{}-{}s.png'.format(self.video.file, self.video_crop.start))
                break
            
            # bw frame:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # crop frame:
            x = self.video.x()
            y = self.video.y()
            frame = frame[y:y+w, x:x+w]/255
            
            # blur:
            #frame = cv2.GaussianBlur(frame, (13, 13), 0)
            
            if self.previous is None:
                self.previous = frame.copy()
            else:
                counter += 1        
                gradient = np.abs(frame - self.previous)
                gradients.append(np.sum(gradient))
                #print('sum', gradients[-1])
                
                self.integrated += gradient
                
                # update previous
                self.previous = frame.copy()
                
                # Display the resulting frame
                img_integrated.show('Frame-{}'.format(self.video_crop.start))
                
                # Press Q on keyboard to  exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                  break
         
        # When everything done, release the video capture object
        cap.release()

# Define videos:

# Video 3:
# 15-35s: pause
# 60-80s: fire
# 85-105s: wild fire
# 140s-160s: cool down
# center: 981/502 @ 360px
video3 = Video('MOV_0090', 981, 502, 360)
video3_no_fire = VideoCrop(video3, 15)
video3_fire = VideoCrop(video3, 60)
video3_wild_fire = VideoCrop(video3, 85)
video3_cooldown1 = VideoCrop(video3, 120)
video3_cooldown2 = VideoCrop(video3, 140)

no_fire = VideoCropProcessor(video3_no_fire)
fire = VideoCropProcessor(video3_fire)
cooldown = VideoCropProcessor(video3_cooldown1)

# Video 2:
# 15-35s: pause
# 50-65: fire
# 65-85: wild fire
# 120-end: cooldown
video2 = Video('MOV_0089', 1005, 477, 614)
video2_no_fire = VideoCrop(video2, 15, duration=5)
video2_fire = VideoCrop(video2, 50, duration=5)
video2_cooldown = VideoCrop(video2, 140, duration=5)

#no_fire = VideoCropProcessor(video2_no_fire)
#fire = VideoCropProcessor(video2_fire)
#cooldown = VideoCropProcessor(video2_cooldown)

### run all 3 and show differences between fire and no fire ###
no_fire.run()
fire.run()
cooldown.run()

# show a single frame
#single_frame = NormImage(no_fire.previous)
#single_frame.show('single')
#cv2.waitKey(300)

diff_thermal = NormImage(np.abs(fire.integrated - no_fire.integrated))
diff_thermal.show('diff_thermal')
diff_thermal.save('diff_thermal.png')
diff_thermal.stats('diff_thermal')
diff_no_thermal = NormImage(np.abs(cooldown.integrated - no_fire.integrated))
diff_no_thermal.show('diff_no_thermal')
diff_no_thermal.save('diff_no_thermal.png')
diff_no_thermal.stats('diff_no_thermal')
        

cv2.waitKey(300)
 
# Closes all the frames
#cv2.destroyAllWindows()

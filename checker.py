import cv2
import os
import glob

def inone():
    sequence='30'
    path='../data/first/birdview/old/'
    impath = os.path.join(path, sequence)
    list = glob.glob(impath + '/*.png')
    cv2.setWindowTitle("show", "1")
    for i in list:
        label = i.replace('png', 'txt')
        img = cv2.imread(i)
        w, h, c = img.shape
        f = open(label)
        lines = f.readlines()
        f.close()
        for l in lines:
            l = l.split(" ")
            xmid = float(l[1])
            ymid = float(l[2])
            width = float(l[3])
            height = float(l[4])
            xmin = int((xmid - width / 2) * w)
            ymin = int((ymid - height / 2) * h)
            xmax = int((xmid + width / 2) * w)
            ymax = int((ymid + height / 2) * h)
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.imshow("show", img)
        cv2.waitKey()
    cv2.destroyAllWindows()

#inone()

def checklabel():
    #num = 'first'
    num = 'second'
    for sequence in range(22,23):
        impath = 'F:/daxie/daxie/data/%s/birdview/feature_3c/%d' % (num, sequence)
        labelpath = 'F:/daxie/daxie/data/%s/labels/%d' % (num, sequence)

        #impath=os.path.join(impath,sequence)
        #labelpath=os.path.join(labelpath,sequence)
        list=os.listdir(impath)
        list.sort(key=lambda x:int(x[:-4]))
        print(sequence)
        cv2.setWindowTitle("show","1")
        for i in list:
            img=cv2.imread(os.path.join(impath,i))
            w,h,c=img.shape
            labelfile=labelpath+'/'+i
            labelfile=labelfile.replace('png', 'txt')
            if not os.path.exists(labelfile):
                #print(labelfile)
                open(labelfile,'w')

            f=open(os.path.join(labelpath,i.replace('png','txt')))
            lines=f.readlines()
            f.close()
            for l in lines:
                l=l.split(" ")
                xmid=float(l[1])
                ymid = float(l[2])
                width=float(l[3])
                height=float(l[4])
                xmin=int((xmid-width/2)*w)
                ymin = int((ymid - height / 2) * h)
                xmax = int((xmid + width / 2) * w)
                ymax = int((ymid + height / 2) * h)
                img=cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,0,0),2)
            cv2.imshow("show",img)
            cv2.waitKey()
        cv2.destroyAllWindows()

checklabel()










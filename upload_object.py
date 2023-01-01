import yaml
import os
import cv2
import time 
class Upload_Object:

    # 放入標籤 , train的照片數量 , val的照片數量
    def __init__(self,opt):
        
        ratio = opt.split
        self.tar_path = opt.save
        self.root = opt.root
        self.label = opt.label
        self.train_data_num = opt.num*ratio
        self.val_data_num = opt.num* (1-ratio)
        self.index = int(self.read_yaml()['nc'])
        self.percentage = (1-opt.region)/2
        self.source = opt.source

    # 讀取yaml
    def read_yaml(self):
        
        with open(os.path.join(self.root,"./dataset.yaml"),'r',encoding= 'utf-8') as f:
            data = yaml.load(f,Loader=yaml.FullLoader)
        return data
    
    # 更新yaml
    def update_yaml(self):
        
        data = self.read_yaml()
        
        # 新增本次新的物品標籤
        data["names"].append(self.label)
        # 更新nc數量 , 為加1
        data['nc'] = len(data["names"])
        # 寫檔
        with open(os.path.join(self.root,"./dataset.yaml"),'w',encoding= 'utf-8') as f:
            yaml.dump(data,f)
    
    # 開始偵測的部份
    def start_detect(self):
        
        # 開啟鏡頭
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        # 計次數
        num = 1 

        # 開始抓取圖片並分類到對應的檔案
        while True:
            img_path = f"{self.label}_{num}"
            ret , frame = cap.read()
            if not ret:
                print("Cannot receive Frame")

            # 建立資料夾
          
            if not os.path.isdir(self.tar_path):
                os.mkdir(self.tar_path)
                os.mkdir(os.path.join(self.tar_path,"./images"))
                os.mkdir(os.path.join(self.tar_path,"./images/train"))
                os.mkdir(os.path.join(self.tar_path,"./images/val"))
                os.mkdir(os.path.join(self.tar_path,"./labels"))
                os.mkdir(os.path.join(self.tar_path,"./labels/train"))
                os.mkdir(os.path.join(self.tar_path,"./labels/val"))            
            # 根據計數到對應的資料夾
            if num <= self.train_data_num:
                mode ="train/"
            else:
                mode = "val/"
            
            # 根據圖片大小做label , 取以中央80%的物件
            with open(os.path.join(self.tar_path,f"./labels/{mode}{img_path}.txt") , "w") as f:
                
                
                h ,w ,d = frame.shape
                x_min , y_min , x_max ,y_max= w*self.percentage ,h*self.percentage, w*(1-self.percentage) ,  h*(1-self.percentage) 
                
                x_center = float(x_min+x_max)/2 * float(1.0/w)
                y_center = float(y_min + y_max)/2.0 *float(1.0/h)
                yolo_w = float(x_max-x_min)/w
                yolo_h = float(y_max-y_min)/h
                
                str = f"{self.index} {x_center} {y_center} {yolo_w} {yolo_h}\n"
                
                f.write(str)
            img =frame.copy()
            img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), cv2.LINE_AA)
            # DEBUG用 ,顯示 
            cv2.imshow("frame",img)
           
            # 存圖
            cv2.imwrite(os.path.join(self.tar_path,f"./images/{mode}{img_path}.png"),frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.5)
            num+=1
            # 達到指定數量,結束
            if num > self.train_data_num+self.val_data_num:
                break

        # DEBUG用 , 關閉顯示視窗
        cv2.destroyAllWindows()
        # 更新yaml檔
        self.update_yaml()
        
        

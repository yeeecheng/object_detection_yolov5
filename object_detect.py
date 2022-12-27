import torch
import cv2
import pandas
import json
import subprocess
import os
import argparse

def detect(opt):
    
    yolov5_path = os.path.join(opt.root,"./yolov5")
    weight_path = opt.weight
    
    model = torch.hub.load(yolov5_path, 'custom', weight_path, source='local' )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    while True:
        
        ret,frame = cap.read()
        if not ret:
            print("Cannot receive Frame")
        
        img_path = "./detect.png"
        frame = cv2.resize(frame,(320,320))
        cv2.imwrite(img_path,frame)
        # IoU門檻值
        model.iou = opt.iou
        # 信心門檻值
        model.conf = opt.conf
        # 辨識
        results =model(img_path,size=(320,320))
        # Results
        results.print()  
        results.xyxy[0]  
       
        # 轉化成json
        results =results.pandas().xyxy[0].to_json()
        results =json.loads(results)
            
        # # 存json
        with open(opt.save,"w") as f:
            json.dump(results,f,ensure_ascii=False)
        

def main(opt):
    detect(opt)
    
def parse_opt(known=False):
   
    ROOT =os.getcwd()
    parser =argparse.ArgumentParser()
    parser.add_argument("--root",type=str , default=ROOT )
    parser.add_argument("--weight",type=str , default=os.path.join(ROOT,"./best-int8_edgetpu.tflite") )
    parser.add_argument("--iou",type=float,default=0.3)
    parser.add_argument("--conf",type=float,default=0.5)
    parser.add_argument("--save",type=str ,default=os.path.join(ROOT,'./detect_result.json'))
    return parser.parse_args()[0] if known else parser.parse_args()


def run(**kwargs):

    opt = parse_opt(True)
    for k,v in kwargs.items():
        setattr(opt,k,v)
    main(opt)
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


#subprocess.run("python3 ./yolov5/detect.py --weight ./best-int8_edgetpu.tflite --data ./dataset.yaml --imgsz 320 --iou-thres 0.3 --conf-thres 0.6 --source 0".split(" "))
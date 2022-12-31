import  upload_object
import argparse
import os


def main(opt):
    
    # 拍圖片 , 更新yaml
    UO =upload_object.Upload_Object(opt)
    UO.start_detect()



def parse_opt(known=False):
    ROOT =os.getcwd()
    parser =argparse.ArgumentParser()
    parser.add_argument("--root",type=str , default=ROOT )
    parser.add_argument("--save",type=str,default=os.path.join(ROOT,"./object_detect_dataset"),help="the path where dataset save")
    parser.add_argument("--label",type=str ,default="obj",help="new obj label")
    parser.add_argument("--num",type=int,default=400,help="the number you want to train")
    parser.add_argument("--split",type=float , default=0.9 ,help="the ratio you want to split the dataset")
    parser.add_argument("--region",type=float ,default=0.8,help="want to capture img region as label")
    parser.add_argument("--source",type=int,default=0,help="camera")
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
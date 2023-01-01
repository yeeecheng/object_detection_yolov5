# Objectt Detection

## 注意 
* #### 檔案中的best-int8_edgetpu.tflite為使用imgsz320 batch 16 epcoh 20 
* #### detect寫入txt的內容，如果沒有偵測到會是空的，有偵測到東西就會呈現對應的label，句末沒有加"\n" 
* #### 你需要把path的地方改成你目前這個檔案的位置<br>例如server路徑為C://Desktop/Convert_TFlite_Server，那就需要將path的地方更改為C://Desktop/Convert_TFlite_Server

![image](https://user-images.githubusercontent.com/88101776/210137823-266ce6fb-4569-4427-a7ea-46d902a9d677.png)


## 註冊新的物件

```cmd
python3 logIn_obj.py --label <label_name> 
```

## 物件偵測

```cmd
python3 object_detect.py --weight <weight> 
```
## 開啟client

```cmd
python3 client.py --root <根目錄> --data <要訓練資料集的路徑 --ip <連線ip addr>  --port <連線port>
```

# crossroad_TrafficCount

![aidrone2](https://user-images.githubusercontent.com/78409601/109888906-b57aeb80-7cc7-11eb-8423-6380a3acba35.png)

위의 서비스는 [yolov5](https://github.com/ultralytics/yolov5) 모델을 기반으로 제작되었습니다.
*UPDATE: 4.0ver에 적용되었습니다.

서비스 모델을 사용하기 위해서는 git의 코드를 다운받아야 합니다.
```bash
$ git clone https://github.com/metabuild-radarAI/crossroad_TrafficCount.git
$ cd crossroad_TrafficCount
```

conda 가상환경 및 pip을 이용하여 환경을 세팅할수 있도록 하였으며 conda를 추천합니다.
가상환경 설치는 아래와 같습니다.
```bash
$ conda env create -f crossroad_service.yml
```

서비스를 사용하기 위해 학습하는 명령어는 아래와 같습니다.
```bash
$ python train.py --weights yolov5s.pt --data (data 경로) --cfg models/yolov5s.yaml --epochs 300
```

다중 GPU를 풀 가동하기 위해서는 아래의 명령어로 사용합니다.
```bash
$ python -m torch.distributed.launch --nproc_per_node (gpu 개수) train.py --weights yolov5s.pt --data (data 경로) --cfg yolov5s.yaml --epochs 300
```

서비스 사용 방법중 동영상의 경우 아래와 같습니다.
```bash
$ python detect_aidrone.py --weights 300ep_uhd_c6_best.pt --source drone_video2.MP4
```
이제부터 기존의 학습 파일은 [여기](https://url.kr/nk8e6q)에서 다운받을 수 있습니다.
또한 예제의 영상은 [여기](https://url.kr/3o94sm)에서 다운받을 수 있습니다.

학습한 [pt파일](https://url.kr/nk8e6q)과 예제에 사용된 [영상](https://url.kr/3o94sm)은 링크에 첨부되어있습니다.

영상의 Counting 영역을 지정하기 위한 오토툴은 현재 구상중에 있습니다.
그 전까지는 영상을 실행하여서 w키를 눌러서 이미지를 저장시키며 aidrone.png를 통해 영역의 좌표를 직접 계산하여야 합니다.




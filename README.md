# crossroad_TrafficCount

<img src="https://github.com/metabuild-radarAI/crossroad_TrafficCount/tree/main/images/aidrones2.png" width="1000">

위의 서비스는 yolov5 모델을 기반으로 제작되었다.

https://github.com/ultralytics/yolov5

가상환경 설치는 아래와 같다.
```bash
$ conda env create -f crossroad_service.yml
```

서비스 사용은 동영상의 경우 아래와 같다.
```bash
$ python detect_aidrone.py --weights 300ep_uhd_c6_best.pt --source drone_video2.MP4
```

test 문서

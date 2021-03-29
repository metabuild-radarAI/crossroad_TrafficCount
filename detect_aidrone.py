import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
import random
from scipy import stats

from models.experimental import attempt_load
from utils.datasets_aidrone import LoadStreams, LoadImages_aidrone
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging, box_iou, bbox_iou)
from utils.torch_utils import select_device, load_classifier, time_synchronized

#4k영상을 기준으로 작성되었음을 알려드립니다.

def image_with_color(size):
    color_image = np.zeros(shape=[size[0], size[1]*2, 3], dtype=np.uint8)
    return color_image

def overlay(front_image, back_image, position=(0, 0)):
	x_offset, y_offset = position
	back_image[y_offset:y_offset + front_image.shape[0], x_offset:x_offset + front_image.shape[1]] = front_image
	return back_image

def init_track(tracks, results, id_count, conf_thres):
    for item in results:
        if item['score'] > conf_thres:
            id_count += 1
            item['age'] = 1
            item['id'] = id_count
            #class 추적
            item['cls'] = item['clss'][-1]
            if not ('ct' in item):
                bbox = item['xyxy']
                item['ct'] = [(bbox[0] + bbox[2]) / 2,
                                (bbox[1] + bbox[3]) / 2]
            tracks.append(item)
    return tracks, id_count

def predict(t, past_location, v_x,v_y):
    prediction_x = int(float(past_location[0]) + (t*v_x))
    prediction_y = int(float(past_location[1]) + (t*v_y))
    return prediction_x,prediction_y

def alpha_beta_tracking(detect_bbox, track_bbox, alpha=1, beta=0.1):
    delta_t = detect_bbox['time'] - track_bbox['time']
    if delta_t == 0:
        delta_t = 1
    expected_location_x,expected_location_y = predict(delta_t, track_bbox['ct'],track_bbox['v_x'],track_bbox['v_y'])
    error_x = int(detect_bbox['ct'][0]) - expected_location_x
    error_y = int(detect_bbox['ct'][1]) - expected_location_y
    track_bbox['ct'][0] = expected_location_x + alpha * error_x
    track_bbox['ct'][1] = expected_location_y + alpha * error_y
    track_bbox['v_x'] = float(track_bbox['v_x']) + (beta / delta_t) * error_x
    track_bbox['v_y'] = float(track_bbox['v_y']) + (beta / delta_t) * error_y
    track_bbox['time'] = detect_bbox['time']
    track_bbox['age'] +=1
    track_bbox['xyxy'] = detect_bbox['xyxy']
    
    #clss 누적
    if len(track_bbox['clss']) == 9:
        track_bbox['clss'][0] = detect_bbox['clss'][0]
    else : 
        track_bbox['clss'].append(detect_bbox['clss'][0])
    track_bbox['cls'] = int(stats.mode(track_bbox['clss'])[0])
    '''
    #class 단일
    track_bbox['cls'] = track_bbox['clss'][0]
    '''
    return track_bbox

def set_det(classlist):
    for i in range(len(classlist)):
        classlist[i]=list(set(classlist[i]))
    return classlist

def area_count(area, detect_box, id_list):
    ct = 0
    for key, value in area.items():
        if (value[0]<= detect_box['ct'][0] <= value[2]) and (value[1]<=detect_box['ct'][1]<=value[3]):
            id_list[int(detect_box['cls'])][ct].append(detect_box['id'])
            id_list[int(detect_box['cls'])][ct] = list(set(id_list[int(detect_box['cls'])][ct]))
        ct+=1
    return id_list

# def area_list(id_list,n=1):
#     a=[]
#     b=[]
#     for i in range(len(id_list)):
#         a.append(len(id_list[i][n-1]))
#         b.append(len(id_list[i][n+3]))
#     return a,b

def area_list2(id_list,m=0,n=0):
    return len(id_list[m][n])

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    
    #Time
    prevTime = 0
    curTime = 0
    prevTime1 = 0
    
    #210223 Tracker 
    alpha=1
    beta=0.1
    v_x=1
    v_y=1
    id_count=0
    ms = 0
    conf_thres = opt.conf_thres
    tracks = []
    circle_tracks=[]
    frame_ori = 0
    
    #AREA
    area_s={}
    area_s['west out'] = [5,162,142,276]
    area_s['west in'] = [5,277,142,434]
    area_s['south out'] =[277,474,484,535]
    area_s['south in'] =[485,474,672,535]
    area_s['east out'] =[843,294,955,371]
    area_s['east in'] =[843,138,955,293]
    area_s['north out'] =[509,5,693,78]
    area_s['north in'] =[310,5,508,78]
    
    area_t=[]
    for i in range(6):
        area_c = []
        for j in range(8):
            area_c.append([])
        area_t.append(area_c)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        #210118 화면크기 배율 설정 영역
        fx = 0.25
        fy = 0.25
        
        save_img = True
        dataset = LoadImages_aidrone(source, img_size=imgsz, fx = fx, fy = fy)
        #print(dataset)
        vid0 = cv2.VideoCapture(source)
        ret00, frame00 = vid0.read()
        video_size = frame00.shape
        video_size2 = [int(video_size[0]*fy), int(video_size[1]*fx), 3]
        bg_img = image_with_color(video_size2) # x축 2배로 그림        
        
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    #f = open('tracks.txt', 'w')
    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz,  imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        frame_ori += 1
        if frame_ori % 2 ==0:
            #curTime = time.time()
            over_img = overlay(im0s, bg_img, position=(int(video_size[1]/2), 0))
            cv2.imshow(path, over_img)
            vid_writer.write(over_img)
            pass
        else :
            # print(frame_ori)
            curTime1 = time.time()
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                ms += 1
                # if ms % 2 ==0 : 
                #     continue
                # else :
                #print(ms)
                curTime = time.time()
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                #0257
                person_count, car_count, bus_count, truck_count,bike_count, etc_count = 0,0,0,0,0,0
                bg_img[0:540,0:480] = np.zeros((540,480,3),np.uint8)
                bg_img[0:540,1440:1920] = np.zeros((540,480,3),np.uint8)
                #210118 우측 배경
                bg_img[:,int(video_size2[1]*1.5):] = np.zeros((int(video_size2[0]),int(video_size2[1]/2),3),np.uint8)

                #
                p_z = 'Person : '
                c_z = 'Car : '
                b_z = 'Bus : '
                t_z = 'Truck : '
                bi_z = 'Bike : '
                e_z = 'etc : '
                cv2.putText(bg_img, p_z, (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                cv2.putText(bg_img, c_z, (0,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                cv2.putText(bg_img, b_z, (0,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                cv2.putText(bg_img, t_z, (0,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                cv2.putText(bg_img, bi_z, (0,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                cv2.putText(bg_img, e_z, (0,180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                #cv2.putText(bg_img, str(ms), (0,240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                
                
                # if ms == 513:
                #     raise StopIteration
                
                
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
                        #numdetect += '%ss : %g \n' % (names[int(c)], n)  # count number in image 숫자세기 
                    
                    #태깅용 카운트
                    car_count = (det[:,-1]==0).sum()
                    truck_count = (det[:,-1]==1).sum()
                    bus_count = (det[:,-1]==2).sum()
                    person_count = (det[:,-1]==3).sum()
                    bike_count = (det[:,-1]==4).sum()
                    etc_count = (det[:,-1]==5).sum()
                    
                    p_d = 'Person : %g' % (person_count)
                    c_d = 'Car : %g' % (car_count)
                    b_d = 'Bus : %g' % (bus_count)
                    t_d = 'Truck : %g' % (truck_count)
                    bi_d = 'Bike : %g' % (bike_count)
                    e_d = 'etc : %g' % (etc_count)
                    detections = []
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        detection={}
                        detection['xyxy'] = xyxy
                        detection['coord'] = [int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])]
                        detection['ct'] = [int((xyxy[0] + xyxy[2]) / 2),int((xyxy[1] + xyxy[3]) / 2)]
                        detection['score'] = conf
                        detection['clss'] = [int(cls)]
                        detection['time'] = ms
                        detection['v_x'] = 1
                        detection['v_y'] = 1
                        detections.append(detection)
                        #print(xyxy)

                    if ms == 1:
                        tracks, id_count = init_track(tracks, detections,id_count,conf_thres)
                    #print('\n')
                    #print(ms, tracks,'\n', detections)
                    #print(float(bbox_iou(torch.tensor(detections[0]['xyxy']),torch.tensor(tracks[0]['xyxy']))))
                    #print(torch.tensor(detections[0]['xyxy']),torch.tensor(tracks[0]['xyxy']))

                    dets = []
                    new_tracks = []
                    tracks2 = []
                    t11 = time.time()

                    for det in detections :
                        iou_test=[]
                        for i in range(len(tracks)):
                            iou_test.append(float(bbox_iou(torch.tensor(det['xyxy']),torch.tensor(tracks[i]['xyxy']))))
                        if len(iou_test)==0:
                            continue
                        else :
                            max_iou = max(iou_test)
                            
                        if max_iou > 0.1:
                            iou_index = iou_test.index(max_iou)
                            new_track = alpha_beta_tracking(det, tracks[iou_index], alpha=alpha, beta=beta)
                            new_tracks.append(new_track)
                            del tracks[iou_index]
                        else :
                            dets.append(det)
                    
                    tracks2, id_count = init_track(tracks2, dets,id_count,conf_thres)
                    # print('*************************************************************')
                    # print(new_tracks)        
                    # print('-------------------------------------------------------------')
                    # print(tracks2)        
                    # print('ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo')
                    # print(tracks)
                    
                    if len(tracks) > 0:
                        new_tracks = new_tracks # + tracks
                    #print(len(new_tracks),len(tracks2), id_count)
                    tracks = new_tracks + tracks2
                    # print('ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo')
                    # print(tracks)
                    circle_tracks.append(tracks)
                    if len(circle_tracks)>10:
                        del circle_tracks[0]
                    # print('33333333',circle_tracks, len(circle_tracks))
                    '''    
                    for i in circle_tracks:
                        for j in circle_tracks[i]:
                            cv2.circle(im0, circle_tracks[i][j]['ct'], 1, (0,0,255), -1)
                    '''
                    #
                    
                    '''
                    f.write('\n')
                    f.write(str(new_tracks))
                    f.write('\n')
                    f.write(str(tracks2))
                    f.write('\n')
                    f.write(str(tracks))
                    f.write('\n')
                    '''
                    
                    #print(ms, '\n', detections, '\n', new_tracks, '\n', tracks2, '\n', tracks)
                    #print(ms,time.time()-t11, len(tracks))
                    
                    #for det in detections:
                    for det in tracks:
                        xyxy,conf,cls = det['xyxy'], det['score'], det['cls']
                        #영역 카운팅 알고리즘
                        area_t = area_count(area_s, det, area_t)
                        #print('value,',len(tracks),':', area_t)
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(txt_path + '.txt', 'a') as fd:
                                fd.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                        if save_img or view_img:  # Add bbox to image
                            #label = '%s %.2f' % (names[int(cls)], conf)
                            #확률 삭제
                            if cls==0 or cls==1 or cls==2 or cls==3 or cls==4 or cls==5:
                                label = '%s, %s, %s' % (det['id'], names[int(cls)], det['age'])
                                #좌표확인용
                                #label = '%s, %s, %s, %s, %s' % (det['id'], names[int(cls)], det['age'], det['ct'][0], det['ct'][1])
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2) # 네모굵기
                                
                                #좌측상단에 태깅 추가
                                cv2.putText(bg_img, p_d, (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                                cv2.putText(bg_img, c_d, (0,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                                cv2.putText(bg_img, b_d, (0,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                                cv2.putText(bg_img, t_d, (0,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                                cv2.putText(bg_img, bi_d, (0,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                                cv2.putText(bg_img, e_d, (0,180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

                x1 = 10 #class 들여쓰기
                x2 = 12 #방향 들여쓰기
                for raw in range(int(len(area_t)/2)+1): #0~(6/2+1)
                    for col in range(len(area_t[0])+1): #0~9
                        c1 = (1453+80*raw,3+60*col)
                        c2 = (1530+80*raw,60+60*col)
                        if raw ==0 and col ==0:
                            cv2.rectangle(bg_img, c1,c2,(180,180,180),-1,cv2.LINE_AA)
                            
                        else:
                            if raw ==1 and col == 0:
                                cv2.rectangle(bg_img, c1,c2,(255,255,255),-1,cv2.LINE_AA)
                                cv2.putText(bg_img,' Car', (c1[0]+x1,c2[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 3/5, (0,0,0),1,lineType=cv2.LINE_AA)
                                    
                            elif raw ==2 and col == 0:
                                cv2.rectangle(bg_img, c1,c2,(255,255,255),-1,cv2.LINE_AA)
                                cv2.putText(bg_img,'Truck', (c1[0]+x1,c2[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 3/5, (0,0,0),1,lineType=cv2.LINE_AA)    
                                
                            elif raw ==3 and col == 0:
                                cv2.rectangle(bg_img, c1,c2,(255,255,255),-1,cv2.LINE_AA)
                                cv2.putText(bg_img,' Bus', (c1[0]+x1,c2[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 3/5, (0,0,0),1,lineType=cv2.LINE_AA)    
                                
                            elif col == 1:
                                cv2.rectangle(bg_img, c1,c2,(120,120,120),-1,cv2.LINE_AA)
                                if raw ==0:
                                    cv2.putText(bg_img,'West', (c1[0]+x2,c1[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 3/5, (0,0,0),1,lineType=cv2.LINE_AA)
                                    cv2.putText(bg_img,' Out', (c1[0]+x2,c1[1]+45), cv2.FONT_HERSHEY_SIMPLEX, 3/5, (0,0,0),1,lineType=cv2.LINE_AA)

                            elif col == 2:
                                cv2.rectangle(bg_img, c1,c2,(120,120,120),-1,cv2.LINE_AA)
                                if raw ==0:
                                    cv2.putText(bg_img,'West', (c1[0]+x2,c1[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 3/5, (0,0,0),1,lineType=cv2.LINE_AA)
                                    cv2.putText(bg_img,'  In', (c1[0]+x2,c1[1]+45), cv2.FONT_HERSHEY_SIMPLEX, 3/5, (0,0,0),1,lineType=cv2.LINE_AA)

                            elif col == 3:
                                cv2.rectangle(bg_img, c1,c2,(255,120,120),-1,cv2.LINE_AA)
                                if raw ==0:
                                    cv2.putText(bg_img,'South', (c1[0]+x2-2,c1[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 3/5, (0,0,0),1,lineType=cv2.LINE_AA)
                                    cv2.putText(bg_img,' Out', (c1[0]+x2-2,c1[1]+45), cv2.FONT_HERSHEY_SIMPLEX, 3/5, (0,0,0),1,lineType=cv2.LINE_AA)

                            elif col == 4:
                                cv2.rectangle(bg_img, c1,c2,(255,120,120),-1,cv2.LINE_AA)
                                if raw ==0:
                                    cv2.putText(bg_img,'South', (c1[0]+x2-2,c1[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 3/5, (0,0,0),1,lineType=cv2.LINE_AA)
                                    cv2.putText(bg_img,'  In', (c1[0]+x2-2,c1[1]+45), cv2.FONT_HERSHEY_SIMPLEX, 3/5, (0,0,0),1,lineType=cv2.LINE_AA)

                            elif col == 5:
                                cv2.rectangle(bg_img, c1,c2,(120,255,120),-1,cv2.LINE_AA)
                                if raw ==0:
                                    cv2.putText(bg_img,'East', (c1[0]+x2,c1[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 3/5, (0,0,0),1,lineType=cv2.LINE_AA)
                                    cv2.putText(bg_img,' Out', (c1[0]+x2,c1[1]+45), cv2.FONT_HERSHEY_SIMPLEX, 3/5, (0,0,0),1,lineType=cv2.LINE_AA)

                            elif col == 6:
                                cv2.rectangle(bg_img, c1,c2,(120,255,120),-1,cv2.LINE_AA)
                                if raw ==0:
                                    cv2.putText(bg_img,'East', (c1[0]+x2,c1[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 3/5, (0,0,0),1,lineType=cv2.LINE_AA)
                                    cv2.putText(bg_img,'  In', (c1[0]+x2,c1[1]+45), cv2.FONT_HERSHEY_SIMPLEX, 3/5, (0,0,0),1,lineType=cv2.LINE_AA)

                            elif col == 7:
                                cv2.rectangle(bg_img, c1,c2,(120,120,255),-1,cv2.LINE_AA)
                                if raw ==0:
                                    cv2.putText(bg_img,'North', (c1[0]+x2-2,c1[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 3/5, (0,0,0),1,lineType=cv2.LINE_AA)
                                    cv2.putText(bg_img,' Out', (c1[0]+x2-2,c1[1]+45), cv2.FONT_HERSHEY_SIMPLEX, 3/5, (0,0,0),1,lineType=cv2.LINE_AA)

                            elif col == 8:
                                cv2.rectangle(bg_img, c1,c2,(120,120,255),-1,cv2.LINE_AA)
                                if raw ==0:
                                    cv2.putText(bg_img,'North', (c1[0]+x2-2,c1[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 3/5, (0,0,0),1,lineType=cv2.LINE_AA)
                                    cv2.putText(bg_img,'  In', (c1[0]+x2-2,c1[1]+45), cv2.FONT_HERSHEY_SIMPLEX, 3/5, (0,0,0),1,lineType=cv2.LINE_AA)

                            if raw ==0 or col == 0:
                                pass
                            else:
                                str_count = area_list2(area_t,m=raw-1,n=col-1)
                                #print(raw,col,str_count)
                                cv2.putText(bg_img,str(str_count), (c1[0]+30,c2[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 3/5, (0,0,0),1,lineType=cv2.LINE_AA)
                
                #print(area_t)
                            
                # Print time (inference + NMS)
                #-----------------------------------------------------------------------------#
                #print('%s Done. (%.3fs)' % (s, t2 - t1))
                sec = curTime - prevTime
                prevTime = curTime
                fpss = 2/sec
                #print('Estimated fps 1 : {0}'.format(fps))

                # Stream results
                if view_img:
                    strs = "FPS : %0.1f" %fpss
                    #cv2.putText(bg_img, strs, (0,210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('w'):
                        cv2.imwrite("test.png",im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration


                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
                    else:
                        ##210118------------------------------------------------------------
                        strs = "FPS : %0.1f" %fpss
                        cv2.putText(bg_img, strs, (0,210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                        video_size = im0.shape
                        
                        over_img = overlay(im0, bg_img, position=(int(video_size[1]/2), 0))
                        over_img_size = over_img.shape
                        cv2.imshow(p, over_img)
                        print(p)
                        #time.sleep(0.5)
                        if cv2.waitKey(1) == ord('w'):
                            cv2.imwrite("/images/aidrone.png",im0)
                            cv2.imwrite("/images/aidrone2.png",over_img)
                        if cv2.waitKey(1) == ord('q'):  # q to quit
                            raise StopIteration
                        
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)

                            w1 = int(over_img_size[1])
                            h1 = int(over_img_size[0])
                            #print(w1,h1)
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w1, h1))
                        vid_writer.write(over_img)
                        ##
                            
            sec1 = curTime1 - prevTime1
            prevTime1 = curTime1
            fps1 = 1/sec
            #print('Estimated fps 2 : {0}'.format(fps1))
            
            

    if save_txt or save_img:
        #print('Results saved to %s' % Path(out))
        if platform.system() == 'Darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()

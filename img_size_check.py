#-*- coding:utf-8 -*-
import os
import shutil
import time
import json


route_base = 'd:/drone_mv'
#route_base = 'C:\Dataset\201224_drone'
old_p = ''
folder_name = ''
folder_name1 = ''
k_1 = ''
err_list = []
result_folder = 'result/'
err_txt = open(result_folder + 'error.txt', 'wt')
semiclass = ['car','truck','bus','person','bike','etc vehicle', 'not applicable']
blank = '\n'
s_c2 = [0]*len(semiclass)
ct_wh_error = 0
ct_wh_error1 = 0
file_count = 0
file_count1 = 0
total_inf = []


for p, d, f in os.walk(route_base):
    if 'labels' in p:
        s_c = [0]*len(semiclass)
        ct_wh_error = 0
        file_count = 0
        for file in f:
            file_count +=1
            file_count1 +=1
            if old_p != p:
                print("p:", p)
                if "PM" in p:
                    folder_name = p[p.find("PM"):p.find("0m")+1]
                elif "AM" in p:
                    folder_name = p[p.find("AM"):p.find("0m")+1]
                anno_txt = result_folder + folder_name + '.txt'
                annot = open(anno_txt, 'wt', encoding='UTF-8')
            old_p = p
            
            if 'DS' in file:
                pass       
            else :               
                file_name = p + '/' + file
                #print(file_name)
                
                #time.sleep(1)
                with open(file_name, 'rt', encoding='UTF8') as f:
                    json_data = json.load(f)

                #속성정보 읽기
                data1=json_data['annotations']
                for k in data1 :
                    #'가이드' 클래스 제거
                    if k['label']=={} or k['label']=='not applicable':
                        k_1 = str(file_name)
                        err_list.append(k_1)
                        pass
                    #그 외 클래스 추출
                    else:
                        # 변수 정의
                        point = k['points']

                        #좌표구성
                        xl = point[0][0] # 좌측 x
                        xr = point[2][0] # 우측 x
                        yb = point[0][1] # 아래 y
                        yu = point[2][1] # 위 y                    
                                                
                        w = int(point[2][0] - point[0][0])
                        h = int(point[2][1] - point[0][1])
                        x = int((point[2][0] + point[0][0])/2)
                        y = int((point[2][1] + point[0][1])/2)
                        
                        #210119 클래스 갯수 확인
                        s_c[semiclass.index(k['label'])]+=1
                        s_c2[semiclass.index(k['label'])]+=1 
                        
                        if xl<0 or xr <0 or yb <0 or yu <0 or w<0 or h<0 or x <0 or y<0 :
                            err_file = file_name+': '+ 'xl, xr, yb, yu '+ str(xl)+ ' '+ str(xr)+ ' '+ str(yb)+ ' '+ str(yu)+ ' '
                            annot.write(str(err_file)+'\n')
                            err_txt.write(str(err_file)+'\n')
                            #annot.write(blank)
                            if w<40 or h< 40:
                                err_file = file_name+'"w<40,h<40": '+ str(w)+ ' '+ str(h)
                                annot.write(str(err_file)+'\n')
                                annot.write(blank)
                                err_txt.write(str(err_file)+'\n')
                                err_txt.write(blank)
                            ct_wh_error += 1
                            ct_wh_error1 += 1
                         
                        
        
            #폴더별 속성기록
        s_c= str(semiclass)+': '+ str(s_c)
        #s_c = str(s_c)
        annot.write(s_c)
        annot.write(blank)
        error_rate ="error rate : "+ str(ct_wh_error) + ' / ' + str(file_count)
        #error_rate = str(error_rate)
        annot.write(error_rate)
        annot.close()  
        total_inf.append(s_c)
        total_inf.append(error_rate)
    else:
        pass
error_rate1 =str(ct_wh_error1) + '/' + str(file_count1)
s_c2= str(semiclass)+ ': ' + str(s_c2)
s_c2 = str(s_c2)
print(s_c2, type(s_c2))

err_txt.write(s_c2)
err_txt.write(blank)
err_txt.write(str(error_rate1))
err_txt.close()
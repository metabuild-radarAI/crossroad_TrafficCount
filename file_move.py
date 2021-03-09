#-*- coding:utf-8 -*-
import os
import shutil
import time

#D:/drone/crossroad 내부에 image, json 폴더가 있으며 하부에 여러 depth의 폴더가 있고 그 아래에 파일이 존재하는 상황


class Transform:
    def __init__(self, filename, path):
        self.filename = filename
        self.path = path
        self.time_check(self.filename,self.path)
        #print("init :", self.filename, self.path)

    def time_check(self, filename, path):
        self.filename = filename
        time_am = '/AM-'
        time_pm = '/PM-'
        if 'AM' in self.filename or '-07h' in self.filename or '-08h' in self.filename or '-09h' in self.filename or '-10h' in self.filename or '-11h' in self.filename :
            self.path += time_am
            self.weather(self.filename, self.path)
            #print('time1', self.filename, self.path)
        elif 'PM' in self.filename or '-12h' in self.filename or '-13h' in self.filename or '-14h' in self.filename or '-15h' in self.filename or '-16h' in self.filename or '-17h' in self.filename or '-18h' in self.filename :
            self.path += time_pm
            self.weather(self.filename, self.path)
            #print('time2', self.filename, self.path)
        else :
            return print('tiem?')

    def weather(self, filename, path):
        weather_sun = 'SUN-'
        weather_cloud = 'CLD-'
        if 'SUN' in filename:
            self.path +=weather_sun
            self.altitude(filename, path)
            #print('weather1', self.filename, self.path)
        elif 'CLD' in filename:
            self.path +=weather_cloud
            self.altitude(filename, path)
            #print('weather2', self.filename, self.path)
        else :
            return print('weather?')

    def altitude(self, filename, path):
        altitude_40 = '40m'
        altitude_50 = '50m'
        altitude_60 = '60m'
        altitude_70 = '70m'
        altitude_80 = '80m'
        altitude_90 = '90m'
        altitude_100 = '100m'
        if '-40m-' in filename or '-40M-' in filename:
            self.path +=altitude_40
            self.jpg_json(filename, path)
        elif '-50m-' in filename or '-50M-' in filename:
            self.path +=altitude_50
            self.jpg_json(filename, path)
        elif '-60m-' in filename or '-60M-' in filename:
            self.path +=altitude_60
            self.jpg_json(filename, path)
        elif '-70m-' in filename or '-70M-' in filename:
            self.path +=altitude_70
            self.jpg_json(filename, path)
        elif '-80m-' in filename or '-80M-' in filename:
            self.path +=altitude_80
            self.jpg_json(filename, path)
        elif '-90m-' in filename or '-90M-' in filename:
            self.path +=altitude_90
            self.jpg_json(filename, path)
        elif '-100m-' in filename or '-100M-' in filename or '-100 m-' in filename:
            self.path +=altitude_100
            self.jpg_json(filename, path)
        else :
            return print('altitude?')    

    def dir_check(self, path):
        if os.path.isdir(self.path):
            pass
        else:
            os.makedirs(self.path)
            
    def jpg_json(self, filename, path):
        if filename[-4:] == '.jpg':
            self.path += mv_image
            self.dir_check(path)
            return self.path
        if filename[-5:] == '.json':
            self.path += mv_json
            self.dir_check(path)
            return self.path
        else :
            return print('jpg/json?')    

if __name__ == '__main__':
    route_base = 'E:/drone_0215'
    route_image = '/image'
    route_json = '/json'

    mv_image = '/images'
    mv_json = '/labels_json'
    old_p = ''
    os.chdir(route_base)
    #print(os.getcwd())
    for p, d, f in os.walk(route_base):
        for fn in f:
            if 'DS' in fn:
                pass          
            else:              
                if old_p != p:
                    print(p)
                old_p = p
                
                path = 'D:/drone_mv'
                old_fn = fn
                if '이진구' in fn:
                    fn = fn.replace("이진구","LJK")
                    fn = fn.replace("맑음","SUN")
                    fn = fn.replace("흐림","CLD")
                if '박종원' in fn:
                    fn = fn.replace("박종원","PJY")
                    fn = fn.replace("맑음","SUN")
                    fn = fn.replace("흐림","CLD")            
                spath = Transform(fn,path)
                #print(spath.path)
                #time.sleep(2)
                path = spath.path
                old_path = p+'/'+old_fn
                new_path = path+'/'+fn
                #print('#################################################')
                #print(old_path, new_path)
                #shutil.move(old_path,new_path)
                
                if os.path.isfile(new_path):
                    pass
                else :
                    shutil.copy(old_path,new_path)
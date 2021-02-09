# imports
import os
from datetime import date as dt
import json
import urllib.request
import cv2
from skimage import io
import textwrap
import numpy as np

class WisePhoto:
    '''
    A class to generate a photo from NASA's Astronomy
    Photo of the Day API. Methods are image processing functions from
    OpenCV that can be applied to the APOD.

    param:: date = the date of the desired APOD in 'YYYY-MM-DD' format. If not provided, today's photo will be retrieved
    '''
    def __init__(self, date=None):

        if not date:
            self.date = str(dt.today())
        else:
            self.date = date

        # Photo attributes
        self.photo_request = 'http://0.0.0.0:5000/v1/apod/?concept_tags=True&date=' + self.date
        try:
            self.photo = self.get_photo()
        except:
            print('There is no APOD for the provided date!')

        # Quote attribues
        if date:
            self.zen_request = 'https://zenquotes.io/api/random'
        else:
            self.zen_request = 'https://zenquotes.io/api/today'
        self.font = cv2.FONT_HERSHEY_TRIPLEX
        self.fontScale = 2
        self.color = (255,255,255)
        self.thickness = 2
        

    def get_photo(self):
        request = urllib.request.Request(self.photo_request)     
        response = urllib.request.urlopen(request)
        data = response.read()
        values = json.loads(data)  
        img_url = values['url']
        rgb = io.imread(img_url)
        r,g,b = cv2.split(rgb)
        bgr = cv2.merge([b,g,r])
        return bgr

    def get_quote(self):
        request = urllib.request.Request(self.zen_request)
        response = urllib.request.urlopen(request)
        data = response.read()
        values = json.loads(data)[0]
        quote = values['q']
        return quote

    def get_text_size(self, quote):
        text = quote
        text_size = cv2.getTextSize(text=text, 
                                    fontFace=self.font, 
                                    fontScale=self.fontScale, 
                                    thickness=self.thickness)
        px_w, px_h = text_size[0]
        return (px_w, px_h)

    def get_accent_color(self, img):
        #TODO : clean up this function
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
        k_in = lab.reshape((-1, 3))
        # print(f'lab dytpe: {lab.dtype}')
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(k_in, 5, None, criteria, 10, flags)
        centers = np.uint8(centers)
        # print(f'centers:\n{centers}')

        bgr_centers = []
        bgr = cv2.cvtColor(np.uint8([centers]), cv2.COLOR_Lab2BGR)
        bgr_centers.append(np.squeeze(bgr))
        center = bgr_centers[0][3]
        center = (int(center[0]), int(center[1]), int(center[2]))
        # print(f'bgr centers:\n{bgr_centers}')
        # print(f'bgr centers:\n{center.dtype}')
        return center

    def wise_photo(self):
        bg = self.photo
        bg_h, bg_w, _ = bg.shape
        # print(f'shape: {bg.shape}')
        quote = self.get_quote()
        qu_w, qu_h = self.get_text_size(quote)
    
        chars = len(quote)
        px_per_char = qu_w // chars

        qu_bbox_w = int(0.75 * bg_w)
        qu_bbox_x = int(0.25 * bg_w / 2) 

        char_w = qu_bbox_w // px_per_char
        
        # print(f'img h: {bg_h}, img w: {bg_w}')
        # print('Text width: ', char_w)
        
        wrapped_text = textwrap.wrap(quote, width=char_w)
        num_lines = len(wrapped_text) - 1
        # print(wrapped_text)

        color = self.get_accent_color(bg)
        # print(f'color: {color}')
        
        for i, line in enumerate(wrapped_text):
            gap = int(qu_h + (0.3 * qu_h))
            qu_bbox_h = gap * num_lines
            ori_y = (bg_h - qu_bbox_h) // 2
            y = ori_y + (i * gap)

            # start new line according to width in px of that line
            # x = (bg_w - (len(line) * px_per_char)) // 2
            
            # start new line at same x position
            x = qu_bbox_x
            

            wise_photo = cv2.putText(bg, line, (x, y), self.font,
                            self.fontScale, 
                            color, 
                            self.thickness, 
                            lineType = cv2.LINE_AA)

        return wise_photo

# Starting OpenCV tutorials on APOD:

    # 2021-02-01
    def show(self, name, img):
        cv2.imshow(name, img)
        k = cv2.waitKey(0)
        if k == 27: 
            cv2.destroyAllWindows()
        elif k == ord('s'): 
            cv2.imwrite(f'./out/{name}.png', img)

    # 2021-02-02 - draw line
    def draw_line(self, img, start, end, color, thickness):
        ''' draw line on input img
        :param img: image to be drawn on 
        :param type: BRG img
        
        :param start: pixel location of line's origin 
        :param type: (x,y) as tuple
            
        :param end: pixel location of line's end 
        :param type: (x,y) as tuple
            
        :param color: desired color of line; default is white
        :param type: BGR tuple
            
        :param thickness: desired thickness of line
        :param type: int
        '''
        color = self.color
        lined_img = cv2.line(img, start, end, color, thickness)
        return lined_img

    # 2021-02-03 - no APOD! refactored class

    # 2021-02-04 - draw polygon
    def draw_polygon(self, img, pts, close=True, fill=False, color=None):
        '''draw polygon on input img

        :param img: image to be drawn on
        :param type: BGR img 
        
        :param pts: coordinates of polygon vertices 
        :param type: list in the following format: [[10,5],[20,30],[70,20],[50,10]]

        :param close: Default is True, which will close the shape. \
                False will draw polylines joining the points, \
                but will not close the shape.
        :param type: Boolean

        :param fill: Default is False, which will not fill in polygon with specified color. \
                True results in polygon filled with specified color.
        :param type: Boolean
        
        :param color: desired color of polygon; default is White
        :param type: tuple of BGR values 
        '''
        color = self.color
        pts =  np.array(pts, np.int32)
        pts = pts.reshape((-1,1,2))
        
        if fill:
            img = cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
        else:
            img = cv2.polylines(img, [pts], close, color)
        
        return img

    # 2021-02-05 - mouse event options
        # ['EVENT_FLAG_ALTKEY', 'EVENT_FLAG_CTRLKEY', 'EVENT_FLAG_LBUTTON', 'EVENT_FLAG_MBUTTON', 
        # 'EVENT_FLAG_RBUTTON', 'EVENT_FLAG_SHIFTKEY', 'EVENT_LBUTTONDBLCLK', 'EVENT_LBUTTONDOWN', 
        # 'EVENT_LBUTTONUP', 'EVENT_MBUTTONDBLCLK', 'EVENT_MBUTTONDOWN', 'EVENT_MBUTTONUP', 'EVENT_MOUSEHWHEEL', 
        # 'EVENT_MOUSEMOVE', 'EVENT_MOUSEWHEEL', 'EVENT_RBUTTONDBLCLK', 'EVENT_RBUTTONDOWN', 'EVENT_RBUTTONUP']
    def circle_from_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(self.photo, (x,y), 100, (255,255,255), -1)

    def click_draw_cirle(self, image):
        img = image
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.circle_from_click)

        while(1):
            cv2.imshow('image', img)
            if cv2.waitKey(20) & 0xFF == 27:
                break
        cv2.destroyAllWindows()






if __name__ == '__main__':
    apod = WisePhoto(date = '2021-01-29')
    # pts = [[24, 497], [174, 502], [219, 627], [137, 701], [21, 627]]
    # poly = apod.draw_polygon(apod.photo, pts, fill=True, color=(0, 255, 255))

    img = apod.wise_photo()
    apod.show('Wise Photo', img)
    apod.draw_polygon()

    # colors = apod.get_accent_color(apod.photo)
    # apod.show('k means', colors)
   




# imports
import os
from datetime import date as dt
import json
import urllib.request
import cv2
from skimage import io
import matplotlib.pyplot as plt
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

        self.photo_shape = self.photo.shape

        # Quote attribues
        self.zen_request_today = 'https://zenquotes.io/api/today'
        self.zen_request_random = 'https://zenquotes.io/api/random'
        

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
        request = urllib.request.Request(self.zen_request_today)
        response = urllib.request.urlopen(request)
        data = response.read()
        values = json.loads(data)[0]
        quote = values['q']
        return quote

    def wise_photo(self):
        bg = self.get_photo()
        text_width = 300
        print('Text width: ', text_width)
        # text_height = int(0.8 * bg.shape[1])
        # org_x = int((bg.shape[0] - text_width)/2)
        # org_y = int((bg.shape[1] - text_height)/2)
        text = self.get_quote()
        wrapped_text = textwrap.wrap(text, width=text_width)
        print(wrapped_text)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 3
        # org = (org_x, org_y)
        color = (255,255,255)
        thickness = 5

        for i, line in enumerate(wrapped_text):
            textsize = cv2.getTextSize(line, font, fontScale, 
                        thickness)[0]

            gap = textsize[1] + 10

            y = int((bg.shape[0] + textsize[1]) / 2) + i * gap
            x = int((bg.shape[1] - textsize[0]) / 2)

            wise_photo = cv2.putText(bg, line, (x, y), font,
                            fontScale, 
                            color, 
                            thickness, 
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
        ''' img : image to be drawn on
            start : pixel location of line's origin as tuple
            end : pixel location of line's end as tuple
            color : desired color as rgb tuple
            thickness : int as desired thickness
        '''
        lined_img = cv2.line(img, start, end, color, thickness)
        return lined_img

    # 2021-02-03 - no APOD! refactored class

    # 2021-02-04 - draw polygon
    def draw_polygon(self, img, pts, close=True, fill=False, color=(255,255,255)):
        '''
        draw polygon on input img
        img: image to be drawn on 
        pts: list of coordinates of vertices in the following format: [[10,5],[20,30],[70,20],[50,10]]
        close: Boolean. Default is True, which will close the shape. False will draw polylines joining the points, but will not close the shape
        fill: Boolean. Default is False, which will not fill in polygon with specified color. True results in polygon filled with specified color.
        color: tuple of BGR values of desired color of polygon; default is White
        '''
        pts =  np.array(pts, np.int32)
        pts = pts.reshape((-1,1,2))
        
        if fill:
            img = cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
        else:
            img = cv2.polylines(img, [pts], close, color)
        
        return img


if __name__ == '__main__':
    apod = WisePhoto()
    pts = [[24, 497], [174, 502], [219, 627], [137, 701], [21, 627]]
    poly = apod.draw_polygon(apod.photo, pts, fill=True, color=(0, 255, 255))
    apod.show('poly-img', poly)

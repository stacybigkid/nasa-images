# imports
import os
from datetime import date as dt
import json
import urllib.request
import cv2
from skimage import io
import textwrap
import numpy as np
import random
from scipy.spatial import ConvexHull

class WisePhoto:
    '''
    A class to generate a photo from NASA's Astronomy Photo of the Day API. 
    Methods are image processing functions from OpenCV that can be applied to the APOD.

    :param date: the date of the desired APOD in 'YYYY-MM-DD' format. \
                    default is today's date.
    :param type: str
    '''
    def __init__(self):

        # NASA API key required in root directory config.json
        # generate one here: https://api.nasa.gov/index.html
        # see example_config.json for formatting
        with open('./config.json') as j:       
            credentials = json.load(j)
            self.key = credentials['key']

        # date
        self.date = str(dt.today())
        # self.date = '2021-01-29'

        # photo attributes
        # try getting APOD with today's date
        self.photo = self.get_photo()
        
        # if today's date returns None, try again with
        # random date until img populates self.photo
        max_tries = 5
        try_count = 1
        while (self.photo is None and try_count < max_tries):
            self.photo = self.get_photo(self.generate_date())
            try_count += 1

        # quote attribues
        self.font = cv2.FONT_HERSHEY_TRIPLEX
        self.fontScale = 2
        self.color = (255,255,255)
        self.thickness = 2

        # storage
        out_path = './out'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        self.out = out_path
        
        library_path = './library'
        if not os.path.exists(library_path):
            os.makedirs(library_path)    
        self.library = library_path      

    # methods
    def get_photo(self, date=None):
        if not date:
            date = self.date
        else:
            date = date

        photo_request = f'https://api.nasa.gov/planetary/apod?api_key={self.key}&date={date}'
        request = urllib.request.Request(photo_request)     
        response = urllib.request.urlopen(request)
        data = response.read()
        values = json.loads(data) 
        
        media_type = values['media_type']
        if media_type == 'image': 
            img_url = values['url']
            rgb = io.imread(img_url)
            if len(rgb.shape) == 2:
                rgb = cv2.cvtColor(rgb,cv2.COLOR_GRAY2RGB)
            r,g,b = cv2.split(rgb)
            bgr = cv2.merge([b,g,r])
            print(f"Date: {date} has APOD")
            return bgr
        else: 
            # print(f"Date: {date} has no APOD")
            return None
    
    def get_quote(self):
        zen_request = 'https://zenquotes.io/api/today'
        request = urllib.request.Request(zen_request)
        response = urllib.request.urlopen(request)
        data = response.read()
        values = json.loads(data)[0]
        quote = values['q']
        return quote

    def generate_date(self):
        mmdd = lambda x: f'0{x}' if (len(str(x)) < 2) else str(x)
        random_year = str(random.randint(1996, 2021))
        random_month = mmdd(random.randint(1, 12))
        random_day = mmdd(random.randint(1,31))
        return f'{random_year}-{random_month}-{random_day}'

    def get_text_size(self, quote):
        text = quote
        text_size = cv2.getTextSize(text=text, 
                                    fontFace=self.font, 
                                    fontScale=self.fontScale, 
                                    thickness=self.thickness)
        px_w, px_h = text_size[0]
        return (px_w, px_h)

    def get_accent_color(self, img, k=10):
        ''' calculate proportion of each of k colors in input img
        :param img: img where k colors should be extracted and quantified 
        :type: BGR img

        :param k: number of colors to separate img into, must eb 3 or more; \
                default=5
        :type: int 
        '''
        assert k >= 3, 'k must be set to 3 or more!'

        # preprocess img
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
        k_in = lab.reshape((-1, 3))

        # k means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                    10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, centers = cv2.kmeans(k_in, k, None, criteria, 10, flags)
        # print('Centers: ', centers)
                
        # calculate proportions of labels in img
        unique, counts = np.unique(labels, return_counts=True)
        pp = np.divide(counts, len(labels))
        pp = zip(unique, pp)
        pp = sorted(pp, key=lambda p: p[1], reverse=True) 
        # print(f"Unique, proportions:\n {[p for p in pp]}")

        # return least common color for text
        color_label = pp[-1][0]
        # print('color label: ', color_label)

        # reformat color centers to bgr tuples
        bgr_centers = cv2.cvtColor(np.uint8([centers]), 
                                cv2.COLOR_Lab2BGR).astype(int)
        bgr_centers = np.squeeze(bgr_centers)
        bgr_centers = [tuple(c) for c in bgr_centers]
        # print('BGR Centers: ', bgr_centers)

        # change dtype to int from int32
        color = bgr_centers[color_label]
        color = tuple([int(i) for i in color])

        return color
    
    # slope and drawLine functions from stack overflow answer:
    # https://stackoverflow.com/questions/59578855/opencv-python-how-do-i-draw-a-line-using-the-gradient-and-the-first-point
    def slope(self, x1,y1,x2,y2):
        ###finding slope
        if x2!=x1:
            return((y2-y1)/(x2-x1))
        else:
            return 'NA'

    def drawLine(self, image,x1,y1,x2,y2):
        m=self.slope(x1,y1,x2,y2)
        h,w=image.shape[:2]
        if m!='NA':
            ### extend the line to x=0 and x=width
            ### and calculating the y associated with it
            ##starting point
            px=0
            py=-(x1-0)*m+y1
            ##ending point
            qx=w
            qy=-(x2-w)*m+y2
        else:
        ### if slope is zero, draw a line with x=x1 and y=0 and y=height
            px,py=x1,0
            qx,qy=x1,h
        mask = cv2.line(image, (int(px), int(py)), (int(qx), int(qy)), (255, 255, 255), 2)
        return mask

    def get_non_edges(self, img):
        img = img
        self.show('original', img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 7)
        edges = cv2.Canny(blur,100,150)
        # d_k = np.ones((5,5), np.uint8)
        # dilate = cv2.dilate(edges, kernel=d_k, iterations=3)
        # o_k = np.ones((3,3), np.uint8)
        # opened = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel=o_k)
        self.show('Canny', edges)
        return edges

    def get_non_features(self, img):
        # load img, convert to greyscale, 
        # extract keypoint (kp) features
        img = img
        self.show('today', img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kps = sift.detect(gray, None)

        # sort kps largest to smallest
        # keep top 1% 
        sorted_kps = sorted(kps, 
                            key=lambda kp: kp.size, reverse=True)
        fraction = int(0.01 * len(kps))
        if len(fraction) < 3:
            fraction = int(0.02 * len(kps))
        sorted_kps = sorted_kps[:fraction]
        
        # to view keypoints found in img
        img = cv2.drawKeypoints(gray, 
                                sorted_kps[:fraction], 
                                img, 
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # reformat coords into list of lists
        # get enclosing shape of keypoints
        # hull vertices are returned in counterclockwise order
        coords = [list(kp.pt) for kp in sorted_kps]
        ch = ConvexHull(coords)
        hull = [list(coords[i]) for i in ch.vertices]
        
        # create inverse mask from hull vertices
        mask = np.zeros((img.shape[0], img.shape[1]))
        mask = self.draw_polygon(mask, 
                                hull, 
                                close=True, 
                                fill=True, 
                                color=self.color).astype(np.uint8)
        self.show('mask', mask)
        mask = cv2.bitwise_not(mask)

        # iterate over pairs of hull vertices
        # draw line that extends to img edges 
        # from each vertex
        hull = hull + [hull[0]]
        edges = []
        for i, v in enumerate(hull[:-1]):
            x1, y1 = v
            x2, y2 = hull[i+1]

            line = [x1, y1, x2, y2]
            slope = self.slope(x1, y1, x2, y2)
            
            bg = np.zeros(img.shape[:2])
            line_mask = self.drawLine(bg, 
                                      x1, 
                                      y1, 
                                      x2, 
                                      y2).astype(np.uint8)
            # self.show('line', line)

            h, w = line_mask.shape[:2]

            corners = [(0,0),
                        (w-1, 0),
                        (w-1, h-1), 
                        (0, h-1)]

            # flood fill the line mask from each corner
            corner_data = []
            for i, corner in enumerate(corners):
                corner_mask = line_mask.copy()
                _, _, _, rect = cv2.floodFill(corner_mask, 
                                              mask=None, 
                                              seedPoint=corner, 
                                              newVal=255)
                area = np.divide(np.sum(corner_mask), 255)
                corner_data.append([corner_mask, rect, area, line, slope])

            # sort line masks by area
            # keep smallest corner mask in array edges
            corner_data = sorted(corner_data, 
                                 key=lambda corner: corner[2])
            edges.append(corner_data[0])
            

        # sort all edges by size
        # keep largest edge mask for text in wise_photo
        text_edges = sorted(edges, 
                            key=lambda edge: edge[2], reverse=True)
        text_edge = text_edges[0]
        print(f'chosen line: {text_edge[-2]}, slope: {text_edge[-1]}')
        return text_edge

    def wise_photo(self):
        bg = self.photo
        bg_h, bg_w, _ = bg.shape
        # print(f'shape: {bg.shape}')
        quote = self.get_quote()
        qu_w, qu_h = self.get_text_size(quote)

        # get_non_features returns in the following order:
        # 0 : mask of region with fewest features
        # 1 : bounding box of the masked region
        # 2 : area of the masked region
        # -2 : coorinates of line as list [x1, y1, x2, y2]
        # -1 : slope of line

        # feature_line = self.get_non_features(bg)
        # print(f'Line: {feature_line[-1]}')
        chars = len(quote)
        px_per_char = qu_w // chars

        qu_bbox_w = int(0.75 * bg_w)
        qu_bbox_x = int(0.25 * bg_w / 2) 

        char_w = qu_bbox_w // px_per_char
        
        # print(f'img h: {bg_h}, img w: {bg_w}')
        # print('Text width: ', char_w)
        
        wrapped_text = textwrap.wrap(quote, width=char_w)
        num_lines = len(wrapped_text)
        gap = int(qu_h + (0.3 * qu_h))
        qu_bbox_h = gap * num_lines

        if qu_bbox_h > bg_h:
            font_scale = self.fontScale - 1
        else:
            font_scale = self.fontScale
        # print(wrapped_text)

        color = self.get_accent_color(bg)
        # print(f'color: {color}')
        
        for i, line in enumerate(wrapped_text):

            ori_y = (bg_h - qu_bbox_h) // 2
            y = ori_y + (i * gap)

            # x = y / feature_line[-1]
            # x = int(x)

            # start new line according to width in px of that line
            # x = (bg_w - (len(line) * px_per_char)) // 2
            
            # start new line at same x position
            x = qu_bbox_x
            
            wise_photo = cv2.putText(bg, line, (x, y), self.font,
                            font_scale, 
                            color, 
                            self.thickness, 
                            lineType = cv2.LINE_AA)

        cv2.imwrite(os.path.join(self.library, f'{self.date}.png'), wise_photo)

        return wise_photo

    # 2021-02-01
    def show(self, name, img):
        cv2.imshow(name, img)
        k = cv2.waitKey(0)
        if k == 27: 
            cv2.destroyAllWindows()
        elif k == ord('s'): 
            cv2.imwrite(os.path.join(self.out, f'{name}.png'), img)



if __name__ == '__main__':
    
    apod = WisePhoto()
    # apod.wise_photo()
    apod.get_photo()
    apod.show(apod.date, apod.photo)
    # apod.get_non_features(apod.photo)

    # pts = [[24, 497], [174, 502], [219, 627], [137, 701], [21, 627]]
    # poly = apod.draw_polygon(apod.photo, pts, fill=True, color=(0, 255, 255))

    # img = apod.wise_photo()
    # apod.show(apod.date, img)

    # img = apod.get_non_edges(apod.photo)
    # apod.show('Mask', img)


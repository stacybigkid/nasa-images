import os
import cv2

class OpenImage:
    def __init__(self, input_img):

        if isinstance(input_img, str):
            self.img = cv2.imread(input_img)
        else:
            self.img = input_img
        
    def show(self, name, img):
        cv2.imshow(name, img)
        k = cv2.waitKey(0)
        if k == 27: 
            cv2.destroyAllWindows()
        elif k == ord('s'): 
            cv2.imwrite(os.path.join(self.out, f'{name}.png'), img)

    # 2021-02-02 - draw line
    def draw_line(self, img, start, end, color, thickness):
        ''' draw line on input img
        :param img: image to be drawn on 
        :param type: BRG img
        
        :param start: pixel location of line's origin 
        :param type: (x,y) tuple
            
        :param end: pixel location of line's end 
        :param type: (x,y) tuple
            
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
        :param type: bool

        :param fill: Default is False, which will not fill in polygon with specified color. \
                True results in polygon filled with specified color.
        :param type: bool
        
        :param color: desired color of polygon; default is White
        :param type: BGR tuple 
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

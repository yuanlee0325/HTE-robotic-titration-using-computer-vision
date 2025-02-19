from PIL import Image, ImageDraw
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2

class readjson():
    '''
    # readjson read json and create a mask
    
    '''
    
    def __init__(self,
                 path=r'C:\Users\bdutta\work\pys\AI_algos\Liv\Camera_Test\color_palette_images\col_file2.json',
                ):
        
        self.path=path
        with open(path, "r",encoding="utf-8") as f:
            self.dj = json.load(f)
            
        self.polys = []
        for sh in self.dj['shapes']:
            self.polys.append(sh['points'])
            
    def get_mask(self,fname='TestJson2',instance_offset=1,verbose=True):
        
        dj=self.dj
        oldmask=0
        count=0
        num_patches=0
        for points in self.polys:
            shape_type=None
            
            count+=instance_offset

            mask = np.zeros((dj['imageHeight'],dj['imageWidth']), dtype=np.uint8)
            mask = Image.fromarray(mask)


            draw = ImageDraw.Draw(mask)
            
            xy = [tuple(point) for point in points]
            
            if shape_type == "circle":
                assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
                (cx, cy), (px, py) = xy
                d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
                
            elif shape_type == "rectangle":
                assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
                draw.rectangle(xy, outline=1, fill=1)
                
            else:
                assert len(xy) > 2, "Polygon must have points more than 2"
                #print('polygon running')
                draw.polygon(xy=xy, outline=1, fill=count)
           
            mask = count*np.array(mask, dtype=bool) + oldmask

            oldmask=mask
            num_patches+=1

        if verbose : plt.imshow(mask)
        
        self.mask=mask
        self.num_patches=num_patches
        #print(rj.mask.shape)
        cv2.imwrite(fname+'.png',self.mask)
        #return mask

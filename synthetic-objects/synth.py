import glob
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import random
import os
import copy as cp
import xml.etree.ElementTree as ET
from PIL import Image



images = glob.glob('./Classes/**/*.png', recursive=True)
annotation_file = ET.parse("/Users/toddgleed/aalwaysAI_stuff/datasets/Unity/ManualSynth/result.xml")

annotation_path = "Users/toddgleed/aalwaysAI_stuff/datasets/Unity/ManualSynth/Annotations/"
annotation_name = "result.xml"

#annotation_file.write(os.path.join(annotation_path, annotation_name))

numImages = 3250

i = 1
while i<numImages:
    # reset the image background - future can randomize through different bg images
    path = r"/Users/toddgleed/aalwaysAI_stuff/datasets/Unity/ManualSynth/backgrounds/"
    random_filename = random.choice([
        x for x in os.listdir(path)
            if os.path.isfile(os.path.join(path, x))
    ])
    
    img_bg = Image.open("./backgrounds/" + random_filename)
    width, height = img_bg.size 

    

    #Randomly determine number of objects to put on image
    a = random.randint(1,5)

    #Randomly sample examples of all class examples
    sample_list = random.choices(images, k=a)
    ai = 0
    bbs = []
    for img in sample_list:
        class_name = img.split('/')[2]
        
        img = Image.open(img).convert("RGBA")
        x, y = img.size
        lane = int(round(width/6))
        xstart = 80 + (ai * lane)
        print('x:' + str(width) + 'xstart:' + str(xstart))
        
        x1 = random.randint(xstart, xstart+lane)
        y1 = random.randint(80, height-180)
        x3 = x+x1
        y3 = y+y1
        x2 = min(x+x1, width)
        y2 = min(y+y1, height)
        img_bg.paste(img, (x1, y1, x3, y3), img)

        bbs.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
        ai+=1
        #root.append(element)

        #subelement = root[0][1].makeelement('name')
        #ET.SubElement(root[1],'name')
        #root[1][0].text = class_name

    #Create an XMLFile
    #new_annotation_name = annotation_file
    new_annotation_file = cp.deepcopy(annotation_file)
    root = new_annotation_file.getroot()
    for bbx in bbs:
        object = ET.SubElement(root, "object")
        name = ET.SubElement(object, "name")
        pose = ET.SubElement(object, "pose")
        truncated = ET.SubElement(object, "truncated")
        difficult = ET.SubElement(object, "difficult")
        occluded = ET.SubElement(object, "occluded")
        bndbox = ET.SubElement(object, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        ymin = ET.SubElement(bndbox, "ymin")
        xmax = ET.SubElement(bndbox, "xmax")
        ymax = ET.SubElement(bndbox, "ymax")
        name.text = class_name
        pose.text = 'Unspecified'
        truncated.text = '0'
        difficult.text = '0'
        occluded.text = '0'
        xmin.text = str(bbx.x1)
        xmax.text = str(bbx.x2)
        ymin.text = str(bbx.y1)
        ymax.text = str(bbx.y2)


    
    new_annotation_file.find('filename').text = "result" + str(i) + ".jpg"
    for w in root.iter('width'):
        w.text = str(width)
    for h in root.iter('height'):
        h.text = str(height)
    new_annotation_file.write("Annotations/result" + str(i) + ".xml")  
         
    img_bg.save('JPEGImages/result' + str(i) + '.jpg')
    i += 1
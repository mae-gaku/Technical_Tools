import pandas as pd
import numpy as np
from lxml import etree
# import xmlAnnotation.etree.cElementTree as ET
# import xml.etree.ElementTree as ET
from lxml import etree as ET
# filename,width,height,xmin,ymin,xmax,ymax,class

# fields = ['filename','width','height','xmin','ymin','xmax','ymax']
fields = ['image_id','width','height','xmin','ymin','xmax','ymax']
df = pd.read_csv('csv path', usecols=fields)


# Change the name of the file.
# This will replace the / with -
def nameChange(x):
    x = x.replace("/", "-")
    return x


df['NAME_ID'] = df['image_id'].apply(nameChange)

for i in range(0, 10000):
    height = df['height'].iloc[i]
    width = df['width'].iloc[i]
    depth = 3

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = 'images'
    ET.SubElement(annotation, 'filename').text = str(df['NAME_ID'].iloc[i])
    ET.SubElement(annotation, 'path').text = "/media/sf_virtualbox/My_code/nemoto_codes/dataset/" + str(df['NAME_ID'].iloc[i])
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = "Unknown"
    
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)
    ET.SubElement(annotation, 'segmented').text = '0'
    ob = ET.SubElement(annotation, 'object')
    ET.SubElement(ob, 'name').text = 'Comp'
    ET.SubElement(ob, 'pose').text = 'Unspecified'
    ET.SubElement(ob, 'truncated').text = '0'
    ET.SubElement(ob, 'difficult').text = '0'
    bbox = ET.SubElement(ob, 'bndbox')
    ET.SubElement(bbox, 'xmin').text = str(df['xmin'].iloc[i])
    ET.SubElement(bbox, 'ymin').text = str(df['ymin'].iloc[i])
    ET.SubElement(bbox, 'xmax').text = str(df['xmax'].iloc[i])
    ET.SubElement(bbox, 'ymax').text = str(df['ymax'].iloc[i])

    fileName = str(df['NAME_ID'].iloc[i])
    import os
    fileName = os.path.splitext(os.path.basename(fileName))[0]
    tree = ET.ElementTree(annotation)
    tree.write("" + fileName + ".xml", pretty_print=True)
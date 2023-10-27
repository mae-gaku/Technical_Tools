import xml.etree.ElementTree as ET
import os

base_dir = "/media/sf_virtualbox/My_code/work_nemoto_codes/fire_extinguisher_v1"
files = [f for f in os.listdir(f"{base_dir}/labels")]

for file in files:
  print(file)

  if file == "classes.txt":
    continue
  tree = ET.parse(f"{base_dir}/labels/{file}")
  root = tree.getroot()

  for i in root.iter('size'):
    size = i
    height = float(size.findtext('height'))
    width = float(size.findtext('width'))

  new_filename = file.replace(".xml", ".txt")
  for i in root.findall('object'):
    for j in i.iter('bndbox'):
      xmin = float(j[0].text)
      ymin = float(j[1].text)
      xmax = float(j[2].text)
      ymax = float(j[3].text)

      el_1 = (xmin + xmax) / 2 / width
      el_2 = (ymin + ymax) / 2 / height
      el_3 = (xmax - xmin) / width
      el_4 = (ymax - ymin) / height

      for i in root.iter('object'):
        text = f"0 {el_1:.10f} {el_2:.10f} {el_3:.10f} {el_4:.10f}\n"


      if not os.path.exists(f"{base_dir}/txt/{new_filename}"):
        with open(f"{base_dir}/txt/{new_filename}", "w", newline="\n") as f:
          f.write(text)

      else:
        with open(f"{base_dir}/txt/{new_filename}", "a", newline="\n") as f:
          f.write(text)
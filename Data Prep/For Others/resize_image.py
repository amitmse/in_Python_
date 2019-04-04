import PIL
from PIL import Image
import os


os.chdir("E:\\Moni\\test")

def get_file_names(directory):
		raw_csv_file_name_list = []
		for root, directories, files in os.walk(directory):
			for filename in files:
				raw_csv_file_name_list.append(filename)
		return raw_csv_file_name_list
	
raw_csv_file_name = list(set(get_file_names("E:\\Moni\\test")))


def resize_image(file_name):
	#https://opensource.com/life/15/2/resize-images-python
	basewidth = 750
	img = Image.open(file_name)
	wpercent = (basewidth / float(img.size[0]))
	hsize = int((float(img.size[1]) * float(wpercent)))
	img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
	return img.save(file_name)

for file in raw_csv_file_name:
	resize_image(file)

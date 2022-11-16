from PIL import Image
# import matplotlib.pyplot as plt

img = Image.open('input_file.jpg')
img_gray = img.convert('L')
img_gray.show()
img_binary = img.convert('1')
img_binary.show()
# plt.show(img_gray)
# img.save('output_file.jpg')
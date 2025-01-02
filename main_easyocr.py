import cv2
import easyocr
import matplotlib as plt

path=r'D:\javeed\project\01.jpeg'

image=cv2.imread(path)
image = cv2.resize(image, (1000, 800))
reader=easyocr.Reader(['en'], gpu=False)

result=reader.readtext(image,detail=1, paragraph=False)

# print(result)

for cord, text, confi in result:
    # print(cord, text, confi)
    TL, TR, BR, BL = cord
    # bbox=(TL,BR)
    tx,ty=TL[0],TL[1]
    bx,by=BR[0],BR[1]
    cv2.rectangle(image,(tx,ty),(bx,by),(0,255,0),5)
    cv2.putText(image,text,(tx-100,ty-50),cv2.FONT_HERSHEY_COMPLEX_SMALL,3,(0,255,0),8)

# plt.imshow(image)
# plt.show()

cv2.imshow('image with text', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
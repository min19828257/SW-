import cv2
from binary_cls_3 import test
from PIL import Image

#이미지 resize()
def resize(img,cnt1,cnt2,cnt3,cnt4):

    dst = img.copy()
    dst = img[cnt4:cnt2,cnt3:cnt1]
    return dst

#값 측정
def calculate(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    result = test(img)
    return result

def main():
    img = cv2.imread('./hani.jpg')

    # cv2.waitKey()
#    img = cv2.rectangle(img, (150, 100), (400, 250), (0, 255, 0), 3)

    cnt1 = 200 ;cnt3=0;cnt2=200;cnt4=0
    count =  1
    while True:
        img = cv2.imread('./hani.jpg')

        print(cnt2, " : ", cnt4)

        img = cv2.rectangle(img, (cnt1, cnt2), (cnt3, cnt4), (0, 255, 0), 3)
        height = img.shape[0]
        width = img.shape[1]
        resize_img = resize(img,cnt1,cnt2,cnt3,cnt4)

        print(height,width)
        #cv2.imshow("dfasd",resize_img)
        #cv2.waitKey(0)
        result = calculate(resize_img)

        if(result >0.3):
            cv2.imshow("df",resize_img)
            cv2.waitKey(0)

        cv2.imshow('loop',img)
        k = cv2.waitKey(1)
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()
            break

        cnt3 +=10;cnt1+=10

        if(cnt1== 780 and cnt3 == 680 and cnt2 ==600 and cnt4==500):
            cnt1 = 200; cnt3 = 0; cnt2 = 200; cnt4 = 0
        if(cnt1 == 780 and cnt3 == 580):
            cnt1 = 200;cnt3 = 0;cnt2 += 200*count;cnt4 += 200*count
            count+=1

if __name__ == "__main__":
    main()
import cv2
import pytesseract

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img_lst = ["work.png"]

for i, img_nm in enumerate(img_lst):
    img = cv2.imread(img_nm)
    gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (h, w) = gry.shape[:2]
    if i == 0:
        thr = gry
    else:
        gry = cv2.resize(gry, (w * 2, h * 2))
        erd = cv2.erode(gry, None, iterations=1)
        if i == len(img_lst)-1:
            thr = cv2.threshold(erd, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        else:
            thr = cv2.threshold(erd, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    bnt = cv2.bitwise_not(thr)
    txt = pytesseract.image_to_string(bnt, config="--psm 6 digits")
    print("".join([t for t in txt if t.isalnum()]))
    cv2.imshow("bnt", bnt)
    cv2.waitKey(0)

import cv2
from background_extraction import extract_bg
from contour_utility import do_detect_circle, find_permitted_numbers, detect_the_numbers


input_video = './test 1.avi'
output_video = './test 1.mp4'
model = './cnn.h5'

background_image = extract_bg(input_video)

cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'MPEG'), fps, (W, H))

print("fps =", fps)

cX_gray, cY_gray, radius_gray_circle, color_gray_circle = do_detect_circle(background_image, 0)
print("center of static circle (x axis)=", cX_gray, "(y axis)=", cY_gray)

frame_number = 0
while True:
    success, image = cap.read()
    if not success:
        break
    frame_number += 1
    print("frame =", frame_number)
    subtracted_image = cv2.subtract(image, background_image)
    permitted_contours_gray = find_permitted_numbers(subtracted_image, cX_gray, cY_gray, radius_gray_circle)
    numbers_gray = detect_the_numbers(permitted_contours_gray, subtracted_image, model, frame_number)
    cX, cY, radius_blue_circle, color_blue_circle = do_detect_circle(subtracted_image, frame_number)
    permitted_contours_blue = find_permitted_numbers(subtracted_image, cX, cY, radius_blue_circle)
    numbers_blue = detect_the_numbers(permitted_contours_blue, subtracted_image, model)
    cv2.putText(image, 'blue circle:' + str(numbers_blue) + '=' + str(sum(numbers_blue)), (10, H - 30),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
    cv2.putText(image, 'gray circle:' + str(numbers_gray) + '=' + str(sum(numbers_gray)), (10, H - 50),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
    cv2.putText(image, 'blue circle:center=' + str((cX, cY)) + ',radius=' + str(radius_blue_circle), (W - 400, 45),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
    cv2.putText(image, 'gray circle:center=' + str((cX_gray, cY_gray)) + ',radius=' + str(radius_gray_circle), (W - 400, 65),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
    cv2.putText(image, 'fps=' + str(int(fps)) + "/" + 'gray circle(BGR)=' + str(color_gray_circle) + '/blue circle(BGR)=' + str(color_blue_circle) , (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
    out.write(image)


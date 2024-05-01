import os
import numpy as np
import cv2
from rembg import remove


# 폴더 내의 npy 모두 읽어오기
def read_npy_files_in_folder(folder_path):
    npy_files = []
    obj_name = []
    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            file_path = os.path.join(folder_path, file)
            npy_data = np.load(file_path)
            npy_files.append(npy_data)
            # print(file)
            obj_name.append(file)
    return npy_files, obj_name


def goodMatch_orb(desc, desc2):
    # 첫번재 이웃의 거리가 두 번째 이웃 거리의 75% 이내인 것만 추출
    ratio = 0.75
    # BF-Hamming 생성
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)  # NORM_L2 -> sift, surf에 적합     #cv.NORM_HAMMING -> ORB에 적합
    matches = matcher.knnMatch(desc, desc2, k=2)
    good_matches = [first for first, second in matches \
                    if first.distance < second.distance * ratio]
    return len(good_matches)


def goodMatch_sift(desc, desc2):
    # 첫번재 이웃의 거리가 두 번째 이웃 거리의 75% 이내인 것만 추출
    ratio = 0.75
    # BF-Hamming 생성
    matcher = cv2.BFMatcher(cv2.NORM_L2)  # NORM_L2 -> sift, surf에 적합     #cv.NORM_HAMMING -> ORB에 적합
    matches = matcher.knnMatch(desc, desc2, k=2)
    good_matches = [first for first, second in matches \
                    if first.distance < second.distance * ratio]
    return len(good_matches)



# <ORB로 검색>
folder_path = "./descriptor_fixedNumber/orb"

#폴더 내의 모든 npy 파일 읽어오기
obj_data_desc_list, obj_name_list = read_npy_files_in_folder(folder_path)

totalObj_num = len(obj_data_desc_list)
print("저장된 물체 수 : " + str(totalObj_num))
print("-------------------------------------")
#accuarcy 결과
orb_correct_num = 0

#검색할 이미지 폴더
image_path = "./data/searchData"
for count, search_object in enumerate(os.listdir(image_path)):
    _search_object_name = os.path.basename(search_object)
    search_object_name = os.path.splitext(_search_object_name)[0] #파일 이름에서 확장자 제거
    image = cv2.imread(image_path + "/" + search_object)
      #이미지 크기 조정
    height, width, _ = image.shape
    if height >= width:
        max_length = height
    else:
        max_length = width
    if max_length >= 1400:
        ratio = 1400 / max_length
        image = cv2.resize(image, (int(width * ratio), int(height * ratio)),  interpolation=cv2.INTER_AREA)
    image_result = remove(image)
    #검색할 이미지 orb 추출
    detector_orb = cv2.ORB_create()
    kp, desc = detector_orb.detectAndCompute(image_result, None)

    most_similar_obj_num = 0 #가장 비슷한 물체 번호
    most_similar_obj_num_decs = -1 #가장 많은 특징점 수
    #print("일치한 특징점 수")
    for i, obj_data_desc in  enumerate(obj_data_desc_list):
        goodMatch = goodMatch_orb(desc, obj_data_desc)
        #print(obj_name_list[i] + " : " +str(goodMatch))
        #print("i : " + str(i))
        if most_similar_obj_num_decs < goodMatch:
            most_similar_obj_num_decs = goodMatch
            most_similar_obj_num = i
            #print(most_similar_obj_num)
    print("------------")
    result_obj = os.path.splitext(str(obj_name_list[most_similar_obj_num]))[0]
    if search_object_name == result_obj:
        print(str(count+1) + "/" + str(totalObj_num))
        orb_correct_num += 1
    else:
        print("틀린 물체")
        print("입력한 사진 : " + search_object_name)
        print("조회된 물체 이름 : " + result_obj)

print("------------")
print("ORB ACCUARCY")
print(str(orb_correct_num) + "/" + str(totalObj_num))
print(str(orb_correct_num/totalObj_num) + "%")


#############################################################################################


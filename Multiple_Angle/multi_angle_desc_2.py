import cv2
import numpy as np
import os
from rembg import remove

#np.set_printoptions(threshold=np.inf)
#폴더에 있는 모든 이미지 특징점 저장 #ORB -> 빠름
def featurePoint_orb(folder_path):#output_file
    detector_orb = cv2.ORB_create()
    _ObjName = os.listdir(folder_path)[0]
    all_features = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image = cv2.imread(file_path)
            #이미지 크기 조정
            height, width, _ = image.shape
            if height >= width:
                max_length = height
            else:
                max_length = width
            if max_length >= 1400:
                ratio = 1400 / max_length
                image = cv2.resize(image, (int(width * ratio), int(height * ratio)),  interpolation=cv2.INTER_AREA)

            #image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #COLOR_BGR2HSV #COLOR_BGR2RGB
            image_result = remove(image) #배경 제거
            kp, desc = detector_orb.detectAndCompute(image_result, None)  # detector.compute(image, keypoins, descriptors):
            print(filename)
            print('keypoint:', len(kp), ' descriptor:', desc.shape) #orb 기본 검출 -> 500개
            """
            # 키 포인트 그리기
            img_draw = cv2.drawKeypoints(image_result, kp, None, \
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow('ORB', img_draw)
            cv2.waitKey()
            cv2.destroyAllWindows()
            """
            print("------------------------------------")
            all_features.append(desc)
            #all_features.append((kp, desc))
            #print(desc[0])
            _ObjName = _ObjName.split('.')[0]
            ObjName = _ObjName[:-1]
    return all_features, ObjName

def featurePoint_sift(folder_path):#output_file
    detector_sift = cv2.SIFT_create()
    _ObjName = os.listdir(folder_path)[0]
    all_features = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image = cv2.imread(file_path)
            #이미지 크기 조정
            height, width, _ = image.shape
            if height >= width:
                max_length = height
            else:
                max_length = width
            if max_length >= 1400:
                ratio = 1400 / max_length
                image = cv2.resize(image, (int(width * ratio), int(height * ratio)),  interpolation=cv2.INTER_AREA)
            #image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_result = remove(image)  # 배경 제거
            kp, desc = detector_sift.detectAndCompute(image_result, None)  # detector.compute(image, keypoins, descriptors):
            print(filename)
            print('keypoint:', len(kp), ' descriptor:', desc.shape)
            """
            # 키 포인트 그리기
            img_draw = cv2.drawKeypoints(image_result, kp, None, \
                                         flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS) #flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
                                                                                            # flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            # 이미지 크기 조정
            #img_draw = cv2.resize(img_draw, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
            cv2.imshow('SIFT', img_draw)
            cv2.waitKey()
            cv2.destroyAllWindows()
            """
            print("------------------------------------")
            all_features.append(desc)
            #all_features.append((kp, desc))
            #print(desc[0])
            _ObjName = _ObjName.split('.')[0]
            ObjName = _ObjName[:-1]
    return all_features, ObjName

# 특징점 필터링 -> 겹치는거
def filter_matching_features_orb(all_features):
    filtered_features = np.empty((0, 32), dtype=np.uint8) #[] #desc만
    ratio = 0.75
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    print("여러 각도 사진 갯수 : "+str(len(all_features)))
    print("----------------------------------------------------")
    for i in range(len(all_features)):
        if i == 0:
            filtered_features = np.append(filtered_features, all_features[0], axis=0) #
            #print(all_features[0][1])
            print(str(i+1) + "번째 : 추가된 특징점 갯수 : " + str(len(all_features[0])))
            #print(filtered_features)
            print("---------------------------------")
        else:
            #print(filtered_features.dtype)
            #print(all_features[i].dtype)
            """
            <knnMatch>
            matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k, mask, compactResult)
            queryDescriptors : : 특징 디스크립터 배열, 매칭의 기준이 될 디스크립터
            trainDescriptors: 특징 디스크립터 배열, 매칭의 대상이 될 디스크립터
            all_features[i]에서 filtered_features의 매치 찾기
            """
            matches = matcher.knnMatch(all_features[i], filtered_features, k=2) #500개 #all_features[i] 먼저

            good_matches = [first for first, second in matches \
                            if first.distance < second.distance * ratio]
            #same_matches = [first for first, second in matches \
            #                if first.distance == second.distance * ratio]
            #bad_matches = [first for first, second in matches \
            #                if first.distance > second.distance * ratio]
            print("잘 맞는 특징점 수 : " + str(len(good_matches)))
            bad_desc_Idx = [] #지금까지 저장된 특징점과 다른 특징점의 Idx
            #print("비어있는지 확인 : " + str(len(bad_desc_Idx)))
            # matches 리스트의 각 요소에 대해 반복
            for first, second in matches:
                if first.distance > second.distance * ratio: #distance: 매칭된 특징점 사이의 거리
                    bad_desc_Idx.append(first.queryIdx) #queryIdx: 매칭을 수행하는 기준이 되는 이미지의 특징점 인덱스
                                                        #trainIdx: 매칭 과정에서 비교 대상이 되는 이미지(학습이미지)의 특징점 인덱스
            #print(matches[0].trainIdx)
            #print(matches[1].trainIdx)
            print("다른 특징점 갯수 : " + str(len(bad_desc_Idx)))
            # 겹치지 않는 특징점(desc)를 filtered_features에 추가
            filtered_features_count = len(filtered_features)
            for Idx in bad_desc_Idx:
                filtered_features = np.append(filtered_features, [all_features[i][Idx]], axis=0) #다른 특징점 저장
            print(str(i+1) + "번째 : 추가된 특징점 갯수 : " + str(len(filtered_features) - filtered_features_count))
            print("저장된 특징점 총 수 : " + str(len(filtered_features)))
            print("---------------------------------")
    return filtered_features

def filter_matching_features_sift(all_features):
    filtered_features = np.empty((0, 128), dtype=np.uint8) #[] #desc만
    ratio = 0.75
    #matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    print("여러 각도 사진 갯수 : "+str(len(all_features)))
    print("----------------------------------------------------")
    for i in range(len(all_features)):
        if i == 0:
            filtered_features = np.append(filtered_features, all_features[0], axis=0) #
            print(str(i+1) + "번째 : 추가된 특징점 갯수 : " + str(len(all_features[0])))
            print("---------------------------------")
        else:
            matches = matcher.knnMatch(all_features[i], filtered_features, 2)
            good_matches = [first for first, second in matches \
                            if first.distance < second.distance * ratio]
            print("전체 특징점 수 : " + str(len(matches)))
            print("잘 맞는 특징점 수 : " + str(len(good_matches)))
            bad_desc_Idx = [] #지금까지 저장된 특징점과 다른 특징점의 Idx
            #print("비어있는지 확인 : " + str(len(bad_desc_Idx)))
            # matches 리스트의 각 요소에 대해 반복
            for first, second in matches:
                if first.distance > second.distance * ratio: #distance: 매칭된 특징점 사이의 거리
                    bad_desc_Idx.append(first.queryIdx) #queryIdx: 매칭을 수행하는 기준이 되는 이미지의 특징점 인덱스
                                                        #trainIdx: 매칭 과정에서 비교 대상이 되는 이미지(학습이미지)의 특징점 인덱스
            print("다른 특징점 갯수 : " + str(len(bad_desc_Idx)))
            # 겹치지 않는 특징점(desc)를 filtered_features에 추가
            filtered_features_count = len(filtered_features)
            for Idx in bad_desc_Idx:
                filtered_features = np.append(filtered_features, [all_features[i][Idx]], axis=0) #다른 특징점 저장
            print(str(i+1) + "번째 : 추가된 특징점 갯수 : " + str(len(filtered_features) - filtered_features_count))
            print("저장된 특징점 총 수 : " + str(len(filtered_features)))
            print("---------------------------------")
    return filtered_features


folder_path = "./data/New/whiteOutTape_b"
# <ORB 특징 추출>
all_feature, objName = featurePoint_orb(folder_path) #폴더 위치
# 특징점 필터링
filtered_features = filter_matching_features_orb(all_feature)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("최종 특징점 shape : " + str(filtered_features.shape))
print("물체 이름 : " + objName)
print("총 사진 갯수 : " + str(len(all_feature)))
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#특징점 numpy 배열 저장
np.save('./descriptor/orb/' + objName, filtered_features)


# <SIFT 특징 추출>
all_feature, objName = featurePoint_sift(folder_path) #폴더 위치
# 특징점 필터링
filtered_features = filter_matching_features_sift(all_feature)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("최종 특징점 shape : " + str(filtered_features.shape))
print("물체 이름 : " + objName)
print("총 사진 갯수 : " + str(len(all_feature)))
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#특징점 numpy 배열 저장
np.save('./descriptor/sift/' + objName, filtered_features)

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
            image_result = remove(image) #배경 제거
            kp, desc = detector_orb.detectAndCompute(image_result, None)  # detector.compute(image, keypoins, descriptors):
            """
            # 키 포인트 그리기
            img_draw = cv2.drawKeypoints(image_result, kp, None, \
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow('ORB', img_draw)
            cv2.waitKey()
            cv2.destroyAllWindows()
            """
            all_features.append(desc)
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
            image_result = remove(image)  # 배경 제거
            kp, desc = detector_sift.detectAndCompute(image_result, None)  # detector.compute(image, keypoins, descriptors):
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
            all_features.append(desc)
            _ObjName = _ObjName.split('.')[0]
            ObjName = _ObjName[:-1]
    return all_features, ObjName

# 특징점 필터링 -> 겹치는거
def filter_matching_features_orb(all_features):
    filtered_features = np.empty((0, 32), dtype=np.uint8) #[] #desc만
    ratio = 0.75
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    for i in range(len(all_features)):
        if i == 0:
            filtered_features = np.append(filtered_features, all_features[0], axis=0) #
        else:
            """
            <knnMatch>
            matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k, mask, compactResult)
            queryDescriptors : : 특징 디스크립터 배열, 매칭의 기준이 될 디스크립터
            trainDescriptors: 특징 디스크립터 배열, 매칭의 대상이 될 디스크립터
            all_features[i]에서 filtered_features의 매치 찾기
            """
            matches = matcher.knnMatch(all_features[i], filtered_features, k=2) #500개 #all_features[i] 먼저
            bad_desc_Idx = [] #지금까지 저장된 특징점과 다른 특징점의 Idx
            # matches 리스트의 각 요소에 대해 반복
            for first, second in matches:
                if first.distance > second.distance * ratio: #distance: 매칭된 특징점 사이의 거리
                    bad_desc_Idx.append(first.queryIdx) #queryIdx: 매칭을 수행하는 기준이 되는 이미지의 특징점 인덱스
                                                        #trainIdx: 매칭 과정에서 비교 대상이 되는 이미지(학습이미지)의 특징점 인덱스
            #5개 이상이면 그만
            f_number = filtered_features.shape[0]
            if f_number + len(bad_desc_Idx) > 5000:
                max_num = 5000 - f_number
                for Idx in bad_desc_Idx[:min(len(bad_desc_Idx), max_num)]:
                    filtered_features = np.append(filtered_features, [all_features[i][Idx]], axis=0)  # 다른 특징점 저장
                return filtered_features
            else:
                for Idx in bad_desc_Idx:
                    filtered_features = np.append(filtered_features, [all_features[i][Idx]], axis=0)  # 다른 특징점 저장
    return filtered_features

def filter_matching_features_sift(all_features):
    filtered_features = np.empty((0, 128), dtype=np.uint8) #[] #desc만
    ratio = 0.75
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    for i in range(len(all_features)):
        if i == 0:
            filtered_features = np.append(filtered_features, all_features[0], axis=0) #
        else:
            matches = matcher.knnMatch(all_features[i], filtered_features, 2)
            bad_desc_Idx = [] #지금까지 저장된 특징점과 다른 특징점의 Idx

            for first, second in matches:
                if first.distance > second.distance * ratio: #distance: 매칭된 특징점 사이의 거리
                    bad_desc_Idx.append(first.queryIdx) #queryIdx: 매칭을 수행하는 기준이 되는 이미지의 특징점 인덱스
                                                        #trainIdx: 매칭 과정에서 비교 대상이 되는 이미지(학습이미지)의 특징점 인덱스

            #5000개 이상이면 그만
            #현재 저장된 특징점
            f_number = filtered_features.shape[0]
            if f_number + len(bad_desc_Idx) > 5000:
                max_num = 5000 - f_number
                for Idx in bad_desc_Idx[:min(len(bad_desc_Idx), max_num)]:
                    filtered_features = np.append(filtered_features, [all_features[i][Idx]], axis=0)  # 다른 특징점 저장
                return filtered_features
            else:
                for Idx in bad_desc_Idx:
                    filtered_features = np.append(filtered_features, [all_features[i][Idx]], axis=0) #다른 특징점 저장
    return filtered_features

#폴더 내 모든 물건 특징점 추출
top_path =  "./data/storedData"

subDirectories = [f.path for f in os.scandir(top_path) if f.is_dir()]

for subdir in subDirectories:
    folder_path = top_path + "/" + os.path.basename(subdir)
    print(folder_path)

    """
    # <ORB 특징 추출>
    all_feature, objName = featurePoint_orb(folder_path) #폴더 위치
    # 특징점 필터링
    filtered_features = filter_matching_features_orb(all_feature)
    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    if filtered_features.shape[0] < 2500:
        print("특징점 갯수 부족")
    else :
        print("orb 최종 특징점 shape : " + str(filtered_features.shape))
        #print("물체 이름 : " + objName)
        #print("총 사진 갯수 : " + str(len(all_feature)))
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #특징점 numpy 배열 저장
        np.save('./descriptor_fixedNumber/orb/' + objName, filtered_features)
    print("------------------------")
    """
    #"""
    # <SIFT 특징 추출>
    all_feature, objName = featurePoint_sift(folder_path) #폴더 위치
    # 특징점 필터링
    filtered_features = filter_matching_features_sift(all_feature)
    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    if filtered_features.shape[0] < 2500:
        print("특징점 갯수 부족")
    else:
        print("sift 최종 특징점 shape : " + str(filtered_features.shape))
        #특징점 numpy 배열 저장
        np.save('./descriptor_fixedNumber/sift/' + objName, filtered_features)
    print("------------------------")
    #"""
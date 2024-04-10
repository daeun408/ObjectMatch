import os

# 변경할 폴더의 경로
folder_path = "./data/sunsrcreen_calming"
objName = os.path.basename(folder_path)

#print(objName)
# 폴더 내의 모든 파일 목록 가져오기
files = os.listdir(folder_path)

# 숫자
number = 0

# 폴더 내의 각 파일에 대해 반복
for file in files:
    # 파일의 확장자 확인하여 사진 파일인지 확인
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        if number == 0:
            new_name = f"{objName}.jpg"
            os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_name))
        else:
            new_name = f"{objName}{number}.jpg"
            # 파일명 변경
            os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_name))
        number += 1
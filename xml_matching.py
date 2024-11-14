import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np

def extract_heading_and_look_side(xml_path):
    """Extract heading angle and look side from XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    heading = float(root.find(".//heading").text)
    look_side = root.find(".//look_side").text
    return heading, look_side

def calculate_rotation_angle(base_heading, target_heading):
    """Calculate rotation angle between two headings, accounting for the shortest rotation path."""
    rotation_angle = target_heading - base_heading
    # Adjust rotation to be within the range -180 to 180 degrees
    if rotation_angle > 180:
        rotation_angle -= 360
    elif rotation_angle < -180:
        rotation_angle += 360
    return rotation_angle

# XML 파일 경로 리스트
xml_files = [
    "./tif/ICEYE_X8_GRD_SLH_434396_20220328T121951.xml",
    "./tif/ICEYE_X9_GRD_SLH_397343_20220329T122542.xml",
    "./tif/ICEYE_X9_GRD_SLH_397344_20220330T013450.xml",
    "./tif/ICEYE_X2_GRD_SLH_445296_20220331T014838.xml"
]

# 첫 번째 파일을 기준으로 설정
base_heading, base_look_side = extract_heading_and_look_side(xml_files[0])
print(f"Base heading (first image): {base_heading} degrees, Look side: {base_look_side}")

# 각 XML 파일에 대해 회전 각도 및 좌우 반전 여부 계산
rotation_angles = {}
for xml_file in xml_files[1:]:
    heading, look_side = extract_heading_and_look_side(xml_file)
    rotation_angle = calculate_rotation_angle(base_heading, heading)
    
    # 좌우 반전 여부를 결정: look_side가 "right"이면 좌우 반전
    flip_horizontal = (look_side == "right")
    
    # 특정 조건에 맞는 경우에만 시계 방향으로 60도 추가 회전 적용
    if Path(xml_file).stem in ["ICEYE_X9_GRD_SLH_397344_20220330T013450", "ICEYE_X2_GRD_SLH_445296_20220331T014838"]:
        rotation_angle -= 60  # 시계 방향으로 60도 추가 회전
    
    # 회전 각도와 좌우 반전 여부를 저장
    rotation_angles[Path(xml_file).stem] = (rotation_angle, flip_horizontal)
    
    print(f"File: {xml_file}, Heading: {heading} degrees, Rotation Angle: {rotation_angle} degrees, Flip Horizontal: {flip_horizontal}")

# 이미지 파일을 변환하고 적용
input_folder = Path("./tif")  # 원본 .tif 이미지 폴더
output_folder = Path("./jpg")  # 변환된 .jpg 파일을 저장할 폴더
output_folder.mkdir(parents=True, exist_ok=True)

# 이미지 변환 파라미터
target_size = 1024
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))

# 모든 .tif 파일에 대해 변환 및 처리
for tif_file in input_folder.glob("*.tif"):
    file_stem = tif_file.stem
    
    # 해당 이미지의 회전 각도와 좌우 반전 여부
    rotation_angle, flip_horizontal = rotation_angles.get(file_stem, (0, False))
    
    # 이미지 읽기
    image = cv2.imread(str(tif_file), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Failed to load {tif_file}")
        continue

    # 이미지 정규화 및 대비 향상
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    enhanced_image = clahe.apply(image)
    contrast_image = cv2.convertScaleAbs(enhanced_image, alpha=1.5, beta=40)
    
    # 좌우 반전 적용 (look_side가 "right"인 경우에만 수행)
    if flip_horizontal:
        contrast_image = cv2.flip(contrast_image, 1)

    # 회전 적용
    height, width = contrast_image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_image = cv2.warpAffine(contrast_image, rotation_matrix, (width, height))

    # 이미지 리사이즈
    if width > height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))
    resized_image = cv2.resize(rotated_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # JPG 파일로 저장
    output_path = output_folder / f"{file_stem}.jpg"
    cv2.imwrite(str(output_path), resized_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved rotated and flipped image as: {output_path}")

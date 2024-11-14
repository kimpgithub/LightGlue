from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 디바이스 설정 (GPU가 있으면 GPU 사용, 아니면 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SuperPoint 특징점 추출기 및 LightGlue 매칭기 초기화
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)

# 이미지 경로 설정
image0_path = Path("./jpg/ICEYE_X8_GRD_SLH_434396_20220328T121951.jpg")
image1_path = Path("./jpg/ICEYE_X2_GRD_SLH_445296_20220331T014838.jpg")

# 이미지 로드 및 디바이스로 이동
image0 = load_image(image0_path).to(device)
image1 = load_image(image1_path).to(device)

# 특징점 추출
feats0 = extractor.extract(image0)
feats1 = extractor.extract(image1)

# 특징점 매칭
matches01 = matcher({"image0": feats0, "image1": feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

# 매칭 결과에서 키포인트와 매칭된 점 추출
kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]

# 매칭이 있는지 확인
if matches.shape[0] > 0:
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    # 첫 번째 결과 저장 (매칭 라인 포함)
    fig, axes = plt.subplots(1, 1, figsize=(15, 10))
    viz2d.plot_images([image0.cpu(), image1.cpu()])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f"Stop after {matches01['stop']} layers", fs=20)

    output_path_matches = Path("output_match_result_with_lines.jpg")
    plt.savefig(output_path_matches, bbox_inches='tight')
    print(f"Saved match result with lines as: {output_path_matches}")
    plt.close(fig)
else:
    print("No matches found between the images.")

# 두 번째 결과 저장 (키포인트만 포함)
fig, axes = plt.subplots(1, 1, figsize=(15, 10))
kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
viz2d.plot_images([image0.cpu(), image1.cpu()])
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)

output_path_keypoints = Path("output_match_result_with_keypoints.jpg")
plt.savefig(output_path_keypoints, bbox_inches='tight')
print(f"Saved match result with keypoints as: {output_path_keypoints}")
plt.close(fig)
import streamlit as st
import cv2
import numpy as np
from matplotlib import pyplot as plt
# from PIL import Image # PIL은 st.image에 직접 사용될 때 유용하지만, 여기선 cv2로 처리

# --- 이미지 처리 및 스펙트럼 계산 함수 ---
def get_magnitude_spectrum_from_uploaded_file(uploaded_file):
    """
    Streamlit의 UploadedFile 객체를 입력받아 흑백 변환 후 푸리에 변환을 수행하고,
    로그 스케일의 크기 스펙트럼을 반환합니다.
    """
    if uploaded_file is None:
        return None, None, "오류: 파일이 업로드되지 않았습니다."

    try:
        # UploadedFile 객체에서 바이트 데이터를 읽어 NumPy 배열로 변환 후 OpenCV로 디코딩
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            # 업로드된 파일의 포인터를 처음으로 되돌려 파일 이름을 로깅 등에 사용
            uploaded_file.seek(0)
            return None, None, f"오류: '{uploaded_file.name}' 파일에서 이미지를 로드할 수 없습니다. 파일이 손상되었거나 지원하지 않는 이미지 형식일 수 있습니다."

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        uploaded_file.seek(0)
        return None, None, f"이미지 로드 및 흑백 변환 중 오류 발생 ({uploaded_file.name}): {e}"

    # DFT를 수행하기 위해 입력 이미지를 float32로 변환
    dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)

    # 저주파 성분을 중앙에 위치시키기 위해 주파수 영역을 이동
    dft_shift = np.fft.fftshift(dft)

    # 크기 스펙트럼 계산 (시각화를 위해 로그 스케일 적용)
    # log(0) 방지를 위해 작은 값(1e-9) 추가
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1e-9)

    return img_gray, magnitude_spectrum, None # 성공 시 오류 메시지는 None


# --- Streamlit 앱 UI 구성 ---
st.set_page_config(layout="wide") # 페이지 레이아웃을 넓게 사용
st.title("🖼️ 이미지 주파수 스펙트럼 분석기")
st.markdown("""
이 애플리케이션은 두 이미지(예: 실제 이미지와 딥페이크 의심 이미지)의 주파수 스펙트럼을
시각적으로 비교하여 분석할 수 있도록 도와줍니다. 푸리에 변환을 통해 이미지의 주파수 성분을 확인하고,
가짜 이미지에서 나타날 수 있는 인공적인 패턴이나 아티팩트를 관찰해볼 수 있습니다.
""")

# Matplotlib 스타일 설정
plt.style.use('seaborn-v0_8-darkgrid') # 기존 코드의 스타일 유지

# 파일 업로더를 위한 두 개의 컬럼 생성
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. 원본/실제 이미지 업로드")
    uploaded_real_file = st.file_uploader("실제(Real) 이미지를 업로드하세요.", type=["jpg", "jpeg", "png"], key="real_uploader")

with col2:
    st.subheader("2. 가짜/딥페이크 의심 이미지 업로드")
    uploaded_fake_file = st.file_uploader("가짜(Fake) 의심 이미지를 업로드하세요.", type=["jpg", "jpeg", "png"], key="fake_uploader")

# 두 파일이 모두 업로드되었는지 확인 후 분석 진행
if uploaded_real_file is not None and uploaded_fake_file is not None:
    st.header("📊 분석 결과 비교")

    # 각 파일에 대해 스펙트럼 계산
    real_img_gray, real_magnitude_spectrum, error_real = get_magnitude_spectrum_from_uploaded_file(uploaded_real_file)
    fake_img_gray, fake_magnitude_spectrum, error_fake = get_magnitude_spectrum_from_uploaded_file(uploaded_fake_file)

    # 오류 처리
    if error_real:
        st.error(f"실제 이미지 처리 중 오류: {error_real}")
    if error_fake:
        st.error(f"가짜 의심 이미지 처리 중 오류: {error_fake}")

    # 두 이미지 모두 성공적으로 처리된 경우에만 시각화
    if real_img_gray is not None and real_magnitude_spectrum is not None and \
       fake_img_gray is not None and fake_magnitude_spectrum is not None:

        fig, axs = plt.subplots(2, 2, figsize=(14, 10)) # 플롯 크기 조정

        # 실제 이미지 원본 (흑백)
        axs[0, 0].imshow(real_img_gray, cmap='gray')
        axs[0, 0].set_title(f'실제 이미지 (흑백): {uploaded_real_file.name}')
        axs[0, 0].axis('off') # 축 정보 숨기기

        # 실제 이미지 주파수 스펙트럼
        axs[0, 1].imshow(real_magnitude_spectrum, cmap='gray')
        axs[0, 1].set_title(f'실제 이미지 - 주파수 스펙트럼')
        axs[0, 1].axis('off')

        # 가짜 의심 이미지 원본 (흑백)
        axs[1, 0].imshow(fake_img_gray, cmap='gray')
        axs[1, 0].set_title(f'가짜 의심 (흑백): {uploaded_fake_file.name}')
        axs[1, 0].axis('off')

        # 가짜 의심 이미지 주파수 스펙트럼
        axs[1, 1].imshow(fake_magnitude_spectrum, cmap='gray')
        axs[1, 1].set_title(f'가짜 의심 - 주파수 스펙트럼')
        axs[1, 1].axis('off')

        plt.tight_layout() # 서브플롯 간 간격 자동 조절
        st.pyplot(fig) # Streamlit에 Matplotlib 플롯 표시

        st.markdown("""
        ---
        ### 📝 주파수 스펙트럼 관찰 시 참고 사항
        * **전체적인 패턴:** 실제 이미지는 일반적으로 자연스러운 주파수 분포를 보입니다. 인공적으로 생성되거나 조작된 이미지는 특정 주파수 대역에서 부자연스러운 패턴, 주기적인 노이즈, 또는 예상치 못한 선/격자무늬 아티팩트를 보일 수 있습니다.
        * **고주파 성분 (가장자리, 디테일):** 딥페이크는 때때로 실제 이미지보다 고주파 디테일이 부족하거나(블러 처리된 듯한 느낌), 반대로 특정 종류의 생성 모델은 불필요한 고주파 노이즈를 추가할 수 있습니다.
        * **주기적 아티팩트:** 일부 GAN(생성적 적대 신경망)이나 업샘플링 과정(예: Transposed Convolution)은 스펙트럼 상에 미세한 격자무늬(체커보드 패턴)나 특정 방향으로 정렬된 선 형태의 아티팩트를 남길 수 있습니다.
        * **스펙트럼의 '에너지' 분포:** 일반적으로 이미지의 에너지는 저주파(중심부)에 집중되고 고주파(주변부)로 갈수록 자연스럽게 감쇠합니다. 이 감쇠 패턴이 부자연스럽다면 조작을 의심해볼 수 있습니다.
        * **압축 아티팩트:** JPEG과 같은 손실 압축 알고리즘은 8x8 픽셀 블록 단위로 처리하므로, 강하게 압축된 이미지의 스펙트럼에는 블록 경계에 해당하는 주기적인 패턴이 나타날 수 있습니다.

        **주의:** 주파수 스펙트럼 분석은 이미지의 진위 여부를 판단하는 데 유용한 **보조 도구** 중 하나입니다. 시각적 비교만으로는 정확한 판별이 어려울 수 있으며, 특히 정교하게 만들어진 가짜 이미지의 경우 더욱 그렇습니다. 숙련된 분석가도 스펙트럼만으로 100% 확신하기는 어려우므로, 다양한 분석 방법을 함께 활용하는 것이 좋습니다.
        """)
    elif not error_real and not error_fake: # 이미지는 로드되었으나 스펙트럼 계산에 실패한 다른 경우 (이론상 위의 if에서 걸러짐)
        st.warning("이미지 처리 중 알 수 없는 오류로 시각화를 진행할 수 없습니다.")

elif uploaded_real_file or uploaded_fake_file: # 둘 중 하나만 업로드된 경우
    st.info("분석을 시작하려면 두 개의 이미지를 모두 업로드해주세요.")
else: # 아무것도 업로드되지 않은 초기 상태
    st.info("위에서 분석할 실제 이미지와 가짜 의심 이미지를 업로드해주세요.")

# 사이드바 정보
st.sidebar.header("앱 정보")
st.sidebar.markdown("""
이 애플리케이션은 업로드된 두 이미지의 주파수 스펙트럼을 계산하고 시각화하여 비교 분석할 수 있도록 돕습니다.
Colab에서 제공된 Python 스크립트를 기반으로 제작되었습니다.

**주요 기능:**
- 이미지 파일 업로드 (2개)
- OpenCV를 사용한 이미지 흑백 변환
- NumPy를 사용한 2D 푸리에 변환(DFT) 수행
- Matplotlib을 사용한 로그 스케일 크기 스펙트럼 시각화
""")
st.sidebar.markdown("---")
st.sidebar.markdown("만든이: AI (사용자 코드 기반)")

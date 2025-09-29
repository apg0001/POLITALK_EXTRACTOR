import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from gui_manager import run_gui
from login import show_login_window
import torch
import glob


def set_java_home():
    # JDK가 설치될 가능성이 높은 디렉토리 목록
    possible_paths = [
        r"C:\Program Files\Java",
        r"C:\Program Files (x86)\Java"
    ]

    java_home = None

    # 설치된 JDK 디렉토리를 검색
    for path in possible_paths:
        if os.path.exists(path):
            # JDK 디렉토리 찾기 (jdk-로 시작하는 폴더)
            jdk_paths = glob.glob(os.path.join(path, "jdk-*"))
            if jdk_paths:
                # 가장 최신 버전을 선택 (알파벳/숫자순 정렬 기준)
                java_home = sorted(jdk_paths)[-1]
                break

    if java_home:
        # JAVA_HOME 설정
        os.environ["JAVA_HOME"] = java_home
        # PATH 환경 변수에 추가
        os.environ["Path"] = os.environ["Path"] + \
            ";" + os.path.join(java_home, "bin")
        print(f"JAVA_HOME 설정 완료: {java_home}")
    else:
        print("JDK를 찾을 수 없습니다. JDK를 설치하거나 JAVA_HOME을 수동으로 설정하세요.")


if __name__ == "__main__":
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version: {torch.version.cuda}")

    # 실행
    set_java_home()

    # 확인
    print(f"JAVA_HOME: {os.getenv('JAVA_HOME')}")

    is_logged_in = show_login_window()

    if is_logged_in:
        run_gui()
    else:
        print("로그인 실패. 프로그램 종료.")
    # run_gui()

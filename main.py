import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import glob
import torch
from gui_manager import run_gui
from login import LoginWindow


class ApplicationLauncher:
    """애플리케이션 실행을 담당하는 클래스
    
    프로그램의 전체 실행 흐름을 관리합니다.
    JAVA_HOME 설정, CUDA 확인, 로그인, GUI 실행 등의 기능을 제공합니다.
    """
    
    def __init__(self):
        """ApplicationLauncher 초기화"""
        # self.java_home = None
        self.login_window = LoginWindow()

    # def set_java_home(self):
    #     """JDK 자동 감지 및 JAVA_HOME 설정"""
    #     possible_paths = [
    #         r"C:\Program Files\Java",
    #         r"C:\Program Files (x86)\Java"
    #     ]

    #     for path in possible_paths:
    #         if os.path.exists(path):
    #             jdk_paths = glob.glob(os.path.join(path, "jdk-*"))
    #             if jdk_paths:
    #                 self.java_home = sorted(jdk_paths)[-1]
    #                 break

    #     if self.java_home:
    #         os.environ["JAVA_HOME"] = self.java_home
    #         os.environ["Path"] = os.environ["Path"] + ";" + os.path.join(self.java_home, "bin")
    #         print(f"JAVA_HOME 설정 완료: {self.java_home}")
    #     else:
    #         print("JDK를 찾을 수 없습니다. JDK를 설치하거나 JAVA_HOME을 수동으로 설정하세요.")

    def check_cuda_availability(self):
        """CUDA 가용성 확인"""
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"CUDA Version: {torch.version.cuda}")

    def run(self):
        """애플리케이션 실행
        
        전체 프로그램 실행 흐름을 관리합니다:
        1. CUDA 가용성 확인
        2. JAVA_HOME 설정
        3. 로그인 처리
        4. GUI 실행
        """
        # CUDA 가용성 확인
        self.check_cuda_availability()

        # JAVA_HOME 설정
        # self.set_java_home()
        # print(f"JAVA_HOME: {os.getenv('JAVA_HOME')}")

        # 로그인 후 GUI 실행
        is_logged_in = self.login_window.show_login_window()

        if is_logged_in:
            run_gui()
        else:
            print("로그인 실패. 프로그램 종료.")


def main():
    """메인 함수"""
    launcher = ApplicationLauncher()
    launcher.run()


if __name__ == "__main__":
    main()
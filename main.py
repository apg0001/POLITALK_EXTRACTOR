import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
        self.login_window = LoginWindow()

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
        self.check_cuda_availability()

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
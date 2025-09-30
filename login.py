import tkinter as tk
from tkinter import messagebox


class LoginWindow:
    """로그인 창을 담당하는 클래스
    
    사용자 인증을 위한 로그인 인터페이스를 제공합니다.
    Tkinter를 사용하여 간단한 로그인 창을 구현합니다.
    """
    
    def __init__(self):
        """LoginWindow 초기화
        
        로그인 창의 상태와 위젯들을 초기화합니다.
        """
        self.root = None
        self.login_success = False
        self.entry_id = None
        self.entry_password = None

    def show_login_window(self):
        """로그인 창 표시 및 로그인 처리"""
        self.login_success = False
        self._create_login_window()
        self.root.mainloop()
        return self.login_success

    def _create_login_window(self):
        """로그인 창 생성"""
        self.root = tk.Tk()
        self.root.title("로그인 창")
        self.root.geometry("300x200")

        # 창 닫기 이벤트 설정
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # ID 입력 라벨 및 엔트리
        label_id = tk.Label(self.root, text="ID:")
        label_id.pack(pady=5)
        
        self.entry_id = tk.Entry(self.root)
        self.entry_id.pack(pady=5)
        self.entry_id.insert(0, "admin")

        # 비밀번호 입력 라벨 및 엔트리
        label_password = tk.Label(self.root, text="비밀번호:")
        label_password.pack(pady=5)
        
        self.entry_password = tk.Entry(self.root, show="*")
        self.entry_password.pack(pady=5)
        self.entry_password.insert(0, "password")

        # 로그인 버튼
        login_button = tk.Button(self.root, text="로그인", command=self._login)
        login_button.pack(pady=20)

        # 엔터 키 이벤트 바인딩
        self.root.bind('<Return>', self._login)

    def _login(self, event=None):
        """로그인 처리"""
        user_id = self.entry_id.get()
        password = self.entry_password.get()

        if user_id == "admin" and password == "password":
            self.login_success = True
            self.root.destroy()
        else:
            messagebox.showerror("로그인 실패", "ID 또는 비밀번호가 잘못되었습니다.")

    def _on_closing(self):
        """창 닫기 시 동작"""
        self.login_success = False
        self.root.destroy()


def show_login_window():
    """로그인 창 표시 함수 (하위 호환성)"""
    login_window = LoginWindow()
    return login_window.show_login_window()
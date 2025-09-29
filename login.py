import tkinter as tk
from tkinter import messagebox

def show_login_window():
    # 로그인 성공 여부를 저장하는 변수
    login_success = False

    def login(event=None):  # event 매개변수 추가
        nonlocal login_success  # 함수 외부의 변수를 사용하기 위해 nonlocal 선언
        user_id = entry_id.get()
        password = entry_password.get()

        # ID와 비밀번호 확인
        if user_id == "admin" and password == "password":
            # messagebox.showinfo("로그인 성공", "환영합니다!")
            login_success = True  # 로그인 성공으로 설정
            root.destroy()  # 창 닫기
        else:
            messagebox.showerror("로그인 실패", "ID 또는 비밀번호가 잘못되었습니다.")

    # 창 닫기 시 동작 설정
    def on_closing():
        nonlocal login_success
        login_success = False  # 창을 닫으면 로그인 실패로 설정
        root.destroy()

    # tkinter 윈도우 생성
    root = tk.Tk()
    root.title("로그인 창")
    root.geometry("300x200")

    # 창 닫기 이벤트 설정
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # ID 입력 라벨 및 엔트리
    label_id = tk.Label(root, text="ID:")
    label_id.pack(pady=5)
    entry_id = tk.Entry(root)
    entry_id.pack(pady=5)
    entry_id.insert(0, "admin")  # 기본 ID 값 설정

    # 비밀번호 입력 라벨 및 엔트리
    label_password = tk.Label(root, text="비밀번호:")
    label_password.pack(pady=5)
    entry_password = tk.Entry(root, show="*")
    entry_password.pack(pady=5)
    entry_password.insert(0, "password")  # 기본 비밀번호 값 설정

    # 로그인 버튼
    login_button = tk.Button(root, text="로그인", command=login)
    login_button.pack(pady=20)

    # 엔트리 위젯에 이벤트 바인딩 추가
    root.bind('<Return>', login)  # 엔터 키 이벤트 바인딩

    root.mainloop()

    return login_success  # 로그인 성공 여부 반환
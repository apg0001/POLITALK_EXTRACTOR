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
        
        # 색상 테마 정의
        self.colors = {
            'primary': '#2E86AB',      # 메인 블루
            'secondary': '#A23B72',    # 보라색
            'accent': '#F18F01',       # 오렌지
            'success': '#C73E1D',      # 빨간색
            'background': '#F5F5F5',   # 연한 회색
            'surface': '#FFFFFF',      # 흰색
            'text': '#2C3E50',         # 진한 회색
            'text_light': '#7F8C8D',   # 연한 회색
            'border': '#BDC3C7',       # 테두리 회색
            'hover': '#3498DB'         # 호버 색상
        }

    def show_login_window(self):
        """로그인 창 표시 및 로그인 처리"""
        self.login_success = False
        self._create_login_window()
        self.root.mainloop()
        return self.login_success

    def _create_login_window(self):
        """로그인 창 위젯 생성 및 배치"""
        self.root = tk.Tk()
        self.root.title("🔐 로그인")
        self.root.geometry("400x600")
        self.root.configure(bg=self.colors['background'])
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # 창 중앙에 배치
        self._center_window()
        
        # 메인 컨테이너
        main_frame = tk.Frame(self.root, bg=self.colors['background'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)
        
        # 로고/제목 섹션
        self._create_header(main_frame)
        
        # 로그인 폼 섹션
        self._create_login_form(main_frame)
        
        # 푸터 섹션
        self._create_footer(main_frame)

    def _center_window(self):
        """창을 화면 중앙에 배치"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def _create_header(self, parent):
        """헤더 섹션 생성"""
        header_frame = tk.Frame(parent, bg=self.colors['background'])
        header_frame.pack(fill=tk.X, pady=(0, 30))
        
        # 제목
        title_label = tk.Label(
            header_frame,
            text="🔐 로그인",
            font=("맑은 고딕", 24, "bold"),
            fg=self.colors['primary'],
            bg=self.colors['background']
        )
        title_label.pack()
        
        # 부제목
        subtitle_label = tk.Label(
            header_frame,
            text="CSV to Excel 변환기에 로그인하세요",
            font=("맑은 고딕", 12),
            fg=self.colors['text_light'],
            bg=self.colors['background']
        )
        subtitle_label.pack(pady=(5, 0))

    def _create_login_form(self, parent):
        """로그인 폼 섹션 생성"""
        form_frame = tk.Frame(parent, bg=self.colors['surface'], relief=tk.RAISED, bd=1)
        form_frame.pack(fill=tk.X, pady=(0, 20))
        
        # 내부 패딩
        inner_frame = tk.Frame(form_frame, bg=self.colors['surface'])
        inner_frame.pack(fill=tk.X, padx=30, pady=30)
        
        # ID 입력
        id_frame = tk.Frame(inner_frame, bg=self.colors['surface'])
        id_frame.pack(fill=tk.X, pady=(0, 20))
        
        id_label = tk.Label(
            id_frame,
            text="👤 사용자 ID",
            font=("맑은 고딕", 12, "bold"),
            fg=self.colors['text'],
            bg=self.colors['surface']
        )
        id_label.pack(anchor=tk.W, pady=(0, 8))
        
        self.entry_id = tk.Entry(
            id_frame,
            font=("맑은 고딕", 12),
            relief=tk.SOLID,
            bd=1,
            bg=self.colors['surface'],
            fg=self.colors['text'],
            insertbackground=self.colors['primary']
        )
        self.entry_id.pack(fill=tk.X, pady=(0, 5))
        self.entry_id.insert(0, "admin")
        
        # 비밀번호 입력
        password_frame = tk.Frame(inner_frame, bg=self.colors['surface'])
        password_frame.pack(fill=tk.X, pady=(0, 20))
        
        password_label = tk.Label(
            password_frame,
            text="🔒 비밀번호",
            font=("맑은 고딕", 12, "bold"),
            fg=self.colors['text'],
            bg=self.colors['surface']
        )
        password_label.pack(anchor=tk.W, pady=(0, 8))
        
        self.entry_password = tk.Entry(
            password_frame,
            font=("맑은 고딕", 12),
            show="*",
            relief=tk.SOLID,
            bd=1,
            bg=self.colors['surface'],
            fg=self.colors['text'],
            insertbackground=self.colors['primary']
        )
        self.entry_password.pack(fill=tk.X, pady=(0, 5))
        self.entry_password.insert(0, "password")
        
        # 로그인 버튼
        login_button = tk.Button(
            inner_frame,
            text="🚀 로그인",
            command=self._login,
            font=("맑은 고딕", 14, "bold"),
            bg=self.colors['primary'],
            fg='white',
            relief=tk.FLAT,
            bd=0,
            padx=40,
            pady=12,
            cursor='hand2'
        )
        login_button.pack(pady=(10, 0))
        
        # 버튼 호버 효과
        self._add_button_hover_effect(login_button, self.colors['primary'], self.colors['hover'])
        
        # Enter 키 바인딩
        self.root.bind('<Return>', self._login)

    def _create_footer(self, parent):
        """푸터 섹션 생성"""
        footer_frame = tk.Frame(parent, bg=self.colors['background'])
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # 도움말 텍스트
        help_label = tk.Label(
            footer_frame,
            text="💡 기본 계정: admin / password",
            font=("맑은 고딕", 10),
            fg=self.colors['text_light'],
            bg=self.colors['background']
        )
        help_label.pack()

    def _add_button_hover_effect(self, button, original_color, hover_color):
        """버튼에 호버 효과 추가"""
        def on_enter(event):
            button.configure(bg=hover_color)
        
        def on_leave(event):
            button.configure(bg=original_color)
        
        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)

    def _login(self, event=None):
        """로그인 시도 및 결과 처리"""
        user_id = self.entry_id.get()
        password = self.entry_password.get()

        if user_id == "admin" and password == "password":
            self.login_success = True
            self.root.destroy()
        else:
            # 오류 메시지 창 스타일링
            error_window = tk.Toplevel(self.root)
            error_window.title("❌ 로그인 실패")
            error_window.geometry("300x150")
            error_window.configure(bg=self.colors['background'])
            error_window.resizable(False, False)
            
            # 중앙 배치
            error_window.transient(self.root)
            error_window.grab_set()
            
            # 오류 메시지
            error_label = tk.Label(
                error_window,
                text="❌ 로그인 실패",
                font=("맑은 고딕", 14, "bold"),
                fg=self.colors['success'],
                bg=self.colors['background']
            )
            error_label.pack(pady=20)
            
            message_label = tk.Label(
                error_window,
                text="ID 또는 비밀번호가 잘못되었습니다.",
                font=("맑은 고딕", 10),
                fg=self.colors['text'],
                bg=self.colors['background']
            )
            message_label.pack(pady=5)
            
            # 확인 버튼
            ok_button = tk.Button(
                error_window,
                text="확인",
                command=error_window.destroy,
                font=("맑은 고딕", 10, "bold"),
                bg=self.colors['primary'],
                fg='white',
                relief=tk.FLAT,
                bd=0,
                padx=20,
                pady=8,
                cursor='hand2'
            )
            ok_button.pack(pady=15)

    def _on_closing(self):
        """창 닫기 이벤트 처리"""
        self.login_success = False
        self.root.destroy()
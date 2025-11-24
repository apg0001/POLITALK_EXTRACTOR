from tkinter import filedialog, messagebox, ttk
import tkinter as tk
import datetime
import os
import sys
import traceback
from file_manager import FileProcessor


class CSVExcelConverterGUI:
    """CSV to Excel 변환기 GUI 클래스
    
    Tkinter를 사용하여 사용자 인터페이스를 제공합니다.
    파일 선택, 변환 실행, 진행상황 표시 등의 기능을 포함합니다.
    """
    
    def __init__(self):
        """CSVExcelConverterGUI 초기화
        
        GUI 컴포넌트들과 파일 프로세서를 초기화합니다.
        """
        self.root = None
        self.file_processor = FileProcessor()
        self.csv_file_entry = None
        self.excel_file_entry = None
        self.progress_bar = None
        self.progress_label = None
        self.run_button = None
        
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

    def run_gui(self):
        """Tkinter 기반 GUI 실행"""
        self.root = tk.Tk()
        self.root.title("행합치기 및 중복제거")
        self.root.geometry("900x700")
        self.root.configure(bg=self.colors['background'])
        
        # 창 중앙에 배치
        self._center_window()
        
        # 아이콘 설정 (icon.png 우선, icon.ico 있으면 함께 시도)
        try:
            if getattr(sys, 'frozen', False):
                base_path = os.path.dirname(sys.executable)
            else:
                base_path = os.path.dirname(os.path.abspath(__file__))

            # 우선순위: 실행 경로의 파일 → MEIPASS 리소스 → 현재 작업 디렉토리
            candidates = [
                os.path.join(base_path, "icon.png"),
                os.path.join(getattr(sys, "_MEIPASS", base_path)),
                os.path.join(os.path.abspath("."), "icon.png"),
            ]
            icon_png = next((p for p in candidates if isinstance(p, str) and p.endswith("icon.png") and os.path.exists(p)), None)

            if icon_png:
                # PhotoImage가 가비지 컬렉션 되지 않도록 참조 유지
                self._icon_img = tk.PhotoImage(file=icon_png)
                self.root.iconphoto(True, self._icon_img)

            # Windows 작업 표시줄용 .ico가 있다면 추가 설정
            ico_path = os.path.join(base_path, "icon.ico")
            if os.path.exists(ico_path):
                try:
                    self.root.iconbitmap(ico_path)
                except Exception:
                    pass

        except Exception as _:
            pass

        self._create_widgets()
        self.root.mainloop()
    
    def _center_window(self):
        """창을 화면 중앙에 배치"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def _create_widgets(self):
        """GUI 위젯 생성"""
        # 메인 컨테이너
        main_frame = tk.Frame(self.root, bg=self.colors['background'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 헤더 섹션
        self._create_header(main_frame)
        
        # 파일 선택 섹션
        self._create_file_section(main_frame)
        
        # 진행률 섹션
        self._create_progress_section(main_frame)
        
        # 버튼 섹션
        self._create_button_section(main_frame)
        
        # 푸터 섹션
        self._create_footer(main_frame)

    def _create_header(self, parent):
        """헤더 섹션 생성"""
        header_frame = tk.Frame(parent, bg=self.colors['background'])
        header_frame.pack(fill=tk.X, pady=(0, 30))
        
        # 제목
        title_label = tk.Label(
            header_frame, 
            text="행합치기 및 중복제거", 
            font=("맑은 고딕", 24, "bold"),
            fg=self.colors['primary'],
            bg=self.colors['background']
        )
        title_label.pack()
        
        # 부제목
        subtitle_label = tk.Label(
            header_frame,
            text="AI 기반 발언문 분석 및 Excel 변환 도구",
            font=("맑은 고딕", 12),
            fg=self.colors['text_light'],
            bg=self.colors['background']
        )
        subtitle_label.pack(pady=(5, 0))

    def _create_file_section(self, parent):
        """파일 선택 섹션 생성"""
        file_frame = tk.Frame(parent, bg=self.colors['surface'], relief=tk.RAISED, bd=1)
        file_frame.pack(fill=tk.X, pady=(0, 20))
        
        # 내부 패딩
        inner_frame = tk.Frame(file_frame, bg=self.colors['surface'])
        inner_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # CSV 파일 선택
        csv_section = tk.Frame(inner_frame, bg=self.colors['surface'])
        csv_section.pack(fill=tk.X, pady=(0, 15))
        
        csv_label = tk.Label(
            csv_section,
            text="📁 CSV 파일 선택",
            font=("맑은 고딕", 14, "bold"),
            fg=self.colors['text'],
            bg=self.colors['surface']
        )
        csv_label.pack(anchor=tk.W, pady=(0, 8))
        
        csv_input_frame = tk.Frame(csv_section, bg=self.colors['surface'])
        csv_input_frame.pack(fill=tk.X)
        
        self.csv_file_entry = tk.Entry(
            csv_input_frame,
            font=("맑은 고딕", 11),
            relief=tk.SOLID,
            bd=1,
            bg=self.colors['surface'],
            fg=self.colors['text'],
            insertbackground=self.colors['primary']
        )
        self.csv_file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        csv_button = tk.Button(
            csv_input_frame,
            text="📂 찾아보기",
            command=self._select_csv_file,
            font=("맑은 고딕", 10, "bold"),
            bg=self.colors['primary'],
            fg='white',
            relief=tk.FLAT,
            bd=0,
            padx=20,
            pady=8,
            cursor='hand2'
        )
        csv_button.pack(side=tk.RIGHT)
        
        # Excel 파일 저장 위치
        excel_section = tk.Frame(inner_frame, bg=self.colors['surface'])
        excel_section.pack(fill=tk.X)
        
        excel_label = tk.Label(
            excel_section,
            text="💾 Excel 파일 저장 위치",
            font=("맑은 고딕", 14, "bold"),
            fg=self.colors['text'],
            bg=self.colors['surface']
        )
        excel_label.pack(anchor=tk.W, pady=(0, 8))
        
        excel_input_frame = tk.Frame(excel_section, bg=self.colors['surface'])
        excel_input_frame.pack(fill=tk.X)
        
        self.excel_file_entry = tk.Entry(
            excel_input_frame,
            font=("맑은 고딕", 11),
            relief=tk.SOLID,
            bd=1,
            bg=self.colors['surface'],
            fg=self.colors['text'],
            insertbackground=self.colors['primary']
        )
        self.excel_file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        excel_button = tk.Button(
            excel_input_frame,
            text="📁 저장 위치",
            command=self._select_excel_file,
            font=("맑은 고딕", 10, "bold"),
            bg=self.colors['secondary'],
            fg='white',
            relief=tk.FLAT,
            bd=0,
            padx=20,
            pady=8,
            cursor='hand2'
        )
        excel_button.pack(side=tk.RIGHT)

    def _create_progress_section(self, parent):
        """진행률 섹션 생성"""
        progress_frame = tk.Frame(parent, bg=self.colors['surface'], relief=tk.RAISED, bd=1)
        progress_frame.pack(fill=tk.X, pady=(0, 20))
        
        # 내부 패딩
        inner_frame = tk.Frame(progress_frame, bg=self.colors['surface'])
        inner_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # 진행률 라벨
        self.progress_label = tk.Label(
            inner_frame,
            text="⏳ 변환 실행 버튼을 눌러주세요.",
            font=("맑은 고딕", 12),
            fg=self.colors['text'],
            bg=self.colors['surface']
        )
        self.progress_label.pack(pady=(0, 10))
        
        # 진행률 바 스타일 설정
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            "Custom.Horizontal.TProgressbar",
            background=self.colors['primary'],
            troughcolor=self.colors['border'],
            borderwidth=0,
            lightcolor=self.colors['primary'],
            darkcolor=self.colors['primary']
        )
        
        self.progress_bar = ttk.Progressbar(
            inner_frame,
            orient="horizontal",
            length=400,
            mode="determinate",
            style="Custom.Horizontal.TProgressbar"
        )
        self.progress_bar.pack(pady=(0, 10))

    def _create_button_section(self, parent):
        """버튼 섹션 생성"""
        button_frame = tk.Frame(parent, bg=self.colors['background'])
        button_frame.pack(fill=tk.X, pady=(0, 20))
        
        # 실행 버튼
        self.run_button = tk.Button(
            button_frame,
            text="변환 시작",
            command=self._process_file,
            font=("맑은 고딕", 16, "bold"),
            bg=self.colors['accent'],
            fg='white',
            relief=tk.FLAT,
            bd=0,
            padx=40,
            pady=15,
            cursor='hand2'
        )
        self.run_button.pack()
        
        # 버튼 호버 효과
        self._add_button_hover_effect(self.run_button, self.colors['accent'], self.colors['hover'])

    def _create_footer(self, parent):
        """푸터 섹션 생성"""
        footer_frame = tk.Frame(parent, bg=self.colors['background'])
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # 상태 표시
        status_label = tk.Label(
            footer_frame,
            text="💡 CSV 파일을 선택하고 변환을 시작하세요",
            font=("맑은 고딕", 10),
            fg=self.colors['text_light'],
            bg=self.colors['background']
        )
        status_label.pack()

    def _add_button_hover_effect(self, button, original_color, hover_color):
        """버튼에 호버 효과 추가"""
        def on_enter(event):
            button.configure(bg=hover_color)
        
        def on_leave(event):
            button.configure(bg=original_color)
        
        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)

    def _select_csv_file(self):
        """CSV 파일 선택 대화상자"""
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV 파일", "*.csv"), ("모든 파일", "*.*")],
            title="CSV 파일을 선택하세요"
        )
        if file_path:
            self.csv_file_entry.delete(0, tk.END)
            self.csv_file_entry.insert(0, file_path)

            # 현재 날짜를 YYMMDD 형식으로 설정
            formatted_date = datetime.datetime.now().strftime('%y%m%d')
            excel_file_path = file_path.replace(".csv", f"_AI변환{formatted_date}.xlsx")
            
            self.excel_file_entry.delete(0, tk.END)
            self.excel_file_entry.insert(0, excel_file_path)

    def _select_excel_file(self):
        """Excel 저장 위치 선택"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel 파일", "*.xlsx"), ("모든 파일", "*.*")],
            title="Excel 파일 저장 위치를 선택하세요"
        )
        if file_path:
            self.excel_file_entry.delete(0, tk.END)
            self.excel_file_entry.insert(0, file_path)

    def _reset_gui_error(self):
        """오류 발생 시 GUI를 초기화하고 재시작"""
        messagebox.showinfo("재시작", "오류가 발생하여 프로그램을 재시작합니다.")
        self.root.destroy()
        self.run_gui()
        
    def _reset_gui(self):
        """저장 완료 시 GUI를 초기화하고 재시작"""
        messagebox.showinfo("완료", "저장이 완료되어 프로그램을 재시작합니다.")
        self.root.destroy()
        self.run_gui()

    def _process_file(self):
        """CSV 데이터를 Excel로 변환하는 함수"""
        try:
            self.run_button.config(state=tk.DISABLED, text="⏳ 처리 중...")
            self.run_button.configure(bg=self.colors['text_light'])
            
            csv_file = self.csv_file_entry.get()
            excel_file = self.excel_file_entry.get()

            if not csv_file or not excel_file:
                raise ValueError("CSV 파일과 Excel 파일을 모두 선택해야 합니다.")

            # CSV에서 데이터 추출 및 Excel 저장
            extracted_data = self.file_processor.extract_text_from_csv(csv_file, self.progress_bar, self.progress_label)
            print(f"csv에서 추출된 데이터 수 {len(extracted_data)}")
            
            # CSV 파일 내용 병합
            merged_data = self.file_processor.merge_data(extracted_data, self.progress_bar, self.progress_label)
            print(f"병합 후 데이터 수 {len(merged_data)}")
            
            # 중복 내용 제거
            duplicate_removed_data = self.file_processor.remove_duplicates(merged_data, self.progress_bar, self.progress_label)
            print(f"중복 제거 후 데이터 수 {len(duplicate_removed_data)}")
            
            # Excel 파일 저장
            self.file_processor.save_data_to_excel(duplicate_removed_data, excel_file, self.progress_bar, self.progress_label)

            # 성공 메시지
            success_window = tk.Toplevel(self.root)
            success_window.title("✅ 변환 완료")
            success_window.geometry("400x300")
            success_window.configure(bg=self.colors['background'])
            success_window.resizable(False, False)
            
            # 중앙 배치
            success_window.transient(self.root)
            success_window.grab_set()
            
            # 성공 메시지
            success_label = tk.Label(
                success_window,
                text="변환이 완료되었습니다!",
                font=("맑은 고딕", 16, "bold"),
                fg=self.colors['success'],
                bg=self.colors['background']
            )
            success_label.pack(pady=20)
            
            file_label = tk.Label(
                success_window,
                text=f"📁 저장 위치: {excel_file}",
                font=("맑은 고딕", 10),
                fg=self.colors['text'],
                bg=self.colors['background'],
                wraplength=350
            )
            file_label.pack(pady=10)
            
            # 확인 버튼
            ok_button = tk.Button(
                success_window,
                text="확인",
                command=success_window.destroy,
                font=("맑은 고딕", 12, "bold"),
                bg=self.colors['primary'],
                fg='white',
                relief=tk.FLAT,
                bd=0,
                padx=30,
                pady=10,
                cursor='hand2'
            )
            ok_button.pack(pady=20)
            
            # 버튼 상태 복원
            self.run_button.config(state=tk.NORMAL, text="변환 시작")
            self.run_button.configure(bg=self.colors['accent'])

        except ValueError as ve:
            messagebox.showwarning("입력 오류", str(ve))
            self.run_button.config(state=tk.NORMAL, text="변환 시작")
            self.run_button.configure(bg=self.colors['accent'])
        except Exception as e:
            error_details = traceback.format_exc()
            messagebox.showerror("오류 발생", f"예상치 못한 오류가 발생했습니다.\n{str(e)}")
            self.run_button.config(state=tk.NORMAL, text="변환 시작")
            self.run_button.configure(bg=self.colors['accent'])


def run_gui():
    """GUI 실행 함수 (하위 호환성)"""
    gui = CSVExcelConverterGUI()
    gui.run_gui()
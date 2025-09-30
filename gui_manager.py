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

    def run_gui(self):
        """Tkinter 기반 GUI 실행"""
        self.root = tk.Tk()
        self.root.title("CSV - Excel 변환기")
        self.root.geometry("600x400")

        self._create_widgets()
        self.root.mainloop()

    def _create_widgets(self):
        """GUI 위젯 생성"""
        # CSV 파일 선택
        csv_file_label = tk.Label(self.root, text="CSV 파일 선택:")
        csv_file_label.pack(pady=5)
        
        self.csv_file_entry = tk.Entry(self.root, width=70)
        self.csv_file_entry.pack(pady=5)
        
        csv_file_button = tk.Button(self.root, text="파일 선택", command=self._select_csv_file)
        csv_file_button.pack(pady=5)

        # Excel 저장 위치
        excel_file_label = tk.Label(self.root, text="Excel 파일 저장 위치:")
        excel_file_label.pack(pady=5)
        
        self.excel_file_entry = tk.Entry(self.root, width=70)
        self.excel_file_entry.pack(pady=5)
        
        excel_file_button = tk.Button(self.root, text="다른 이름으로 저장", command=self._select_excel_file)
        excel_file_button.pack(pady=5)

        # Progressbar 설정
        self.progress_label = tk.Label(self.root, text="변환 실행 버튼을 눌러주세요.")
        self.progress_label.pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(self.root, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.pack(pady=5)

        # 실행 버튼
        self.run_button = tk.Button(self.root, text="변환 실행", command=self._process_file, fg='green')
        self.run_button.pack(pady=20)

    def _select_csv_file(self):
        """CSV 파일 선택 대화상자"""
        file_path = filedialog.askopenfilename(filetypes=[("CSV 파일", "*.csv")])
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
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                 filetypes=[("Excel 파일", "*.xlsx")],
                                                 title="엑셀 파일 저장")
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
        messagebox.showinfo("재시작", "저장이 완료되어 프로그램을 재시작합니다.")
        self.root.destroy()
        self.run_gui()

    def _process_file(self):
        """CSV 데이터를 Excel로 변환하는 함수"""
        try:
            self.run_button.config(state=tk.DISABLED)
            
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

            messagebox.showinfo("완료", f"엑셀 파일이 '{excel_file}'로 저장되었습니다.")
            self.run_button.config(state=tk.NORMAL)

        except ValueError as ve:
            messagebox.showwarning("입력 오류", str(ve))
            self._reset_gui_error()
        except Exception as e:
            error_details = traceback.format_exc()
            messagebox.showerror("오류 발생", f"예상치 못한 오류가 발생했습니다.\n{str(e)}:{str(error_details)}")
            self._reset_gui_error()


def run_gui():
    """GUI 실행 함수 (하위 호환성)"""
    gui = CSVExcelConverterGUI()
    gui.run_gui()
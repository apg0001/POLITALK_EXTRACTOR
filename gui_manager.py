from tkinter import filedialog, messagebox, ttk
import tkinter as tk
import datetime
import os
import sys
import traceback
from file_manager import *
from text_manager import *

def run_gui():
    """Tkinter 기반 GUI 실행."""
    root = tk.Tk()
    root.title("CSV - Excel 변환기")
    root.geometry("600x400")  # GUI 크기 설정

    def reset_gui_error():
        """오류 발생 시 GUI를 초기화하고 재시작"""
        messagebox.showinfo("재시작", "오류가 발생하여 프로그램을 재시작합니다.")
        root.destroy()
        run_gui()
        
    def reset_gui():
        """저장 완료 시 GUI를 초기화하고 재시작"""
        messagebox.showinfo("재시작", "저장이 완료되어 프로그램을 재시작합니다.")
        root.destroy()
        run_gui()
    
    def select_csv_file():
        """CSV 파일 선택 대화상자"""
        file_path = filedialog.askopenfilename(filetypes=[("CSV 파일", "*.csv")])
        if file_path:
            csv_file_entry.delete(0, tk.END)
            csv_file_entry.insert(0, file_path)

            # 현재 날짜를 YYMMDD 형식으로 설정
            formatted_date = datetime.datetime.now().strftime('%y%m%d')
            excel_file_path = file_path.replace(".csv", f"_AI변환{formatted_date}.xlsx")
            
            excel_file_entry.delete(0, tk.END)
            excel_file_entry.insert(0, excel_file_path)

    def select_excel_file():
        """Excel 저장 위치 선택"""
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                 filetypes=[("Excel 파일", "*.xlsx")],
                                                 title="엑셀 파일 저장")
        if file_path:
            excel_file_entry.delete(0, tk.END)
            excel_file_entry.insert(0, file_path)

    def process_file():
        """CSV 데이터를 Excel로 변환하는 함수"""
        try:
            run_button.config(state=tk.DISABLED)
            
            csv_file = csv_file_entry.get()
            excel_file = excel_file_entry.get()

            if not csv_file or not excel_file:
                raise ValueError("CSV 파일과 Excel 파일을 모두 선택해야 합니다.")

            # CSV에서 데이터 추출 및 Excel 저장
            extracted_data = extract_text_from_csv(csv_file, progress_bar, progress_label)
            print(f"csv에서 추출된 데이터 수 {len(extracted_data)}")
            # CSV 파일 내용 병합
            merged_data = merge_data(extracted_data, progress_bar, progress_label)
            print(f"병합 후 데이터 수 {len(merged_data)}")
            # 중복 내용 제거
            duplicate_removed_data = remove_duplicates(merged_data, progress_bar, progress_label)
            print(f"중복 제거 후 데이터 수 {len(duplicate_removed_data)}")
            # save_data_to_excel(extracted_data, excel_file, progress_bar, progress_label)
            save_data_to_excel(duplicate_removed_data, excel_file, progress_bar, progress_label)
            # save_data_to_csv(duplicate_removed_data, excel_file.replace(".xlsx", ".csv"), progress_bar, progress_label)

            messagebox.showinfo("완료", f"엑셀 파일이 '{excel_file}'로 저장되었습니다.")
            run_button.config(state=tk.NORMAL)

        except ValueError as ve:
            messagebox.showwarning("입력 오류", str(ve))
            reset_gui_error()
        except Exception as e:
            error_details = traceback.format_exc()  # 전체 스택 추적
            messagebox.showerror("오류 발생", f"예상치 못한 오류가 발생했습니다.\n{str(e)}:{str(error_details)}")
            reset_gui_error()

    # CSV 파일 선택
    csv_file_label = tk.Label(root, text="CSV 파일 선택:")
    csv_file_label.pack(pady=5)
    csv_file_entry = tk.Entry(root, width=70)
    csv_file_entry.pack(pady=5)
    csv_file_button = tk.Button(root, text="파일 선택", command=select_csv_file)
    csv_file_button.pack(pady=5)

    # Excel 저장 위치
    excel_file_label = tk.Label(root, text="Excel 파일 저장 위치:")
    excel_file_label.pack(pady=5)
    excel_file_entry = tk.Entry(root, width=70)
    excel_file_entry.pack(pady=5)
    excel_file_button = tk.Button(root, text="다른 이름으로 저장", command=select_excel_file)
    excel_file_button.pack(pady=5)

    # Progressbar 설정
    progress_label = tk.Label(root, text="변환 실행 버튼을 눌러주세요.")
    progress_label.pack(pady=5)
    progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
    progress_bar.pack(pady=5)

    # 실행 버튼
    run_button = tk.Button(root, text="변환 실행", command=process_file, fg='green')
    run_button.pack(pady=20)

    root.mainloop()

# GUI 실행
# run_gui()
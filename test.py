import os
import pandas as pd
import PyPDF2

def extract_pdf_info(folder_path):
    pdf_info = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            try:
                subject, human_result, title = filename.replace('.pdf', '').split('-', 2)
                # 读取PDF文件内容
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                    pdf_info.append({
                        '标题': title,
                        '正文': text,
                        '题材': subject,
                        '人类评审结果': human_result
                    })
            except:
                print(f"Error processing file: {filename}")
    return pdf_info

def save_to_csv(data, csv_path):
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

folder_path = 'test_data'
csv_path = 'dataset/pdf_info.csv'
data = extract_pdf_info(folder_path)
save_to_csv(data, csv_path)
print(f"信息已保存到 {csv_path}")
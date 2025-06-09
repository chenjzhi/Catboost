import os
import pandas as pd
import PyPDF2

def extract_pdf_info(folder_path):
    pdf_info = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            try:
                title = filename.replace('.pdf','')
                # 读取PDF文件内容
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                    pdf_info.append({
                        '标题': title,
                        '正文': text,
                        '人类评审结果': 'F'
                    })
            except:
                print(f"Error processing file: {filename}")
    return pdf_info

def save_to_csv(data, csv_path):
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

folder_path = 'positive_sample'
csv_path = 'dataset/pdf_infs.csv'

data = extract_pdf_info(folder_path)
save_to_csv(data, csv_path)
print(f"信息已保存到 {csv_path}")
import io
import re
import logging

logger = logging.getLogger(__name__)

def extract_text_from_file(uploaded_file) -> str:
    file_bytes = uploaded_file.read()
    file_name = uploaded_file.name.lower()
    
    extracted_text = ""
    
    try:
        if file_name.endswith('.pdf'):
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        elif file_name.endswith('.txt'):
            try:
                extracted_text = file_bytes.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    extracted_text = file_bytes.decode('gbk')
                except UnicodeDecodeError:
                    extracted_text = file_bytes.decode('utf-8', errors='ignore')
        else:
            raise ValueError("本教练只认 PDF 和 TXT 文件，别拿奇奇怪怪的文件糊弄我！")
            
    except Exception as e:
        logger.error(f"文件解析底层崩溃: {e}")
        raise ValueError(f"文件损坏或格式太复杂。底层报错：{str(e)[:50]}")
        
    if len(extracted_text.strip()) < 100:
        raise ValueError("提取出的文字太少！你的简历可能全是复杂的表格或图片，本教练看不了这种花里胡哨的排版。请直接复制纯文本粘贴到下方！")
        
    if '<w:' in extracted_text or '<v:' in extracted_text:
        raise ValueError("检测到复杂的底层排版代码，解析失败。请直接复制纯文本粘贴到下方！")
    
    if '<html' in extracted_text.lower() or '<!doctype' in extracted_text.lower():
        raise ValueError("检测到HTML格式内容，解析失败。请直接复制纯文本粘贴到下方！")
        
    chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', extracted_text))
    total_chars = len(extracted_text.replace(" ", "").replace("\n", ""))
    
    if total_chars > 100 and chinese_chars / total_chars < 0.3:
        raise ValueError("提取出的内容乱码太多，大概率是扫描件。请直接复制纯文本粘贴到下方！")
        
    return extracted_text.strip()

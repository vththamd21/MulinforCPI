import pandas as pd
import os

# ĐƯỜNG DẪN ĐẾN FILE CSV CỦA BẠN
# Thay đổi đường dẫn này trỏ đến file dữ liệu của bạn
input_csv_path = '/Users/vuhongtham/Desktop/workspace/PerceiverCPI/toy_dataset/davis.csv'  # Ví dụ: 'data/test.csv'
output_fasta_path = 'sequences_to_map.fasta'

def create_fasta(input_path, output_path):
    # Đọc dữ liệu
    df = pd.read_csv(input_path)
    
    # Lấy các sequence duy nhất để không phải map trùng lặp
    unique_sequences = df['sequence'].unique()
    
    print(f"Tìm thấy {len(unique_sequences)} sequence duy nhất.")
    
    with open(output_path, 'w') as f:
        for i, seq in enumerate(unique_sequences):
            # Format FASTA: >Header\nSequence
            # Ta dùng chính sequence làm header để dễ map lại sau này
            # (Lưu ý: Header không được quá dài, nhưng UniProt xử lý tốt)
            f.write(f">seq_{i}\n{seq}\n")
            
    print(f"Đã xuất file FASTA tại: {output_path}")

if __name__ == "__main__":
    create_fasta(input_csv_path, output_fasta_path)
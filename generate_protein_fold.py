import torch
import esm
from tqdm import tqdm
import numpy as np
import pdb
import pandas as pd
import os
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# FIX: Dùng 'cuda' để tự động chọn GPU đầu tiên (thường là cuda:0) thay vì ép cứng cuda:1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
print("Loading ESM-Fold model...")
model = esm.pretrained.esmfold_v1().to(device)
model = model.eval()

# Tắt gradient để tiết kiệm bộ nhớ khi suy luận (inference)
torch.set_grad_enabled(False)

def main(data_path):
    # FIX: data_path là thư mục chứa các file csv đầu vào
    files = os.listdir(data_path)
    for file in files:
        if file.endswith('.csv'):
            print(f"Processing file: {file}")
            
            # FIX: Sửa 'data_folder' thành 'data_path' vì biến 'data_folder' không tồn tại
            input_csv_path = os.path.join(data_path, file)
            
            # Tạo thư mục đầu ra
            re_fold = os.path.join(data_path, 'esm1' + file[:-4])
            if not os.path.isdir(re_fold): os.makedirs(re_fold)

            data1 = pd.read_csv(input_csv_path)
            data = data1.copy()
            
            # Lấy danh sách chuỗi protein duy nhất và sắp xếp theo độ dài
            uni_sequences = sorted(list(set(data['sequence'])), key=len)

            # FIX: Tối ưu hóa việc map sequence sang index (nhanh hơn vòng lặp for cũ)
            seq_to_idx = {seq: i for i, seq in enumerate(uni_sequences)}
            data['sequence'] = data['sequence'].map(seq_to_idx)
                
            # Lưu file từ điển mapping (Dictionary)
            list_mach_df = pd.DataFrame(list(seq_to_idx.items()), columns=['sequence', 'index'])
            list_mach_df.to_csv(os.path.join(re_fold, file + 'dic.csv'), index=False)
            
            # Lưu file data đã map index
            data.to_csv(os.path.join(re_fold, 'esm' + file), index=False)

            print(f"Predicting structures for {len(uni_sequences)} unique sequences...")
            for index, sequence in enumerate(tqdm(uni_sequences)):
                # Giới hạn độ dài sequence để tránh OOM (Out of Memory) nếu cần
                # sequence = sequence[:1024] 
                
                try:
                    with torch.no_grad():
                        output = model.infer_pdb(sequence)
                        
                    with open(os.path.join(re_fold, "{}_result.pdb".format(index)), "w") as f:
                        f.write(output)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"Skipping sequence index {index} due to OOM (Length: {len(sequence)})")
                        torch.cuda.empty_cache()
                    else:
                        print(f"Error on sequence index {index}: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python generate_protein_fold.py <path_to_data_folder>")
    else:
        data_path_arg = str(sys.argv[1])
        main(data_path=data_path_arg)

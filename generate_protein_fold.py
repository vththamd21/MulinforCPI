import torch
import esm
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import sys
import gc

# 1. Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading ESM-Fold model...")
# 2. Load model
model = esm.pretrained.esmfold_v1()
model = model.eval().to(device)

# --- SỬA LỖI QUAN TRỌNG ---
# OpenFold Attention kernel bị lỗi với Float16 trên một số GPU/Môi trường.
# Chúng ta KHOÁ dòng model.half() lại để chạy ở Float32 (mặc định).
# model.half()  <-- ĐÃ VÔ HIỆU HÓA DÒNG NÀY ĐỂ SỬA LỖI "Unsupported datatype"

# Kích hoạt Axial Attention Chunking
# Giá trị 64 giúp tiết kiệm bộ nhớ cực tốt, cho phép chạy Float32 mà không lo OOM với chuỗi dài.
model.set_chunk_size(64)

# Tắt gradient
torch.set_grad_enabled(False)

def predict_structure(sequence, model, attempt=1):
    """Hàm dự đoán có khả năng tự giảm chunk_size nếu gặp lỗi OOM"""
    try:
        with torch.no_grad():
            output = model.infer_pdb(sequence)
        return output
    except RuntimeError as e:
        # Xử lý lỗi OOM nếu xảy ra (dù hiếm với chunk=64)
        if "out of memory" in str(e) and attempt == 1:
            print(f"Warning: OOM detected. Clearing cache and trying smaller chunk size (32)...")
            torch.cuda.empty_cache()
            gc.collect()
            model.set_chunk_size(32) # Giảm chunk xuống mức thấp nhất
            try:
                with torch.no_grad():
                    output = model.infer_pdb(sequence)
                model.set_chunk_size(64) # Reset lại cho protein sau
                return output
            except RuntimeError as e2:
                model.set_chunk_size(64) # Reset lại
                raise e2
        else:
            raise e

def main(data_path):
    if not os.path.isdir(data_path):
        print(f"Error: {data_path} is not a directory.")
        return

    files = os.listdir(data_path)
    for file in files:
        if file.endswith('.csv') and 'dic' not in file:
            print(f"Processing file: {file}")
            
            input_csv_path = os.path.join(data_path, file)
            re_fold = os.path.join(data_path, 'esm1' + file[:-4])
            if not os.path.isdir(re_fold): os.makedirs(re_fold)

            data1 = pd.read_csv(input_csv_path)
            data = data1.copy()
            
            # Lọc và sắp xếp
            data = data[data['sequence'].str.len() > 10]
            uni_sequences = sorted(list(set(data['sequence'])), key=len)

            print(f"Mapping sequences to indices...")
            seq_to_idx = {seq: i for i, seq in enumerate(uni_sequences)}
            data['sequence'] = data['sequence'].map(seq_to_idx)
                
            # Lưu file phụ trợ
            list_mach_df = pd.DataFrame(list(seq_to_idx.items()), columns=['sequence', 'index'])
            list_mach_df.to_csv(os.path.join(re_fold, file + 'dic.csv'), index=False)
            data.to_csv(os.path.join(re_fold, 'esm' + file), index=False)

            print(f"Predicting structures for {len(uni_sequences)} unique sequences...")
            
            for index, sequence in enumerate(tqdm(uni_sequences)):
                # Logic cắt ngắn an toàn (chỉ cắt khi cực dài > 2000aa để bảo vệ FP32)
                if len(sequence) > 2000:
                    sequence = sequence[:2000]

                output_file = os.path.join(re_fold, "{}_result.pdb".format(index))
                
                if os.path.exists(output_file):
                    continue

                try:
                    pdb_string = predict_structure(sequence, model)
                    with open(output_file, "w") as f:
                        f.write(pdb_string)
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"SKIPPING sequence {index} (Length: {len(sequence)}) due to persistent OOM.")
                        torch.cuda.empty_cache()
                        gc.collect()
                    else:
                        print(f"Error on sequence {index}: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python generate_protein_fold.py <path_to_data_folder>")
    else:
        data_path_arg = str(sys.argv[1])
        main(data_path=data_path_arg)
import torch
import esm
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import sys
import gc

# 1. Cấu hình thiết bị và tối ưu hóa bộ nhớ
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tối ưu hóa: Sử dụng TF32 trên Ampere GPUs (như A100/A10) nếu có để tăng tốc
if torch.cuda.is_available() and hasattr(torch.backends.cuda, "matmul"):
    torch.backends.cuda.matmul.allow_tf32 = True

print("Loading ESM-Fold model...")
# 2. Load model
model = esm.pretrained.esmfold_v1()
model = model.eval().to(device)

# --- TỐI ƯU HÓA QUAN TRỌNG CHO PROTEIN DÀI ---
# Chuyển model sang float16 (half precision) để tiết kiệm 50% VRAM
# Lưu ý: ESMFold v1 hoạt động tốt ở FP16
try:
    model.half()
    print("Model converted to Float16 for memory efficiency.")
except:
    print("Keep model in Float32.")

# Kích hoạt Axial Attention Chunking
# Đây là chìa khóa để xử lý chuỗi dài > 1000 aa trên GPU giới hạn
# Giá trị 64 là cân bằng tốt giữa tốc độ và bộ nhớ.
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
        if "out of memory" in str(e) and attempt == 1:
            # Chiến lược cứu hộ: Xóa cache, giảm chunk size và thử lại
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
    # Kiểm tra đường dẫn
    if not os.path.isdir(data_path):
        print(f"Error: {data_path} is not a directory.")
        return

    files = os.listdir(data_path)
    for file in files:
        if file.endswith('.csv') and 'dic' not in file: # Tránh đọc nhầm file dic
            print(f"Processing file: {file}")
            
            input_csv_path = os.path.join(data_path, file)
            re_fold = os.path.join(data_path, 'esm1' + file[:-4])
            if not os.path.isdir(re_fold): os.makedirs(re_fold)

            data1 = pd.read_csv(input_csv_path)
            data = data1.copy()
            
            # Lọc các chuỗi protein hợp lệ (không quá ngắn)
            data = data[data['sequence'].str.len() > 10]
            
            # Sắp xếp để xử lý từ ngắn đến dài (giúp quản lý bộ nhớ tốt hơn)
            uni_sequences = sorted(list(set(data['sequence'])), key=len)

            print(f"Mapping sequences to indices...")
            seq_to_idx = {seq: i for i, seq in enumerate(uni_sequences)}
            data['sequence'] = data['sequence'].map(seq_to_idx)
                
            # Lưu các file phụ trợ
            list_mach_df = pd.DataFrame(list(seq_to_idx.items()), columns=['sequence', 'index'])
            list_mach_df.to_csv(os.path.join(re_fold, file + 'dic.csv'), index=False)
            data.to_csv(os.path.join(re_fold, 'esm' + file), index=False)

            print(f"Predicting structures for {len(uni_sequences)} unique sequences...")
            
            for index, sequence in enumerate(tqdm(uni_sequences)):
                # Bỏ qua việc cắt ngắn (truncation) vì ta đã có chunking
                # Chỉ cắt nếu chuỗi thực sự quá khủng khiếp (> 4000aa cho T4 GPU)
                if len(sequence) > 4000:
                    print(f"Truncating sequence {index} from {len(sequence)} to 4000 aa.")
                    sequence = sequence[:4000]

                output_file = os.path.join(re_fold, "{}_result.pdb".format(index))
                
                # Kiểm tra nếu file đã tồn tại thì bỏ qua (Resume capability)
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
    # Hỗ trợ chạy trực tiếp trên Kaggle cell
    # Ví dụ: python generate_protein_fold.py /kaggle/working/data
    if len(sys.argv) < 2:
        print("Usage: python generate_protein_fold.py <path_to_data_folder>")
        # Mặc định cho debug
        # main('/kaggle/working/MulinforCPI/data/newcomp') 
    else:
        data_path_arg = str(sys.argv[1])
        main(data_path=data_path_arg)

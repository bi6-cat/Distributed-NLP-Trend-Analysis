import json
import os
import re

def fix_dict():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "slang_dict.json")
    output_file = os.path.join(current_dir, "slang_dict_copy.json")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Dùng OrderedDict để giữ thứ tự và phát hiện key lặp
    seen_keys = {}       # key -> value (lần đầu gặp)
    duplicate_keys = []  # danh sách các key bị lặp

    def add_pair(k, v):
        if k in seen_keys:
            duplicate_keys.append((k, seen_keys[k], v))
        else:
            seen_keys[k] = v

    # --- Bước 1: Xử lý các dòng dạng Tab (key\tvalue) ---
    tab_lines = re.findall(r'^([^\t\n"{}]+)\t([^\t\n]+)', content, re.MULTILINE)
    for k, v in tab_lines:
        key = k.strip()
        val = v.strip().rstrip(',')
        if key:
            add_pair(key, val)

    # --- Bước 2: Tìm TẤT CẢ cặp "key": "value" trong toàn bộ file ---
    json_pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]*)"', content)
    for k, v in json_pairs:
        add_pair(k, v)

    # --- Báo cáo các key bị lặp ---
    if duplicate_keys:
        print(f"\n⚠️  Phát hiện {len(duplicate_keys)} từ bị lặp (đã giữ lần xuất hiện ĐẦU TIÊN):")
        for k, first_val, dup_val in duplicate_keys:
            same = "  [nghĩa giống nhau]" if first_val == dup_val else f"  [nghĩa khác: '{dup_val}' → bỏ]"
            print(f"   • \"{k}\": \"{first_val}\"{same}")
    else:
        print("✅ Không có từ nào bị lặp!")

    print(f"\n📦 Tổng số từ (sau khi loại trùng): {len(seen_keys)}")

    # --- Ghi ra file copy ---
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(seen_keys, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Đã ghi file: slang_dict_copy.json\n")

if __name__ == "__main__":
    fix_dict()

# TEAM GUIDE — Git Workflow & Contribution Guide

> **Hướng dẫn ngắn gọn**  
> **Cập nhật:** Tháng 3/2026  

---

## 1. Git Workflow

### Branch Structure

```
main                              → Production
  │
  └── develop                     → Integration branch (merge vào đây)
        │
        ├── feature/<tên-feature> → Tính năng mới
        ├── fix/<tên-bug>         → Sửa bug
        └── hotfix/<tên-issue>    → Sửa khẩn cấp
```

### Quy trình cơ bản

```bash
# 1. Tạo branch mới từ develop
git checkout develop
git pull origin develop
git checkout -b feature/ten-feature

# 2. Code và commit
git add .
git commit -m "feat(scope): mô tả ngắn gọn"

# 3. Push lên remote
git push origin feature/ten-feature

# 4. Tạo Pull Request trên GitHub → develop
# 5. Chờ review & merge
# 6. Sync lại develop
git checkout develop
git pull origin develop
```

---

## 2. Quy tắc đặt tên Branch

### Format: `<type>/<mô-tả-ngắn>`

| Type | Dùng khi | Ví dụ |
|------|----------|-------|
| `feature/` | Tính năng mới | `feature/voz-crawler` |
| `fix/` | Sửa bug | `fix/hdfs-timeout` |
| `hotfix/` | Sửa khẩn cấp | `hotfix/data-loss` |

**Lưu ý:**
- Dùng chữ thường, ngăn cách bằng `-`
- Ngắn gọn, tối đa 50 ký tự
- KHÔNG dùng tên cá nhân hoặc số issue

---

## 3. Commit Convention

### Format: `<type>(<scope>): <mô tả>`

```bash
feat(crawler): add VOZ crawler
fix(hdfs): handle connection timeout
docs(readme): update setup guide
```

### Các type commit:

| Type | Khi nào dùng | Ví dụ |
|------|--------------|-------|
| `feat` | Tính năng mới | `feat(crawler): add VOZ pagination` |
| `fix` | Sửa bug | `fix(hdfs): handle timeout` |
| `docs` | Cập nhật docs | `docs(readme): add setup guide` |
| `refactor` | Refactor code | `refactor(spark): optimize groupBy` |
| `test` | Thêm tests | `test(crawler): add unit tests` |
| `chore` | Maintenance | `chore(deps): update pandas` |

### Scope (phạm vi):
- `crawler`, `ingestion`, `hdfs`, `spark`, `nlp`, `dashboard`, `infra`

### Quy tắc:
- ✅ Chữ thường, không dấu chấm cuối
- ✅ Dùng động từ: "add", "fix", "update" (không dùng "added", "fixed")
- ✅ Ngắn gọn, tối đa 72 ký tự
- ❌ KHÔNG commit code không chạy được

---

## 4. Pull Request (PR) Process

### Khi tạo PR:

1. **Điền thông tin đầy đủ:**
   ```markdown
   ## Mô tả
   - Thêm VOZ crawler với Pydantic validation
   - Xử lý pagination và rate limiting
   
   ## Thay đổi
   - `crawlers/voz_crawler.py`: Crawler chính
   - `schemas/models.py`: Pydantic models
   - `tests/test_voz_crawler.py`: Unit tests
   
   ## Test
   - [x] Unit tests pass
   - [x] Đã test với 100 posts
   
   ## Liên quan
   Closes #123
   ```

2. **Assign reviewer:** Ít nhất 1 người liên quan
3. **Self-review:** Kiểm tra lại code trước khi assign
4. **Đảm bảo:**
   - ✅ Code chạy được
   - ✅ Không có file thừa (`.pyc`, `.env`)
   - ✅ Tests pass
   - ✅ Follow coding standards

### Khi review PR của người khác:

1. **Review trong 24 giờ** (12h cho urgent)
2. **Comment constructive:**
   - ✅ "Có thể thêm error handling ở dòng 45 không?"
   - ✅ "Suggest dùng list comprehension cho performance tốt hơn"
   - ❌ "Code này tệ"
   - ❌ "Tại sao lại làm thế?"

3. **Approve hoặc Request Changes**
4. **Không approve nếu:**
   - Code không chạy
   - Thiếu tests
   - Logic sai

### Sau khi merge:

```bash
# Sync lại develop
git checkout develop
git pull origin develop

# Xóa branch cũ
git branch -d feature/old-branch
```

---

## 5. Các lệnh Git thường dùng

### Branch operations
```bash
# Xem danh sách branch
git branch -a

# Chuyển branch
git checkout develop

# Tạo branch mới
git checkout -b feature/new-feature

# Xóa branch local
git branch -d old-branch

# Xóa branch remote
git push origin --delete old-branch
```

### Sync & Update
```bash
# Pull updates từ develop
git checkout develop
git pull origin develop

# Merge develop vào branch hiện tại
git merge develop

# Sync branch với remote
git fetch origin
```

### Commit operations
```bash
# Stage files
git add .                    # Tất cả files
git add file1.py file2.py    # Specific files

# Commit
git commit -m "feat(scope): message"

# Sửa commit cuối
git commit --amend

# Undo commit (giữ changes)
git reset --soft HEAD~1
```

### Stash (lưu tạm changes)
```bash
# Lưu tạm changes
git stash

# Lấy lại changes
git stash pop

# Xem danh sách stash
git stash list
```

### Xử lý conflicts
```bash
# Khi có conflict sau git merge
# 1. Mở file có conflict
# 2. Tìm và sửa phần:
#    <<<<<<< HEAD
#    (your code)
#    =======
#    (their code)
#    >>>>>>> branch-name
# 3. Giữ code đúng, xóa markers <<< === >>>
# 4. Add và commit
git add <resolved-file>
git commit -m "merge: resolve conflicts"
```

### Kiểm tra status
```bash
# Xem status
git status

# Xem lịch sử commits
git log --oneline --graph

# Xem diff
git diff                      # Chưa stage
git diff --staged            # Đã stage
```

---


**Good luck!**

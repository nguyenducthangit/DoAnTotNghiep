"""
FEDERATED LEARNING PROJECT - SETUP CODE FOR GOOGLE COLAB
==========================================================

Đoạn code này phải được chạy ở ĐẦU NOTEBOOK để:
1. Mount Google Drive
2. Fix path issues (ModuleNotFoundError, FileNotFoundError)
3. Cho phép import từ utils/ và load configs/ một cách tự nhiên

Author: Nguyen Duc Thang
Last Updated: 2025-12-28
"""

import os
import sys

print("="*80)
print("FEDERATED LEARNING PROJECT - SETUP")
print("="*80)

# ============================================================================
# 1. MOUNT GOOGLE DRIVE
# ============================================================================
print("\n📂 Step 1: Mounting Google Drive...")
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("   ✅ Google Drive mounted successfully")
except Exception as e:
    print(f"   ⚠️  Not running in Colab or Drive already mounted: {e}")


# ============================================================================
# 2. THIẾT LẬP WORKING DIRECTORY
# ============================================================================
print("\n📍 Step 2: Setting working directory...")

# QUAN TRỌNG: Thay đổi đường dẫn này theo cấu trúc Drive của bạn!
# Ví dụ các đường dẫn phổ biến:
#   - '/content/drive/MyDrive/Notebooks'
#   - '/content/drive/My Drive/Notebooks'
#   - '/content/drive/MyDrive/Projects/Notebooks'

PROJECT_ROOT = '/content/drive/MyDrive/Notebooks'  # <-- THAY ĐỔI ĐƯỜNG DẪN NÀY!

try:
    os.chdir(PROJECT_ROOT)
    current_dir = os.getcwd()
    print(f"   ✅ Changed working directory to: {current_dir}")
except FileNotFoundError:
    print(f"   ❌ ERROR: Directory not found: {PROJECT_ROOT}")
    print(f"   Please update PROJECT_ROOT variable to match your Google Drive structure.")
    print(f"   Current directory: {os.getcwd()}")
    raise
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    raise


# ============================================================================
# 3. THÊM VÀO sys.path
# ============================================================================
print("\n🔧 Step 3: Adding project to sys.path...")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"   ✅ Added {PROJECT_ROOT} to sys.path")
else:
    print(f"   ℹ️  {PROJECT_ROOT} already in sys.path")


# ============================================================================
# 4. XÁC MINH CẤU TRÚC THƯ MỤC
# ============================================================================
print("\n🔍 Step 4: Verifying project structure...")

required_dirs = ['utils', 'configs']
all_dirs_exist = True

for dir_name in required_dirs:
    dir_path = os.path.join(PROJECT_ROOT, dir_name)
    if os.path.exists(dir_path):
        print(f"   ✅ Found directory: {dir_name}/")
    else:
        print(f"   ❌ Missing directory: {dir_name}/")
        all_dirs_exist = False

if not all_dirs_exist:
    print("\n   ⚠️  WARNING: Some required directories are missing!")
    print("   Please ensure you have the following structure:")
    print("   Notebooks/")
    print("   ├── utils/")
    print("   │   ├── data_utils.py")
    print("   │   ├── model_utils.py")
    print("   │   └── fl_utils.py")
    print("   └── configs/")
    print("       └── training_config.yaml")


# ============================================================================
# 5. KIỂM TRA FILES QUAN TRỌNG
# ============================================================================
print("\n📄 Step 5: Checking important files...")

important_files = {
    'configs/training_config.yaml': 'Training configuration',
    'utils/data_utils.py': 'Data utilities',
    'utils/model_utils.py': 'Model utilities',
    'utils/fl_utils.py': 'Federated Learning utilities'
}

all_files_exist = True

for file_path, description in important_files.items():
    full_path = os.path.join(PROJECT_ROOT, file_path)
    if os.path.exists(full_path):
        print(f"   ✅ Found: {file_path} ({description})")
    else:
        print(f"   ❌ Missing: {file_path} ({description})")
        all_files_exist = False

if not all_files_exist:
    print("\n   ⚠️  WARNING: Some important files are missing!")


# ============================================================================
# 6. TEST IMPORT
# ============================================================================
print("\n🧪 Step 6: Testing module imports...")

try:
    from utils import data_utils
    print("   ✅ Successfully imported: utils.data_utils")
except ImportError as e:
    print(f"   ❌ Failed to import utils.data_utils: {e}")

try:
    from utils import model_utils
    print("   ✅ Successfully imported: utils.model_utils")
except ImportError as e:
    print(f"   ❌ Failed to import utils.model_utils: {e}")

try:
    from utils import fl_utils
    print("   ✅ Successfully imported: utils.fl_utils")
except ImportError as e:
    print(f"   ❌ Failed to import utils.fl_utils: {e}")


# ============================================================================
# 7. TEST CONFIG LOADING
# ============================================================================
print("\n⚙️  Step 7: Testing config file access...")

config_file = 'configs/training_config.yaml'
try:
    with open(config_file, 'r') as f:
        print(f"   ✅ Can read config file: {config_file}")
except FileNotFoundError:
    print(f"   ❌ Cannot find config file: {config_file}")
except Exception as e:
    print(f"   ❌ Error reading config: {e}")


# ============================================================================
# SETUP COMPLETED
# ============================================================================
print("\n" + "="*80)
print("✅ SETUP COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nYou can now use the following imports in your notebook:")
print("  from utils import data_utils")
print("  from utils import model_utils")
print("  from utils import fl_utils")
print("\nAnd load config files like:")
print("  with open('configs/training_config.yaml', 'r') as f:")
print("      config = yaml.safe_load(f)")
print("="*80 + "\n")

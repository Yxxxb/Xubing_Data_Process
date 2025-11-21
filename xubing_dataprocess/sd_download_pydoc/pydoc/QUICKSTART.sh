#!/bin/bash
# 快速入门脚本 - PyDoc 批量下载

echo "================================================================"
echo "PyDoc Batch Downloader - Quick Start"
echo "================================================================"
echo ""
echo "This script will help you get started with downloading Python package docs"
echo ""

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查 Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.6+"
    exit 1
fi

echo -e "${GREEN}✓${NC} Python found: $(python --version)"

# 检查依赖
echo ""
echo "Checking dependencies..."

MISSING_DEPS=()

python -c "import requests" 2>/dev/null || MISSING_DEPS+=("requests")
python -c "import bs4" 2>/dev/null || MISSING_DEPS+=("beautifulsoup4")

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠${NC} Missing dependencies: ${MISSING_DEPS[*]}"
    echo ""
    read -p "Install missing dependencies? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install "${MISSING_DEPS[@]}"
    else
        echo "Please install manually: pip install ${MISSING_DEPS[*]}"
        exit 1
    fi
else
    echo -e "${GREEN}✓${NC} All dependencies installed"
fi

echo ""
echo "================================================================"
echo "Choose an option:"
echo "================================================================"
echo "1. Quick test (download 5 small packages to test)"
echo "2. Download top 100 packages"
echo "3. Download top 1000 packages"
echo "4. Download top 10000 packages"
echo "5. Custom download"
echo ""
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "Running quick test..."
        python test_quick.py
        ;;
    2)
        echo ""
        echo "Downloading top 100 packages..."
        python 6_master_download.py -n 100 -j 8
        ;;
    3)
        echo ""
        echo "Downloading top 1000 packages..."
        echo "This may take several hours depending on your system"
        read -p "Continue? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python 6_master_download.py -n 1000 -j 8
        fi
        ;;
    4)
        echo ""
        echo "Downloading top 10000 packages..."
        echo "This may take 10+ hours depending on your system"
        read -p "Continue? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python 6_master_download.py -n 10000 -j 16
        fi
        ;;
    5)
        echo ""
        read -p "Number of top packages to download: " num_packages
        read -p "Number of parallel jobs (default 8): " num_jobs
        num_jobs=${num_jobs:-8}
        
        echo ""
        echo "Downloading top $num_packages packages with $num_jobs parallel jobs..."
        python 6_master_download.py -n "$num_packages" -j "$num_jobs"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "================================================================"
echo "Done!"
echo "================================================================"
echo ""
echo "Documentation saved to: /pfs/training-data/xubingye/data/code_doc/pydoc/"
echo ""
echo "To view statistics:"
echo "  ls -lh /pfs/training-data/xubingye/data/code_doc/pydoc/"
echo "  du -sh /pfs/training-data/xubingye/data/code_doc/pydoc/"
echo ""
echo "To continue/resume download:"
echo "  python 6_master_download.py -n <number> -j <jobs> --skip-fetch"
echo ""


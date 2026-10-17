#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "=== Building GEMMrvStudio AppImage ==="

# 1. Generate icon
echo "[1/4] Generating application icon..."
python3 -c "
from PyQt5 import QtGui, QtCore, QtSvg
import sys
app = QtGui.QGuiApplication(sys.argv)
svg_renderer = QtSvg.QSvgRenderer('resources/icons/gemm.svg')
image = QtGui.QImage(256, 256, QtGui.QImage.Format_ARGB32)
image.fill(QtCore.Qt.transparent)
painter = QtGui.QPainter(image)
svg_renderer.render(painter)
painter.end()
image.save('packaging/gemmrvstudio.png')
"

# 2. PyInstaller freeze
echo "[2/4] Freezing application with PyInstaller..."
pyinstaller --noconfirm packaging/gemmrvstudio.spec

# 3. Assemble AppDir
echo "[3/4] Assembling AppDir structure..."
rm -rf AppDir
mkdir -p AppDir/usr/bin AppDir/usr/lib AppDir/usr/share/icons/hicolor/256x256/apps
cp -r dist/gemm_dashboard/* AppDir/usr/bin/
cp packaging/gemmrvstudio.desktop AppDir/
cp packaging/gemmrvstudio.png AppDir/
cp packaging/gemmrvstudio.png AppDir/usr/share/icons/hicolor/256x256/apps/
cp packaging/AppRun AppDir/
chmod +x AppDir/AppRun

# 4. Package AppImage
echo "[4/4] Building standalone AppImage..."
if [ ! -f "packaging/appimagetool" ]; then
    curl -L -o packaging/appimagetool "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"
    chmod +x packaging/appimagetool
fi

ARCH=x86_64 ./packaging/appimagetool --appimage-extract-and-run AppDir GEMMrvStudio-x86_64.AppImage

echo "=== AppImage build successful: ${REPO_ROOT}/GEMMrvStudio-x86_64.AppImage ==="

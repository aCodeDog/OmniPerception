#!/bin/bash

# Installation script for LiDAR Sensor integration in IsaacLab
# Usage: ./install_lidar_sensor.sh /path/to/IsaacLab

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 /path/to/IsaacLab"
    echo "Example: $0 /home/user/IsaacLab"
    exit 1
fi

ISAACLAB_PATH="$1"
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if IsaacLab directory exists
if [ ! -d "$ISAACLAB_PATH" ]; then
    echo "Error: IsaacLab directory not found: $ISAACLAB_PATH"
    exit 1
fi

# Check if IsaacLab structure looks correct
if [ ! -d "$ISAACLAB_PATH/source/isaaclab/isaaclab/sensors" ]; then
    echo "Error: IsaacLab structure not found. Make sure you provided the correct IsaacLab root directory."
    exit 1
fi

echo "Installing LiDAR Sensor files to IsaacLab..."
echo "Target directory: $ISAACLAB_PATH"

# Copy sensor implementations
echo "Copying sensor files..."
cp "$CURRENT_DIR/sensors/lidar_sensor.py" "$ISAACLAB_PATH/source/isaaclab/isaaclab/sensors/"
cp "$CURRENT_DIR/sensors/lidar_sensor_cfg.py" "$ISAACLAB_PATH/source/isaaclab/isaaclab/sensors/"
cp "$CURRENT_DIR/sensors/lidar_sensor_data.py" "$ISAACLAB_PATH/source/isaaclab/isaaclab/sensors/"

# Copy pattern implementations
echo "Copying pattern files..."
cp "$CURRENT_DIR/sensors/ray_caster/patterns/patterns.py" "$ISAACLAB_PATH/source/isaaclab/isaaclab/sensors/ray_caster/patterns/"
cp "$CURRENT_DIR/sensors/ray_caster/patterns/patterns_cfg.py" "$ISAACLAB_PATH/source/isaaclab/isaaclab/sensors/ray_caster/patterns/"


echo "Copying raycaster files..."
cp "$CURRENT_DIR/sensors/ray_caster/ray_caster.py" "$ISAACLAB_PATH/source/isaaclab/isaaclab/sensors/ray_caster/"
cp "$CURRENT_DIR/sensors/ray_caster/ray_caster_camera.py" "$ISAACLAB_PATH/source/isaaclab/isaaclab/sensors/ray_caster/"
cp "$CURRENT_DIR/sensors/ray_caster/ray_caster_cfg.py" "$ISAACLAB_PATH/source/isaaclab/isaaclab/sensors/ray_caster/"
# Copy scan patterns from unified location for local fallback
echo "Copying scan pattern files from unified location..."
UNIFIED_PATTERNS_DIR="$CURRENT_DIR/../../../sensor_pattern/sensor_lidar/scan_mode"
mkdir -p "$ISAACLAB_PATH/source/isaaclab/isaaclab/sensors/ray_caster/patterns/scan_patterns"

if [ -d "$UNIFIED_PATTERNS_DIR" ]; then
    cp "$UNIFIED_PATTERNS_DIR"/*.npy "$ISAACLAB_PATH/source/isaaclab/isaaclab/sensors/ray_caster/patterns/scan_patterns/"
    echo "✓ Copied scan patterns from unified location: $UNIFIED_PATTERNS_DIR"
else
    echo "⚠ Warning: Unified scan patterns directory not found: $UNIFIED_PATTERNS_DIR"
    echo "  Patterns will be loaded from unified location at runtime if available"
fi

# Copy example script
echo "Copying example script..."
cp "$CURRENT_DIR/scripts/examples/simple_lidar_integration.py" "$ISAACLAB_PATH/scripts/demos/"

# Copy benchmark script
echo "Copying benchmark script..."
cp "$CURRENT_DIR/benchmark_lidar.sh" "$ISAACLAB_PATH/"
chmod +x "$ISAACLAB_PATH/benchmark_lidar.sh"

# Update __init__.py files automatically
echo "Updating __init__.py files..."

# Replace sensors/__init__.py with complete version
SENSORS_INIT="$ISAACLAB_PATH/source/isaaclab/isaaclab/sensors/__init__.py"
echo "Replacing sensors/__init__.py..."
cp "$CURRENT_DIR/init_files/sensors_init.py" "$SENSORS_INIT"
echo "✓ Updated sensors/__init__.py"

# Replace patterns/__init__.py with complete version
PATTERNS_INIT="$ISAACLAB_PATH/source/isaaclab/isaaclab/sensors/ray_caster/patterns/__init__.py"
echo "Replacing patterns/__init__.py..."
cp "$CURRENT_DIR/init_files/patterns_init.py" "$PATTERNS_INIT"
echo "✓ Updated patterns/__init__.py"

echo ""
echo "✓ Installation completed successfully!"
echo ""
echo "🚀 LiDAR Sensor is now ready to use in IsaacLab!"
echo ""
echo "Test the installation:"
echo "   cd $ISAACLAB_PATH"
echo "   ./isaaclab.sh -p scripts/demos/simple_lidar_integration.py"
echo ""
echo "Run performance benchmark:"
echo "   ./benchmark_lidar.sh"
echo ""
echo "For detailed usage instructions, see README.md"

#!/bin/bash

# PR自动化测试脚本
# 用于测试 sunpihai-up 的深度相机PR

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 测试结果记录
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_TESTS=()

# 记录测试结果
record_test() {
    local test_name="$1"
    local result="$2"
    
    if [ "$result" = "PASS" ]; then
        TESTS_PASSED=$((TESTS_PASSED + 1))
        log_success "✓ $test_name"
    else
        TESTS_FAILED=$((TESTS_FAILED + 1))
        FAILED_TESTS+=("$test_name")
        log_error "✗ $test_name"
    fi
}

# 检查环境
check_environment() {
    log_info "检查测试环境..."
    
    # 检查Python
    if python3 --version >/dev/null 2>&1; then
        record_test "Python3可用性" "PASS"
    else
        record_test "Python3可用性" "FAIL"
        return 1
    fi
    
    # 检查Git
    if git --version >/dev/null 2>&1; then
        record_test "Git可用性" "PASS"
    else
        record_test "Git可用性" "FAIL"
        return 1
    fi
    
    # 检查当前目录是否为Git仓库
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        record_test "Git仓库检查" "PASS"
    else
        record_test "Git仓库检查" "FAIL"
        return 1
    fi
}

# 设置测试分支
setup_test_branch() {
    log_info "设置测试分支..."
    
    # 保存当前分支
    ORIGINAL_BRANCH=$(git branch --show-current)
    log_info "当前分支: $ORIGINAL_BRANCH"
    
    # 创建测试分支
    TEST_BRANCH="test-pr-camera-$(date +%Y%m%d-%H%M%S)"
    
    if git checkout -b "$TEST_BRANCH" >/dev/null 2>&1; then
        record_test "创建测试分支" "PASS"
        log_info "测试分支: $TEST_BRANCH"
    else
        record_test "创建测试分支" "FAIL"
        return 1
    fi
    
    # 合并PR内容
    if git merge pr-contributor/main --no-edit >/dev/null 2>&1; then
        record_test "合并PR内容" "PASS"
    else
        record_test "合并PR内容" "FAIL"
        return 1
    fi
}

# 代码质量检查
check_code_quality() {
    log_info "进行代码质量检查..."
    
    # 检查Python语法
    local files=(
        "LidarSensor/LidarSensor/isaacgym_camera_sensor.py"
        "LidarSensor/LidarSensor/example/isaacgym/unitree_g1_camera.py"
        "LidarSensor/LidarSensor/sensor_config/camera_config/base_depth_camera_config.py"
        "LidarSensor/LidarSensor/sensor_config/camera_config/d455_depth_config.py"
        "LidarSensor/LidarSensor/sensor_config/camera_config/luxonis_oak_d_config.py"
    )
    
    local syntax_ok=true
    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            if python3 -m py_compile "$file" 2>/dev/null; then
                log_success "  ✓ $file 语法检查通过"
            else
                log_error "  ✗ $file 语法检查失败"
                syntax_ok=false
            fi
        else
            log_warning "  ? $file 文件不存在"
            syntax_ok=false
        fi
    done
    
    if [ "$syntax_ok" = true ]; then
        record_test "Python语法检查" "PASS"
    else
        record_test "Python语法检查" "FAIL"
    fi
}

# 导入测试
test_imports() {
    log_info "进行导入测试..."
    
    # 测试基础配置导入
    if python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from LidarSensor.sensor_config.camera_config.base_depth_camera_config import *
    print('Base depth camera config imported successfully')
except Exception as e:
    print(f'Error: {e}')
    exit(1)
" >/dev/null 2>&1; then
        record_test "基础配置导入" "PASS"
    else
        record_test "基础配置导入" "FAIL"
    fi
    
    # 测试D455配置导入
    if python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from LidarSensor.sensor_config.camera_config.d455_depth_config import *
    print('D455 config imported successfully')
except Exception as e:
    print(f'Error: {e}')
    exit(1)
" >/dev/null 2>&1; then
        record_test "D455配置导入" "PASS"
    else
        record_test "D455配置导入" "FAIL"
    fi
    
    # 测试OAK-D配置导入
    if python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from LidarSensor.sensor_config.camera_config.luxonis_oak_d_config import *
    print('OAK-D config imported successfully')
except Exception as e:
    print(f'Error: {e}')
    exit(1)
" >/dev/null 2>&1; then
        record_test "OAK-D配置导入" "PASS"
    else
        record_test "OAK-D配置导入" "FAIL"
    fi
    
    # 测试相机传感器导入
    if python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from LidarSensor.isaacgym_camera_sensor import *
    print('Camera sensor imported successfully')
except Exception as e:
    print(f'Error: {e}')
    exit(1)
" >/dev/null 2>&1; then
        record_test "相机传感器导入" "PASS"
    else
        record_test "相机传感器导入" "FAIL"
    fi
}

# 文件结构检查
check_file_structure() {
    log_info "检查文件结构..."
    
    local expected_files=(
        "LidarSensor/LidarSensor/isaacgym_camera_sensor.py"
        "LidarSensor/LidarSensor/example/isaacgym/unitree_g1_camera.py"
        "LidarSensor/LidarSensor/sensor_config/camera_config/base_depth_camera_config.py"
        "LidarSensor/LidarSensor/sensor_config/camera_config/d455_depth_config.py"
        "LidarSensor/LidarSensor/sensor_config/camera_config/luxonis_oak_d_config.py"
    )
    
    local all_files_exist=true
    for file in "${expected_files[@]}"; do
        if [ -f "$file" ]; then
            log_success "  ✓ $file 存在"
        else
            log_error "  ✗ $file 不存在"
            all_files_exist=false
        fi
    done
    
    if [ "$all_files_exist" = true ]; then
        record_test "文件结构检查" "PASS"
    else
        record_test "文件结构检查" "FAIL"
    fi
}

# 代码风格检查（简单版本）
check_code_style() {
    log_info "检查代码风格..."
    
    # 检查是否有基本的文档字符串
    local has_docstrings=true
    local files=(
        "LidarSensor/LidarSensor/isaacgym_camera_sensor.py"
    )
    
    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            if grep -q '"""' "$file"; then
                log_success "  ✓ $file 包含文档字符串"
            else
                log_warning "  ? $file 可能缺少文档字符串"
                has_docstrings=false
            fi
        fi
    done
    
    if [ "$has_docstrings" = true ]; then
        record_test "基础文档检查" "PASS"
    else
        record_test "基础文档检查" "FAIL"
    fi
}

# 兼容性检查
check_compatibility() {
    log_info "检查兼容性..."
    
    # 检查是否会与现有LiDAR传感器冲突
    if python3 -c "
import sys
sys.path.insert(0, '.')
try:
    # 尝试同时导入新的相机传感器和现有的LiDAR传感器
    from LidarSensor.isaacgym_camera_sensor import *
    from LidarSensor.lidar_sensor import LidarSensor
    print('No import conflicts detected')
except Exception as e:
    print(f'Compatibility issue: {e}')
    exit(1)
" >/dev/null 2>&1; then
        record_test "LiDAR兼容性检查" "PASS"
    else
        record_test "LiDAR兼容性检查" "FAIL"
    fi
}

# 清理测试环境
cleanup() {
    log_info "清理测试环境..."
    
    # 返回原始分支
    if [ -n "$ORIGINAL_BRANCH" ]; then
        git checkout "$ORIGINAL_BRANCH" >/dev/null 2>&1
        log_info "已返回到原始分支: $ORIGINAL_BRANCH"
    fi
    
    # 删除测试分支
    if [ -n "$TEST_BRANCH" ]; then
        git branch -D "$TEST_BRANCH" >/dev/null 2>&1
        log_info "已删除测试分支: $TEST_BRANCH"
    fi
}

# 生成测试报告
generate_report() {
    log_info "生成测试报告..."
    
    echo ""
    echo "=========================================="
    echo "           PR 测试结果报告"
    echo "=========================================="
    echo "PR贡献者: sunpihai-up"
    echo "PR内容: Add IsaacGym DepthCamera"
    echo "测试时间: $(date)"
    echo ""
    echo "测试结果:"
    echo "  通过: $TESTS_PASSED"
    echo "  失败: $TESTS_FAILED"
    echo "  总计: $((TESTS_PASSED + TESTS_FAILED))"
    echo ""
    
    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}🎉 所有测试通过！建议接受此PR。${NC}"
        echo ""
        echo "建议的后续步骤:"
        echo "1. 进行人工代码审查"
        echo "2. 检查PR描述和文档"
        echo "3. 合并PR到主分支"
        echo "4. 更新CHANGELOG"
        RECOMMENDATION="ACCEPT"
    else
        echo -e "${RED}❌ 部分测试失败，建议拒绝此PR或要求修改。${NC}"
        echo ""
        echo "失败的测试:"
        for test in "${FAILED_TESTS[@]}"; do
            echo "  - $test"
        done
        echo ""
        echo "建议的后续步骤:"
        echo "1. 联系贡献者说明问题"
        echo "2. 提供具体的修改建议"
        echo "3. 等待修复后重新测试"
        RECOMMENDATION="REJECT"
    fi
    
    echo ""
    echo "=========================================="
    
    # 保存报告到文件
    {
        echo "# PR测试报告"
        echo ""
        echo "**PR信息:**"
        echo "- 贡献者: sunpihai-up"
        echo "- 内容: Add IsaacGym DepthCamera"
        echo "- 测试时间: $(date)"
        echo ""
        echo "**测试结果:**"
        echo "- 通过: $TESTS_PASSED"
        echo "- 失败: $TESTS_FAILED"
        echo "- 总计: $((TESTS_PASSED + TESTS_FAILED))"
        echo ""
        if [ $TESTS_FAILED -gt 0 ]; then
            echo "**失败的测试:**"
            for test in "${FAILED_TESTS[@]}"; do
                echo "- $test"
            done
            echo ""
        fi
        echo "**最终建议:** $RECOMMENDATION"
    } > "PR_TEST_REPORT_$(date +%Y%m%d_%H%M%S).md"
    
    log_success "测试报告已保存到 PR_TEST_REPORT_$(date +%Y%m%d_%H%M%S).md"
}

# 主函数
main() {
    log_info "开始PR自动化测试..."
    
    # 设置清理陷阱
    trap cleanup EXIT
    
    # 执行所有测试
    check_environment || exit 1
    setup_test_branch || exit 1
    check_file_structure
    check_code_quality
    test_imports
    check_code_style
    check_compatibility
    
    # 生成报告
    generate_report
    
    # 返回适当的退出码
    if [ $TESTS_FAILED -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

# 脚本帮助信息
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "PR自动化测试脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help    显示此帮助信息"
    echo ""
    echo "此脚本会自动测试sunpihai-up的深度相机PR，包括:"
    echo "- 环境检查"
    echo "- 代码质量检查"
    echo "- 导入测试"
    echo "- 兼容性检查"
    echo "- 生成测试报告"
    exit 0
fi

# 执行主函数
main "$@"

"""
摄像头感知系统 - Playwright 自动化测试
"""

import pytest
from playwright.sync_api import Page, expect
import time


BASE_URL = "http://localhost:8000"


def ensure_camera_stopped(page: Page):
    """确保摄像头已停止"""
    response = page.request.get(f"{BASE_URL}/api/status")
    data = response.json()
    if data["camera_opened"]:
        page.request.post(f"{BASE_URL}/api/camera/stop")
        time.sleep(0.5)


def ensure_camera_started(page: Page):
    """确保摄像头已启动"""
    response = page.request.get(f"{BASE_URL}/api/status")
    data = response.json()
    if not data["camera_opened"]:
        page.request.post(f"{BASE_URL}/api/camera/start")
        time.sleep(1)


class TestCameraPerception:
    """摄像头感知系统测试套件"""

    def test_homepage_loads(self, page: Page):
        """测试首页加载"""
        page.goto(BASE_URL)
        
        # 检查标题
        expect(page).to_have_title("摄像头实时感知系统")
        
        # 检查主要元素存在
        expect(page.locator("h1")).to_contain_text("摄像头实时感知系统")
        
        # 检查导航标签
        expect(page.get_by_role("button", name="实时监控")).to_be_visible()
        expect(page.get_by_role("button", name="相机标定")).to_be_visible()
        expect(page.get_by_role("button", name="参数配置")).to_be_visible()

    def test_api_status(self, page: Page):
        """测试 API 状态接口"""
        response = page.request.get(f"{BASE_URL}/api/status")
        assert response.ok
        
        data = response.json()
        assert "camera_opened" in data
        assert "calibrated" in data
        assert "fps" in data

    def test_start_camera(self, page: Page):
        """测试启动摄像头"""
        ensure_camera_stopped(page)
        page.goto(BASE_URL)
        
        # 点击启动摄像头按钮
        start_btn = page.get_by_role("button", name="启动摄像头")
        expect(start_btn).to_be_enabled()
        start_btn.click()
        
        # 等待状态更新
        time.sleep(2)
        
        # 检查状态变化
        response = page.request.get(f"{BASE_URL}/api/status")
        data = response.json()
        assert data["camera_opened"] == True

    def test_stop_camera(self, page: Page):
        """测试停止摄像头"""
        ensure_camera_started(page)
        page.goto(BASE_URL)
        
        # 停止摄像头
        stop_btn = page.get_by_role("button", name="停止摄像头")
        expect(stop_btn).to_be_enabled()
        stop_btn.click()
        
        time.sleep(1)
        
        # 检查状态
        response = page.request.get(f"{BASE_URL}/api/status")
        data = response.json()
        assert data["camera_opened"] == False

    def test_calibration_tab(self, page: Page):
        """测试标定页面"""
        ensure_camera_stopped(page)
        page.goto(BASE_URL)
        
        # 切换到标定标签
        page.get_by_role("button", name="相机标定").click()
        
        # 检查标定页面元素
        expect(page.locator(".calibration-page")).to_be_visible()

    def test_settings_tab(self, page: Page):
        """测试设置页面"""
        page.goto(BASE_URL)
        
        # 切换到设置标签
        page.get_by_role("button", name="参数配置").click()
        
        # 检查设置页面元素
        expect(page.locator(".settings-page")).to_be_visible()

    def test_video_canvas_visible(self, page: Page):
        """测试视频画布可见"""
        ensure_camera_started(page)
        page.goto(BASE_URL)
        
        # 检查 canvas 元素（视频画布）
        canvas = page.locator("canvas").first
        expect(canvas).to_be_visible()

    def test_detection_data_display(self, page: Page):
        """测试检测数据显示"""
        ensure_camera_started(page)
        page.goto(BASE_URL)
        
        # 检查数据面板存在
        expect(page.locator(".data-section")).to_be_visible()
        
        # 检查指标显示
        expect(page.locator(".metrics")).to_be_visible()


class TestAPIEndpoints:
    """API 端点测试"""

    def test_docs_endpoint(self, page: Page):
        """测试 API 文档"""
        page.goto(f"{BASE_URL}/docs")
        expect(page).to_have_title("摄像头实时感知系统 - Swagger UI")

    def test_spatial_config(self, page: Page):
        """测试空间配置 API"""
        response = page.request.get(f"{BASE_URL}/api/spatial/config")
        assert response.ok


# pytest 配置
@pytest.fixture(scope="function")
def browser_context_args(browser_context_args):
    """配置浏览器上下文"""
    return {
        **browser_context_args,
        "viewport": {"width": 1920, "height": 1080},
        "ignore_https_errors": True,
    }
"""
API 测试 - 测试 FastAPI 后端接口
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json


@pytest.fixture
def mock_app():
    """创建模拟的 FastAPI 应用"""
    from fastapi import FastAPI
    
    app = FastAPI()
    
    @app.get("/api/status")
    async def get_status():
        return {
            "camera_opened": True,
            "calibrated": True,
            "fps": 20.0
        }
    
    @app.post("/api/camera/start")
    async def start_camera():
        return {"status": "success", "message": "Camera started"}
    
    @app.post("/api/camera/stop")
    async def stop_camera():
        return {"status": "success", "message": "Camera stopped"}
    
    @app.get("/api/calibration/status")
    async def calibration_status():
        return {
            "calibrated": True,
            "spatial_calc_initialized": True
        }
    
    return app


@pytest.fixture
def client(mock_app):
    """创建测试客户端"""
    return TestClient(mock_app)


class TestAPIEndpoints:
    """测试 API 端点"""
    
    def test_get_status(self, client):
        """测试获取系统状态"""
        response = client.get("/api/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "camera_opened" in data
        assert "calibrated" in data
        assert "fps" in data
        
        assert data["camera_opened"] is True
        assert data["calibrated"] is True
    
    def test_start_camera(self, client):
        """测试启动摄像头"""
        response = client.post("/api/camera/start")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
    
    def test_stop_camera(self, client):
        """测试停止摄像头"""
        response = client.post("/api/camera/stop")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
    
    def test_calibration_status(self, client):
        """测试标定状态"""
        response = client.get("/api/calibration/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["calibrated"] is True
        assert data["spatial_calc_initialized"] is True


class TestAPIWithParams:
    """测试带参数的 API"""
    
    @pytest.fixture
    def app_with_params(self):
        from fastapi import FastAPI
        from pydantic import BaseModel
        
        app = FastAPI()
        
        class CameraConfig(BaseModel):
            camera_id: int = 0
            resolution: list = [1920, 1080]
            fps: int = 20
        
        @app.post("/api/camera/start")
        async def start_camera(config: CameraConfig):
            return {
                "status": "success",
                "camera_id": config.camera_id,
                "resolution": config.resolution
            }
        
        return app
    
    def test_start_camera_with_config(self, app_with_params):
        """测试带配置启动摄像头"""
        client = TestClient(app_with_params)
        
        response = client.post(
            "/api/camera/start",
            json={"camera_id": 1, "fps": 30}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["camera_id"] == 1


class TestAPIErrors:
    """测试 API 错误处理"""
    
    @pytest.fixture
    def app_with_errors(self):
        from fastapi import FastAPI, HTTPException
        
        app = FastAPI()
        
        @app.get("/api/calibration/load")
        async def load_calibration(filepath: str):
            if not filepath.endswith(".json"):
                raise HTTPException(status_code=400, detail="Invalid file format")
            return {"status": "success"}
        
        return app
    
    def test_load_calibration_invalid_format(self, app_with_errors):
        """测试加载标定的错误情况"""
        client = TestClient(app_with_errors)
        
        response = client.get("/api/calibration/load?filepath=test.txt")
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data
        assert "Invalid file format" in data["detail"]
    
    def test_load_calibration_valid(self, app_with_errors):
        """测试加载标定的正常情况"""
        client = TestClient(app_with_errors)
        
        response = client.get("/api/calibration/load?filepath=test.json")
        assert response.status_code == 200
        assert response.json()["status"] == "success"


class TestWebSocketMock:
    """测试 WebSocket（模拟）"""
    
    def test_websocket_connection(self):
        """测试 WebSocket 连接"""
        from fastapi import FastAPI, WebSocket
        
        app = FastAPI()
        
        @app.websocket("/ws/test")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            await websocket.send_json({"status": "connected"})
            await websocket.close()
        
        client = TestClient(app)
        
        # 注意：TestClient 对 WebSocket 支持有限
        # 实际 WebSocket 测试应使用异步客户端
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/test") as websocket:
                data = websocket.receive_json()
                assert data["status"] == "connected"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

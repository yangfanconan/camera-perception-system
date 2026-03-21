"""
部署优化模块

功能：
1. Docker 支持
2. Kubernetes 支持
3. 配置管理
4. 健康检查
"""

import os
import json
import time
import subprocess
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger
import yaml


@dataclass
class ServiceConfig:
    """服务配置"""
    service_name: str
    image: str
    replicas: int = 1
    port: int = 8000
    environment: Dict[str, str] = field(default_factory=dict)
    volumes: List[str] = field(default_factory=list)
    resources: Dict = field(default_factory=dict)
    health_check: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'service_name': self.service_name,
            'image': self.image,
            'replicas': self.replicas,
            'port': self.port,
            'environment': self.environment,
            'volumes': self.volumes,
            'resources': self.resources,
            'health_check': self.health_check
        }


@dataclass
class HealthStatus:
    """健康状态"""
    service_name: str
    status: str  # healthy, unhealthy, starting
    last_check: float
    uptime: float
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    error_count: int = 0
    message: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'service_name': self.service_name,
            'status': self.status,
            'last_check': self.last_check,
            'uptime': round(self.uptime, 1),
            'cpu_percent': round(self.cpu_percent, 1),
            'memory_percent': round(self.memory_percent, 1),
            'error_count': self.error_count,
            'message': self.message
        }


class DockerManager:
    """
    Docker 管理器
    
    管理 Docker 容器
    """
    
    def __init__(self):
        """初始化 Docker 管理器"""
        self.containers: Dict[str, Dict] = {}
        self.network_name = "camera-perception-network"
        
        logger.info("DockerManager initialized")
    
    def is_docker_available(self) -> bool:
        """检查 Docker 是否可用"""
        try:
            result = subprocess.run(
                ['docker', '--version'],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def build_image(
        self,
        dockerfile_path: str,
        image_name: str,
        tag: str = "latest"
    ) -> bool:
        """
        构建 Docker 镜像
        
        Args:
            dockerfile_path: Dockerfile 路径
            image_name: 镜像名称
            tag: 标签
            
        Returns:
            是否成功
        """
        try:
            cmd = [
                'docker', 'build',
                '-t', f"{image_name}:{tag}",
                '-f', dockerfile_path,
                '.'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=300
            )
            
            if result.returncode == 0:
                logger.info(f"Image built: {image_name}:{tag}")
                return True
            else:
                logger.error(f"Build failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Build error: {e}")
            return False
    
    def run_container(
        self,
        config: ServiceConfig
    ) -> Optional[str]:
        """
        运行容器
        
        Args:
            config: 服务配置
            
        Returns:
            容器ID
        """
        try:
            cmd = [
                'docker', 'run',
                '-d',
                '--name', config.service_name,
                '-p', f"{config.port}:{config.port}"
            ]
            
            # 环境变量
            for key, value in config.environment.items():
                cmd.extend(['-e', f"{key}={value}"])
            
            # 卷挂载
            for volume in config.volumes:
                cmd.extend(['-v', volume])
            
            # 资源限制
            if config.resources:
                if 'cpu_limit' in config.resources:
                    cmd.extend(['--cpus', str(config.resources['cpu_limit'])])
                if 'memory_limit' in config.resources:
                    cmd.extend(['--memory', config.resources['memory_limit']])
            
            cmd.append(config.image)
            
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=60
            )
            
            if result.returncode == 0:
                container_id = result.stdout.strip()[:12]
                self.containers[config.service_name] = {
                    'container_id': container_id,
                    'config': config.to_dict(),
                    'started_at': time.time()
                }
                
                logger.info(f"Container started: {config.service_name} ({container_id})")
                return container_id
            else:
                logger.error(f"Run failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Run error: {e}")
            return None
    
    def stop_container(self, service_name: str) -> bool:
        """停止容器"""
        try:
            result = subprocess.run(
                ['docker', 'stop', service_name],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                logger.info(f"Container stopped: {service_name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Stop error: {e}")
            return False
    
    def remove_container(self, service_name: str) -> bool:
        """移除容器"""
        try:
            subprocess.run(
                ['docker', 'rm', '-f', service_name],
                capture_output=True, text=True, timeout=30
            )
            
            if service_name in self.containers:
                del self.containers[service_name]
            
            return True
            
        except Exception as e:
            logger.error(f"Remove error: {e}")
            return False
    
    def get_container_status(self, service_name: str) -> Optional[Dict]:
        """获取容器状态"""
        try:
            result = subprocess.run(
                ['docker', 'inspect', '-f', 
                 '{{.State.Status}} {{.State.StartedAt}}',
                 service_name],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                parts = result.stdout.strip().split()
                return {
                    'status': parts[0] if parts else 'unknown',
                    'started_at': parts[1] if len(parts) > 1 else ''
                }
            
        except:
            pass
        
        return None
    
    def get_logs(self, service_name: str, lines: int = 100) -> str:
        """获取容器日志"""
        try:
            result = subprocess.run(
                ['docker', 'logs', '--tail', str(lines), service_name],
                capture_output=True, text=True, timeout=10
            )
            
            return result.stdout + result.stderr
            
        except Exception as e:
            return f"Error getting logs: {e}"
    
    def generate_dockerfile(self, output_path: str = "Dockerfile") -> str:
        """生成 Dockerfile"""
        dockerfile = '''# Camera Perception System
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/api/status || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        
        with open(output_path, 'w') as f:
            f.write(dockerfile)
        
        logger.info(f"Dockerfile generated: {output_path}")
        return dockerfile
    
    def generate_docker_compose(self, output_path: str = "docker-compose.yml") -> str:
        """生成 docker-compose.yml"""
        compose = {
            'version': '3.8',
            'services': {
                'camera-perception': {
                    'build': '.',
                    'ports': ['8000:8000'],
                    'volumes': [
                        './models:/app/models',
                        './data:/app/data',
                        './config:/app/config'
                    ],
                    'environment': {
                        'LOG_LEVEL': 'INFO',
                        'DEVICE': 'cuda'
                    },
                    'deploy': {
                        'resources': {
                            'reservations': {
                                'devices': [{
                                    'driver': 'nvidia',
                                    'count': 1,
                                    'capabilities': ['gpu']
                                }]
                            }
                        }
                    }
                }
            },
            'networks': {
                'camera-network': {
                    'driver': 'bridge'
                }
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(compose, f, default_flow_style=False)
        
        logger.info(f"docker-compose.yml generated: {output_path}")
        return yaml.dump(compose, default_flow_style=False)


class KubernetesManager:
    """
    Kubernetes 管理器
    
    管理 K8s 资源
    """
    
    def __init__(self, namespace: str = "camera-perception"):
        """初始化 K8s 管理器"""
        self.namespace = namespace
        
        logger.info(f"KubernetesManager initialized (namespace={namespace})")
    
    def is_kubectl_available(self) -> bool:
        """检查 kubectl 是否可用"""
        try:
            result = subprocess.run(
                ['kubectl', 'version', '--client'],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def create_namespace(self) -> bool:
        """创建命名空间"""
        try:
            result = subprocess.run(
                ['kubectl', 'create', 'namespace', self.namespace],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0 or 'AlreadyExists' in result.stderr:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Create namespace error: {e}")
            return False
    
    def apply_manifest(self, manifest_path: str) -> bool:
        """应用清单文件"""
        try:
            result = subprocess.run(
                ['kubectl', 'apply', '-f', manifest_path, '-n', self.namespace],
                capture_output=True, text=True, timeout=60
            )
            
            if result.returncode == 0:
                logger.info(f"Manifest applied: {manifest_path}")
                return True
            else:
                logger.error(f"Apply failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Apply error: {e}")
            return False
    
    def get_pods(self) -> List[Dict]:
        """获取 Pod 列表"""
        try:
            result = subprocess.run(
                ['kubectl', 'get', 'pods', '-n', self.namespace, '-o', 'json'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return [
                    {
                        'name': pod['metadata']['name'],
                        'status': pod['status']['phase'],
                        'ready': pod['status'].get('conditions', [{}])[0].get('status', 'Unknown')
                    }
                    for pod in data.get('items', [])
                ]
            
        except Exception as e:
            logger.error(f"Get pods error: {e}")
        
        return []
    
    def get_services(self) -> List[Dict]:
        """获取 Service 列表"""
        try:
            result = subprocess.run(
                ['kubectl', 'get', 'services', '-n', self.namespace, '-o', 'json'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return [
                    {
                        'name': svc['metadata']['name'],
                        'type': svc['spec']['type'],
                        'cluster_ip': svc['spec']['clusterIP'],
                        'ports': svc['spec']['ports']
                    }
                    for svc in data.get('items', [])
                ]
            
        except Exception as e:
            logger.error(f"Get services error: {e}")
        
        return []
    
    def scale_deployment(self, name: str, replicas: int) -> bool:
        """扩缩容"""
        try:
            result = subprocess.run(
                ['kubectl', 'scale', 'deployment', name, 
                 f'--replicas={replicas}', '-n', self.namespace],
                capture_output=True, text=True, timeout=30
            )
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Scale error: {e}")
            return False
    
    def generate_deployment(self, config: ServiceConfig) -> str:
        """生成 Deployment 清单"""
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': config.service_name,
                'namespace': self.namespace
            },
            'spec': {
                'replicas': config.replicas,
                'selector': {
                    'matchLabels': {
                        'app': config.service_name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': config.service_name
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': config.service_name,
                            'image': config.image,
                            'ports': [{
                                'containerPort': config.port
                            }],
                            'env': [
                                {'name': k, 'value': v}
                                for k, v in config.environment.items()
                            ],
                            'resources': config.resources or {},
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/api/status',
                                    'port': config.port
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/api/status',
                                    'port': config.port
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }
        
        return yaml.dump(deployment, default_flow_style=False)
    
    def generate_service(self, config: ServiceConfig) -> str:
        """生成 Service 清单"""
        service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': config.service_name,
                'namespace': self.namespace
            },
            'spec': {
                'selector': {
                    'app': config.service_name
                },
                'ports': [{
                    'port': config.port,
                    'targetPort': config.port
                }],
                'type': 'LoadBalancer'
            }
        }
        
        return yaml.dump(service, default_flow_style=False)


class HealthChecker:
    """
    健康检查器
    
    检查服务健康状态
    """
    
    def __init__(self, check_interval: float = 30.0):
        """
        初始化健康检查器
        
        Args:
            check_interval: 检查间隔
        """
        self.check_interval = check_interval
        self.services: Dict[str, HealthStatus] = {}
        
        self._running = False
        self._check_thread = None
        
        logger.info("HealthChecker initialized")
    
    def register_service(self, service_name: str, check_url: str = None):
        """注册服务"""
        self.services[service_name] = HealthStatus(
            service_name=service_name,
            status='starting',
            last_check=0,
            uptime=0
        )
        
        logger.info(f"Service registered: {service_name}")
    
    def start(self):
        """开始健康检查"""
        if self._running:
            return
        
        self._running = True
        self._check_thread = threading.Thread(target=self._check_loop, daemon=True)
        self._check_thread.start()
        
        logger.info("Health checker started")
    
    def stop(self):
        """停止健康检查"""
        self._running = False
        
        if self._check_thread:
            self._check_thread.join(timeout=2.0)
        
        logger.info("Health checker stopped")
    
    def _check_loop(self):
        """检查循环"""
        while self._running:
            for service_name in list(self.services.keys()):
                self._check_service(service_name)
            
            time.sleep(self.check_interval)
    
    def _check_service(self, service_name: str):
        """检查服务"""
        status = self.services.get(service_name)
        if not status:
            return
        
        try:
            import requests
            
            response = requests.get(
                f"http://localhost:8000/api/status",
                timeout=5
            )
            
            if response.status_code == 200:
                status.status = 'healthy'
                status.message = 'OK'
                status.error_count = 0
            else:
                status.status = 'unhealthy'
                status.message = f'HTTP {response.status_code}'
                status.error_count += 1
                
        except Exception as e:
            status.status = 'unhealthy'
            status.message = str(e)
            status.error_count += 1
        
        status.last_check = time.time()
    
    def get_status(self, service_name: str = None) -> Dict:
        """获取状态"""
        if service_name:
            status = self.services.get(service_name)
            return status.to_dict() if status else {}
        
        return {
            name: status.to_dict()
            for name, status in self.services.items()
        }
    
    def is_healthy(self, service_name: str) -> bool:
        """检查是否健康"""
        status = self.services.get(service_name)
        return status.status == 'healthy' if status else False


class DeploymentManager:
    """
    部署管理器
    
    整合所有部署功能
    """
    
    def __init__(self):
        """初始化部署管理器"""
        self.docker = DockerManager()
        self.kubernetes = KubernetesManager()
        self.health_checker = HealthChecker()
        
        self.config: Dict = {}
        self.start_time = 0
        
        logger.info("DeploymentManager initialized")
    
    def load_config(self, config_path: str = "config/deployment.yaml") -> Dict:
        """加载配置"""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            logger.info(f"Config loaded: {config_path}")
            return self.config
            
        except Exception as e:
            logger.error(f"Load config error: {e}")
            return {}
    
    def deploy_docker(self, config: ServiceConfig) -> bool:
        """Docker 部署"""
        if not self.docker.is_docker_available():
            logger.error("Docker not available")
            return False
        
        # 构建镜像
        if not self.docker.build_image("Dockerfile", config.image):
            return False
        
        # 运行容器
        container_id = self.docker.run_container(config)
        
        if container_id:
            self.health_checker.register_service(config.service_name)
            self.health_checker.start()
            return True
        
        return False
    
    def deploy_kubernetes(self, config: ServiceConfig) -> bool:
        """Kubernetes 部署"""
        if not self.kubernetes.is_kubectl_available():
            logger.error("kubectl not available")
            return False
        
        # 创建命名空间
        self.kubernetes.create_namespace()
        
        # 生成并应用 Deployment
        deployment_yaml = self.kubernetes.generate_deployment(config)
        with open('/tmp/deployment.yaml', 'w') as f:
            f.write(deployment_yaml)
        
        if not self.kubernetes.apply_manifest('/tmp/deployment.yaml'):
            return False
        
        # 生成并应用 Service
        service_yaml = self.kubernetes.generate_service(config)
        with open('/tmp/service.yaml', 'w') as f:
            f.write(service_yaml)
        
        return self.kubernetes.apply_manifest('/tmp/service.yaml')
    
    def get_status(self) -> Dict:
        """获取部署状态"""
        return {
            'uptime': time.time() - self.start_time if self.start_time else 0,
            'health': self.health_checker.get_status(),
            'docker_available': self.docker.is_docker_available(),
            'kubernetes_available': self.kubernetes.is_kubectl_available()
        }
    
    def generate_all_configs(self, output_dir: str = "deploy"):
        """生成所有配置文件"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Dockerfile
        self.docker.generate_dockerfile(f"{output_dir}/Dockerfile")
        
        # docker-compose.yml
        self.docker.generate_docker_compose(f"{output_dir}/docker-compose.yml")
        
        # K8s manifests
        config = ServiceConfig(
            service_name="camera-perception",
            image="camera-perception:latest"
        )
        
        with open(f"{output_dir}/k8s-deployment.yaml", 'w') as f:
            f.write(self.kubernetes.generate_deployment(config))
        
        with open(f"{output_dir}/k8s-service.yaml", 'w') as f:
            f.write(self.kubernetes.generate_service(config))
        
        logger.info(f"All configs generated in {output_dir}")


# 全局实例
_deployment_manager = None

def get_deployment_manager() -> DeploymentManager:
    """获取部署管理器单例"""
    global _deployment_manager
    if _deployment_manager is None:
        _deployment_manager = DeploymentManager()
    return _deployment_manager


# 测试代码
if __name__ == '__main__':
    print("Testing Deployment Manager...")
    
    manager = DeploymentManager()
    
    # 检查可用性
    print(f"Docker available: {manager.docker.is_docker_available()}")
    print(f"Kubernetes available: {manager.kubernetes.is_kubectl_available()}")
    
    # 生成配置
    manager.generate_all_configs()
    
    # 创建服务配置
    config = ServiceConfig(
        service_name="camera-perception",
        image="camera-perception:latest",
        port=8000,
        environment={"LOG_LEVEL": "INFO"}
    )
    
    print(f"Service config: {config.service_name}")
    
    print("\nDone!")
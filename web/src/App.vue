<template>
  <div class="app-container">
    <!-- 错误提示 -->
    <div v-if="errorMessage" class="error-banner" @click="clearError">
      <span class="error-icon">⚠️</span>
      <span>{{ errorMessage }}</span>
      <span class="error-close">×</span>
    </div>

    <!-- 顶部导航 -->
    <header class="header">
      <div class="header-left">
        <h1>🎥 摄像头实时感知系统</h1>
        <div class="status-bar">
          <span :class="['status-dot', { online: systemStatus.camera_opened }]"></span>
          <span>{{ systemStatus.camera_opened ? '摄像头已连接' : '摄像头未连接' }}</span>
          <span v-if="systemStatus.calibrated" class="calibrated">✓ 已标定</span>
        </div>
      </div>
      <nav class="nav-tabs">
        <button :class="['tab', { active: currentTab === 'monitor' }]" @click="currentTab = 'monitor'">
          📺 实时监控
        </button>
        <button :class="['tab', { active: currentTab === 'calibration' }]" @click="currentTab = 'calibration'">
          🔧 相机标定
        </button>
        <button :class="['tab', { active: currentTab === 'settings' }]" @click="currentTab = 'settings'">
          ⚙️ 参数配置
        </button>
      </nav>
    </header>

    <main class="main-content">
      <!-- 实时监控页面 -->
      <div v-if="currentTab === 'monitor'" class="monitor-page">
        <!-- 左侧：视频显示区 -->
        <div class="video-section">
          <div class="video-container" ref="videoContainer">
            <canvas ref="videoCanvas" :width="canvasWidth" :height="canvasHeight"></canvas>
            <div class="video-overlay">
              <div v-if="!systemStatus.camera_opened" class="no-signal">
                <div class="no-signal-icon">📷</div>
                <div>无视频信号</div>
                <div class="no-signal-hint">请点击"启动摄像头"按钮</div>
              </div>
            </div>
          </div>

          <!-- 控制按钮 -->
          <div class="controls">
            <button class="btn btn-primary" @click="startCamera" :disabled="systemStatus.camera_opened">
              ▶ 启动摄像头
            </button>
            <button class="btn btn-danger" @click="stopCamera" :disabled="!systemStatus.camera_opened">
              ⏹ 停止摄像头
            </button>
            <button class="btn btn-success" @click="loadCalibration" :disabled="!systemStatus.camera_opened">
              📐 加载标定
            </button>
            <button class="btn btn-warning" @click="exportData" :disabled="detectionData.persons.length === 0">
              📊 导出数据
            </button>
          </div>
        </div>

        <!-- 右侧：数据面板 -->
        <div class="data-section" :style="{ maxHeight: videoContainerHeight + 'px' }">
          <!-- 检测结果 -->
          <div class="panel">
            <h2>📊 检测结果</h2>
            <div class="metrics">
              <div class="metric">
                <span class="label">人数</span>
                <span class="value">{{ detectionData.persons.length }}</span>
              </div>
              <div class="metric">
                <span class="label">手数</span>
                <span class="value">{{ detectionData.hands.length }}</span>
              </div>
              <div class="metric">
                <span class="label">FPS</span>
                <span class="value">{{ systemStatus.fps }}</span>
              </div>
            </div>
            <!-- 性能监控 -->
            <div v-if="detectionData.processing" class="performance-metrics">
              <div class="perf-row">
                <span>检测: {{ detectionData.processing.detect_time_ms }}ms</span>
                <span>空间: {{ detectionData.processing.spatial_time_ms }}ms</span>
                <span>总计: {{ detectionData.processing.total_time_ms }}ms</span>
              </div>
            </div>
          </div>

          <!-- 人体数据 -->
          <div class="panel person-panel" v-if="detectionData.persons.length > 0">
            <h2>👤 人体数据</h2>
            <div v-for="(person, index) in detectionData.persons" :key="index" class="person-data">
              <div class="person-header">
                <span class="person-id">人 {{ index + 1 }}</span>
                <span :class="['motion-badge', person.motion_state]">
                  {{ getMotionStateLabel(person.motion_state) }}
                </span>
                <span class="body-part-badge" :title="'画面占比: ' + ((person.bbox_area_ratio || 0) * 100).toFixed(1) + '%'">
                  {{ getBodyPartLabel(person.body_part) }}
                </span>
              </div>
              <div class="data-row">
                <span>📏 距离：</span>
                <span class="highlight">{{ person.distance }} m</span>
                <span v-if="person.distance_confidence" class="confidence-mini">
                  ({{ (person.distance_confidence * 100).toFixed(0) }}%)
                </span>
              </div>
              <div class="data-row">
                <span>📐 身高：</span>
                <span class="highlight">{{ person.height }} cm</span>
              </div>
              <div class="data-row">
                <span>🚀 速度：</span>
                <span>{{ person.velocity?.toFixed(2) || 0 }} m/s</span>
              </div>
              <div class="data-row">
                <span>📍 估计方法：</span>
                <span class="method-tag">{{ person.estimate_method }}</span>
              </div>
              <div class="data-row">
                <span>🎯 置信度：</span>
                <span>头部 {{ ((person.head_confidence || 0) * 100).toFixed(0) }}%</span>
                <span v-if="person.pose_confidence"> | 姿态 {{ ((person.pose_confidence || 0) * 100).toFixed(0) }}%</span>
              </div>
              <div class="data-row" v-if="person.bbox_area_ratio">
                <span>📦 画面占比：</span>
                <span>{{ (person.bbox_area_ratio * 100).toFixed(1) }}%</span>
              </div>
              <div class="data-row" v-if="person.keypoints">
                <span>🔑 关键点：</span>
                <span>{{ Object.keys(person.keypoints).length }} 个</span>
              </div>
              <div class="data-row" v-if="person.body_part_confidence">
                <span>🎯 部位置信度：</span>
                <span>{{ (person.body_part_confidence * 100).toFixed(0) }}%</span>
              </div>
              <div class="data-row" v-if="person.bbox">
                <span>📐 边界框：</span>
                <span>{{ person.bbox[2] }}×{{ person.bbox[3] }}px</span>
              </div>
              <div class="data-row" v-if="person.estimate_method">
                <span>🔧 估计方法：</span>
                <span class="method-tag">{{ person.estimate_method }}</span>
              </div>
              <div class="data-row" v-if="person.image_size">
                <span>🖼️ 图像尺寸：</span>
                <span>{{ person.image_size[0] }}×{{ person.image_size[1] }}</span>
              </div>
              <div class="data-row">
                <span>🗺️ 顶视图：</span>
                <span>({{ person.topview?.x }}, {{ person.topview?.y }})</span>
              </div>
            </div>
          </div>

          <!-- 手部数据 -->
          <div class="panel hand-panel" v-if="detectionData.hands.length > 0">
            <h2>✋ 手部数据</h2>
            <div v-for="(hand, index) in detectionData.hands" :key="index" class="hand-data">
              <div class="person-header">
                <span class="person-id">手 {{ index + 1 }} ({{ hand.hand_type || 'Unknown' }})</span>
              </div>
              <div class="data-row">
                <span>📏 手大小：</span>
                <span class="highlight">{{ hand.size }} cm</span>
              </div>
              <div class="data-row">
                <span>📏 距离：</span>
                <span class="highlight">{{ hand.distance }} m</span>
              </div>
              <div class="data-row">
                <span>🗺️ 顶视图：</span>
                <span>({{ hand.topview?.x }}, {{ hand.topview?.y }})</span>
              </div>
            </div>
          </div>

          <!-- 顶视图 -->
          <div class="panel">
            <h2>🗺️ 顶视图</h2>
            <canvas ref="topviewCanvas" width="800" height="600" class="topview-canvas"></canvas>
            <div class="topview-legend">
              <div class="legend-item">
                <span class="legend-dot person-dot"></span>
                <span>人</span>
              </div>
              <div class="legend-item">
                <span class="legend-dot hand-dot"></span>
                <span>手</span>
              </div>
              <div class="legend-item">
                <span class="legend-dot camera-dot"></span>
                <span>摄像头</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 相机标定页面 -->
      <div v-if="currentTab === 'calibration'" class="calibration-page">
        <div class="calibration-container">
          <div class="calibration-left">
            <div class="panel">
              <h2>📐 相机标定</h2>
              <p class="calibration-hint">
                使用棋盘格标定板进行相机标定，获取相机内参和畸变系数。
                <br/>请打印 9×6 的棋盘格标定板，并从不同角度拍摄至少 15 张图片。
              </p>
              
              <div class="calibration-steps">
                <h3>标定步骤：</h3>
                <ol>
                  <li>打印棋盘格标定板（9×6 角点）</li>
                  <li>从不同角度/距离拍摄标定板照片</li>
                  <li>上传照片或实时采集</li>
                  <li>执行标定并保存参数</li>
                </ol>
              </div>
              
              <div class="calibration-actions">
                <button class="btn btn-primary" @click="startCalibrationCapture">
                  📷 实时采集
                </button>
                <label class="btn btn-secondary">
                  📁 上传图片
                  <input type="file" multiple accept="image/*" @change="uploadCalibrationImages" style="display: none">
                </label>
                <button class="btn btn-success" @click="runCalibration" :disabled="!canCalibrate">
                  🔧 执行标定
                </button>
              </div>
              
              <div v-if="calibrationStatus.collecting" class="capture-status">
                <div class="progress-bar">
                  <div class="progress" :style="{ width: calibrationProgress + '%' }"></div>
                </div>
                <p>已采集 {{ calibrationStatus.imageCount }} / 15 张有效图片</p>
              </div>
            </div>
            
            <div class="panel" v-if="calibrationResult">
              <h2>📊 标定结果</h2>
              <div class="result-grid">
                <div class="result-item">
                  <span class="result-label">重投影误差</span>
                  <span class="result-value">{{ calibrationResult.reprojection_error.toFixed(4) }} px</span>
                </div>
                <div class="result-item">
                  <span class="result-label">有效图片</span>
                  <span class="result-value">{{ calibrationResult.num_images }} 张</span>
                </div>
                <div class="result-item">
                  <span class="result-label">fx</span>
                  <span class="result-value">{{ calibrationResult.fx.toFixed(2) }}</span>
                </div>
                <div class="result-item">
                  <span class="result-label">fy</span>
                  <span class="result-value">{{ calibrationResult.fy.toFixed(2) }}</span>
                </div>
                <div class="result-item">
                  <span class="result-label">cx</span>
                  <span class="result-value">{{ calibrationResult.cx.toFixed(2) }}</span>
                </div>
                <div class="result-item">
                  <span class="result-label">cy</span>
                  <span class="result-value">{{ calibrationResult.cy.toFixed(2) }}</span>
                </div>
              </div>
              <div class="distortion-coeffs">
                <h4>畸变系数：</h4>
                <code>k1={{ calibrationResult.dist_coeffs[0]?.toFixed(6) }}, 
                      k2={{ calibrationResult.dist_coeffs[1]?.toFixed(6) }},
                      p1={{ calibrationResult.dist_coeffs[2]?.toFixed(6) }},
                      p2={{ calibrationResult.dist_coeffs[3]?.toFixed(6) }}</code>
              </div>
            </div>
          </div>
          
          <div class="calibration-right">
            <div class="panel">
              <h2>📷 标定预览</h2>
              <canvas ref="calibrationCanvas" width="640" height="480" class="calibration-preview"></canvas>
              <div class="calibration-controls">
                <button class="btn btn-primary" @click="toggleCalibrationPreview" :disabled="!systemStatus.camera_opened">
                  {{ calibrationPreviewActive ? '停止预览' : '开始预览' }}
                </button>
              </div>
            </div>
            
            <div class="panel">
              <h2>⚙️ 外参配置</h2>
              <div class="form-group">
                <label>摄像头安装高度 (米)</label>
                <input type="number" v-model.number="extrinsics.height" step="0.1" class="form-input">
              </div>
              <div class="form-group">
                <label>俯角 (度)</label>
                <input type="number" v-model.number="extrinsics.pitch" step="1" class="form-input">
              </div>
              <button class="btn btn-primary" @click="saveExtrinsics">保存外参</button>
            </div>
          </div>
        </div>
      </div>

      <!-- 参数配置页面 -->
      <div v-if="currentTab === 'settings'" class="settings-page">
        <div class="settings-container">
          <div class="panel">
            <h2>📷 相机配置</h2>
            <div class="form-group">
              <label>摄像头设备 ID</label>
              <input type="number" v-model.number="settings.cameraId" class="form-input">
            </div>
            <div class="form-group">
              <label>分辨率</label>
              <select v-model="settings.resolution" class="form-input">
                <option value="1920x1080">1920×1080 (Full HD)</option>
                <option value="1280x720">1280×720 (HD)</option>
                <option value="640x480">640×480 (VGA)</option>
              </select>
            </div>
            <div class="form-group">
              <label>帧率 (FPS)</label>
              <input type="number" v-model.number="settings.fps" min="1" max="60" class="form-input">
            </div>
            <button class="btn btn-primary" @click="saveSettings">保存相机配置</button>
          </div>
          
          <div class="panel">
            <h2>🎯 检测配置</h2>
            <div class="form-group">
              <label>置信度阈值</label>
              <input type="range" v-model.number="settings.confThreshold" min="0.1" max="0.9" step="0.05" class="form-range">
              <span>{{ settings.confThreshold.toFixed(2) }}</span>
            </div>
            <div class="form-group">
              <label>关键点平滑</label>
              <label class="switch">
                <input type="checkbox" v-model="settings.smoothEnabled">
                <span class="slider"></span>
              </label>
            </div>
            <button class="btn btn-primary" @click="saveSettings">保存检测配置</button>
          </div>
          
          <div class="panel">
            <h2>📏 空间计量配置</h2>
            <div class="form-group">
              <label>参考肩宽 (米)</label>
              <input type="number" v-model.number="settings.refShoulderWidth" step="0.01" class="form-input">
              <small>用于距离估算的参考值</small>
            </div>
            <div class="form-group">
              <label>顶视图比例 (像素/米)</label>
              <input type="number" v-model.number="settings.topviewScale" step="1" class="form-input">
            </div>
            <button class="btn btn-primary" @click="saveSettings">保存计量配置</button>
          </div>

          <div class="panel">
            <h2>👤 头部尺寸参数</h2>
            <p class="config-hint">用于近距离距离估计的参考尺寸</p>
            <div class="form-group">
              <label>参考头宽 (米)</label>
              <input type="number" v-model.number="headParams.refHeadWidth" step="0.01" class="form-input">
              <small>成人平均 14-16cm</small>
            </div>
            <div class="form-group">
              <label>双眼间距 (米)</label>
              <input type="number" v-model.number="headParams.refEyeDistance" step="0.001" class="form-input">
              <small>平均 6.3cm</small>
            </div>
            <div class="form-group">
              <label>双耳间距 (米)</label>
              <input type="number" v-model.number="headParams.refEarDistance" step="0.01" class="form-input">
              <small>平均 14.5cm</small>
            </div>
            <div class="form-group">
              <label>眼到鼻距离 (米)</label>
              <input type="number" v-model.number="headParams.refEyeNoseDistance" step="0.001" class="form-input">
              <small>平均 3.5cm</small>
            </div>
            <div class="form-group">
              <label>头部高度 (米)</label>
              <input type="number" v-model.number="headParams.refHeadHeight" step="0.01" class="form-input">
              <small>平均 22cm</small>
            </div>
            <button class="btn btn-primary" @click="saveHeadParams">保存头部参数</button>
          </div>

          <div class="panel">
            <h2>🚀 移动速度参数</h2>
            <p class="config-hint">卡尔曼滤波器参数，用于距离平滑</p>
            <div class="form-group">
              <label>最大移动速度 (米/秒)</label>
              <input type="number" v-model.number="kalmanParams.maxSpeed" step="0.5" min="0.5" max="10" class="form-input">
              <small>步行约1.5m/s，奔跑可达10m/s</small>
            </div>
            <div class="form-group">
              <label>过程噪声</label>
              <input type="range" v-model.number="kalmanParams.processNoise" min="0.01" max="1" step="0.01" class="form-range">
              <span>{{ kalmanParams.processNoise.toFixed(2) }}</span>
              <small>越小越信任预测模型</small>
            </div>
            <div class="form-group">
              <label>测量噪声</label>
              <input type="range" v-model.number="kalmanParams.measurementNoise" min="0.1" max="1" step="0.05" class="form-range">
              <span>{{ kalmanParams.measurementNoise.toFixed(2) }}</span>
              <small>越小越信任测量值</small>
            </div>
            <button class="btn btn-primary" @click="saveKalmanParams">保存速度参数</button>
          </div>

          <div class="panel">
            <h2>📍 近距离估计优化</h2>
            <p class="config-hint">近距离（&lt;1.5m）时的距离估计优化参数</p>
            <div class="form-group">
              <label>近距离阈值 (米)</label>
              <input type="number" v-model.number="closeRangeParams.threshold" step="0.1" min="0.5" max="3" class="form-input">
              <small>低于此距离启用优化算法</small>
            </div>
            <div class="form-group">
              <label>超近距离阈值 (米)</label>
              <input type="number" v-model.number="closeRangeParams.ultraCloseThreshold" step="0.1" min="0.3" max="1" class="form-input">
              <small>低于此距离主要依赖头部估计</small>
            </div>
            <div class="form-group">
              <label>近距离头部权重</label>
              <input type="range" v-model.number="closeRangeParams.headWeight" min="0" max="1" step="0.1" class="form-range">
              <span>{{ closeRangeParams.headWeight.toFixed(1) }}</span>
              <small>近距离时头部估计的权重</small>
            </div>
            <div class="form-group">
              <label>近距离身体权重</label>
              <input type="range" v-model.number="closeRangeParams.bodyWeight" min="0" max="1" step="0.1" class="form-range">
              <span>{{ closeRangeParams.bodyWeight.toFixed(1) }}</span>
              <small>近距离时身体估计的权重</small>
            </div>
            <div class="form-group">
              <label>启用透视校正</label>
              <label class="switch">
                <input type="checkbox" v-model="closeRangeParams.usePerspectiveCorrection">
                <span class="slider"></span>
              </label>
              <small>近距离时应用透视校正</small>
            </div>
            <button class="btn btn-primary" @click="saveCloseRangeParams">保存近距离参数</button>
          </div>
          
          <div class="panel">
            <h2>🔧 系统校准</h2>
            <div class="form-group">
              <label>已知身高校准</label>
              <div class="inline-inputs">
                <input type="number" v-model.number="calibrationInput.height" placeholder="身高 (cm)" class="form-input">
                <button class="btn btn-secondary" @click="calibrateHeight">应用</button>
              </div>
            </div>
            <div class="form-group">
              <label>已知距离校准</label>
              <div class="inline-inputs">
                <input type="number" v-model.number="calibrationInput.distance" placeholder="距离 (m)" class="form-input">
                <button class="btn btn-secondary" @click="calibrateDistance">应用</button>
              </div>
            </div>
            <button class="btn btn-danger" @click="resetCalibration">重置所有校准</button>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, onUnmounted, computed } from 'vue'
import axios from 'axios'

// ==================== 状态定义 ====================

// 当前页面
const currentTab = ref('monitor')

// 系统状态
const systemStatus = reactive({
  camera_opened: false,
  calibrated: false,
  fps: 0
})

// 检测数据
const detectionData = reactive({
  persons: [],
  hands: []
})

// Canvas 引用
const videoCanvas = ref(null)
const videoContainer = ref(null)
const topviewCanvas = ref(null)
const calibrationCanvas = ref(null)

// 画布尺寸（响应式）
const canvasWidth = ref(1280)
const canvasHeight = ref(720)
const videoContainerHeight = ref(720)

// WebSocket 连接
let videoWs = null
let dataWs = null

// API 基础 URL
const API_BASE = ''

// 标定相关
const calibrationStatus = reactive({
  collecting: false,
  imageCount: 0
})

const calibrationProgress = computed(() => {
  return Math.min(100, (calibrationStatus.imageCount / 15) * 100)
})

const canCalibrate = computed(() => {
  return calibrationStatus.imageCount >= 15
})

const calibrationResult = ref(null)
const calibrationPreviewActive = ref(false)

// 外参配置
const extrinsics = reactive({
  height: 1.8,
  pitch: 30.0
})

// 系统设置
const settings = reactive({
  cameraId: 0,
  resolution: '1920x1080',
  fps: 20,
  confThreshold: 0.5,
  smoothEnabled: true,
  refShoulderWidth: 0.45,
  topviewScale: 10
})

// 头部尺寸参数
const headParams = reactive({
  refHeadWidth: 0.15,
  refEyeDistance: 0.063,
  refEarDistance: 0.145,
  refEyeNoseDistance: 0.035,
  refHeadHeight: 0.22
})

// 卡尔曼滤波参数
const kalmanParams = reactive({
  maxSpeed: 3.0,
  processNoise: 0.1,
  measurementNoise: 0.3
})

// 近距离估计参数
const closeRangeParams = reactive({
  threshold: 1.5,
  ultraCloseThreshold: 0.5,
  headWeight: 0.7,
  bodyWeight: 0.3,
  usePerspectiveCorrection: true
})

// 相机安装参数（用于三角函数距离估计）
const cameraParams = reactive({
  height: 1.8,
  pitchAngle: 30.0
})

// 错误消息
const errorMessage = ref('')

// 清除错误
const clearError = () => {
  errorMessage.value = ''
}

// 显示错误
const showError = (msg) => {
  errorMessage.value = msg
  // 3秒后自动清除
  setTimeout(() => {
    errorMessage.value = ''
  }, 5000)
}

// 校准输入
const calibrationInput = reactive({
  height: 0,
  distance: 0
})

// ==================== 摄像头控制 ====================

const startCamera = async () => {
  try {
    await axios.post(`${API_BASE}/api/camera/start`, {
      camera_id: settings.cameraId,
      resolution: settings.resolution.split('x').map(Number),
      fps: settings.fps
    })
    systemStatus.camera_opened = true
    connectWebSocket()
  } catch (error) {
    console.error('启动摄像头失败:', error)
    alert('启动摄像头失败：' + (error.response?.data?.detail || error.message))
  }
}

const stopCamera = async () => {
  try {
    await axios.post(`${API_BASE}/api/camera/stop`)
    systemStatus.camera_opened = false
    disconnectWebSocket()
  } catch (error) {
    console.error('停止摄像头失败:', error)
  }
}

const loadCalibration = async () => {
  try {
    await axios.post(`${API_BASE}/api/calibration/load?filepath=calibration_data/calib_params.json`)
    const res = await axios.get(`${API_BASE}/api/calibration/status`)
    systemStatus.calibrated = res.data.calibrated
    alert('标定加载成功')
  } catch (error) {
    console.error('加载标定失败:', error)
    alert('加载标定失败：' + error.response?.data?.detail)
  }
}

// ==================== WebSocket 连接 ====================

const connectWebSocket = () => {
  // 视频流 WebSocket - 直接连接后端
  videoWs = new WebSocket('ws://localhost:8000/ws/video')
  videoWs.binaryType = 'arraybuffer'

  // 存储最新的检测数据用于绘制
  let latestDetectionData = { persons: [], hands: [] }

  videoWs.onmessage = (event) => {
    if (!videoCanvas.value) return

    const ctx = videoCanvas.value.getContext('2d')
    const img = new Image()
    img.onload = () => {
      // 绘制视频帧
      ctx.drawImage(img, 0, 0, canvasWidth.value, canvasHeight.value)
      
      // 绘制检测标注（在视频帧之上）
      drawAnnotations(latestDetectionData)
    }
    img.src = URL.createObjectURL(new Blob([event.data]))
  }

  // 数据 WebSocket - 直接连接后端
  dataWs = new WebSocket('ws://localhost:8000/ws/data')

  dataWs.onmessage = (event) => {
    const data = JSON.parse(event.data)

    // 确保数据完整性
    if (!data) {
      console.warn('Received empty data')
      return
    }

    // 更新检测数据（确保数组存在）
    detectionData.persons = data.persons || []
    detectionData.hands = data.hands || []

    // 更新 FPS
    systemStatus.fps = Math.round(1000 / (data.timestamp_diff || 50))

    // 保存最新数据用于绘制
    latestDetectionData = {
      persons: data.persons || [],
      hands: data.hands || []
    }

    // 绘制顶视图（独立画布）
    drawTopView(latestDetectionData)
  }
  
  // 心跳
  const heartbeatInterval = setInterval(() => {
    if (dataWs && dataWs.readyState === WebSocket.OPEN) {
      dataWs.send(JSON.stringify({ type: 'heartbeat' }))
    }
  }, 5000)
  
  // 清理
  dataWs.onclose = () => clearInterval(heartbeatInterval)
}

const disconnectWebSocket = () => {
  if (videoWs) {
    videoWs.close()
    videoWs = null
  }
  if (dataWs) {
    dataWs.close()
    dataWs = null
  }
}

// ==================== 辅助函数 ====================

const getMotionStateLabel = (state) => {
  const stateMap = {
    'static': '静止',
    'moving': '移动中',
    'walking': '行走',
    'running': '奔跑',
    'unknown': '未知'
  }
  return stateMap[state] || state || '未知'
}

const getBodyPartLabel = (part) => {
  const partMap = {
    'full_body': '全身',
    'half_body': '半身',
    'upper_body': '上半身',
    'lower_body': '下半身',
    'head_only': '仅头部',
    'unknown': '未知'
  }
  return partMap[part] || part || '未知'
}

// ==================== 绘制函数 ====================

const drawAnnotations = (data) => {
  if (!videoCanvas.value) return

  const ctx = videoCanvas.value.getContext('2d')
  const scaleX = canvasWidth.value / 1920
  const scaleY = canvasHeight.value / 1080

  // 绘制人体
  data.persons?.forEach(person => {
    // 检查 bbox 是否存在
    if (!person.bbox || !Array.isArray(person.bbox) || person.bbox.length < 4) {
      console.warn('Invalid person bbox:', person)
      return  // 跳过当前person，继续下一个
    }
    const [x, y, w, h] = person.bbox
    ctx.strokeStyle = '#00ff00'
    ctx.lineWidth = 2 * Math.min(scaleX, scaleY)
    ctx.strokeRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY)

    // 绘制关键点
    if (person.keypoints) {
      Object.values(person.keypoints).forEach(kp => {
        ctx.fillStyle = '#ff0000'
        ctx.beginPath()
        ctx.arc(kp[0] * scaleX, kp[1] * scaleY, 5 * Math.min(scaleX, scaleY), 0, Math.PI * 2)
        ctx.fill()
      })
    }

    // 绘制文字
    ctx.fillStyle = '#00ff00'
    ctx.font = `${14 * Math.min(scaleX, scaleY)}px Arial`
    ctx.fillText(`距离：${person.distance}m`, x * scaleX, y * scaleY - 10)
    if (person.height > 0) {
      ctx.fillText(`身高：${person.height}cm`, x * scaleX, y * scaleY - 30)
    }
  })

  // 绘制手部
  data.hands?.forEach(hand => {
    // 检查 bbox 是否存在
    if (!hand.bbox || !Array.isArray(hand.bbox) || hand.bbox.length < 4) {
      console.warn('Invalid hand bbox:', hand)
      return
    }
    const [x, y, w, h] = hand.bbox
    ctx.strokeStyle = '#0088ff'
    ctx.lineWidth = 2 * Math.min(scaleX, scaleY)
    ctx.strokeRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY)

    // 绘制关键点
    if (hand.keypoints && hand.keypoints.length >= 21) {
      ctx.strokeStyle = '#ff00ff'
      ctx.lineWidth = 1 * Math.min(scaleX, scaleY)
      hand.keypoints.forEach(kp => {
        ctx.fillStyle = '#ff00ff'
        ctx.beginPath()
        ctx.arc(kp[0] * scaleX, kp[1] * scaleY, 3 * Math.min(scaleX, scaleY), 0, Math.PI * 2)
        ctx.fill()
      })
    }
  })
}

const drawTopView = (data) => {
  if (!topviewCanvas.value) return
  
  const ctx = topviewCanvas.value.getContext('2d')
  const width = topviewCanvas.value.width
  const height = topviewCanvas.value.height
  
  // 清空
  ctx.fillStyle = '#1a1a2e'
  ctx.fillRect(0, 0, width, height)
  
  // 网格
  ctx.strokeStyle = '#333355'
  ctx.lineWidth = 1
  const gridSize = 50
  for (let x = 0; x < width; x += gridSize) {
    ctx.beginPath()
    ctx.moveTo(x, 0)
    ctx.lineTo(x, height)
    ctx.stroke()
  }
  for (let y = 0; y < height; y += gridSize) {
    ctx.beginPath()
    ctx.moveTo(0, y)
    ctx.lineTo(width, y)
    ctx.stroke()
  }
  
  // 原点（摄像头）
  ctx.fillStyle = '#ffffff'
  ctx.beginPath()
  ctx.arc(400, 300, 8, 0, Math.PI * 2)
  ctx.fill()
  ctx.fillStyle = '#ffffff'
  ctx.font = '12px Arial'
  ctx.fillText('摄像头', 410, 300)
  
  // 人体位置
  data.persons?.forEach((person, index) => {
    if (!person.topview) return
    
    ctx.fillStyle = '#00ff00'
    ctx.beginPath()
    ctx.arc(person.topview.x, person.topview.y, 15, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = '#ffffff'
    ctx.font = '12px Arial'
    ctx.fillText(`人${index + 1}`, person.topview.x - 15, person.topview.y - 20)
  })
  
  // 手部位置
  data.hands?.forEach((hand, index) => {
    if (!hand.topview) return
    
    ctx.fillStyle = '#0088ff'
    ctx.beginPath()
    ctx.arc(hand.topview.x, hand.topview.y, 10, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = '#ffffff'
    ctx.font = '12px Arial'
    ctx.fillText(`手${index + 1}`, hand.topview.x - 15, hand.topview.y - 15)
  })
}

// ==================== 标定功能 ====================

const startCalibrationCapture = async () => {
  try {
    calibrationStatus.collecting = true
    calibrationStatus.imageCount = 0

    // 开始采集
    await axios.get(`${API_BASE}/api/calibration/start_capture`)

    // 自动采集循环
    const captureInterval = setInterval(async () => {
      try {
        // 采集一帧
        const response = await axios.post(`${API_BASE}/api/calibration/capture_frame`)
        calibrationStatus.imageCount = response.data.image_count

        // 检查是否完成
        if (response.data.image_count >= 15) {
          clearInterval(captureInterval)
          calibrationStatus.collecting = false
          alert('采集完成！请点击"执行标定"')
        }
      } catch (err) {
        // 继续尝试
        console.log('Capture attempt:', err.response?.data?.detail || err.message)
      }
    }, 1000) // 每秒尝试采集一次

    // 保存 interval 以便可以取消
    calibrationStatus.intervalId = captureInterval

  } catch (error) {
    console.error('采集失败:', error)
    alert('采集失败：' + error.message)
    calibrationStatus.collecting = false
  }
}

const uploadCalibrationImages = async (event) => {
  const files = event.target.files
  if (!files.length) return
  
  const formData = new FormData()
  for (let file of files) {
    formData.append('images', file)
  }
  
  try {
    await axios.post(`${API_BASE}/api/calibration/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    alert(`已上传 ${files.length} 张图片`)
  } catch (error) {
    console.error('上传失败:', error)
    alert('上传失败：' + error.message)
  }
}

const runCalibration = async () => {
  try {
    const response = await axios.post(`${API_BASE}/api/calibration/run`)
    calibrationResult.value = response.data

    if (response.data.status === 'success') {
      systemStatus.calibrated = true
      alert('标定成功！')
    }
  } catch (error) {
    console.error('标定失败:', error)
    alert('标定失败：' + error.response?.data?.detail)
  }
}

let calibrationPreviewInterval = null

const toggleCalibrationPreview = async () => {
  calibrationPreviewActive.value = !calibrationPreviewActive.value

  if (calibrationPreviewActive.value) {
    // 启动预览 - 每100ms获取一帧
    calibrationPreviewInterval = setInterval(async () => {
      try {
        const response = await axios.get(`${API_BASE}/api/calibration/preview`)
        if (response.data.status === 'success' && response.data.image) {
          // 绘制到canvas
          const canvas = calibrationCanvas.value
          if (!canvas) return
          
          const ctx = canvas.getContext('2d')
          const img = new Image()
          img.onload = () => {
            // 保持宽高比缩放
            const scale = Math.min(
              canvas.width / img.width,
              canvas.height / img.height
            )
            const x = (canvas.width - img.width * scale) / 2
            const y = (canvas.height - img.height * scale) / 2
            
            ctx.clearRect(0, 0, canvas.width, canvas.height)
            ctx.drawImage(img, x, y, img.width * scale, img.height * scale)
            
            // 绘制状态信息
            ctx.fillStyle = response.data.found ? 'rgba(0, 255, 0, 0.8)' : 'rgba(255, 0, 0, 0.8)'
            ctx.fillRect(10, canvas.height - 40, 200, 30)
            ctx.fillStyle = '#fff'
            ctx.font = '14px Arial'
            ctx.fillText(
              response.data.found ? `✓ 检测到棋盘格 (${response.data.corners_count}角点)` : '✗ 未检测到棋盘格',
              20,
              canvas.height - 20
            )
          }
          img.src = 'data:image/jpeg;base64,' + response.data.image
        }
      } catch (error) {
        console.error('预览获取失败:', error)
      }
    }, 100)
  } else {
    // 停止预览
    if (calibrationPreviewInterval) {
      clearInterval(calibrationPreviewInterval)
      calibrationPreviewInterval = null
    }
    // 清空canvas
    const canvas = calibrationCanvas.value
    if (canvas) {
      const ctx = canvas.getContext('2d')
      ctx.clearRect(0, 0, canvas.width, canvas.height)
    }
  }
}

const saveExtrinsics = async () => {
  try {
    await axios.post(`${API_BASE}/api/calibration/extrinsics`, {
      height: extrinsics.height,
      pitch_angle: extrinsics.pitch
    })
    alert('外参已保存')
  } catch (error) {
    console.error('保存失败:', error)
    alert('保存失败')
  }
}

// ==================== 设置功能 ====================

const saveSettings = async () => {
  try {
    // 保存到后端
    await axios.post(`${API_BASE}/api/settings`, {
      camera: {
        id: settings.cameraId,
        resolution: settings.resolution,
        fps: settings.fps
      },
      detection: {
        conf_threshold: settings.confThreshold,
        smooth_enabled: settings.smoothEnabled
      },
      spatial: {
        ref_shoulder_width: settings.refShoulderWidth,
        topview_scale: settings.topviewScale
      }
    })
    
    // 同时保存到 localStorage 实现前端持久化
    localStorage.setItem('cameraSettings', JSON.stringify({
      cameraId: settings.cameraId,
      resolution: settings.resolution,
      fps: settings.fps
    }))
    localStorage.setItem('detectionSettings', JSON.stringify({
      confThreshold: settings.confThreshold,
      smoothEnabled: settings.smoothEnabled
    }))
    localStorage.setItem('spatialSettings', JSON.stringify({
      refShoulderWidth: settings.refShoulderWidth,
      topviewScale: settings.topviewScale
    }))
    
    alert('设置已保存')
  } catch (error) {
    console.error('保存失败:', error)
    alert('保存失败')
  }
}

// 从 localStorage 加载设置
const loadLocalSettings = () => {
  try {
    const cameraSettings = localStorage.getItem('cameraSettings')
    if (cameraSettings) {
      const parsed = JSON.parse(cameraSettings)
      settings.cameraId = parsed.cameraId ?? settings.cameraId
      settings.resolution = parsed.resolution ?? settings.resolution
      settings.fps = parsed.fps ?? settings.fps
    }
    
    const detectionSettings = localStorage.getItem('detectionSettings')
    if (detectionSettings) {
      const parsed = JSON.parse(detectionSettings)
      settings.confThreshold = parsed.confThreshold ?? settings.confThreshold
      settings.smoothEnabled = parsed.smoothEnabled ?? settings.smoothEnabled
    }
    
    const spatialSettings = localStorage.getItem('spatialSettings')
    if (spatialSettings) {
      const parsed = JSON.parse(spatialSettings)
      settings.refShoulderWidth = parsed.refShoulderWidth ?? settings.refShoulderWidth
      settings.topviewScale = parsed.topviewScale ?? settings.topviewScale
    }
    
    console.log('本地设置已加载')
  } catch (error) {
    console.error('加载本地设置失败:', error)
  }
}

const calibrateHeight = async () => {
  try {
    await axios.post(`${API_BASE}/api/calibration/height`, {
      known_height: calibrationInput.height
    })
    alert('身高校准已应用')
  } catch (error) {
    console.error('校准失败:', error)
    alert('校准失败')
  }
}

const calibrateDistance = async () => {
  try {
    await axios.post(`${API_BASE}/api/calibration/distance`, {
      known_distance: calibrationInput.distance
    })
    alert('距离校准已应用')
  } catch (error) {
    console.error('校准失败:', error)
    alert('校准失败')
  }
}

// 保存头部尺寸参数
const saveHeadParams = async () => {
  try {
    await axios.post(`${API_BASE}/api/spatial/head_params`, {
      ref_head_width: headParams.refHeadWidth,
      ref_eye_distance: headParams.refEyeDistance,
      ref_ear_distance: headParams.refEarDistance,
      ref_eye_nose_distance: headParams.refEyeNoseDistance,
      ref_head_height: headParams.refHeadHeight
    })
    alert('头部参数已保存')
  } catch (error) {
    console.error('保存失败:', error)
    alert('保存失败：' + (error.response?.data?.detail || error.message))
  }
}

// 保存卡尔曼滤波参数
const saveKalmanParams = async () => {
  try {
    await axios.post(`${API_BASE}/api/spatial/kalman_params`, {
      max_speed: kalmanParams.maxSpeed,
      process_noise: kalmanParams.processNoise,
      measurement_noise: kalmanParams.measurementNoise
    })
    alert('速度参数已保存')
  } catch (error) {
    console.error('保存失败:', error)
    alert('保存失败：' + (error.response?.data?.detail || error.message))
  }
}

// 保存近距离估计参数
const saveCloseRangeParams = async () => {
  try {
    await axios.post(`${API_BASE}/api/spatial/close_range_params`, {
      threshold: closeRangeParams.threshold,
      ultra_close_threshold: closeRangeParams.ultraCloseThreshold,
      head_weight: closeRangeParams.headWeight,
      body_weight: closeRangeParams.bodyWeight,
      use_perspective_correction: closeRangeParams.usePerspectiveCorrection
    })
    alert('近距离参数已保存')
  } catch (error) {
    console.error('保存失败:', error)
    alert('保存失败：' + (error.response?.data?.detail || error.message))
  }
}

// 保存相机安装参数
const saveCameraParams = async () => {
  try {
    await axios.post(`${API_BASE}/api/calibration/extrinsics`, {
      height: cameraParams.height,
      pitch_angle: cameraParams.pitchAngle
    })
    alert('相机参数已保存')
  } catch (error) {
    console.error('保存失败:', error)
    alert('保存失败：' + (error.response?.data?.detail || error.message))
  }
}

// 加载空间计量配置
const loadSpatialConfig = async () => {
  try {
    const response = await axios.get(`${API_BASE}/api/spatial/config`)
    const data = response.data
    
    if (data.head_params) {
      headParams.refHeadWidth = data.head_params.ref_head_width || 0.15
      headParams.refEyeDistance = data.head_params.ref_eye_distance || 0.063
      headParams.refEarDistance = data.head_params.ref_ear_distance || 0.145
      headParams.refEyeNoseDistance = data.head_params.ref_eye_nose_distance || 0.035
      headParams.refHeadHeight = data.head_params.ref_head_height || 0.22
    }
    
    if (data.kalman_params) {
      kalmanParams.maxSpeed = data.kalman_params.max_speed || 3.0
      kalmanParams.processNoise = data.kalman_params.process_noise || 0.1
      kalmanParams.measurementNoise = data.kalman_params.measurement_noise || 0.3
    }
    
    if (data.close_range_params) {
      closeRangeParams.threshold = data.close_range_params.threshold || 1.5
      closeRangeParams.ultraCloseThreshold = data.close_range_params.ultra_close_threshold || 0.5
      closeRangeParams.headWeight = data.close_range_params.head_weight || 0.7
      closeRangeParams.bodyWeight = data.close_range_params.body_weight || 0.3
      closeRangeParams.usePerspectiveCorrection = data.close_range_params.use_perspective_correction ?? true
    }
  } catch (error) {
    console.error('加载配置失败:', error)
  }
}

const resetCalibration = async () => {
  if (!confirm('确定要重置所有校准参数吗？')) return
  
  try {
    await axios.post(`${API_BASE}/api/calibration/reset`)
    alert('校准已重置')
  } catch (error) {
    console.error('重置失败:', error)
  }
}

const exportData = async () => {
  try {
    const response = await axios.get(`${API_BASE}/api/export/data`, {
      responseType: 'blob'
    })
    
    const blob = new Blob([response.data], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `detection_data_${new Date().toISOString().split('T')[0]}.json`
    a.click()
    URL.revokeObjectURL(url)
  } catch (error) {
    console.error('导出失败:', error)
    alert('导出失败')
  }
}

// ==================== 获取系统状态 ====================

const fetchStatus = async () => {
  try {
    const res = await axios.get(`${API_BASE}/api/status`)
    systemStatus.camera_opened = res.data.camera_opened
    systemStatus.calibrated = res.data.calibrated
  } catch (error) {
    console.error('获取状态失败:', error)
  }
}

// ==================== 生命周期 ====================

// 计算画布尺寸（响应式）
const updateCanvasSize = () => {
  if (videoContainer.value) {
    const container = videoContainer.value
    const width = container.offsetWidth
    const height = container.offsetHeight
    
    // 保持 16:9 比例
    const targetWidth = width
    const targetHeight = Math.round(width * 9 / 16)
    
    canvasWidth.value = targetWidth
    canvasHeight.value = targetHeight
    videoContainerHeight.value = targetHeight
  }
}

onMounted(() => {
  fetchStatus()
  setInterval(fetchStatus, 2000)

  // 初始化画布尺寸
  updateCanvasSize()

  // 监听窗口大小变化
  window.addEventListener('resize', updateCanvasSize)

  // 加载本地持久化设置
  loadLocalSettings()

  // 加载空间计量配置
  loadSpatialConfig()
})

onUnmounted(() => {
  disconnectWebSocket()
  window.removeEventListener('resize', updateCanvasSize)
})
</script>

<style scoped>
/* ==================== 全局样式 ==================== */
.app-container {
  min-height: 100vh;
  background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
  color: #ffffff;
}

.header {
  padding: 20px;
  background: rgba(26, 26, 46, 0.9);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid #333355;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 15px;
}

@media (max-width: 768px) {
  .header {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .nav-tabs {
    width: 100%;
    overflow-x: auto;
  }
}

.header-left h1 {
  margin: 0 0 10px 0;
  font-size: 24px;
  background: linear-gradient(90deg, #00ff00, #0088ff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.status-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 14px;
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #ff4444;
}

.status-dot.online {
  background: #00ff00;
  box-shadow: 0 0 10px #00ff00;
}

.calibrated {
  color: #00ff00;
  margin-left: 10px;
}

.nav-tabs {
  display: flex;
  gap: 10px;
}

.tab {
  padding: 10px 20px;
  background: rgba(74, 74, 106, 0.5);
  color: #ffffff;
  border: 1px solid transparent;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.3s;
}

.tab:hover {
  background: rgba(74, 74, 106, 0.8);
}

.tab.active {
  background: linear-gradient(90deg, #4a4a6a, #5a5a7a);
  border-color: #00ff00;
}

.main-content {
  padding: 20px;
}

/* ==================== 监控页面 ==================== */
.monitor-page {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
}

.video-section {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
}

.video-container {
  position: relative;
  background: #000;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 20px rgba(0, 255, 0, 0.1);
  width: 100%;
  aspect-ratio: 16 / 9;
}

.video-container canvas {
  width: 100%;
  height: 100%;
  display: block;
  object-fit: contain;
}

.video-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.5);
}

.no-signal {
  text-align: center;
  color: #666;
}

.no-signal-icon {
  font-size: 64px;
  margin-bottom: 20px;
}

.no-signal-hint {
  font-size: 14px;
  color: #888;
  margin-top: 10px;
}

.controls {
  display: flex;
  gap: 10px;
  margin-top: 15px;
  flex-wrap: wrap;
}

@media (max-width: 600px) {
  .controls {
    flex-direction: column;
  }
  
  .controls .btn {
    width: 100%;
  }
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.3s;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-primary {
  background: linear-gradient(90deg, #4a4a6a, #5a5a7a);
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: linear-gradient(90deg, #5a5a7a, #6a6a8a);
}

.btn-danger {
  background: linear-gradient(90deg, #dc3545, #c82333);
  color: white;
}

.btn-success {
  background: linear-gradient(90deg, #28a745, #218838);
  color: white;
}

.btn-warning {
  background: linear-gradient(90deg, #ffc107, #e0a800);
  color: #000;
}

.btn-secondary {
  background: linear-gradient(90deg, #6c757d, #5a6268);
  color: white;
}

.data-section {
  width: 350px;
  display: flex;
  flex-direction: column;
  gap: 15px;
  overflow-y: auto;
  max-height: 720px;
}

@media (max-width: 1200px) {
  .data-section {
    width: 100%;
    flex-direction: row;
    flex-wrap: wrap;
    max-height: none;
  }
  
  .data-section .panel {
    flex: 1;
    min-width: 280px;
  }
}

.panel {
  background: rgba(26, 26, 46, 0.8);
  border-radius: 12px;
  padding: 20px;
  border: 1px solid #333355;
}

.panel h2 {
  margin: 0 0 15px 0;
  font-size: 18px;
  color: #00ff00;
}

.metrics {
  display: flex;
  gap: 15px;
}

.metric {
  flex: 1;
  text-align: center;
  background: rgba(0, 0, 0, 0.2);
  padding: 15px;
  border-radius: 8px;
}

.metric .label {
  display: block;
  color: #888;
  font-size: 12px;
  margin-bottom: 8px;
}

.metric .value {
  display: block;
  font-size: 28px;
  font-weight: bold;
  color: #00ff00;
}

.person-data, .hand-data {
  background: rgba(0, 0, 0, 0.2);
  padding: 12px;
  border-radius: 8px;
  margin-bottom: 10px;
}

.person-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 10px;
}

.person-id {
  font-weight: bold;
  color: #00ff00;
}

.confidence {
  font-size: 12px;
  color: #888;
}

.data-row {
  display: flex;
  justify-content: space-between;
  padding: 6px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  font-size: 14px;
}

.data-row:last-child {
  border-bottom: none;
}

.highlight {
  color: #00ff00;
  font-weight: bold;
}

.method-tag {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-family: monospace;
}

.performance-metrics {
  margin-top: 10px;
  padding: 8px;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 6px;
}

.perf-row {
  display: flex;
  gap: 15px;
  font-size: 12px;
  color: #aaa;
}

.perf-row span {
  font-family: monospace;
}

.topview-canvas {
  width: 100%;
  height: auto;
  border-radius: 8px;
  background: #1a1a2e;
}

.topview-legend {
  display: flex;
  gap: 20px;
  margin-top: 10px;
  justify-content: center;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
}

.legend-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.person-dot { background: #00ff00; }
.hand-dot { background: #0088ff; }
.camera-dot { background: #ffffff; }

/* ==================== 错误提示 ==================== */
.error-banner {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  background: linear-gradient(90deg, #ff4444, #ff6666);
  color: white;
  padding: 12px 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  z-index: 1000;
  cursor: pointer;
  animation: slideDown 0.3s ease;
}

@keyframes slideDown {
  from { transform: translateY(-100%); }
  to { transform: translateY(0); }
}

.error-icon {
  font-size: 18px;
}

.error-close {
  font-size: 20px;
  margin-left: 10px;
  opacity: 0.8;
}

/* ==================== 标定页面 ==================== */
.calibration-page {
  max-width: 1400px;
  margin: 0 auto;
}

.calibration-container {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

@media (max-width: 1024px) {
  .calibration-container {
    grid-template-columns: 1fr;
  }
  
  .calibration-right {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
}

.calibration-hint {
  color: #888;
  font-size: 14px;
  line-height: 1.6;
  margin-bottom: 20px;
}

.calibration-steps {
  background: rgba(0, 0, 0, 0.2);
  padding: 15px;
  border-radius: 8px;
  margin-bottom: 20px;
}

.calibration-steps h3 {
  margin: 0 0 10px 0;
  color: #00ff00;
  font-size: 16px;
}

.calibration-steps ol {
  margin: 0;
  padding-left: 20px;
  color: #888;
}

.calibration-actions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.capture-status {
  margin-top: 20px;
  text-align: center;
}

.progress-bar {
  height: 8px;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 10px;
}

.progress {
  height: 100%;
  background: linear-gradient(90deg, #00ff00, #0088ff);
  transition: width 0.3s;
}

.result-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 15px;
  margin-bottom: 20px;
}

@media (max-width: 600px) {
  .result-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

.result-item {
  background: rgba(0, 0, 0, 0.2);
  padding: 12px;
  border-radius: 8px;
  text-align: center;
}

.result-label {
  display: block;
  color: #888;
  font-size: 12px;
  margin-bottom: 5px;
}

.result-value {
  display: block;
  color: #00ff00;
  font-weight: bold;
  font-size: 18px;
}

.distortion-coeffs {
  background: rgba(0, 0, 0, 0.2);
  padding: 12px;
  border-radius: 8px;
}

.distortion-coeffs h4 {
  margin: 0 0 10px 0;
  color: #888;
  font-size: 14px;
}

.distortion-coeffs code {
  color: #00ff00;
  font-family: 'Courier New', monospace;
}

.calibration-preview {
  width: 100%;
  height: auto;
  border-radius: 8px;
  background: #000;
  margin-bottom: 15px;
}

.calibration-controls {
  text-align: center;
}

.form-group {
  margin-bottom: 15px;
}

.form-group label {
  display: block;
  color: #888;
  font-size: 14px;
  margin-bottom: 5px;
}

.form-input {
  width: 100%;
  padding: 10px;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid #333355;
  border-radius: 6px;
  color: #ffffff;
  font-size: 14px;
}

.form-input:focus {
  outline: none;
  border-color: #00ff00;
}

.form-range {
  width: 200px;
  vertical-align: middle;
}

.inline-inputs {
  display: flex;
  gap: 10px;
}

.inline-inputs .form-input {
  flex: 1;
}

/* 开关样式 */
.switch {
  position: relative;
  display: inline-block;
  width: 50px;
  height: 24px;
  vertical-align: middle;
}

.switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.slider {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: #333355;
  transition: 0.4s;
  border-radius: 24px;
}

.slider:before {
  position: absolute;
  content: "";
  height: 18px;
  width: 18px;
  left: 3px;
  bottom: 3px;
  background-color: white;
  transition: 0.4s;
  border-radius: 50%;
}

input:checked + .slider {
  background-color: #00ff00;
}

input:checked + .slider:before {
  transform: translateX(26px);
}

/* ==================== 设置页面 ==================== */
.settings-page {
  max-width: 1200px;
  margin: 0 auto;
}

.settings-container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
}

.config-hint {
  color: #888;
  font-size: 12px;
  margin-bottom: 15px;
  padding: 8px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 4px;
}

/* 运动状态徽章 */
.motion-badge {
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: bold;
  margin-left: 8px;
}

.motion-badge.stationary {
  background: rgba(100, 100, 100, 0.3);
  color: #aaa;
}

.motion-badge.walking {
  background: rgba(0, 200, 100, 0.3);
  color: #00c864;
}

.motion-badge.running {
  background: rgba(255, 100, 0, 0.3);
  color: #ff6400;
}

.motion-badge.unknown {
  background: rgba(100, 100, 100, 0.2);
  color: #888;
}

/* 身体部位徽章 */
.body-part-badge {
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: bold;
  margin-left: 8px;
  background: rgba(100, 150, 255, 0.3);
  color: #6496ff;
  cursor: help;
}

/* 置信度小标签 */
.confidence-mini {
  font-size: 10px;
  color: #888;
  margin-left: 4px;
}

/* 估计方法标签 */
.method-tag {
  font-size: 10px;
  padding: 2px 6px;
  background: rgba(0, 136, 255, 0.2);
  color: #0088ff;
  border-radius: 4px;
  font-family: monospace;
}

/* 数据行增强 */
.data-row {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.data-row:last-child {
  border-bottom: none;
}

.data-row span:first-child {
  min-width: 80px;
  color: #888;
}
</style>

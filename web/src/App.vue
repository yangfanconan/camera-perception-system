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
      <div class="header-right">
        <button class="sound-toggle" @click="toggleSound" :title="soundEnabled ? '关闭报警声音' : '开启报警声音'">
          {{ soundEnabled ? '🔔' : '🔕' }}
        </button>
        <button class="theme-toggle" @click="toggleTheme" :title="isDarkTheme ? '切换到浅色模式' : '切换到深色模式'">
          {{ isDarkTheme ? '☀️' : '🌙' }}
        </button>
      </div>
      <nav class="nav-tabs">
        <button :class="['tab', { active: currentTab === 'monitor' }]" @click="currentTab = 'monitor'">
          📺 实时监控
        </button>
        <button :class="['tab', { active: currentTab === 'dashboard' }]" @click="currentTab = 'dashboard'; fetchDashboardData()">
          📊 数据仪表盘
        </button>
        <button :class="['tab', { active: currentTab === 'alerts' }]" @click="currentTab = 'alerts'; fetchAlerts()">
          🚨 报警管理
          <span v-if="unreadAlerts > 0" class="badge">{{ unreadAlerts }}</span>
        </button>
        <button :class="['tab', { active: currentTab === 'cameras' }]" @click="currentTab = 'cameras'; fetchCameras()">
          📹 多摄像头
        </button>
        <button :class="['tab', { active: currentTab === 'recording' }]" @click="currentTab = 'recording'">
          🎬 录制回放
        </button>
        <button :class="['tab', { active: currentTab === 'analytics' }]" @click="currentTab = 'analytics'; fetchAnalytics()">
          📈 数据分析
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
            <canvas ref="videoCanvas" :width="canvasWidth" :height="canvasHeight"
              @mousedown="onCanvasMouseDown"
              @mousemove="onCanvasMouseMove"
              @mouseup="onCanvasMouseUp"
              :style="{ cursor: isCalibrating ? 'crosshair' : 'default' }"
            ></canvas>
            
            <!-- 框选矩形显示 -->
            <div v-if="selectionRect.show" class="selection-rect"
              :style="{
                left: selectionRect.x + 'px',
                top: selectionRect.y + 'px',
                width: selectionRect.width + 'px',
                height: selectionRect.height + 'px'
              }">
            </div>
            
            <!-- 标定点标记 -->
            <div v-for="(point, idx) in calibrationMarkers" :key="idx" class="calibration-marker"
              :style="{ left: point.x + 'px', top: point.y + 'px', width: point.w + 'px', height: point.h + 'px' }">
              <span class="marker-label">{{ point.distance }}m</span>
            </div>
            
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
            <button 
              :class="['btn', isCalibrating ? 'btn-warning' : 'btn-success']" 
              @click="toggleCalibrationMode"
              :disabled="!systemStatus.camera_opened"
            >
              {{ isCalibrating ? '✕ 取消标定' : '📐 深度标定' }}
            </button>
            <button class="btn btn-warning" @click="exportData" :disabled="detectionData.persons.length === 0">
              📊 导出数据
            </button>
            <button 
              :class="['btn', showDepthHeatmap ? 'btn-info' : 'btn-secondary']" 
              @click="toggleDepthHeatmap" 
              :disabled="!systemStatus.camera_opened"
            >
              🌡️ {{ showDepthHeatmap ? '关闭热力图' : '深度热力图' }}
            </button>
          </div>
          
          <!-- 深度热力图显示 -->
          <div v-if="showDepthHeatmap && depthHeatmap" class="heatmap-container">
            <img :src="'data:image/jpeg;base64,' + depthHeatmap" alt="Depth Heatmap" class="heatmap-image" />
            <div class="heatmap-legend">
              <span class="legend-item"><span class="legend-color near"></span> 近（红）</span>
              <span class="legend-item"><span class="legend-color mid"></span> 中</span>
              <span class="legend-item"><span class="legend-color far"></span> 远（蓝）</span>
            </div>
            <div class="heatmap-info">
              <div class="depth-stat">
                <span class="depth-label">📍 最近距离</span>
                <span class="depth-value near">{{ depthInfo.nearest?.toFixed(2) }} m</span>
              </div>
              <div class="depth-stat">
                <span class="depth-label">📍 最远距离</span>
                <span class="depth-value far">{{ depthInfo.farthest?.toFixed(2) }} m</span>
              </div>
              <div class="depth-stat">
                <span class="depth-label">📊 平均值</span>
                <span class="depth-value avg">{{ depthInfo.mean?.toFixed(2) }}</span>
              </div>
            </div>
            
            <!-- 深度标定状态 -->
            <div class="depth-calibration">
              <div class="calibration-header">
                <h4>📐 深度标定</h4>
                <span :class="['calibration-status', depthCalibration.calibrated ? 'calibrated' : '']">
                  {{ depthCalibration.calibrated ? '✓ 已标定' : '未标定' }}
                </span>
              </div>
              
              <p class="calibration-hint">点击"深度标定"按钮，然后在画面中框选区域，输入真实距离进行标定。</p>

              <div v-if="depthCalibration.points?.length > 0" class="calibration-points">
                <div class="points-header">校准点 ({{ depthCalibration.points.length }}/2+):</div>
                <div v-for="(point, idx) in depthCalibration.points" :key="idx" class="point-item">
                  <span>相对: {{ point[0]?.toFixed(2) }}</span>
                  <span>→</span>
                  <span>真实: {{ point[1]?.toFixed(2) }}m</span>
                </div>
              </div>

              <div v-if="depthCalibration.calibrated" class="calibration-result">
                <span>缩放因子: {{ depthCalibration.scale_factor?.toFixed(4) }}</span>
                <span>偏移: {{ depthCalibration.offset?.toFixed(4) }}</span>
                <button class="btn btn-sm btn-secondary" @click="clearDepthCalibration">清除标定</button>
              </div>
            </div>
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
          
          <!-- 3D 姿态可视化 -->
          <div class="panel" v-if="detectionData.persons.length > 0">
            <Pose3DViewer 
              :keypoints="detectionData.persons[0]?.keypoints || {}"
              :depth="detectionData.persons[0]?.distance || 2.0"
            />
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

      <!-- 数据仪表盘页面 -->
      <div v-if="currentTab === 'dashboard'" class="dashboard-page">
        <div class="dashboard-grid">
          <!-- 系统状态卡片 -->
          <div class="dashboard-card">
            <h3>🖥️ 系统状态</h3>
            <div class="status-grid">
              <div class="status-item">
                <span class="status-label">摄像头</span>
                <span :class="['status-value', systemStatus.camera_opened ? 'online' : 'offline']">
                  {{ systemStatus.camera_opened ? '已连接' : '未连接' }}
                </span>
              </div>
              <div class="status-item">
                <span class="status-label">帧率</span>
                <span class="status-value">{{ systemStatus.fps }} FPS</span>
              </div>
              <div class="status-item">
                <span class="status-label">标定状态</span>
                <span :class="['status-value', systemStatus.calibrated ? 'calibrated' : '']">
                  {{ systemStatus.calibrated ? '已标定' : '未标定' }}
                </span>
              </div>
              <div class="status-item">
                <span class="status-label">检测人数</span>
                <span class="status-value">{{ dashboardData.persons_detected }}</span>
              </div>
            </div>
          </div>

          <!-- 性能监控卡片 -->
          <div class="dashboard-card">
            <h3>⚡ 性能监控</h3>
            <div class="performance-bars">
              <div class="perf-item">
                <span class="perf-label">CPU</span>
                <div class="perf-bar">
                  <div class="perf-fill" :style="{ width: performanceData.cpu_percent + '%' }"
                       :class="{ warning: performanceData.cpu_percent > 70, danger: performanceData.cpu_percent > 90 }"></div>
                </div>
                <span class="perf-value">{{ performanceData.cpu_percent?.toFixed(1) }}%</span>
              </div>
              <div class="perf-item">
                <span class="perf-label">内存</span>
                <div class="perf-bar">
                  <div class="perf-fill" :style="{ width: performanceData.memory_percent + '%' }"
                       :class="{ warning: performanceData.memory_percent > 70, danger: performanceData.memory_percent > 90 }"></div>
                </div>
                <span class="perf-value">{{ performanceData.memory_percent?.toFixed(1) }}%</span>
              </div>
              <div class="perf-item">
                <span class="perf-label">检测耗时</span>
                <div class="perf-bar">
                  <div class="perf-fill" :style="{ width: Math.min(performanceData.detection_time / 2, 100) + '%' }"></div>
                </div>
                <span class="perf-value">{{ performanceData.detection_time?.toFixed(1) }}ms</span>
              </div>
            </div>
          </div>

          <!-- 检测统计卡片 -->
          <div class="dashboard-card">
            <h3>📊 检测统计</h3>
            <div class="stats-grid">
              <div class="stat-item">
                <span class="stat-value">{{ dashboardData.total_persons || 0 }}</span>
                <span class="stat-label">累计人数</span>
              </div>
              <div class="stat-item">
                <span class="stat-value">{{ dashboardData.total_hands || 0 }}</span>
                <span class="stat-label">累计手数</span>
              </div>
              <div class="stat-item">
                <span class="stat-value">{{ dashboardData.total_alerts || 0 }}</span>
                <span class="stat-label">报警次数</span>
              </div>
              <div class="stat-item">
                <span class="stat-value">{{ dashboardData.total_falls || 0 }}</span>
                <span class="stat-label">跌倒事件</span>
              </div>
            </div>
          </div>

          <!-- 跟踪状态卡片 -->
          <div class="dashboard-card">
            <h3>👥 跟踪状态</h3>
            <div class="tracks-list" v-if="tracks.length > 0">
              <div v-for="track in tracks" :key="track.track_id" class="track-item">
                <span class="track-id">#{{ track.track_id }}</span>
                <span class="track-conf">置信度: {{ (track.confidence * 100).toFixed(0) }}%</span>
                <span class="track-age">存在: {{ track.duration?.toFixed(1) }}s</span>
              </div>
            </div>
            <div v-else class="no-data">暂无跟踪数据</div>
          </div>

          <!-- 手势识别卡片 -->
          <div class="dashboard-card">
            <h3>👋 手势识别</h3>
            <div class="gestures-list" v-if="gestures.length > 0">
              <div v-for="gesture in gestures.slice(-5)" :key="gesture.timestamp" class="gesture-item">
                <span class="gesture-icon">{{ getGestureIcon(gesture.gesture) }}</span>
                <span class="gesture-name">{{ getGestureName(gesture.gesture) }}</span>
                <span class="gesture-conf">{{ (gesture.confidence * 100).toFixed(0) }}%</span>
              </div>
            </div>
            <div v-else class="no-data">暂无手势数据</div>
          </div>

          <!-- 动作识别卡片 -->
          <div class="dashboard-card">
            <h3>🏃 动作识别</h3>
            <div class="actions-list" v-if="actions.length > 0">
              <div v-for="action in actions.slice(-5)" :key="action.timestamp" class="action-item">
                <span class="action-icon">{{ getActionIcon(action.action) }}</span>
                <span class="action-name">{{ getActionName(action.action) }}</span>
                <span class="action-duration">{{ action.duration?.toFixed(1) }}s</span>
              </div>
            </div>
            <div v-else class="no-data">暂无动作数据</div>
          </div>
        </div>
      </div>

      <!-- 报警管理页面 -->
      <div v-if="currentTab === 'alerts'" class="alerts-page">
        <div class="alerts-header">
          <h2>🚨 报警管理</h2>
          <div class="alerts-actions">
            <button class="btn btn-secondary" @click="fetchAlerts">🔄 刷新</button>
            <button class="btn btn-warning" @click="acknowledgeAllAlerts">✓ 全部确认</button>
          </div>
        </div>

        <!-- 报警统计 -->
        <div class="alerts-stats">
          <div class="alert-stat critical">
            <span class="stat-num">{{ alertStats.critical || 0 }}</span>
            <span class="stat-text">严重</span>
          </div>
          <div class="alert-stat high">
            <span class="stat-num">{{ alertStats.high || 0 }}</span>
            <span class="stat-text">高</span>
          </div>
          <div class="alert-stat medium">
            <span class="stat-num">{{ alertStats.medium || 0 }}</span>
            <span class="stat-text">中</span>
          </div>
          <div class="alert-stat low">
            <span class="stat-num">{{ alertStats.low || 0 }}</span>
            <span class="stat-text">低</span>
          </div>
        </div>

        <!-- 报警列表 -->
        <div class="alerts-list">
          <div v-if="alerts.length === 0" class="no-alerts">
            <span class="no-alerts-icon">✅</span>
            <span>暂无报警事件</span>
          </div>
          <div v-for="alert in alerts" :key="alert.alert_id" 
               :class="['alert-item', alert.severity, { acknowledged: alert.acknowledged }]">
            <div class="alert-icon">{{ getAlertIcon(alert.alert_type) }}</div>
            <div class="alert-content">
              <div class="alert-title">{{ alert.message }}</div>
              <div class="alert-meta">
                <span class="alert-type">{{ alert.alert_type }}</span>
                <span class="alert-time">{{ formatTime(alert.timestamp) }}</span>
              </div>
            </div>
            <div class="alert-actions">
              <button v-if="!alert.acknowledged" class="btn-small" @click="acknowledgeAlert(alert.alert_id)">
                确认
              </button>
              <span v-else class="acknowledged-badge">已确认</span>
            </div>
          </div>
        </div>

        <!-- 报警区域管理 -->
        <div class="zones-section">
          <h3>📍 报警区域</h3>
          <div class="zones-list">
            <div v-for="zone in zones" :key="zone.id" class="zone-item">
              <span class="zone-name">{{ zone.name }}</span>
              <span :class="['zone-type', zone.type]">{{ zone.type }}</span>
              <span :class="['zone-status', zone.enabled ? 'enabled' : 'disabled']">
                {{ zone.enabled ? '启用' : '禁用' }}
              </span>
              <button class="btn-small" @click="toggleZone(zone.id)">
                {{ zone.enabled ? '禁用' : '启用' }}
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- 多摄像头页面 -->
      <div v-if="currentTab === 'cameras'" class="cameras-page">
        <div class="cameras-header">
          <h2>📹 多摄像头管理</h2>
          <button class="btn btn-primary" @click="addCameraDialog = true">+ 添加摄像头</button>
        </div>
        
        <div class="cameras-grid">
          <div v-for="camera in cameras" :key="camera.camera_id" 
               :class="['camera-card', { active: camera.camera_id === activeCameraId }]">
            <div class="camera-preview">
              <div class="preview-placeholder">
                <span class="camera-icon">📷</span>
              </div>
            </div>
            <div class="camera-info">
              <h4>{{ camera.name }}</h4>
              <div class="camera-status">
                <span :class="['status-dot', camera.is_opened ? 'online' : 'offline']"></span>
                <span>{{ camera.is_opened ? '已连接' : '未连接' }}</span>
              </div>
              <div class="camera-meta">
                <span>FPS: {{ camera.fps }}</span>
                <span>帧数: {{ camera.frame_count }}</span>
              </div>
            </div>
            <div class="camera-actions">
              <button class="btn-small" @click="activateCamera(camera.camera_id)">激活</button>
              <button class="btn-small" @click="openCamera(camera.camera_id)">打开</button>
              <button class="btn-small danger" @click="closeCamera(camera.camera_id)">关闭</button>
            </div>
          </div>
        </div>
      </div>

      <!-- 录制回放页面 -->
      <div v-if="currentTab === 'recording'" class="recording-page">
        <div class="recording-header">
          <h2>🎬 视频录制与回放</h2>
        </div>
        
        <div class="recording-controls">
          <button v-if="!isRecording" class="btn btn-primary" @click="startRecording">🔴 开始录制</button>
          <button v-else class="btn btn-danger" @click="stopRecording">⏹ 停止录制</button>
          <span v-if="isRecording" class="recording-indicator">
            <span class="rec-dot"></span> 录制中...
          </span>
        </div>
        
        <div class="recordings-list">
          <h3>录制列表</h3>
          <div v-if="recordings.length === 0" class="no-data">暂无录制记录</div>
          <div v-for="rec in recordings" :key="rec.session_id" class="recording-item">
            <div class="rec-info">
              <span class="rec-name">{{ rec.file_path?.split('/').pop() }}</span>
              <span class="rec-duration">{{ rec.duration?.toFixed(1) }}s</span>
              <span class="rec-frames">{{ rec.frame_count }} 帧</span>
            </div>
            <div class="rec-actions">
              <button class="btn-small" @click="playRecording(rec)">▶ 播放</button>
              <button class="btn-small" @click="downloadRecording(rec)">⬇ 下载</button>
            </div>
          </div>
        </div>
      </div>

      <!-- 数据分析页面 -->
      <div v-if="currentTab === 'analytics'" class="analytics-page">
        <div class="analytics-header">
          <h2>📈 数据分析与报告</h2>
          <button class="btn btn-primary" @click="generateReport">生成报告</button>
        </div>
        
        <div class="analytics-grid">
          <!-- 统计卡片 -->
          <div class="analytics-card">
            <h3>📊 关键指标</h3>
            <div class="metrics-list">
              <div v-for="(stat, name) in analyticsStats" :key="name" class="metric-row">
                <span class="metric-name">{{ name }}</span>
                <span class="metric-value">{{ stat.value?.toFixed(2) }}</span>
                <span class="metric-unit">{{ stat.unit }}</span>
              </div>
            </div>
          </div>
          
          <!-- 趋势图 -->
          <div class="analytics-card wide">
            <h3>📉 趋势分析</h3>
            <div class="trend-selector">
              <select v-model="selectedMetric" @change="fetchTrend(selectedMetric)">
                <option value="person_count">人数</option>
                <option value="fps">帧率</option>
                <option value="detection_time">检测耗时</option>
              </select>
            </div>
            <div class="trend-chart" v-if="trendData">
              <div class="trend-info">
                <span>趋势: <strong :class="trendData.trend">{{ trendData.trend }}</strong></span>
                <span>R²: {{ trendData.r_squared?.toFixed(3) }}</span>
              </div>
            </div>
          </div>
          
          <!-- 异常检测 -->
          <div class="analytics-card">
            <h3>⚠️ 异常检测</h3>
            <div class="anomalies-list">
              <div v-for="anomaly in anomalies" :key="anomaly.timestamp" class="anomaly-item">
                <span class="anomaly-time">{{ formatTime(anomaly.timestamp) }}</span>
                <span class="anomaly-value">{{ anomaly.value?.toFixed(2) }}</span>
              </div>
              <div v-if="anomalies.length === 0" class="no-data">暂无异常</div>
            </div>
          </div>
          
          <!-- 报告预览 -->
          <div class="analytics-card" v-if="latestReport">
            <h3>📄 最新报告</h3>
            <div class="report-preview">
              <p><strong>{{ latestReport.title }}</strong></p>
              <p class="report-summary">{{ latestReport.summary }}</p>
              <div class="report-meta">
                <span>生成时间: {{ formatTime(latestReport.generated_at) }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, onUnmounted, computed } from 'vue'
import axios from 'axios'
import Pose3DViewer from './components/Pose3DViewer.vue'

// ==================== 状态定义 ====================

// 当前页面
const currentTab = ref('monitor')

// 主题
const isDarkTheme = ref(true)

// 切换主题
const toggleTheme = () => {
  isDarkTheme.value = !isDarkTheme.value
  document.documentElement.setAttribute('data-theme', isDarkTheme.value ? 'dark' : 'light')
  localStorage.setItem('theme', isDarkTheme.value ? 'dark' : 'light')
}

// 初始化主题
const initTheme = () => {
  const savedTheme = localStorage.getItem('theme')
  if (savedTheme) {
    isDarkTheme.value = savedTheme === 'dark'
  }
  document.documentElement.setAttribute('data-theme', isDarkTheme.value ? 'dark' : 'light')
}

// 报警声音
const soundEnabled = ref(true)
const lastAlertCount = ref(0)

// 播放报警声音
const playAlertSound = () => {
  if (!soundEnabled.value) return
  
  // 使用 Web Audio API 生成报警声音
  const audioContext = new (window.AudioContext || window.webkitAudioContext)()
  const oscillator = audioContext.createOscillator()
  const gainNode = audioContext.createGain()
  
  oscillator.connect(gainNode)
  gainNode.connect(audioContext.destination)
  
  oscillator.frequency.value = 800
  oscillator.type = 'sine'
  gainNode.gain.value = 0.3
  
  oscillator.start()
  
  // 闪烁效果
  setTimeout(() => { oscillator.frequency.value = 600 }, 100)
  setTimeout(() => { oscillator.frequency.value = 800 }, 200)
  setTimeout(() => { oscillator.frequency.value = 600 }, 300)
  setTimeout(() => { 
    oscillator.stop()
    audioContext.close()
  }, 400)
}

// 切换声音
const toggleSound = () => {
  soundEnabled.value = !soundEnabled.value
  localStorage.setItem('soundEnabled', soundEnabled.value)
}

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

// 深度热力图
const showDepthHeatmap = ref(false)
const depthHeatmap = ref(null)
const depthInfo = reactive({
  nearest: 0,
  farthest: 0,
  mean: 0
})
let heatmapInterval = null

// 深度标定
const depthCalibration = reactive({
  calibrated: false,
  points: [],
  scale_factor: 1.0,
  offset: 0.0
})
const calibrationRealDistance = ref(null)

// 框选标定
const isCalibrating = ref(false)
const selectionRect = reactive({
  show: false,
  x: 0,
  y: 0,
  width: 0,
  height: 0,
  startX: 0,
  startY: 0
})
const calibrationMarkers = ref([])  // 标定点标记

// 切换标定模式
const toggleCalibrationMode = () => {
  isCalibrating.value = !isCalibrating.value
  if (!isCalibrating.value) {
    selectionRect.show = false
  }
}

// 鼠标按下
const onCanvasMouseDown = (e) => {
  if (!isCalibrating.value) return
  
  const rect = e.target.getBoundingClientRect()
  selectionRect.startX = e.clientX - rect.left
  selectionRect.startY = e.clientY - rect.top
  selectionRect.x = selectionRect.startX
  selectionRect.y = selectionRect.startY
  selectionRect.width = 0
  selectionRect.height = 0
  selectionRect.show = true
}

// 鼠标移动
const onCanvasMouseMove = (e) => {
  if (!isCalibrating.value || !selectionRect.show) return
  
  const rect = e.target.getBoundingClientRect()
  const currentX = e.clientX - rect.left
  const currentY = e.clientY - rect.top
  
  selectionRect.width = Math.abs(currentX - selectionRect.startX)
  selectionRect.height = Math.abs(currentY - selectionRect.startY)
  selectionRect.x = Math.min(currentX, selectionRect.startX)
  selectionRect.y = Math.min(currentY, selectionRect.startY)
}

// 鼠标释放
const onCanvasMouseUp = async (e) => {
  if (!isCalibrating.value || !selectionRect.show) return
  
  // 如果框选区域太小，忽略
  if (selectionRect.width < 20 || selectionRect.height < 20) {
    selectionRect.show = false
    return
  }
  
  // 弹出输入框让用户输入真实距离
  const distance = prompt('请输入该区域距离摄像头的真实距离（米）：')
  if (distance === null) {
    selectionRect.show = false
    return
  }
  
  const realDistance = parseFloat(distance)
  if (isNaN(realDistance) || realDistance <= 0) {
    alert('请输入有效的距离值')
    selectionRect.show = false
    return
  }
  
  try {
    // 发送框选区域到后端获取平均深度
    const res = await axios.post(`${API_BASE}/api/depth/calibrate_region`, {
      x: Math.round(selectionRect.x),
      y: Math.round(selectionRect.y),
      width: Math.round(selectionRect.width),
      height: Math.round(selectionRect.height),
      real_distance: realDistance
    })
    
    if (res.data.status === 'success') {
      Object.assign(depthCalibration, res.data.calibration)
      
      // 添加标记
      calibrationMarkers.value.push({
        x: selectionRect.x,
        y: selectionRect.y,
        w: selectionRect.width,
        h: selectionRect.height,
        distance: realDistance
      })
    }
  } catch (error) {
    console.error('标定失败:', error)
    alert('标定失败，请重试')
  }
  
  selectionRect.show = false
}

// 清除深度标定
const clearDepthCalibration = async () => {
  try {
    await axios.delete(`${API_BASE}/api/depth/calibration`)
    depthCalibration.calibrated = false
    depthCalibration.points = []
    depthCalibration.scale_factor = 1.0
    depthCalibration.offset = 0.0
    calibrationMarkers.value = []
  } catch (error) {
    console.error('清除标定失败:', error)
  }
}

// 获取深度标定状态
const fetchDepthCalibration = async () => {
  try {
    const res = await axios.get(`${API_BASE}/api/depth/calibration`)
    if (res.data.status === 'success') {
      Object.assign(depthCalibration, res.data.calibration)
    }
  } catch (error) {
    console.error('获取标定状态失败:', error)
  }
}

// ==================== 仪表盘数据 ====================
const dashboardData = reactive({
  persons_detected: 0,
  hands_detected: 0,
  total_persons: 0,
  total_hands: 0,
  total_alerts: 0,
  total_falls: 0
})

const performanceData = reactive({
  cpu_percent: 0,
  memory_percent: 0,
  fps: 0,
  detection_time: 0
})

const tracks = ref([])
const gestures = ref([])
const actions = ref([])

// ==================== 报警数据 ====================
const alerts = ref([])
const zones = ref([])
const alertStats = reactive({
  critical: 0,
  high: 0,
  medium: 0,
  low: 0
})
const unreadAlerts = computed(() => alerts.value.filter(a => !a.acknowledged).length)

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

// 深度热力图切换
const toggleDepthHeatmap = async () => {
  showDepthHeatmap.value = !showDepthHeatmap.value

  if (showDepthHeatmap.value) {
    // 开始获取热力图
    fetchDepthHeatmap()
    fetchDepthCalibration()  // 获取标定状态
    heatmapInterval = setInterval(fetchDepthHeatmap, 500)  // 每 500ms 更新一次
  } else {
    // 停止获取热力图
    if (heatmapInterval) {
      clearInterval(heatmapInterval)
      heatmapInterval = null
    }
    depthHeatmap.value = null
  }
}

const fetchDepthHeatmap = async () => {
  try {
    const response = await axios.get(`${API_BASE}/api/depth/heatmap`)
    if (response.data.status === 'success') {
      depthHeatmap.value = response.data.heatmap
      depthInfo.nearest = response.data.nearest_distance
      depthInfo.farthest = response.data.farthest_distance
      depthInfo.mean = response.data.depth_mean
    }
  } catch (error) {
    console.error('获取深度热力图失败:', error)
  }
}

// ==================== 仪表盘数据获取 ====================
const fetchDashboardData = async () => {
  try {
    // 获取性能数据
    const perfRes = await axios.get(`${API_BASE}/api/performance`)
    if (perfRes.data.status === 'success') {
      Object.assign(performanceData, perfRes.data.performance)
    }
    
    // 获取跟踪数据
    const tracksRes = await axios.get(`${API_BASE}/api/tracks`)
    if (tracksRes.data.status === 'success') {
      tracks.value = tracksRes.data.tracks
    }
    
    // 获取手势数据
    const gesturesRes = await axios.get(`${API_BASE}/api/gestures`)
    if (gesturesRes.data.status === 'success') {
      gestures.value = gesturesRes.data.gestures
    }
    
    // 获取记录统计
    const statsRes = await axios.get(`${API_BASE}/api/records/stats`)
    if (statsRes.data.status === 'success') {
      Object.assign(dashboardData, statsRes.data.stats)
    }
  } catch (error) {
    console.error('获取仪表盘数据失败:', error)
  }
}

// ==================== 报警管理 ====================
const fetchAlerts = async () => {
  try {
    const res = await axios.get(`${API_BASE}/api/alerts`)
    if (res.data.status === 'success') {
      const newAlerts = res.data.alerts
      
      // 检测新报警（数量增加）
      if (newAlerts.length > lastAlertCount.value && lastAlertCount.value > 0) {
        playAlertSound()
      }
      lastAlertCount.value = newAlerts.length
      
      alerts.value = newAlerts

      // 计算统计
      alertStats.critical = alerts.value.filter(a => a.severity === 'critical').length
      alertStats.high = alerts.value.filter(a => a.severity === 'high').length
      alertStats.medium = alerts.value.filter(a => a.severity === 'medium').length
      alertStats.low = alerts.value.filter(a => a.severity === 'low').length
    }

    // 获取报警区域
    const zonesRes = await axios.get(`${API_BASE}/api/zones`)
    if (zonesRes.data.status === 'success') {
      zones.value = zonesRes.data.zones
    }
  } catch (error) {
    console.error('获取报警数据失败:', error)
  }
}

const acknowledgeAlert = async (alertId) => {
  try {
    await axios.post(`${API_BASE}/api/alerts/${alertId}/acknowledge`)
    await fetchAlerts()
  } catch (error) {
    console.error('确认报警失败:', error)
  }
}

const acknowledgeAllAlerts = async () => {
  for (const alert of alerts.value.filter(a => !a.acknowledged)) {
    await acknowledgeAlert(alert.alert_id)
  }
}

const toggleZone = async (zoneId) => {
  const zone = zones.value.find(z => z.id === zoneId)
  if (zone) {
    zone.enabled = !zone.enabled
  }
}

// ==================== 辅助函数 ====================
const getGestureIcon = (gesture) => {
  const icons = {
    'thumbs_up': '👍', 'thumbs_down': '👎', 'victory': '✌️', 'ok': '👌',
    'fist': '✊', 'open_palm': '🖐️', 'pointing': '👆', 'rock': '🤘',
    'call_me': '🤙', 'one': '☝️', 'two': '✌️', 'three': '3️⃣',
    'four': '4️⃣', 'five': '🖐️', 'unknown': '❓'
  }
  return icons[gesture] || '❓'
}

const getGestureName = (gesture) => {
  const names = {
    'thumbs_up': '点赞', 'thumbs_down': '踩', 'victory': '胜利', 'ok': 'OK',
    'fist': '握拳', 'open_palm': '张开手掌', 'pointing': '指向', 'rock': '摇滚',
    'call_me': '打电话', 'one': '数字1', 'two': '数字2', 'three': '数字3',
    'four': '数字4', 'five': '数字5', 'unknown': '未知'
  }
  return names[gesture] || gesture
}

const getActionIcon = (action) => {
  const icons = {
    'standing': '🧍', 'sitting': '🪑', 'lying': '🛏️', 'walking': '🚶',
    'running': '🏃', 'jumping': '🦘', 'waving': '👋', 'raising_hand': '🙋',
    'clapping': '👏', 'fighting': '👊', 'falling': '⬇️', 'unknown': '❓'
  }
  return icons[action] || '❓'
}

const getActionName = (action) => {
  const names = {
    'standing': '站立', 'sitting': '坐着', 'lying': '躺着', 'walking': '行走',
    'running': '跑步', 'jumping': '跳跃', 'waving': '挥手', 'raising_hand': '举手',
    'clapping': '拍手', 'fighting': '打架', 'falling': '跌倒', 'unknown': '未知'
  }
  return names[action] || action
}

const getAlertIcon = (type) => {
  const icons = {
    'fall': '⚠️', 'intrusion': '🚨', 'crossing': '🚧', 'loitering': '⏰',
    'crowd': '👥', 'abnormal': '❗', 'exit': '🚪'
  }
  return icons[type] || '🔔'
}

const formatTime = (timestamp) => {
  if (!timestamp) return ''
  const date = new Date(timestamp * 1000)
  return date.toLocaleString('zh-CN')
}

// ==================== 多摄像头管理 ====================
const cameras = ref([])
const activeCameraId = ref(null)
const addCameraDialog = ref(false)

const fetchCameras = async () => {
  try {
    const res = await axios.get(`${API_BASE}/api/cameras`)
    if (res.data.status === 'success') {
      cameras.value = res.data.cameras
    }
  } catch (error) {
    console.error('获取摄像头列表失败:', error)
  }
}

const activateCamera = async (cameraId) => {
  try {
    await axios.post(`${API_BASE}/api/cameras/${cameraId}/activate`)
    activeCameraId.value = cameraId
    await fetchCameras()
  } catch (error) {
    console.error('激活摄像头失败:', error)
  }
}

const openCamera = async (cameraId) => {
  // 打开摄像头逻辑
  console.log('Opening camera:', cameraId)
}

const closeCamera = async (cameraId) => {
  // 关闭摄像头逻辑
  console.log('Closing camera:', cameraId)
}

// ==================== 录制功能 ====================
const isRecording = ref(false)
const recordings = ref([])

const startRecording = async () => {
  try {
    const res = await axios.post(`${API_BASE}/api/recording/start`)
    if (res.data.status === 'success') {
      isRecording.value = true
    }
  } catch (error) {
    console.error('开始录制失败:', error)
  }
}

const stopRecording = async () => {
  try {
    const res = await axios.post(`${API_BASE}/api/recording/stop`)
    if (res.data.status === 'success') {
      isRecording.value = false
      if (res.data.session) {
        recordings.value.unshift(res.data.session)
      }
    }
  } catch (error) {
    console.error('停止录制失败:', error)
  }
}

const playRecording = (rec) => {
  console.log('Playing:', rec.file_path)
}

const downloadRecording = (rec) => {
  window.open(rec.file_path, '_blank')
}

// ==================== 数据分析 ====================
const analyticsStats = ref({})
const trendData = ref(null)
const selectedMetric = ref('person_count')
const anomalies = ref([])
const latestReport = ref(null)

const fetchAnalytics = async () => {
  try {
    const res = await axios.get(`${API_BASE}/api/analytics/statistics`)
    if (res.data.status === 'success') {
      analyticsStats.value = res.data.statistics
    }
  } catch (error) {
    console.error('获取分析数据失败:', error)
  }
}

const fetchTrend = async (metricName) => {
  try {
    const res = await axios.get(`${API_BASE}/api/analytics/trends/${metricName}`)
    if (res.data.status === 'success') {
      trendData.value = res.data.trend
    }
  } catch (error) {
    console.error('获取趋势数据失败:', error)
  }
}

const generateReport = async () => {
  try {
    const res = await axios.get(`${API_BASE}/api/analytics/report`)
    if (res.data.status === 'success') {
      latestReport.value = res.data.report
    }
  } catch (error) {
    console.error('生成报告失败:', error)
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
    
    // 绘制轨迹
    if (person.trajectory && person.trajectory.length > 1) {
      ctx.beginPath()
      ctx.strokeStyle = 'rgba(0, 255, 255, 0.6)'
      ctx.lineWidth = 2 * Math.min(scaleX, scaleY)
      
      person.trajectory.forEach((point, idx) => {
        const px = point.x * scaleX
        const py = point.y * scaleY
        if (idx === 0) {
          ctx.moveTo(px, py)
        } else {
          ctx.lineTo(px, py)
        }
      })
      ctx.stroke()
      
      // 绘制轨迹点（渐变透明度）
      person.trajectory.forEach((point, idx) => {
        const alpha = (idx + 1) / person.trajectory.length
        ctx.fillStyle = `rgba(0, 255, 255, ${alpha * 0.8})`
        ctx.beginPath()
        ctx.arc(point.x * scaleX, point.y * scaleY, 3 * Math.min(scaleX, scaleY), 0, Math.PI * 2)
        ctx.fill()
      })
    }
    
    // 绘制边界框
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
    
    // 绘制速度矢量
    if (person.velocity) {
      const cx = (x + w / 2) * scaleX
      const cy = (y + h / 2) * scaleY
      const vx = person.velocity.vx * 50  // 放大显示
      const vy = person.velocity.vy * 50
      
      ctx.beginPath()
      ctx.strokeStyle = '#ffff00'
      ctx.lineWidth = 3 * Math.min(scaleX, scaleY)
      ctx.moveTo(cx, cy)
      ctx.lineTo(cx + vx, cy + vy)
      ctx.stroke()
      
      // 绘制箭头
      const angle = Math.atan2(vy, vx)
      const arrowSize = 10 * Math.min(scaleX, scaleY)
      ctx.beginPath()
      ctx.fillStyle = '#ffff00'
      ctx.moveTo(cx + vx, cy + vy)
      ctx.lineTo(
        cx + vx - arrowSize * Math.cos(angle - Math.PI / 6),
        cy + vy - arrowSize * Math.sin(angle - Math.PI / 6)
      )
      ctx.lineTo(
        cx + vx - arrowSize * Math.cos(angle + Math.PI / 6),
        cy + vy - arrowSize * Math.sin(angle + Math.PI / 6)
      )
      ctx.closePath()
      ctx.fill()
    }

    // 绘制文字
    ctx.fillStyle = '#00ff00'
    ctx.font = `${14 * Math.min(scaleX, scaleY)}px Arial`
    ctx.fillText(`距离：${person.distance?.toFixed(2) || '?'}m`, x * scaleX, y * scaleY - 10)
    if (person.height > 0) {
      ctx.fillText(`身高：${person.height.toFixed(0)}cm`, x * scaleX, y * scaleY - 30)
    }
    if (person.velocity && person.velocity.speed > 0.1) {
      ctx.fillStyle = '#ffff00'
      ctx.fillText(`速度：${person.velocity.speed.toFixed(2)}m/s`, x * scaleX, y * scaleY - 50)
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
  // 初始化主题
  initTheme()
  
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
/* ==================== 主题变量 ==================== */
:root {
  --bg-primary: #0f0f1a;
  --bg-secondary: #1a1a2e;
  --bg-panel: rgba(26, 26, 46, 0.9);
  --text-primary: #ffffff;
  --text-secondary: #aaaaaa;
  --border-color: #333355;
  --accent-color: #00ff88;
  --accent-secondary: #0088ff;
}

[data-theme="light"] {
  --bg-primary: #f5f5f5;
  --bg-secondary: #ffffff;
  --bg-panel: rgba(255, 255, 255, 0.95);
  --text-primary: #1a1a2e;
  --text-secondary: #666666;
  --border-color: #e0e0e0;
  --accent-color: #00cc6a;
  --accent-secondary: #0066cc;
}

/* ==================== 全局样式 ==================== */
.app-container {
  min-height: 100vh;
  background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
  color: var(--text-primary);
}

.header {
  padding: 20px;
  background: var(--bg-panel);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--border-color);
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 15px;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 15px;
}

.theme-toggle {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  border: 2px solid var(--border-color);
  background: var(--bg-secondary);
  color: var(--text-primary);
  font-size: 20px;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
}

.theme-toggle:hover {
  border-color: var(--accent-color);
  transform: scale(1.1);
}

.sound-toggle {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  border: 2px solid var(--border-color);
  background: var(--bg-secondary);
  color: var(--text-primary);
  font-size: 20px;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 10px;
}

.sound-toggle:hover {
  border-color: var(--accent-color);
  transform: scale(1.1);
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

/* 框选矩形 */
.selection-rect {
  position: absolute;
  border: 2px solid #00ff88;
  background: rgba(0, 255, 136, 0.2);
  pointer-events: none;
  z-index: 10;
}

/* 标定点标记 */
.calibration-marker {
  position: absolute;
  border: 2px dashed #ffcc00;
  background: rgba(255, 204, 0, 0.1);
  pointer-events: none;
  z-index: 5;
}

.marker-label {
  position: absolute;
  top: -20px;
  left: 0;
  background: #ffcc00;
  color: #000;
  padding: 2px 6px;
  font-size: 11px;
  font-weight: bold;
  border-radius: 3px;
}

.video-overlay {
  pointer-events: none;
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

/* 深度热力图样式 */
.heatmap-container {
  margin-top: 15px;
  padding: 15px;
  background: rgba(26, 26, 46, 0.8);
  border-radius: 8px;
  border: 1px solid #333355;
}

.heatmap-image {
  width: 100%;
  max-width: 640px;
  border-radius: 4px;
}

.heatmap-info {
  display: flex;
  justify-content: space-around;
  gap: 20px;
  margin-top: 15px;
  padding: 15px;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 8px;
}

.depth-stat {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 5px;
}

.depth-label {
  font-size: 12px;
  color: #888;
}

.depth-value {
  font-size: 24px;
  font-weight: bold;
}

.depth-value.near {
  color: #00ff88;
}

.depth-value.far {
  color: #ff6b6b;
}

.depth-value.avg {
  color: #4ecdc4;
}

/* 深度标定 */
.depth-calibration {
  margin-top: 15px;
  padding: 15px;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 8px;
  border: 1px solid #333355;
}

.calibration-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.calibration-header h4 {

.calibration-hint {
  font-size: 12px;
  color: #888;
  margin: 10px 0;
  line-height: 1.5;
}
  margin: 0;
  font-size: 14px;
  color: #fff;
}

.calibration-status {
  font-size: 12px;
  padding: 3px 8px;
  border-radius: 4px;
  background: #444;
  color: #888;
}

.calibration-status.calibrated {
  background: #2d5a27;
  color: #7fff7f;
}

.calibration-form {
  margin-bottom: 10px;
}

.calibration-form .form-row {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.calibration-form label {
  font-size: 12px;
  color: #888;
}

.calibration-form input {
  width: 120px;
  padding: 6px 10px;
  border: 1px solid #444;
  border-radius: 4px;
  background: #222;
  color: #fff;
  font-size: 14px;
}

.calibration-form input:focus {
  outline: none;
  border-color: #4ecdc4;
}

.calibration-points {
  margin-top: 10px;
  padding: 10px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 6px;
}

.points-header {
  font-size: 12px;
  color: #888;
  margin-bottom: 8px;
}

.point-item {
  display: flex;
  justify-content: space-between;
  padding: 4px 0;
  font-size: 12px;
  color: #aaa;
}

.calibration-result {
  margin-top: 10px;
  padding: 10px;
  background: rgba(45, 90, 39, 0.3);
  border-radius: 6px;
  display: flex;
  flex-wrap: wrap;
  gap: 15px;
  align-items: center;
  font-size: 12px;
  color: #7fff7f;
}

.btn-sm {
  padding: 4px 10px;
  font-size: 12px;
}

.btn-primary {
  background: #4ecdc4;
  color: #000;
}

.btn-secondary {
  background: #555;
  color: #fff;
}

/* 热力图图例 */
.heatmap-legend {
  display: flex;
  justify-content: center;
  gap: 30px;
  margin-top: 10px;
  padding: 8px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 6px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: #888;
}

.legend-color {
  width: 20px;
  height: 12px;
  border-radius: 2px;
}

.legend-color.far {
  background: linear-gradient(90deg, #0000ff, #00ffff);
}

.legend-color.mid {
  background: linear-gradient(90deg, #00ff00, #ffff00);
}

.legend-color.near {
  background: linear-gradient(90deg, #ff8000, #ff0000);
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

.btn-info {
  background: linear-gradient(90deg, #17a2b8, #138496);
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

/* ==================== 仪表盘样式 ==================== */
.dashboard-page {
  padding: 20px;
}

.dashboard-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
}

.dashboard-card {
  background: var(--bg-panel);
  border-radius: 12px;
  padding: 20px;
  border: 1px solid var(--border-color);
}

.dashboard-card h3 {
  margin: 0 0 15px 0;
  font-size: 16px;
  color: var(--accent-color);
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 15px;
}

.status-item {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.status-label {
  font-size: 12px;
  color: var(--text-secondary);
}

.status-value {
  font-size: 18px;
  font-weight: bold;
}

.status-value.online { color: #00ff88; }
.status-value.offline { color: #ff4444; }
.status-value.calibrated { color: #00aaff; }

.performance-bars {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.perf-item {
  display: flex;
  align-items: center;
  gap: 10px;
}

.perf-label {
  width: 70px;
  font-size: 13px;
  color: var(--text-secondary);
}

.perf-bar {
  flex: 1;
  height: 8px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
  overflow: hidden;
}

.perf-fill {
  height: 100%;
  background: var(--accent-color);
  border-radius: 4px;
  transition: width 0.3s;
}

.perf-fill.warning { background: #ffaa00; }
.perf-fill.danger { background: #ff4444; }

.perf-value {
  width: 60px;
  text-align: right;
  font-size: 13px;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 15px;
}

.stat-item {
  text-align: center;
  padding: 15px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 8px;
}

.stat-value {
  display: block;
  font-size: 28px;
  font-weight: bold;
  color: var(--accent-color);
}

.stat-label {
  font-size: 12px;
  color: var(--text-secondary);
}

.tracks-list, .gestures-list, .actions-list {
  max-height: 200px;
  overflow-y: auto;
}

.track-item, .gesture-item, .action-item {
  display: flex;
  justify-content: space-between;
  padding: 8px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 6px;
  margin-bottom: 5px;
  font-size: 13px;
}

.track-id { color: var(--accent-color); font-weight: bold; }
.gesture-icon, .action-icon { font-size: 20px; margin-right: 10px; }
.no-data { text-align: center; color: var(--text-secondary); padding: 20px; }

/* ==================== 报警页面样式 ==================== */
.alerts-page {
  padding: 20px;
}

.alerts-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.alerts-header h2 {
  margin: 0;
}

.alerts-actions {
  display: flex;
  gap: 10px;
}

.alerts-stats {
  display: flex;
  gap: 20px;
  margin-bottom: 20px;
}

.alert-stat {
  flex: 1;
  text-align: center;
  padding: 20px;
  border-radius: 12px;
  background: var(--bg-panel);
}

.alert-stat.critical { border-left: 4px solid #ff4444; }
.alert-stat.high { border-left: 4px solid #ff8800; }
.alert-stat.medium { border-left: 4px solid #ffaa00; }
.alert-stat.low { border-left: 4px solid #00aaff; }

.alert-stat .stat-num {
  display: block;
  font-size: 32px;
  font-weight: bold;
}

.alert-stat .stat-text {
  font-size: 14px;
  color: var(--text-secondary);
}

.alerts-list {
  background: var(--bg-panel);
  border-radius: 12px;
  padding: 15px;
  margin-bottom: 20px;
}

.no-alerts {
  text-align: center;
  padding: 40px;
  color: var(--text-secondary);
}

.no-alerts-icon {
  font-size: 48px;
  display: block;
  margin-bottom: 10px;
}

.alert-item {
  display: flex;
  align-items: center;
  gap: 15px;
  padding: 15px;
  border-radius: 8px;
  margin-bottom: 10px;
  background: rgba(0, 0, 0, 0.2);
}

.alert-item.critical { border-left: 4px solid #ff4444; }
.alert-item.high { border-left: 4px solid #ff8800; }
.alert-item.medium { border-left: 4px solid #ffaa00; }
.alert-item.low { border-left: 4px solid #00aaff; }
.alert-item.acknowledged { opacity: 0.6; }

.alert-icon {
  font-size: 24px;
}

.alert-content {
  flex: 1;
}

.alert-title {
  font-weight: bold;
  margin-bottom: 5px;
}

.alert-meta {
  font-size: 12px;
  color: var(--text-secondary);
}

.alert-type {
  background: rgba(255, 255, 255, 0.1);
  padding: 2px 8px;
  border-radius: 4px;
  margin-right: 10px;
}

.alert-time {
  color: var(--text-secondary);
}

.acknowledged-badge {
  font-size: 12px;
  color: var(--accent-color);
}

.zones-section {
  background: var(--bg-panel);
  border-radius: 12px;
  padding: 20px;
}

.zones-section h3 {
  margin: 0 0 15px 0;
}

.zone-item {
  display: flex;
  align-items: center;
  gap: 15px;
  padding: 10px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 6px;
  margin-bottom: 8px;
}

.zone-name {
  flex: 1;
  font-weight: bold;
}

.zone-type {
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
}

.zone-type.forbidden { background: rgba(255, 68, 68, 0.3); }
.zone-type.restricted { background: rgba(255, 170, 0, 0.3); }
.zone-type.safe { background: rgba(0, 255, 136, 0.3); }

.zone-status.enabled { color: var(--accent-color); }
.zone-status.disabled { color: #ff4444; }

/* 徽章样式 */
.badge {
  background: #ff4444;
  color: white;
  font-size: 11px;
  padding: 2px 6px;
  border-radius: 10px;
  margin-left: 5px;
}

.btn-small {
  padding: 4px 12px;
  font-size: 12px;
  background: var(--accent-color);
  color: #000;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.btn-small:hover {
  opacity: 0.9;
}

/* ==================== 多摄像头页面样式 ==================== */
.cameras-page {
  padding: 20px;
}

.cameras-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.cameras-header h2 {
  margin: 0;
}

.cameras-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
}

.camera-card {
  background: var(--bg-panel);
  border-radius: 12px;
  overflow: hidden;
  border: 2px solid transparent;
  transition: border-color 0.3s;
}

.camera-card.active {
  border-color: var(--accent-color);
}

.camera-preview {
  height: 150px;
  background: rgba(0, 0, 0, 0.3);
  display: flex;
  align-items: center;
  justify-content: center;
}

.preview-placeholder {
  text-align: center;
}

.camera-icon {
  font-size: 48px;
}

.camera-info {
  padding: 15px;
}

.camera-info h4 {
  margin: 0 0 10px 0;
}

.camera-status {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.camera-meta {
  display: flex;
  gap: 15px;
  font-size: 12px;
  color: var(--text-secondary);
}

.camera-actions {
  padding: 10px 15px;
  display: flex;
  gap: 8px;
  border-top: 1px solid var(--border-color);
}

/* ==================== 录制页面样式 ==================== */
.recording-page {
  padding: 20px;
}

.recording-header {
  margin-bottom: 20px;
}

.recording-header h2 {
  margin: 0;
}

.recording-controls {
  display: flex;
  align-items: center;
  gap: 20px;
  margin-bottom: 30px;
}

.recording-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #ff4444;
}

.rec-dot {
  width: 12px;
  height: 12px;
  background: #ff4444;
  border-radius: 50%;
  animation: pulse 1s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.recordings-list h3 {
  margin-bottom: 15px;
}

.recording-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px;
  background: var(--bg-panel);
  border-radius: 8px;
  margin-bottom: 10px;
}

.rec-info {
  display: flex;
  gap: 20px;
}

.rec-name {
  font-weight: bold;
}

.rec-duration, .rec-frames {
  color: var(--text-secondary);
  font-size: 13px;
}

.rec-actions {
  display: flex;
  gap: 8px;
}

/* ==================== 数据分析页面样式 ==================== */
.analytics-page {
  padding: 20px;
}

.analytics-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.analytics-header h2 {
  margin: 0;
}

.analytics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
}

.analytics-card {
  background: var(--bg-panel);
  border-radius: 12px;
  padding: 20px;
}

.analytics-card.wide {
  grid-column: span 2;
}

.analytics-card h3 {
  margin: 0 0 15px 0;
  font-size: 16px;
  color: var(--accent-color);
}

.metrics-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.metric-row {
  display: flex;
  justify-content: space-between;
  padding: 8px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 6px;
}

.metric-name {
  color: var(--text-secondary);
}

.metric-value {
  font-weight: bold;
}

.metric-unit {
  color: var(--text-secondary);
  font-size: 12px;
}

.trend-selector select {
  padding: 8px 12px;
  border-radius: 6px;
  background: rgba(0, 0, 0, 0.3);
  color: var(--text-color);
  border: 1px solid var(--border-color);
  margin-bottom: 15px;
}

.trend-info {
  display: flex;
  gap: 20px;
}

.trend-info .increasing { color: #00ff88; }
.trend-info .decreasing { color: #ff4444; }
.trend-info .stable { color: #ffaa00; }

.anomalies-list {
  max-height: 200px;
  overflow-y: auto;
}

.anomaly-item {
  display: flex;
  justify-content: space-between;
  padding: 8px;
  background: rgba(255, 68, 68, 0.1);
  border-radius: 6px;
  margin-bottom: 5px;
}

.report-preview {
  padding: 15px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 8px;
}

.report-summary {
  color: var(--text-secondary);
  font-size: 14px;
  margin: 10px 0;
}

.report-meta {
  font-size: 12px;
  color: var(--text-secondary);
}
</style>

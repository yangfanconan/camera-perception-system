<template>
  <div class="pose-3d-container">
    <div class="pose-3d-header">
      <h3>🎭 3D 姿态可视化</h3>
      <div class="controls">
        <button @click="resetCamera" class="btn-small">重置视角</button>
        <button @click="toggleAutoRotate" :class="['btn-small', autoRotate ? 'active' : '']">
          {{ autoRotate ? '停止旋转' : '自动旋转' }}
        </button>
      </div>
    </div>
    <div ref="container" class="pose-3d-canvas"></div>
    <div class="pose-info" v-if="poseData">
      <span>关键点: {{ poseData.keypointCount || 0 }}</span>
      <span>置信度: {{ (poseData.confidence * 100).toFixed(0) }}%</span>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

// Props
const props = defineProps({
  keypoints: {
    type: Object,
    default: () => ({})
  },
  depth: {
    type: Number,
    default: 2.0
  }
})

// Refs
const container = ref(null)
const autoRotate = ref(true)

// Three.js objects
let scene, camera, renderer, controls
let skeletonGroup, jointsGroup, linesGroup
let animationId

// 人体骨骼连接定义 (COCO 格式)
const SKELETON_CONNECTIONS = [
  // 头部
  ['nose', 'L_eye'],
  ['nose', 'R_eye'],
  ['L_eye', 'L_ear'],
  ['R_eye', 'R_ear'],
  // 躯干
  ['L_shoulder', 'R_shoulder'],
  ['L_shoulder', 'L_hip'],
  ['R_shoulder', 'R_hip'],
  ['L_hip', 'R_hip'],
  // 左臂
  ['L_shoulder', 'L_elbow'],
  ['L_elbow', 'L_wrist'],
  // 右臂
  ['R_shoulder', 'R_elbow'],
  ['R_elbow', 'R_wrist'],
  // 左腿
  ['L_hip', 'L_knee'],
  ['L_knee', 'L_ankle'],
  // 右腿
  ['R_hip', 'R_knee'],
  ['R_knee', 'R_ankle']
]

// 关键点颜色
const JOINT_COLORS = {
  nose: 0xff0000,
  L_eye: 0x00ff00,
  R_eye: 0x00ff00,
  L_ear: 0x00ff00,
  R_ear: 0x00ff00,
  L_shoulder: 0xff8800,
  R_shoulder: 0xff8800,
  L_elbow: 0xffff00,
  R_elbow: 0xffff00,
  L_wrist: 0x00ffff,
  R_wrist: 0x00ffff,
  L_hip: 0xff00ff,
  R_hip: 0xff00ff,
  L_knee: 0x8800ff,
  R_knee: 0x8800ff,
  L_ankle: 0x0088ff,
  R_ankle: 0x0088ff
}

// 姿态数据
const poseData = ref(null)

// 初始化 Three.js 场景
const initScene = () => {
  if (!container.value) return

  // 场景
  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x1a1a2e)

  // 相机
  const width = container.value.clientWidth
  const height = container.value.clientHeight
  camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000)
  camera.position.set(0, 0, 5)

  // 渲染器
  renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setSize(width, height)
  renderer.setPixelRatio(window.devicePixelRatio)
  container.value.appendChild(renderer.domElement)

  // 控制器
  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.dampingFactor = 0.05
  controls.autoRotate = autoRotate.value
  controls.autoRotateSpeed = 1.0

  // 光源
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.6)
  scene.add(ambientLight)

  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
  directionalLight.position.set(5, 5, 5)
  scene.add(directionalLight)

  // 网格地面
  const gridHelper = new THREE.GridHelper(10, 20, 0x333355, 0x222244)
  gridHelper.rotation.x = Math.PI / 2
  scene.add(gridHelper)

  // 坐标轴
  const axesHelper = new THREE.AxesHelper(2)
  scene.add(axesHelper)

  // 骨骼组
  skeletonGroup = new THREE.Group()
  scene.add(skeletonGroup)

  // 关节点组
  jointsGroup = new THREE.Group()
  skeletonGroup.add(jointsGroup)

  // 骨骼线组
  linesGroup = new THREE.Group()
  skeletonGroup.add(linesGroup)

  // 开始动画
  animate()
}

// 动画循环
const animate = () => {
  animationId = requestAnimationFrame(animate)
  
  if (controls) {
    controls.update()
  }
  
  if (renderer && scene && camera) {
    renderer.render(scene, camera)
  }
}

// 更新姿态
const updatePose = (keypoints, depth = 2.0) => {
  if (!jointsGroup || !linesGroup) return

  // 清空现有骨骼
  jointsGroup.clear()
  linesGroup.clear()

  if (!keypoints || Object.keys(keypoints).length === 0) {
    poseData.value = null
    return
  }

  // 计算 3D 坐标
  const positions3D = {}
  let totalConfidence = 0
  let validKeypoints = 0

  Object.entries(keypoints).forEach(([name, kp]) => {
    if (kp && kp.length >= 2) {
      // 将 2D 坐标转换为 3D (假设图像中心为原点)
      // x: -1 到 1 (左右)
      // y: -1 到 1 (上下，Y轴向下为正)
      // z: 基于深度估计
      
      const x = (kp[0] - 960) / 960  // 假设图像宽度 1920
      const y = -(kp[1] - 540) / 540  // 假设图像高度 1080，Y轴翻转
      const z = depth * (1 + (Math.random() - 0.5) * 0.1)  // 添加一点随机深度变化
      
      positions3D[name] = new THREE.Vector3(x * depth, y * depth, z * 0.3)
      
      if (kp.length >= 3) {
        totalConfidence += kp[2]
      }
      validKeypoints++
    }
  })

  // 更新姿态数据
  poseData.value = {
    keypointCount: validKeypoints,
    confidence: validKeypoints > 0 ? totalConfidence / validKeypoints : 0
  }

  // 绘制关节点
  Object.entries(positions3D).forEach(([name, pos]) => {
    const geometry = new THREE.SphereGeometry(0.08, 16, 16)
    const material = new THREE.MeshPhongMaterial({
      color: JOINT_COLORS[name] || 0xffffff,
      emissive: JOINT_COLORS[name] || 0xffffff,
      emissiveIntensity: 0.3
    })
    const sphere = new THREE.Mesh(geometry, material)
    sphere.position.copy(pos)
    jointsGroup.add(sphere)
  })

  // 绘制骨骼线
  SKELETON_CONNECTIONS.forEach(([start, end]) => {
    const startPos = positions3D[start]
    const endPos = positions3D[end]
    
    if (startPos && endPos) {
      const material = new THREE.LineBasicMaterial({
        color: 0x00ff88,
        linewidth: 2
      })
      
      const points = [startPos, endPos]
      const geometry = new THREE.BufferGeometry().setFromPoints(points)
      const line = new THREE.Line(geometry, material)
      linesGroup.add(line)
    }
  })

  // 居中骨骼
  if (skeletonGroup) {
    skeletonGroup.position.set(0, 0, 0)
  }
}

// 重置相机
const resetCamera = () => {
  if (camera && controls) {
    camera.position.set(0, 0, 5)
    controls.reset()
  }
}

// 切换自动旋转
const toggleAutoRotate = () => {
  autoRotate.value = !autoRotate.value
  if (controls) {
    controls.autoRotate = autoRotate.value
  }
}

// 监听关键点变化
watch(() => props.keypoints, (newKeypoints) => {
  updatePose(newKeypoints, props.depth)
}, { deep: true })

// 监听深度变化
watch(() => props.depth, (newDepth) => {
  updatePose(props.keypoints, newDepth)
})

// 窗口大小变化
const handleResize = () => {
  if (!container.value || !camera || !renderer) return
  
  const width = container.value.clientWidth
  const height = container.value.clientHeight
  
  camera.aspect = width / height
  camera.updateProjectionMatrix()
  renderer.setSize(width, height)
}

// 生命周期
onMounted(() => {
  initScene()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  if (animationId) {
    cancelAnimationFrame(animationId)
  }
  window.removeEventListener('resize', handleResize)
  
  if (renderer) {
    renderer.dispose()
  }
})

// 暴露方法
defineExpose({
  updatePose,
  resetCamera
})
</script>

<style scoped>
.pose-3d-container {
  background: rgba(26, 26, 46, 0.9);
  border-radius: 8px;
  border: 1px solid #333355;
  overflow: hidden;
}

.pose-3d-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 15px;
  background: rgba(0, 0, 0, 0.3);
  border-bottom: 1px solid #333355;
}

.pose-3d-header h3 {
  margin: 0;
  font-size: 14px;
  color: #00ff88;
}

.controls {
  display: flex;
  gap: 8px;
}

.btn-small {
  padding: 4px 10px;
  font-size: 12px;
  background: #333355;
  color: #fff;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background 0.3s;
}

.btn-small:hover {
  background: #444466;
}

.btn-small.active {
  background: #00ff88;
  color: #000;
}

.pose-3d-canvas {
  width: 100%;
  height: 300px;
}

.pose-info {
  display: flex;
  justify-content: space-around;
  padding: 8px;
  background: rgba(0, 0, 0, 0.3);
  font-size: 12px;
  color: #aaa;
}
</style>
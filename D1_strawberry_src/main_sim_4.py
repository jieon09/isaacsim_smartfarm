"""
main_sim.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Isaac Sim 통합 시뮬레이션 진입점.
  - Nova Carter (카메라) → ROS2로 영상 발행
  - Spot (사족보행)      → UDP로 cmd_vel 수신 / 상태 송신
  - 딸기 Growth 타임랩스 → AquaCrop 모델 기반 색상·크기 변화
실행: isaac-python main_sim.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# Isaac Sim 앱을 가장 먼저 초기화해야 함 (이후 import보다 반드시 앞에 위치)
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "renderer": "RayTracedLighting"})

# Isaac Sim 확장 활성화 (ROS2 브릿지, 카메라 센서 포함)
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.core.nodes")
enable_extension("isaacsim.sensors.camera")
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

import math, time, asyncio, socket, json, os
import numpy as np
import pandas as pd
import omni.usd
import omni.graph.core as og
import omni.kit.app
from pxr import Sdf, UsdGeom, UsdShade, Gf
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy

# ══════════════════════════════════════════════
#  공통 설정
# ══════════════════════════════════════════════

# Omniverse Nucleus 서버에 저장된 농장 환경 USD 파일 경로
base_dir = os.path.dirname(os.path.abspath(__file__))
ASSET_PATH = os.path.join(base_dir, "resource", "kitkit2.usd")

# ── 카메라 설정 ────────────────────────────────
# Nova Carter 왼쪽 카메라 prim 경로 (USD 씬 내부 경로)
PREFERRED_CAMERA_PRIM_PATH = (
    "/World/kitkit2/finalllll/carter_warehouse_navigation"
    "/Nova_Carter_ROS/chassis_link/sensors/left_hawk/left/camera_left"
)
RESOLUTION        = (640, 480)       # 카메라 해상도 (픽셀)
FRAME_ID          = "left_hawk_left_camera"  # ROS2 tf frame 이름
RGB_TOPIC         = "left_hawk/left/image_raw"         # ROS2 RGB 영상 토픽
DEPTH_TOPIC       = "left_hawk/left/depth/image_raw"   # ROS2 깊이 영상 토픽
CAMERA_INFO_TOPIC = "left_hawk/left/camera_info"       # ROS2 카메라 내부 파라미터 토픽
FRAME_SKIP_COUNT  = 0                # 0 = 매 프레임 발행
HFOV_DEG          = 90.0            # 수평 화각 (degrees)
DEPTH_NEAR        = 0.05            # 깊이 클리핑 최솟값 (m)
DEPTH_FAR         = 20.0            # 깊이 클리핑 최댓값 (m)
PIXEL_SIZE_MICRONS = 3.0            # 픽셀 물리 크기 (μm) - 초점거리 계산용

# ── 딸기 성장(Growth) 설정 ──────────────────────
# USD 씬에서 딸기 식물 그룹의 루트 경로
BASE_PATH = (
    "/World/kitkit2/finalllll"
    "/TomatoGrowingZone_WithPotsAndPlants"
    "/Grouped_StrawberryPlant_Instances_v2"
)
ROWS             = ["Row_1_Left", "Row_Center_Middle", "Row_2_Right"]  # 화분 행 이름
PLANTS_PER_ROW   = 5    # 행당 식물 수
FRUITS_PER_PLANT = 4    # 식물당 열매 수
SIM_START        = "2024/04/01"   # AquaCrop 시뮬레이션 시작일
SIM_END          = "2024/09/30"   # AquaCrop 시뮬레이션 종료일
STEP_INTERVAL    = 0.1  # 타임랩스 1스텝당 실제 대기 시간 (초)
DAP_STEP         = 2    # 타임랩스 DAP(파종 후 일수) 증가 단위
SCALE_MIN        = 0.05 # 열매 초기 크기 (비율)
SCALE_MAX        = 1.0  # 열매 최대 크기 (비율)
CC_MAX           = 0.73 # AquaCrop Canopy Cover 최댓값 (런타임에 재계산됨)

# 열매 색상 키프레임: (성장 비율, RGB)
# 0% → 연두색, 60% → 연두색 유지, 80% → 주황색, 100% → 빨간색(익음)
COLOR_KEYFRAMES = [
    (0.00, (0.76, 0.84, 0.44)),
    (0.60, (0.76, 0.84, 0.44)),
    (0.80, (0.84, 0.52, 0.20)),
    (1.00, (0.86, 0.025, 0.018)),
]

# ── Doosan M0609 arm 설정 ──────────────────────
# Isaac Sim Stage Inspector에서 확인한 articulation root prim 경로
# None이면 _find_doosan_prim()이 자동 탐색
DOOSAN_PRIM_PATH    = "/World/kitkit2/finalllll/carter_warehouse_navigation/m0609_rg2_01/base_link"
ARM_JOINT_NAMES     = [f"joint_{i}" for i in range(1, 7)]          # joint_1 ~ joint_6
GRIPPER_JOINT_NAMES = ["finger_joint", "left_outer_knuckle_joint"]  # 그리퍼 관절
ARM_CMD_FILE        = "/tmp/arm_joint_cmd.txt"   # harvest_strawberry_action이 쓰는 파일
GRIPPER_CMD_FILE    = "/tmp/gripper_cmd.txt"     # harvest_strawberry_action이 쓰는 파일

# ── Spot 로봇 설정 ─────────────────────────────
SPOT_PRIM_PATH       = "/World/kitkit2/Robot2Spot"   # USD 씬 내 Spot prim 경로
SPOT_NAME            = "Robot2Spot"
SPOT_INITIAL_POS     = np.array([-3.41, 9.79, 1.0]) # Spot 초기 위치 (m)
CMD_VEL_UDP_HOST     = "127.0.0.1"  # cmd_vel UDP 수신 IP
CMD_VEL_UDP_PORT     = 5005         # cmd_vel UDP 수신 포트 (ROS2 브릿지 → 시뮬레이터)
STATE_UDP_HOST       = "127.0.0.1"  # Spot 상태 UDP 송신 IP
STATE_UDP_PORT       = 5006         # Spot 상태 UDP 송신 포트 (시뮬레이터 → ROS2 브릿지)
LINEAR_X_SCALE       = 4.0          # 선속도 명령 스케일 (Isaac Sim 내부 단위 변환)
ANGULAR_Z_SCALE      = 2.0          # 각속도 명령 스케일
MAX_LINEAR_X         = 2.0          # 선속도 최대값 (m/s)
MAX_ANGULAR_Z        = 2.0          # 각속도 최대값 (rad/s)
CMD_TIMEOUT_SEC      = 0.5          # 이 시간 이상 명령 없으면 정지

# ══════════════════════════════════════════════
#  Growth 헬퍼 함수
# ══════════════════════════════════════════════

def cc_to_color(cc):
    """Canopy Cover 값(0~CC_MAX)을 COLOR_KEYFRAMES 기반 RGB 색상으로 변환."""
    cc = min(max(cc, 0.0), CC_MAX)
    ratio = cc / CC_MAX
    for i in range(len(COLOR_KEYFRAMES) - 1):
        r0, rgb0 = COLOR_KEYFRAMES[i]
        r1, rgb1 = COLOR_KEYFRAMES[i + 1]
        if r0 <= ratio <= r1:
            t = (ratio - r0) / (r1 - r0) if r1 > r0 else 0.0
            return tuple(rgb0[j] + t * (rgb1[j] - rgb0[j]) for j in range(3))
    return COLOR_KEYFRAMES[-1][1]

def make_weather(water_factor, seed=42):
    """AquaCrop 입력용 기상 데이터 DataFrame 생성. water_factor로 강수량 조절."""
    np.random.seed(seed)
    dates = pd.date_range(SIM_START, SIM_END)
    n = len(dates)
    precip = np.random.choice([0.0, 0.0, 0.0, 5.0, 10.0], n) * water_factor
    return pd.DataFrame({
        "MinTemp":       np.random.uniform(15, 20, n).round(1),
        "MaxTemp":       np.random.uniform(25, 32, n).round(1),
        "Precipitation": precip,
        "ReferenceET":   np.random.uniform(3, 6, n).round(2),
        "Date":          dates,
    })

def run_aquacrop(water_factor):
    """AquaCrop 모델로 딸기 성장 시뮬레이션 실행. DAP별 canopy_cover 시리즈 반환."""
    from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent
    model = AquaCropModel(
        sim_start_time=SIM_START, sim_end_time=SIM_END,
        weather_df=make_weather(water_factor),
        soil=Soil("SandyLoam"),
        crop=Crop("Tomato", planting_date="04/01"),
        initial_water_content=InitialWaterContent(value=["FC"]),
    )
    model.run_model(till_termination=True)
    df = model.get_crop_growth()[["dap", "canopy_cover"]].dropna()
    return df.set_index("dap")["canopy_cover"]

def get_plant_base(row, p):
    """특정 행(row)·번호(p) 식물의 USD prim 기본 경로 반환."""
    return f"{BASE_PATH}/{row}_StrawberryPlant_{p:02d}_GRP/PlantAsset"

def get_fruit_paths():
    """
    전체 화분(pot_01 ~ pot_15)별 열매 USD 경로 목록 반환.
    반환: {"pot_01": [".../Fruit_01", ".../Fruit_02", ...], ...}
    """
    paths = {}
    pot_idx = 1
    for row in ROWS:
        for p in range(1, PLANTS_PER_ROW + 1):
            base = get_plant_base(row, p) + "/Plant"
            paths[f"pot_{pot_idx:02d}"] = [f"{base}/Fruit_{f:02d}" for f in range(1, FRUITS_PER_PLANT + 1)]
            pot_idx += 1
    return paths

def get_material(stage, row, p, mat_name):
    """USD 스테이지에서 특정 식물의 재질(Material) prim 반환."""
    return UsdShade.Material(stage.GetPrimAtPath(get_plant_base(row, p) + f"/Materials/{mat_name}"))

def bind_material(stage, fruit_path, mat):
    """열매 FruitBody prim에 재질을 바인딩."""
    body = stage.GetPrimAtPath(fruit_path + "/FruitBody")
    if body.IsValid() and mat:
        UsdShade.MaterialBindingAPI(body).Bind(mat)

def set_diffuse(stage, mat_path_str, rgb):
    """재질의 PreviewSurface 쉐이더 diffuseColor 입력값 설정."""
    shader = UsdShade.Shader(stage.GetPrimAtPath(mat_path_str + "/PreviewSurface"))
    if shader:
        inp = shader.GetInput("diffuseColor")
        if inp:
            inp.Set(Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2])))

def set_scale(stage, path, scale):
    """USD prim의 균일 스케일(XformOp:scale) 설정."""
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return
    xf = UsdGeom.Xformable(prim)
    scale_op = None
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
            break
    if scale_op is None:
        scale_op = xf.AddScaleOp()
    scale_op.Set(Gf.Vec3f(scale, scale, scale))

# ══════════════════════════════════════════════
#  Growth 타임랩스 (비동기)
# ══════════════════════════════════════════════

async def run_timelapse():
    """
    AquaCrop 결과를 기반으로 딸기 열매의 크기·색상을 DAP별로 갱신하는 타임랩스.
    완료 시 /tmp/growth_done.txt 생성 → straight.py(순찰 노드)에 신호 전달.
    """
    global CC_MAX
    print("\n=== Growth 타임랩스 시작 ===")

    # [1/3] 화분별로 다른 water_factor(관개 수준)로 AquaCrop 실행
    print("[1/3] AquaCrop 계산 중...")
    fruit_paths = get_fruit_paths()
    pot_ids = list(fruit_paths.keys())
    # 각 화분에 0.80~1.20 범위의 water_factor를 랜덤 배분
    wf_values = [round(0.8 + i / (len(pot_ids) - 1) * 0.40, 3) for i in range(len(pot_ids))]
    rng = np.random.default_rng(seed=int(time.time()))
    rng.shuffle(wf_values)
    water_factors = dict(zip(pot_ids, wf_values))
    growth_data = {pid: run_aquacrop(wf) for pid, wf in water_factors.items()}
    max_dap = max(s.index.max() for s in growth_data.values())
    CC_MAX = max(float(s.max()) for s in growth_data.values())
    print(f"    완료. 시즌 {max_dap}일  CC_MAX={CC_MAX:.3f}")

    # [2/3] 모든 열매를 초기 상태(작은 크기, 연두색)로 초기화
    print("[2/3] 초기화...")
    stage = omni.usd.get_context().get_stage()
    pot_to_rowp = {}
    pot_idx = 1
    for row in ROWS:
        for p in range(1, PLANTS_PER_ROW + 1):
            pot_to_rowp[f"pot_{pot_idx:02d}"] = (row, p)
            pot_idx += 1

    # 재질 색상을 기본값으로 초기화
    for row in ROWS:
        for p in range(1, PLANTS_PER_ROW + 1):
            for mat_name, rgb in [
                ("Mat_BerryPale", (0.76, 0.84, 0.44)),
                ("Mat_BerryPink", (0.84, 0.52, 0.20)),
                ("Mat_BerryRed",  (0.86, 0.025, 0.018)),
            ]:
                set_diffuse(stage, get_plant_base(row, p) + f"/Materials/{mat_name}", rgb)

    # 열매 크기를 최솟값으로, 재질을 연두색으로 초기화
    for pid, paths in fruit_paths.items():
        row, p = pot_to_rowp[pid]
        pale_mat = get_material(stage, row, p, "Mat_BerryPale")
        set_diffuse(stage, get_plant_base(row, p) + "/Materials/Mat_BerryPale", (0.76, 0.84, 0.44))
        for fp in paths:
            bind_material(stage, fp, pale_mat)
            set_scale(stage, fp, SCALE_MIN)

    await omni.kit.app.get_app().next_update_async()
    print("    초기화 완료!")
    print("[3/3] 타임랩스 실행 중...\n")

    # [3/3] DAP를 DAP_STEP씩 증가시키며 열매 크기·색상 갱신
    peak_cc = {pid: 0.0 for pid in fruit_paths}  # 화분별 최대 CC 누적 (퇴행 방지)
    for dap in range(1, int(max_dap) + 1, DAP_STEP):
        size_ratio = min(dap / max_dap, 1.0)
        scale = SCALE_MIN + size_ratio * (SCALE_MAX - SCALE_MIN)
        for pid, paths in fruit_paths.items():
            row, p = pot_to_rowp[pid]
            cc_series = growth_data[pid]
            idx = int(np.abs(cc_series.index - dap).argmin())
            peak_cc[pid] = max(peak_cc[pid], float(cc_series.iloc[idx]))
            rgb = cc_to_color(peak_cc[pid])
            set_diffuse(stage, get_plant_base(row, p) + "/Materials/Mat_BerryPale", rgb)
            for fp in paths:
                set_scale(stage, fp, scale)

        await omni.kit.app.get_app().next_update_async()
        # STEP_INTERVAL초 동안 렌더 프레임 진행 (타임랩스 속도 조절)
        for _ in range(max(1, int(STEP_INTERVAL * 60))):
            await omni.kit.app.get_app().next_update_async()

        if dap % 15 < DAP_STEP:
            cc8 = peak_cc["pot_08"]
            r, g, b = cc_to_color(cc8)
            print(f"  DAP {dap:3d}일 | scale={scale:.3f} | pot_08 CC={cc8:.3f} RGB=({r:.2f},{g:.2f},{b:.2f})")

    print("\n✅ Growth 완료!")
    # straight.py(순찰 노드)가 파일 존재 여부로 성장 완료를 감지
    with open("/tmp/growth_done.txt", "w") as f:
        f.write("done")
    print("📢 /tmp/growth_done.txt 생성 → straight.py 출발!")

# ══════════════════════════════════════════════
#  카메라 헬퍼 함수
# ══════════════════════════════════════════════

def list_camera_prims(stage):
    """USD 스테이지에서 Camera 타입의 모든 prim 경로 목록 반환."""
    return [str(p.GetPath()) for p in stage.Traverse() if p.GetTypeName() == "Camera"]

def resolve_camera_prim_path(stage, preferred_path):
    """
    지정된 카메라 경로가 유효하면 그대로 사용.
    없으면 씬에서 패턴 매칭으로 자동 탐색.
    """
    if stage.GetPrimAtPath(preferred_path).IsValid():
        return preferred_path
    camera_paths = list_camera_prims(stage)
    matches = [p for p in camera_paths if p.endswith("/sensors/left_hawk/left/camera_left")]
    if len(matches) == 1:
        return matches[0]
    matches = [p for p in camera_paths if p.endswith("/camera_left")]
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError(f"카메라 못 찾음. 카메라 목록:\n" + "\n".join(camera_paths[:20]))

def configure_camera(camera):
    """
    HFOV와 픽셀 크기로부터 초점거리·조리개를 계산해 카메라를 pinhole 모델로 설정.
    YOLO 노드의 3D 좌표 역투영 정확도를 높이기 위해 왜곡 없는 pinhole로 고정.
    """
    width, height = RESOLUTION
    fx = width / (2.0 * math.tan(math.radians(HFOV_DEG) / 2.0))
    ps = PIXEL_SIZE_MICRONS * 1e-6
    camera.set_projection_mode("perspective")
    try: camera.set_lens_distortion_model("pinhole")
    except: pass
    try: camera.set_projection_type("pinhole")
    except: pass
    camera.set_focal_length(ps * fx)
    camera.set_horizontal_aperture(ps * width)
    camera.set_vertical_aperture(ps * height)
    camera.set_focus_distance(10.0)
    camera.set_lens_aperture(0.0)
    camera.set_clipping_range(DEPTH_NEAR, DEPTH_FAR)
    try:
        camera.set_opencv_pinhole_properties(cx=width/2.0, cy=height/2.0, fx=fx, fy=fx, pinhole=[0.0]*8)
    except: pass
    camera.set_clipping_range(DEPTH_NEAR, DEPTH_FAR)
    simulation_app.update()
    print("✅ 카메라 pinhole 설정 완료")

def create_camera_publishers(render_product_path):
    """
    OmniGraph로 ROS2 카메라 퍼블리셔 노드 그래프 생성.
    RGB, Depth, CameraInfo 세 가지 토픽을 Isaac Sim ROS2 브릿지로 발행.
    """
    graph_path = "/ROS2_CameraPublisherGraph"
    stage = omni.usd.get_context().get_stage()
    # 기존 그래프가 있으면 재생성
    if stage.GetPrimAtPath(graph_path).IsValid():
        stage.RemovePrim(Sdf.Path(graph_path))
        simulation_app.update()
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {keys.CREATE_NODES: [
            ("OnPlaybackTick",    "omni.graph.action.OnPlaybackTick"),
            ("ROS2Publish_RGB",   "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("ROS2Publish_Depth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("ROS2Publish_Info",  "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
        ]},
    )
    tick = f"{graph_path}/OnPlaybackTick"
    # RGB·Depth 퍼블리셔 노드 파라미터 설정 및 tick 연결
    for node, t, topic in [
        (f"{graph_path}/ROS2Publish_RGB",   "rgb",   RGB_TOPIC),
        (f"{graph_path}/ROS2Publish_Depth", "depth", DEPTH_TOPIC),
    ]:
        og.Controller.attribute(f"{node}.inputs:renderProductPath").set(render_product_path)
        og.Controller.attribute(f"{node}.inputs:type").set(t)
        og.Controller.attribute(f"{node}.inputs:topicName").set(topic)
        og.Controller.attribute(f"{node}.inputs:frameId").set(FRAME_ID)
        og.Controller.attribute(f"{node}.inputs:frameSkipCount").set(FRAME_SKIP_COUNT)
        og.Controller.connect(f"{tick}.outputs:tick", f"{node}.inputs:execIn")
    # CameraInfo 퍼블리셔 노드 파라미터 설정
    info = f"{graph_path}/ROS2Publish_Info"
    og.Controller.attribute(f"{info}.inputs:renderProductPath").set(render_product_path)
    og.Controller.attribute(f"{info}.inputs:topicName").set(CAMERA_INFO_TOPIC)
    og.Controller.attribute(f"{info}.inputs:frameId").set(FRAME_ID)
    og.Controller.attribute(f"{info}.inputs:frameSkipCount").set(FRAME_SKIP_COUNT)
    og.Controller.connect(f"{tick}.outputs:tick", f"{info}.inputs:execIn")
    print("✅ ROS2 Camera publisher 생성 완료")

# ══════════════════════════════════════════════
#  Doosan arm 헬퍼 함수
# ══════════════════════════════════════════════

def create_arm_controller_graph(arm_prim_path):
    """
    OmniGraph로 ROS2 JointState 구독 → Articulation 관절 구동 그래프 생성.
    /robot1/arm/joint_command (sensor_msgs/JointState) 수신 시
    Isaac Sim articulation의 관절 위치를 직접 제어.
    """
    graph_path = "/ArmControllerGraph"
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(graph_path).IsValid():
        stage.RemovePrim(Sdf.Path(graph_path))
        simulation_app.update()

    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {keys.CREATE_NODES: [
            ("OnPlaybackTick",        "omni.graph.action.OnPlaybackTick"),
            ("SubscribeJointState",   "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
            ("ArticulationController","isaacsim.core.nodes.IsaacArticulationController"),
        ]},
    )

    tick = f"{graph_path}/OnPlaybackTick"
    sub  = f"{graph_path}/SubscribeJointState"
    ctrl = f"{graph_path}/ArticulationController"

    # 토픽 이름 설정
    og.Controller.attribute(f"{sub}.inputs:topicName").set("/robot1/arm/joint_command")

    # Articulation prim 경로 설정
    # inputs:usePath removed (deprecated in this Isaac Sim version)
    og.Controller.attribute(f"{ctrl}.inputs:robotPath").set(arm_prim_path)

    # 노드 연결: tick → subscribe → articulation
    og.Controller.connect(f"{tick}.outputs:tick",             f"{sub}.inputs:execIn")
    og.Controller.connect(f"{sub}.outputs:execOut",           f"{ctrl}.inputs:execIn")
    og.Controller.connect(f"{sub}.outputs:jointNames",        f"{ctrl}.inputs:jointNames")
    og.Controller.connect(f"{sub}.outputs:positionCommand",   f"{ctrl}.inputs:positionCommand")

    print(f"✅ Arm OmniGraph 생성 완료 → /robot1/arm/joint_command")


def _find_doosan_prim(stage):
    """Stage에서 Doosan arm articulation root prim 경로 반환. 없으면 자동 탐색."""
    if DOOSAN_PRIM_PATH:
        return DOOSAN_PRIM_PATH
    from pxr import UsdPhysics
    all_art = [str(p.GetPath()) for p in stage.Traverse()
               if UsdPhysics.ArticulationRootAPI.Get(stage, p.GetPath())]
    with open("/tmp/arm_prims_debug.txt", "w") as f:
        f.write("\n".join(all_art))
    for p in all_art:
        if any(k in p.lower() for k in ("dsr01", "m0609", "doosan", "robot1")):
            print(f"[Arm] prim 자동 선택: {p}")
            return p
    print("[Arm] ⚠️ 키워드 매칭 실패. /tmp/arm_prims_debug.txt 참고")
    return None

def read_arm_cmd():
    """파일에서 관절 위치 6개 float 읽기. 없으면 None."""
    try:
        with open(ARM_CMD_FILE) as f:
            vals = list(map(float, f.read().split()))
        return vals if len(vals) >= 6 else None
    except Exception:
        return None

def read_gripper_cmd():
    """파일에서 그리퍼 위치 1개 float 읽기. 없으면 None."""
    try:
        with open(GRIPPER_CMD_FILE) as f:
            return float(f.read().strip())
    except Exception:
        return None

# ══════════════════════════════════════════════
#  Spot UDP 통신 클래스
# ══════════════════════════════════════════════

def yaw_from_quat_wxyz(q):
    """쿼터니언 (w,x,y,z)에서 yaw 각도(rad) 추출."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))

class UdpStateSender:
    """
    Isaac Sim 내 Spot 위치·자세를 UDP JSON으로 송신.
    수신처: robot2_state_udp_bridge.py (포트 5006)
    패킷 형식: {"x": float, "y": float, "z": float, "yaw": float}
    """
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target = (STATE_UDP_HOST, STATE_UDP_PORT)
        print(f"[UDP state] → {STATE_UDP_HOST}:{STATE_UDP_PORT}")

    def send(self, position, orientation):
        packet = {"x": float(position[0]), "y": float(position[1]),
                  "z": float(position[2]), "yaw": yaw_from_quat_wxyz(orientation)}
        self.sock.sendto(json.dumps(packet).encode(), self.target)

    def close(self): self.sock.close()

class UdpCmdVelReceiver:
    """
    cmd_vel_udp_bridge.py(포트 5005)로부터 Spot 속도 명령을 UDP JSON으로 수신.
    패킷 형식: {"vx": float, "vy": float, "wz": float}
    CMD_TIMEOUT_SEC 이상 수신 없으면 정지 명령 반환.
    """
    def __init__(self):
        self.cmd = np.zeros(3, dtype=np.float32)  # [vx, vy, wz]
        self.last_msg_time = 0.0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((CMD_VEL_UDP_HOST, CMD_VEL_UDP_PORT))
        self.sock.setblocking(False)  # 메인 루프를 블로킹하지 않도록 논블로킹 설정
        print(f"[UDP cmd_vel] ← {CMD_VEL_UDP_HOST}:{CMD_VEL_UDP_PORT}")

    def spin_once(self):
        """소켓 버퍼의 모든 패킷을 소진해 최신 명령만 유지."""
        while True:
            try:
                data, _ = self.sock.recvfrom(1024)
                msg = json.loads(data.decode())
                vx = float(np.clip(msg.get("vx", 0.0) * LINEAR_X_SCALE, -MAX_LINEAR_X, MAX_LINEAR_X))
                wz = float(np.clip(msg.get("wz", 0.0) * ANGULAR_Z_SCALE, -MAX_ANGULAR_Z, MAX_ANGULAR_Z))
                self.cmd[:] = [vx, 0.0, wz]
                self.last_msg_time = time.monotonic()
            except BlockingIOError:
                break  # 버퍼 비어있음 → 정상 종료
            except Exception as e:
                print(f"[UDP cmd_vel] bad packet: {e}")

    def get_command(self):
        """타임아웃 초과 시 정지, 아니면 최신 명령 반환."""
        if time.monotonic() - self.last_msg_time > CMD_TIMEOUT_SEC:
            return np.zeros(3, dtype=np.float32)
        return self.cmd.copy()

    def close(self): self.sock.close()

# ══════════════════════════════════════════════
#  메인 함수
# ══════════════════════════════════════════════

def main():
    # World 먼저 생성 후 reference로 추가해야 physics scene이 깨끗하게 초기화됨
    world = World(stage_units_in_meters=1.0, physics_dt=1/100, rendering_dt=1/50)

    # ── USD 환경 로드 ──────────────────────────
    print("📦 USD 로딩 중...")
    add_reference_to_stage(ASSET_PATH, "/World")

    for _ in range(5):
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    print(f"✅ Stage 로드: {stage.GetRootLayer().identifier}")

    # ── 그리퍼 rotation 보정 (physics reset 전에 적용) ──────────────
    # m0609_rg2_mount 조인트가 link_6 기준 rpy=0,0,0 으로 잘못 조립됨
    # → quick_changer prim을 stage에서 찾아 Y -90° 보정
    _gripper_fixed = False
    for prim in stage.Traverse():
        if prim.GetName() == "quick_changer":
            from pxr import UsdGeom
            xform = UsdGeom.Xformable(prim)
            ops = xform.GetOrderedXformOps()
            # 기존 rotateY op이 없을 때만 추가
            op_names = [op.GetOpName() for op in ops]
            if "xformOp:rotateY" not in op_names:
                xform.AddRotateYOp().Set(-90.0)
            _gripper_fixed = True
            print(f"✅ 그리퍼 rotation 보정: {prim.GetPath()} Y=-90°")
            break
    if not _gripper_fixed:
        print("⚠️ quick_changer prim을 찾지 못했습니다. 그리퍼 보정 생략.")

    # ── Spot 초기화 ────────────────────────────
    # kitkit3.usd에 이미 Spot prim이 포함되어 있어야 함
    spot_prim = stage.GetPrimAtPath(SPOT_PRIM_PATH)
    if not spot_prim.IsValid():
        raise RuntimeError(f"Spot prim을 찾을 수 없습니다: {SPOT_PRIM_PATH}")
    print(f"[Spot] 기존 prim 사용: {SPOT_PRIM_PATH}")
    spot = SpotFlatTerrainPolicy(prim_path=SPOT_PRIM_PATH, name=SPOT_NAME)

    simulation_app.update()
    world.reset()

    # 물리 스텝 콜백용 상태 플래그 (리스트로 감싸 클로저에서 변경 가능하게)
    spot_first_step = [True]
    spot_reset_needed = [False]
    base_command = [np.zeros(3, dtype=np.float32)]

    def on_physics_step(step_size):
        """매 물리 스텝마다 호출: Spot 초기화 → 정책 실행 순으로 처리."""
        if spot_first_step[0]:
            spot.initialize()
            spot_first_step[0] = False
        elif spot_reset_needed[0]:
            world.reset(True)
            spot_reset_needed[0] = False
            spot_first_step[0] = True
        else:
            spot.forward(step_size, base_command[0])

    world.add_physics_callback("spot_physics_step", callback_fn=on_physics_step)

    # ── 카메라 설정 ────────────────────────────
    world.play()


    camera_prim_path = resolve_camera_prim_path(stage, PREFERRED_CAMERA_PRIM_PATH)
    print(f"✅ Camera prim: {camera_prim_path}")

    camera = Camera(prim_path=camera_prim_path, name="left_hawk_left_camera",
                    resolution=RESOLUTION, frequency=30)
    camera.initialize()
    simulation_app.update()
    configure_camera(camera)

    # 렌더 프로덕트가 준비될 때까지 스텝 진행
    for _ in range(5):
        world.step(render=True)

    rp_path = None
    for _ in range(60):
        world.step(render=True)
        rp_path = camera.get_render_product_path()
        if rp_path:
            break
        time.sleep(0.05)

    if not rp_path:
        raise RuntimeError("렌더링 프로덕트를 찾을 수 없습니다.")

    create_camera_publishers(rp_path)

    # ── Doosan arm OmniGraph 설정 ──────────────
    # 이전 실행에서 남은 그리퍼 파일 제거
    if os.path.exists(GRIPPER_CMD_FILE):
        os.remove(GRIPPER_CMD_FILE)

    arm_view              = None
    arm_initialized       = False
    gripper_joint_indices = []
    _arm_init_step        = 0

    arm_prim_path = _find_doosan_prim(stage)
    if arm_prim_path:
        # OmniGraph: ROS2 JointState → Articulation 관절 구동
        create_arm_controller_graph(arm_prim_path)

        # 그리퍼는 파일 IPC로 별도 제어 (Float64 단일값 → 부호 반전 적용)
        try:
            from isaacsim.core.prims import Articulation
            arm_view = Articulation(prim_paths_expr=arm_prim_path, name="doosan_arm")
            world.scene.add(arm_view)
            print(f"✅ Arm Articulation 등록 (그리퍼용) → {arm_prim_path}")
        except Exception as e:
            print(f"⚠️ Arm Articulation 등록 실패: {e}")
            arm_view = None

    # ── Growth 타임랩스 비동기 시작 ────────────
    # Isaac Sim의 이벤트 루프에 코루틴 등록 (시뮬레이션과 병렬 실행)
    asyncio.ensure_future(run_timelapse())

    # ── UDP 소켓 초기화 ────────────────────────
    cmd_vel_node = UdpCmdVelReceiver()  # 포트 5005 수신
    state_sender = UdpStateSender()     # 포트 5006 송신
    last_state_time = 0.0

    print("\n🚀 통합 시뮬레이션 실행 중...")

    try:
        while simulation_app.is_running():
            # UDP 버퍼에서 최신 cmd_vel 수신
            cmd_vel_node.spin_once()
            base_command[0] = cmd_vel_node.get_command()

            # 물리 + 렌더 1스텝 진행
            world.step(render=True)

            # ── Arm 물리 초기화 (10프레임 대기 후 1회) ──────────────────────
            if arm_view is not None and not arm_initialized:
                _arm_init_step += 1
                if _arm_init_step >= 10:
                    try:
                        arm_view.initialize()
                        all_names = arm_view.dof_names
                        arm_joint_indices   = [all_names.index(n) for n in ARM_JOINT_NAMES     if n in all_names]
                        gripper_joint_indices = [all_names.index(n) for n in GRIPPER_JOINT_NAMES if n in all_names]
                        if len(arm_joint_indices) == 6:
                            arm_initialized = True
                            print(f"✅ Arm 초기화 완료 arm={arm_joint_indices} gripper={gripper_joint_indices}")
                        else:
                            print(f"⚠️ Arm 관절 인덱스 매칭 실패: {arm_joint_indices} / 전체: {all_names}")
                    except Exception as e:
                        print(f"⚠️ Arm initialize 오류: {e}")

            # ── 그리퍼 명령 적용 (파일 IPC) ─────────────────────────────────
            # arm 관절은 OmniGraph가 처리, 그리퍼만 파일에서 읽어 적용
            if arm_initialized and gripper_joint_indices:
                gcmd = read_gripper_cmd()
                if gcmd is not None:
                    # finger_joint=+val, left_outer_knuckle_joint=-val (부호 반전)
                    arm_view.set_joint_position_targets(
                        np.array([[gcmd, -gcmd]], dtype=np.float32),
                        joint_indices=gripper_joint_indices,
                    )

            # 20ms마다 Spot 상태를 ROS2 브릿지로 UDP 송신
            now = time.monotonic()
            if now - last_state_time > 0.05:
                pos, quat = spot.robot.get_world_pose()
                state_sender.send(pos, quat)
                last_state_time = now

            # 시뮬레이터가 정지(Stop) 상태가 되면 다음 스텝에서 리셋
            if world.is_stopped():
                spot_reset_needed[0] = True

    finally:
        cmd_vel_node.close()
        state_sender.close()
        simulation_app.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()

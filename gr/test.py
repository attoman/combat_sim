import jsbsim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import sin, cos, tan, atan2, sqrt

def llh_to_ecef(lon, lat, alt):
    """
    경도/위도/고도(LLH)를 ECEF 좌표로 변환
    lon, lat: 라디안
    alt: 피트
    """
    # WGS84 타원체 파라미터
    a = 20902231  # 장반경 (피트)
    f = 1/298.257223563  # 편평률
    b = a * (1 - f)  # 단반경
    
    # 위도/경도/고도를 라디안으로 변환
    lon = np.radians(lon)
    lat = np.radians(lat)
    
    # 지구 타원체의 이심률
    e2 = 1 - (b/a)**2
    
    # 위도에 따른 지구 반경
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    
    # ECEF 좌표 계산
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + alt) * np.sin(lat)
    
    return x, y, z

def ecef_to_ned(x, y, z, ref_lon, ref_lat, ref_alt):
    """
    ECEF 좌표를 NED 좌표로 변환
    ref_lon, ref_lat: 라디안
    ref_alt: 피트
    """
    # 참조점의 ECEF 좌표
    ref_x, ref_y, ref_z = llh_to_ecef(ref_lon, ref_lat, ref_alt)
    
    # 참조점에서의 상대 위치
    dx = x - ref_x
    dy = y - ref_y
    dz = z - ref_z
    
    # 회전 행렬 (ECEF to NED)
    sin_lat = np.sin(ref_lat)
    cos_lat = np.cos(ref_lat)
    sin_lon = np.sin(ref_lon)
    cos_lon = np.cos(ref_lon)
    
    # NED 좌표 계산 (D축 부호 수정)
    N = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    E = -sin_lon * dx + cos_lon * dy
    D = -(cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz)  # 부호 수정
    
    return N, E, D

def llh_to_ned(lon, lat, alt, ref_lon, ref_lat, ref_alt):
    """
    경도/위도/고도(LLH)를 NED 프레임으로 변환
    lon, lat, alt: 현재 위치의 경도, 위도, 고도
    ref_lon, ref_lat, ref_alt: 참조 위치의 경도, 위도, 고도
    """
    # ECEF 좌표로 변환
    x, y, z = llh_to_ecef(lon, lat, alt)
    
    # NED 좌표로 변환
    N, E, D = ecef_to_ned(x, y, z, ref_lon, ref_lat, ref_alt)
    
    return N, E, D

# 시뮬레이션 실행 함수
def run_simulation(aircraft_model, commands, duration=60, dt=0.01):
    # JSBSim 초기화
    fdm = jsbsim.FGFDMExec(None)
    fdm.set_aircraft_path(f"{jsbsim.get_default_root_dir()}/aircraft")
    fdm.load_model(aircraft_model)
    
    # 초기 조건 설정
    fdm['ic/h-sl-ft'] = 10000  # 고도 10,000 ft
    fdm['ic/vt-kts'] = 350     # 속도 350 knots
    
    # 시뮬레이션 시작
    fdm.run_ic()
    
    # 데이터 수집 리스트
    time = []
    # Navigation 프레임 (경도/위도)
    position_lon = []  # 경도
    position_lat = []  # 위도
    position_alt = []  # 고도
    
    # 시뮬레이션 루프
    t = 0
    while t < duration:
        # 기동 명령 적용
        for cmd in commands:
            if cmd['time'] <= t:
                fdm[cmd['property']] = cmd['value']
        
        # 시뮬레이션 한 스텝 진행
        fdm.run()
        
        # Navigation 프레임 데이터 수집
        time.append(t)
        position_lon.append(fdm['position/long-gc-deg'])  # 경도
        position_lat.append(fdm['position/lat-geod-deg'])  # 위도
        position_alt.append(fdm['position/h-sl-ft'])      # 고도
        
        t += dt
    
    # 리스트를 NumPy 배열로 변환
    return np.array(time), np.array(position_lon), np.array(position_lat), np.array(position_alt)

def calculate_direction(lon, lat, alt, idx):
    """진행 방향 벡터 계산"""
    if idx == 0:  # 시작점
        dx = lon[1] - lon[0]
        dy = lat[1] - lat[0]
        dz = alt[1] - alt[0]
    else:  # 종점
        dx = lon[-1] - lon[-2]
        dy = lat[-1] - lat[-2]
        dz = alt[-1] - alt[-2]
    
    # 방향 벡터 정규화
    norm = np.sqrt(dx*dx + dy*dy + dz*dz)
    return dx/norm, dy/norm, dz/norm

def plot_trajectories(time, lon, lat, alt, aircraft_model):
    # 서브플롯 생성
    fig = plt.figure(figsize=(20, 8))
    
    # Navigation 프레임 플롯
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(lon, lat, alt)  # lat와 lon의 순서를 바꿈
    
    # 시작점과 종점을 원으로 표시
    ax1.scatter(lon[0], lat[0], alt[0], c='g', s=100)  # 시작점 (초록색)
    ax1.scatter(lon[-1], lat[-1], alt[-1], c='r', s=100)  # 종점 (빨간색)
    
    ax1.set_ylabel('Latitude (deg)')
    ax1.set_xlabel('Longitude (deg)')
    ax1.set_zlabel('Altitude (ft)')
    ax1.set_title(f'{aircraft_model} Trajectory (Navigation Frame)')
    ax1.grid(True)
    ax1.view_init(elev=20, azim=45)
    
    # NED 프레임 플롯
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 초기 위치를 참조점으로 사용
    ref_lon = lon[0]
    ref_lat = lat[0]
    ref_alt = alt[0]
    
    # NED 좌표로 변환
    N, E, D = llh_to_ned(lon, lat, alt, ref_lon, ref_lat, ref_alt)
    
    # NED 프레임 플롯 (D축 부호 수정)
    ax2.plot(E, N, -D)  # D축 부호 수정
    
    # 시작점과 종점을 원으로 표시
    ax2.scatter(E[0], N[0], -D[0], c='g', s=100)  # 시작점 (초록색)
    ax2.scatter(E[-1], N[-1], -D[-1], c='r', s=100)  # 종점 (빨간색)
    
    ax2.set_xlabel('East (ft)')
    ax2.set_ylabel('North (ft)')
    ax2.set_zlabel('Up (ft)')
    ax2.set_title(f'{aircraft_model} Trajectory (NED Frame)')
    ax2.grid(True)
    ax2.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(f'./{aircraft_model}_trajectories.png')
    plt.close()

# F-16 시뮬레이션
commands_f16 = [
    {'time': 0, 'property': 'fcs/elevator-cmd-norm', 'value': 0.5},   # 엘리베이터 50% 올림
    {'time': 10, 'property': 'fcs/aileron-cmd-norm', 'value': 0.3},   # 에일러론 30% 우측
    {'time': 20, 'property': 'fcs/rudder-cmd-norm', 'value': -0.2},   # 러더 20% 좌측
    {'time': 30, 'property': 'fcs/elevator-cmd-norm', 'value': 0.0},  # 엘리베이터 중립
]
time_f16, lon_f16, lat_f16, alt_f16 = run_simulation('f16', commands_f16)
plot_trajectories(time_f16, lon_f16, lat_f16, alt_f16, 'F-16')

# F-22 시뮬레이션
commands_f22 = [
    {'time': 0, 'property': 'fcs/elevator-cmd-norm', 'value': 0.4},   # 엘리베이터 40% 올림
    {'time': 15, 'property': 'fcs/aileron-cmd-norm', 'value': -0.4},  # 에일러론 40% 좌측
    {'time': 25, 'property': 'fcs/rudder-cmd-norm', 'value': 0.3},    # 러더 30% 우측
    {'time': 35, 'property': 'fcs/elevator-cmd-norm', 'value': -0.2}, # 엘리베이터 20% 내림
]
time_f22, lon_f22, lat_f22, alt_f22 = run_simulation('f22', commands_f22)
plot_trajectories(time_f22, lon_f22, lat_f22, alt_f22, 'F-22')
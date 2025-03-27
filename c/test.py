import jsbsim
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import traceback

# 저장할 디렉토리 설정 (필요 시 사용)
script_dir = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(script_dir, "output")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)

dt = 0.1         # 시간 스텝 (초)
sim_time = 60    # 전체 시뮬레이션 시간 (초)
step_mode = True  # True이면 각 스텝마다 사용자 입력으로 진행

class AircraftAgent:
    def __init__(self, model_name, lat, lon, alt, u_fps, name):
        self.name = name
        # headless 모드: None 인자 사용
        self.fdm = jsbsim.FGFDMExec(None)
        self.fdm.set_dt(dt)
        self.fdm.load_model(model_name)
        # 초기 조건 설정
        self.fdm.set_property_value("ic/lat-gc-deg", lat)
        self.fdm.set_property_value("ic/long-gc-deg", lon)
        self.fdm.set_property_value("ic/alt-ft", alt)
        self.fdm.set_property_value("ic/u-fps", u_fps)
        self.fdm.run_ic()
        # 초기 제어 입력 설정 (엔진, 에일러론, 엘리베이터, 러더)
        self.fdm.set_property_value("fcs/throttle-cmd-norm", 0.8)
        self.fdm.set_property_value("fcs/aileron-cmd-norm", 0.0)
        self.fdm.set_property_value("fcs/elevator-cmd-norm", 0.0)
        self.fdm.set_property_value("fcs/rudder-cmd-norm", 0.0)

    def update(self):
        self.fdm.run()

    def get_position(self):
        lat = self.fdm.get_property_value("position/lat-gc-deg")
        lon = self.fdm.get_property_value("position/long-gc-deg")
        alt = self.fdm.get_property_value("position/alt-ft")
        return lat, lon, alt

    def set_heading_command(self, heading):
        # 현재 heading(psi-deg)을 가져오고 목표 heading과의 차이를 계산
        current_heading = self.fdm.get_property_value("attitude/psi-deg")
        heading_diff = (heading - current_heading + 180) % 360 - 180
        if abs(heading_diff) > 5:
            roll_cmd = np.clip(heading_diff * 0.5, -45, 45)
            self.fdm.set_property_value("fcs/roll-cmd-deg", roll_cmd)
            pitch_cmd = np.clip(heading_diff * 0.2, -20, 20)
            self.fdm.set_property_value("fcs/pitch-cmd-deg", pitch_cmd)
            aileron_cmd = np.clip(heading_diff * 0.1, -1, 1)
            self.fdm.set_property_value("fcs/aileron-cmd-norm", aileron_cmd)
            rudder_cmd = np.clip(heading_diff * 0.05, -1, 1)
            self.fdm.set_property_value("fcs/rudder-cmd-norm", rudder_cmd)
            throttle_cmd = 0.8 + abs(heading_diff) * 0.001
            self.fdm.set_property_value("fcs/throttle-cmd-norm", np.clip(throttle_cmd, 0.5, 1.0))
        else:
            # 목표와 차이가 작으면 기본 제어 입력 유지
            self.fdm.set_property_value("fcs/roll-cmd-deg", 0)
            self.fdm.set_property_value("fcs/pitch-cmd-deg", 0)
            self.fdm.set_property_value("fcs/aileron-cmd-norm", 0)
            self.fdm.set_property_value("fcs/rudder-cmd-norm", 0)
            self.fdm.set_property_value("fcs/throttle-cmd-norm", 0.8)

def compute_bearing(lat1, lon1, lat2, lon2):
    dlon = math.radians(lon2 - lon1)
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360

def main():
    # 두 에이전트 생성: F-16 (추격기)와 F-22 (회피기)
    agent1 = AircraftAgent(model_name="f16", lat=37.0, lon=-122.0, alt=30000, u_fps=800, name="Pursuer")
    agent2 = AircraftAgent(model_name="f22", lat=37.01, lon=-122.01, alt=30000, u_fps=850, name="Evader")
    agents = [agent1, agent2]
    
    num_steps = int(sim_time / dt)
    times = np.linspace(0, sim_time, num_steps)
    
    # 각 항공기의 궤적 데이터를 저장 (경도, 위도, 고도)
    pos1 = []  # Pursuer
    pos2 = []  # Evader
    
    print("시뮬레이션을 step-by-step으로 진행합니다.")
    for i, t in enumerate(times):
        # 각 에이전트의 시뮬레이션 스텝 실행
        for agent in agents:
            agent.update()
        
        # 현재 위치 저장
        lat1, lon1, alt1 = agent1.get_position()
        lat2, lon2, alt2 = agent2.get_position()
        pos1.append((lon1, lat1, alt1))
        pos2.append((lon2, lat2, alt2))
        
        # Pursuer는 Evader를 향하도록 heading 명령 설정
        heading_to_evader = compute_bearing(lat1, lon1, lat2, lon2)
        agent1.set_heading_command(heading_to_evader)
        # Evader는 Pursuer로부터 도망가도록 약간의 랜덤 오프셋 적용
        heading_from_pursuer = (compute_bearing(lat2, lon2, lat1, lon1) + 180) % 360
        random_offset = np.random.normal(0, 15)
        evader_heading = (heading_from_pursuer + random_offset) % 360
        agent2.set_heading_command(evader_heading)
        
        print(f"Step {i+1}/{num_steps}, Time: {t:.1f}s")
        print(f"  Pursuer: lat {lat1:.4f}, lon {lon1:.4f}, alt {alt1:.1f}, heading cmd {heading_to_evader:.2f}°")
        print(f"  Evader:  lat {lat2:.4f}, lon {lon2:.4f}, alt {alt2:.1f}, heading cmd {evader_heading:.2f}°")
        
        if step_mode:
            user_input = input("다음 스텝 진행하려면 Enter, 종료하려면 'q' 입력: ")
            if user_input.lower() == 'q':
                break
    
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    
    # 3D 궤적 플롯 생성 및 표시
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos1[:, 0], pos1[:, 1], pos1[:, 2], 'bo-', label="Pursuer (F-16)", linewidth=2)
    ax.plot(pos2[:, 0], pos2[:, 1], pos2[:, 2], 'ro-', label="Evader (F-22)", linewidth=2)
    ax.scatter(pos1[-1, 0], pos1[-1, 1], pos1[-1, 2], c='blue', s=100, marker='o')
    ax.scatter(pos2[-1, 0], pos2[-1, 1], pos2[-1, 2], c='red', s=100, marker='o')
    ax.set_xlabel("경도 (deg)")
    ax.set_ylabel("위도 (deg)")
    ax.set_zlabel("고도 (ft)")
    ax.set_title("3D 전체 궤적")
    ax.legend()
    
    # 두 에이전트의 데이터 모두를 고려하여 축 범위 설정
    all_lon = np.concatenate((pos1[:,0], pos2[:,0]))
    all_lat = np.concatenate((pos1[:,1], pos2[:,1]))
    all_alt = np.concatenate((pos1[:,2], pos2[:,2]))
    margin_lon = 0.01
    margin_lat = 0.01
    margin_alt = 1000
    ax.set_xlim(all_lon.min()-margin_lon, all_lon.max()+margin_lon)
    ax.set_ylim(all_lat.min()-margin_lat, all_lat.max()+margin_lat)
    ax.set_zlim(all_alt.min()-margin_alt, all_alt.max()+margin_alt)
    
    plt.savefig("./3d_trajectory.png")

if __name__ == '__main__':
    main()

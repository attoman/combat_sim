import jsbsim
import time
import os

# --- 설정 ---
# JSBSim 루트 디렉토리 경로를 지정하세요. (JSBSim 설치 경로에 맞게 수정)
# 예: Linux -> '/usr/share/jsbsim', Windows -> 'C:/Program Files/JSBSim'
# 환경 변수 JSBSIM_ROOT_DIR를 사용하거나 직접 경로를 지정하세요.
jsbsim_root_dir = os.environ.get('JSBSIM_ROOT_DIR', '/usr/share/jsbsim') # <--- 이 경로를 확인하고 수정하세요!

# 시뮬레이션 시간 간격 (초) - JSBSim 내부 스텝과 별개로 Python 루프 주기
dt = 1/60.0

# 시뮬레이션 총 시간 (초)
simulation_time_sec = 120

# --- JSBSim 인스턴스 생성 ---
print(f"JSBSim 루트 디렉토리: {jsbsim_root_dir}")
fdm = jsbsim.FGFDMExec(root_dir=jsbsim_root_dir)

# --- 항공기 모델 로드 ---
print("항공기 모델 로딩: f16")
fdm.load_model('f16')

# --- 초기 조건 설정 ---
# 활주로 출발 위치 (예: KSQL Runway 30 근처, 필요시 공항/활주로 변경)
initial_latitude_deg = 37.5133  # 위도 (도)
initial_longitude_deg = -122.2517 # 경도 (도)
initial_altitude_ft = 10       # 고도 (피트)
initial_heading_deg = 300      # 기수 방향 (도) - 활주로 방향과 일치
initial_speed_kts = 0          # 초기 속도 (노트)

print("초기 조건 설정 중...")
# 위치 및 고도
fdm.set_property_value('ic/lat-gc-deg', initial_latitude_deg)
fdm.set_property_value('ic/long-gc-deg', initial_longitude_deg)
fdm.set_property_value('ic/h-sl-ft', initial_altitude_ft)
# 방향 및 속도
fdm.set_property_value('ic/psi-true-deg', initial_heading_deg)
fdm.set_property_value('ic/vt-kts', initial_speed_kts) # 초기 속도 설정
fdm.set_property_value('ic/ubody-fps', initial_speed_kts * 1.68781) # u-body speed (ft/s)
# 엔진 시동
fdm.set_property_value('propulsion/engine[0]/set-running', 1)
# 연료 (예: 50%)
fdm.set_property_value('propulsion/total-fuel-lbs', 5000) # 파운드 단위

# 초기 조건 적용 (필수)
print("초기 조건 적용...")
result_ic = fdm.run_ic()
if not result_ic:
    print("오류: 초기 조건(IC) 실행 실패!")
    exit(-1)

# 초기 스로틀 및 제어면 설정
fdm.set_property_value('fcs/throttle-cmd-norm', 0) # 스로틀 0%
fdm.set_property_value('fcs/elevator-cmd-norm', 0) # 엘리베이터 중립
fdm.set_property_value('fcs/aileron-cmd-norm', 0)  # 에일러론 중립
fdm.set_property_value('fcs/rudder-cmd-norm', 0)   # 러더 중립
fdm.set_property_value('gear/gear-cmd-norm', 1.0) # 랜딩 기어 내림 (1.0: Down, 0.0: Up)

# --- 시뮬레이션 루프 ---
print("시뮬레이션 시작!")
start_time = time.time()
sim_time = 0.0

takeoff_speed_kts = 150 # 이륙 결정 속도 (노트)
airborne = False
gear_retracted = False

while sim_time < simulation_time_sec:
    # --- JSBSim 실행 ---
    result_run = fdm.run()
    if not result_run:
        print("오류: JSBSim 실행 실패!")
        break

    # --- 현재 상태 읽기 ---
    sim_time = fdm.get_property_value('simulation/sim-time-sec')
    altitude_ft = fdm.get_property_value('position/h-sl-ft')
    speed_kts = fdm.get_property_value('velocities/vt-kts')
    pitch_deg = fdm.get_property_value('attitude/pitch-deg')
    roll_deg = fdm.get_property_value('attitude/roll-deg')
    heading_deg = fdm.get_property_value('attitude/psi-deg')

    # --- 제어 로직 (간단한 이륙 및 상승) ---
    if not airborne:
        # 이륙 전
        if sim_time < 2.0: # 초기 안정화 시간
            fdm.set_property_value('fcs/throttle-cmd-norm', 0.5) # 약간의 스로틀
        else:
            fdm.set_property_value('fcs/throttle-cmd-norm', 1.0) # 최대 스로틀 (Afterburner는 별도 제어 필요)

        # 이륙 속도 도달 시 엘리베이터 조작 (기수 들기)
        if speed_kts > takeoff_speed_kts:
            print(f"시간: {sim_time:.2f}s - 이륙 속도 도달 ({speed_kts:.1f} kts), 기수 들기 시도...")
            fdm.set_property_value('fcs/elevator-cmd-norm', -0.3) # 엘리베이터 당김 (값은 조정 필요)

        # 이륙 확인 (고도 기준)
        if altitude_ft > initial_altitude_ft + 30: # 지상 고도 + 30피트 이상이면 이륙으로 간주
            print(f"시간: {sim_time:.2f}s - 이륙 확인! 고도: {altitude_ft:.1f} ft")
            airborne = True
            # 이륙 후 엘리베이터 약간 완화 (안정적인 상승 각도 유지 위해)
            fdm.set_property_value('fcs/elevator-cmd-norm', -0.1) # 값은 조정 필요
    else:
        # 이륙 후
        # 랜딩 기어 접기 (한 번만 실행)
        if not gear_retracted and altitude_ft > initial_altitude_ft + 100:
            print(f"시간: {sim_time:.2f}s - 랜딩 기어 접기")
            fdm.set_property_value('gear/gear-cmd-norm', 0.0)
            gear_retracted = True

        # (선택 사항) 특정 고도/속도 도달 시 스로틀 조정 등 추가 제어 로직
        # if altitude_ft > 5000 and speed_kts > 300:
        #     fdm.set_property_value('fcs/throttle-cmd-norm', 0.85) # 순항 스로틀

    # --- 상태 출력 (매 초 마다) ---
    if int(sim_time * 10) % 10 == 0: # 0.1초 간격으로 출력 (너무 잦으면 조절)
        print(f"T: {sim_time:.1f}s | Alt: {altitude_ft:.1f}ft | Spd: {speed_kts:.1f}kts | Pitch: {pitch_deg:.1f}° | Roll: {roll_deg:.1f}° | Hdg: {heading_deg:.1f}°")

    # --- 루프 주기 맞추기 ---
    # 실제 경과 시간 고려하여 sleep (간단하게 고정 시간 사용)
    time.sleep(dt)

# --- 시뮬레이션 종료 ---
print("시뮬레이션 종료.")
elapsed_real_time = time.time() - start_time
print(f"총 시뮬레이션 시간: {sim_time:.2f} 초")
print(f"실제 소요 시간: {elapsed_real_time:.2f} 초")
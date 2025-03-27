import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.linalg     # For LQR
import scipy.optimize   # For root finding (trim)
import torch            # RL 필요시 주석 해제
import torch.nn as nn   # RL 필요시 주석 해제
import matplotlib.pyplot as plt
import json
import os
import math
from copy import deepcopy
from unittest.mock import MagicMock
from datetime import datetime

# --- 상수 ---
G = 9.80665 # 중력 가속도 (m/s^2)
RHO0 = 1.225 # 해수면 표준 공기 밀도 (kg/m^3)

# --- 1. 기본 유틸리티 함수 ---
def _normalize_angle(angle_rad):
    """ 각도를 -pi ~ +pi 범위로 정규화 (라디안) """
    while angle_rad > np.pi: angle_rad -= 2 * np.pi
    while angle_rad < -np.pi: angle_rad += 2 * np.pi
    return angle_rad

# --- 2. 물리 계산 함수 ---
def get_isa_conditions(altitude_m):
    """ 고도(m) -> ISA 조건 (온도, 압력, 밀도, 음속) 반환 """
    T0=288.15; P0=101325; R_gas=287.05; GAMMA=1.4; LAPSE_RATE=-0.0065; H_TROPOPAUSE=11000.0
    h = max(0, float(altitude_m))
    if h <= H_TROPOPAUSE: temp=T0+LAPSE_RATE*h; pressure=P0*(temp/T0)**(-G/(LAPSE_RATE*R_gas))
    else: temp_trop=T0+LAPSE_RATE*H_TROPOPAUSE; press_trop=P0*(temp_trop/T0)**(-G/(LAPSE_RATE*R_gas)); temp=temp_trop; pressure=press_trop*np.exp(-G*(h-H_TROPOPAUSE)/(R_gas*temp))
    density=pressure/(R_gas*temp); sound_speed=np.sqrt(GAMMA*R_gas*temp)
    return temp, pressure, density, sound_speed

# --- 3. 좌표 변환 함수 ---
def state_body_to_ned(vec_body, euler_angles_rad):
    """ Body frame vector -> NED frame vector """
    phi, theta, psi = euler_angles_rad; cph, sph=np.cos(phi),np.sin(phi); cth, sth=np.cos(theta),np.sin(theta); cps, sps=np.cos(psi),np.sin(psi)
    C_b_n = np.array([[cth*cps, sph*sth*cps-cph*sps, cph*sth*cps+sph*sps],[cth*sps, sph*sth*sps+cph*cps, cph*sth*sps-sph*cps],[-sth, sph*cth, cph*cth]]); return C_b_n @ vec_body

def state_ned_to_body(vec_ned, euler_angles_rad):
    """ NED frame vector -> Body frame vector """
    phi, theta, psi = euler_angles_rad; cph, sph=np.cos(phi),np.sin(phi); cth, sth=np.cos(theta),np.sin(theta); cps, sps=np.cos(psi),np.sin(psi)
    C_n_b = np.array([[cth*cps, cth*sps, -sth],[sph*sth*cps-cph*sps, sph*sth*sps+cph*cps, sph*cth],[cph*sth*cps+sph*sps, cph*sth*sps-sph*cps, cph*cth]]); return C_n_b @ vec_ned

# --- 4. 기본 BFM 제어 함수 ---

# --- BFM 상태 정의 ---
STATE_NEUTRAL = 0
STATE_OFFENSIVE = 1
STATE_DEFENSIVE = 2
STATE_RECOVERY = 3

# 지면 충돌 방지 임계값
ALTITUDE_CRITICAL = 500.0
ALTITUDE_RECOVERED = 800.0

def calculate_bfm_parameters(self_state, opponent_state):
    """ 두 항공기 상태 기반 BFM 파라미터 계산 """
    params = {'aot_deg': 180.0, 'aspect_deg': 180.0, 'range_m': 99999.0, 'closure_rate_mps': 0.0, 'delta_alt_m': 0.0}
    self_pos=self_state[0:3]; opp_pos=opponent_state[0:3]; self_vel_b=self_state[3:6]; opp_vel_b=opponent_state[3:6]; self_euler=self_state[6:9]; opp_euler=opponent_state[6:9]
    los_vec_ned = opp_pos - self_pos; params['range_m'] = np.linalg.norm(los_vec_ned)
    if params['range_m'] < 1e-6: return params
    los_vec_ned_unit = los_vec_ned / params['range_m']
    self_vel_n = state_body_to_ned(self_vel_b, self_euler); opp_vel_n = state_body_to_ned(opp_vel_b, opp_euler)
    rel_vel_ned = opp_vel_n - self_vel_n; params['closure_rate_mps'] = -np.dot(rel_vel_ned, los_vec_ned_unit); params['delta_alt_m'] = -self_state[2] - (-opponent_state[2])
    opp_fwd_vec_ned = state_body_to_ned(np.array([1,0,0]), opp_euler); opp_tail_vec_ned = -opp_fwd_vec_ned; dot_aot = np.dot(opp_tail_vec_ned, los_vec_ned); norm_tail_o = np.linalg.norm(opp_tail_vec_ned)
    if norm_tail_o > 1e-6: cos_aot = np.clip(dot_aot / (norm_tail_o * params['range_m']), -1.0, 1.0); params['aot_deg'] = np.rad2deg(np.arccos(cos_aot))
    self_fwd_vec_ned = state_body_to_ned(np.array([1,0,0]), self_euler); self_tail_vec_ned = -self_fwd_vec_ned; los_vec_ned_reversed = -los_vec_ned; dot_aspect = np.dot(self_tail_vec_ned, los_vec_ned_reversed); norm_tail_s = np.linalg.norm(self_tail_vec_ned)
    if norm_tail_s > 1e-6: cos_aspect = np.clip(dot_aspect / (norm_tail_s * params['range_m']), -1.0, 1.0); params['aspect_deg'] = np.rad2deg(np.arccos(cos_aspect))
    return params

def opponent_is_threat(bfm_params, aspect_threshold_deg, range_threshold_m):
    aspect = bfm_params.get('aspect_deg', 180.0); rng = bfm_params.get('range_m', 99999.0); return aspect < aspect_threshold_deg and rng < range_threshold_m

def i_am_offensive(bfm_params, aot_threshold_deg, range_threshold_m):
    aot = bfm_params.get('aot_deg', 180.0); rng = bfm_params.get('range_m', 99999.0); return aot < aot_threshold_deg and rng < range_threshold_m

def set_neutral_targets(aircraft_model, opponent_state, bfm_params):
    target_state_dict = {}; self_state = aircraft_model.state; vector_to_opponent_body = state_ned_to_body(opponent_state[0:3] - self_state[0:3], self_state[6:9]); norm_los_b = np.linalg.norm(vector_to_opponent_body)
    if norm_los_b > 1e-6: lat_err_norm = vector_to_opponent_body[1] / norm_los_b; target_state_dict['roll'] = np.clip(-lat_err_norm * np.pi / 4, -np.deg2rad(45), np.deg2rad(45))
    target_state_dict['airspeed'] = np.linalg.norm(self_state[3:6]); target_state_dict['altitude'] = -self_state[2]
    aircraft_model.bfm_target_state_selected = target_state_dict; controls = aircraft_model.get_control_inputs(0, self_state, target_state_dict); return controls

def set_offensive_targets(aircraft_model, opponent_state, bfm_params):
    target_state_dict = {}; self_state = aircraft_model.state; vector_to_opponent_body = state_ned_to_body(opponent_state[0:3] - self_state[0:3], self_state[6:9]); norm_los_b = np.linalg.norm(vector_to_opponent_body)
    if norm_los_b > 1e-6: lat_err_norm = vector_to_opponent_body[1] / norm_los_b; target_state_dict['roll'] = np.clip(-lat_err_norm * np.pi / 2, -np.deg2rad(70), np.deg2rad(70))
    target_state_dict['airspeed'] = np.linalg.norm(self_state[3:6]) * 1.05
    aircraft_model.bfm_target_state_selected = target_state_dict; controls = aircraft_model.get_control_inputs(0, self_state, target_state_dict); return controls

def set_defensive_targets(aircraft_model, opponent_state, bfm_params):
    target_state_dict = {}; self_state = aircraft_model.state; vector_to_opponent_body = state_ned_to_body(opponent_state[0:3] - self_state[0:3], self_state[6:9]); norm_los_b = np.linalg.norm(vector_to_opponent_body)
    if norm_los_b > 1e-6: lat_err_norm = vector_to_opponent_body[1] / norm_los_b; target_state_dict['roll'] = np.sign(-lat_err_norm) * np.deg2rad(100)
    else: target_state_dict['roll'] = self_state[6]
    target_state_dict['pitch_rate'] = np.sign(self_state[7]) * np.deg2rad(30) if abs(self_state[7]) > 0.1 else np.deg2rad(30)
    target_state_dict['throttle'] = 1.0
    aircraft_model.bfm_target_state_selected = target_state_dict; controls = aircraft_model.get_control_inputs(0, self_state, target_state_dict); controls['throttle'] = 1.0; return controls

def set_recovery_targets(aircraft_model):
    target_state_dict = {'roll': 0.0, 'pitch': np.deg2rad(10.0), 'throttle': 1.0}
    aircraft_model.bfm_target_state_selected = target_state_dict; controls = aircraft_model.get_control_inputs(0, aircraft_model.state, target_state_dict); controls['throttle'] = 1.0; return controls

def get_state_machine_bfm_controls(agent, opponent_state, current_bfm_state):
    """ 상태 머신 기반 BFM 제어 로직 (Agent 객체 및 Shared SA 사용) """
    current_state = agent.state; aircraft_id = agent.id; current_alt = -current_state[2]
    next_bfm_state = current_bfm_state; state_changed = False
    aircraft_model = agent.model

    # 1. 최우선 순위: 지면 충돌 회피
    if current_alt < ALTITUDE_CRITICAL:
        if current_bfm_state != STATE_RECOVERY: state_changed = True
        next_bfm_state = STATE_RECOVERY
    elif current_bfm_state == STATE_RECOVERY and current_alt > ALTITUDE_RECOVERED:
        next_bfm_state = STATE_NEUTRAL; state_changed = True
    elif current_bfm_state != STATE_RECOVERY:
        # === Shared SA 기반 판단 ===
        is_self_threatened_by_shared_SA = False
        for detection in agent.shared_enemy_detections:
             threat_state = detection['state_observed']
             threat_params = calculate_bfm_parameters(current_state, threat_state)
             if opponent_is_threat(threat_params, 60, 5000):
                 is_self_threatened_by_shared_SA = True
                 break

        bfm_params = calculate_bfm_parameters(current_state, opponent_state) if opponent_state is not None else {}
        is_threatened = opponent_is_threat(bfm_params, 70, 12000) or is_self_threatened_by_shared_SA
        is_offensive = i_am_offensive(bfm_params, 60, 18000) if opponent_state is not None else False

        threat_escape_range=12000+6000; threat_escape_aspect=70+50
        adv_lose_range=18000+3000; adv_lose_aot=60+30
        prev_state = current_bfm_state
        if current_bfm_state == STATE_NEUTRAL:
            if is_offensive: next_bfm_state = STATE_OFFENSIVE
            elif is_threatened: next_bfm_state = STATE_DEFENSIVE
        elif current_bfm_state == STATE_OFFENSIVE:
            if not i_am_offensive(bfm_params, adv_lose_aot, adv_lose_range): next_bfm_state = STATE_NEUTRAL
            elif is_threatened: next_bfm_state = STATE_DEFENSIVE
        elif current_bfm_state == STATE_DEFENSIVE:
            if not opponent_is_threat(bfm_params, threat_escape_aspect, threat_escape_range) and not is_self_threatened_by_shared_SA:
                if is_offensive: next_bfm_state = STATE_OFFENSIVE
                else: next_bfm_state = STATE_NEUTRAL
        if prev_state != next_bfm_state: state_changed = True

    if next_bfm_state == STATE_RECOVERY: controls = set_recovery_targets(aircraft_model)
    elif next_bfm_state == STATE_OFFENSIVE and opponent_state is not None: controls = set_offensive_targets(aircraft_model, opponent_state, bfm_params)
    elif next_bfm_state == STATE_DEFENSIVE and opponent_state is not None: controls = set_defensive_targets(aircraft_model, opponent_state, bfm_params)
    else:
         opponent_state_for_neutral = opponent_state if opponent_state is not None else current_state
         controls = set_neutral_targets(aircraft_model, opponent_state_for_neutral, bfm_params if opponent_state is not None else {})

    return controls, next_bfm_state

# --- 5. BFM 상태별 제어 함수 ---
def execute_neutral_controls(self_state, opponent_state, bfm_params):
    """ 중립 상태: 기본적인 추적 """
    current_alt = -self_state[2]; max_g = 7.0 if current_alt > 1500 else max(1.5, 7.0 * (current_alt / 1500)**0.5)
    controls = get_simple_bfm_controls(self_state, opponent_state, target_speed_kts=450, max_g=max_g); return controls

def execute_offensive_controls(self_state, opponent_state, bfm_params):
    """ 공격 상태: Lag Pursuit 로직으로 추적 """
    # 1. 기본 제어 한계 및 G-Load 설정
    MAX_ELEVATOR_RAD = np.deg2rad(25.0)
    MAX_AILERON_RAD = np.deg2rad(21.5)
    MAX_RUDDER_RAD = np.deg2rad(30.0)
    current_alt = -self_state[2]
    max_g = 8.0 if current_alt > 1500 else max(1.5, 8.0 * (current_alt / 1500)**0.5)

    # 2. 상태 벡터 추출
    self_pos = self_state[0:3]
    self_vel_body = self_state[3:6]
    self_euler = self_state[6:9]
    self_rates = self_state[9:12]
    V_self = np.linalg.norm(self_vel_body)
    
    opp_pos = opponent_state[0:3]
    opp_vel_body = opponent_state[3:6]
    opp_euler = opponent_state[6:9]
    opp_vel_ned = state_body_to_ned(opp_vel_body, opp_euler)
    V_opp = np.linalg.norm(opp_vel_body)

    # 3. Lag Pursuit 계산
    # 3.1 상대 위치 및 속도 계산
    vector_to_opp = opp_pos - self_pos
    R = np.linalg.norm(vector_to_opp)
    if R < 100: # 근접 시 안전 조치
        return {'throttle': 0.5, 'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0}

    # 3.2 Lag Angle 계산 (상대 속도에 따라 조정)
    # 속도비에 따라 Lag Angle 조정 (더 빠를수록 더 큰 Lag Angle)
    velocity_ratio = V_self / max(V_opp, 1.0)
    base_lag_angle = np.deg2rad(30)  # 기본 30도
    lag_angle = base_lag_angle * min(velocity_ratio, 1.5)  # 최대 45도까지

    # 3.3 목표 조준점 계산 (상대방 위치에서 Lag Angle만큼 뒤)
    opp_heading = math.atan2(opp_vel_ned[1], opp_vel_ned[0])
    lag_point = opp_pos - np.array([
        R * 0.3 * math.cos(opp_heading),
        R * 0.3 * math.sin(opp_heading),
        0  # 고도는 유지
    ])

    # 4. 제어 입력 계산
    # 4.1 Lag Point로의 방향 벡터 (Body Frame)
    vector_to_lag_ned = lag_point - self_pos
    vector_to_lag_body = state_ned_to_body(vector_to_lag_ned, self_euler)
    norm_lag_b = np.linalg.norm(vector_to_lag_body)
    if norm_lag_b < 1e-6:
        return {'throttle': 0.5, 'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0}
    
    # 4.2 Roll 제어 (Lag Point를 향해)
    lat_err_norm = vector_to_lag_body[1] / norm_lag_b
    target_roll = np.clip(-lat_err_norm * np.pi * 0.8, -np.deg2rad(60), np.deg2rad(60))
    roll_error = _normalize_angle(target_roll - self_euler[0])
    target_p = np.clip(roll_error * 2.5, -np.deg2rad(120), np.deg2rad(120))
    p_error = target_p - self_rates[0]
    aileron_cmd = np.clip(p_error * 0.15, -MAX_AILERON_RAD, MAX_AILERON_RAD)

    # 4.3 Pitch 제어 (G-Load 제한 적용)
    vert_err_norm = vector_to_lag_body[2] / norm_lag_b
    target_q = np.clip(vert_err_norm * 1.2, -np.deg2rad(30), np.deg2rad(30))
    q_error = target_q - self_rates[1]
    load_factor = np.sqrt(1 + (self_rates[1]*V_self/G)**2) if V_self > 1 else 1
    g_allowance = max(0, (max_g - load_factor) / max_g)
    elevator_cmd = np.clip(q_error * 0.3 * g_allowance, -MAX_ELEVATOR_RAD, MAX_ELEVATOR_RAD)

    # 4.4 속도 제어 (상대방보다 약간 빠르게)
    target_speed = min(V_opp * 1.1, 480 * 0.5144)  # 최대 480kts
    speed_error = (target_speed - V_self) * 0.01
    throttle_cmd = np.clip(0.5 + speed_error, 0.1, 1.0)

    # 4.5 Yaw 제어 (Beta 최소화)
    beta = math.asin(np.clip(self_vel_body[1]/V_self, -1, 1)) if V_self > 1e-3 else 0
    beta_error = 0.0 - beta
    rudder_cmd = np.clip(beta_error * 0.5, -MAX_RUDDER_RAD, MAX_RUDDER_RAD)

    controls = {
        'throttle': throttle_cmd,
        'elevator': elevator_cmd,
        'aileron': aileron_cmd,
        'rudder': rudder_cmd
    }
    return controls

def execute_defensive_controls(self_state, opponent_state, bfm_params):
    """ 방어 상태: 최대 성능 선회 """
    MAX_ELEVATOR_RAD=np.deg2rad(25.0); MAX_AILERON_RAD=np.deg2rad(21.5); MAX_RUDDER_RAD=np.deg2rad(30.0); current_alt = -self_state[2]
    max_g_allowed = 9.0 if current_alt > 1000 else max(1.5, 9.0 * (current_alt / 1000)**0.5)
    vector_to_opponent_body = state_ned_to_body(opponent_state[0:3] - self_state[0:3], self_state[6:9])
    norm_los_b = np.linalg.norm(vector_to_opponent_body)
    if norm_los_b < 1e-6: return {'throttle': 1.0, 'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0} # Avoid division by zero
    lat_err_norm = vector_to_opponent_body[1] / norm_los_b
    target_roll = np.sign(-lat_err_norm) * np.deg2rad(100); roll_error = _normalize_angle(target_roll - self_state[6])
    target_p = np.clip(roll_error*3.0, -np.deg2rad(150), np.deg2rad(150)); p_error = target_p - self_state[9]; aileron_cmd = np.clip(p_error*0.15, -MAX_AILERON_RAD, MAX_AILERON_RAD)
    elevator_cmd = -MAX_ELEVATOR_RAD * (max_g_allowed / 9.0) # Max pull attempt, limited by allowed G
    throttle_cmd = 1.0; rudder_cmd = np.clip(-lat_err_norm * 0.1, -MAX_RUDDER_RAD, MAX_RUDDER_RAD)
    controls = {'throttle': throttle_cmd, 'elevator': elevator_cmd, 'aileron': aileron_cmd, 'rudder': rudder_cmd}; return controls

def execute_recovery_controls(self_state):
    """ 회복 상태: 지면 충돌 회피 """
    MAX_ELEVATOR_RAD=np.deg2rad(25.0); MAX_AILERON_RAD=np.deg2rad(21.5); MAX_RUDDER_RAD=np.deg2rad(30.0)
    phi, theta, psi = self_state[6:9]; p, q, r = self_state[9:12]; V = np.linalg.norm(self_state[3:6])
    target_roll = 0.0; roll_error = _normalize_angle(target_roll - phi); target_p = np.clip(roll_error * 2.5, -np.deg2rad(120), np.deg2rad(120)); p_error = target_p - p; aileron_cmd = np.clip(p_error * 0.1, -MAX_AILERON_RAD, MAX_AILERON_RAD)
    target_pitch = np.deg2rad(10.0); pitch_error = target_pitch - theta; target_q = np.clip(pitch_error * 1.0, -np.deg2rad(20), np.deg2rad(20)); q_error = target_q - q
    load_factor = np.sqrt(1 + (q*V/G)**2) if V > 1 else 1; max_g_recovery = 4.0; g_allowance = max(0, (max_g_recovery - load_factor) / max_g_recovery)
    elevator_cmd = np.clip(q_error * 0.25 * g_allowance, -MAX_ELEVATOR_RAD, MAX_ELEVATOR_RAD)
    throttle_cmd = 1.0
    beta = math.asin(np.clip(self_state[4]/V,-1,1)) if V>1e-3 else 0; beta_error = 0.0 - beta; rudder_cmd = np.clip(beta_error * 0.5, -MAX_RUDDER_RAD, MAX_RUDDER_RAD)
    controls = {'throttle': throttle_cmd, 'elevator': elevator_cmd, 'aileron': aileron_cmd, 'rudder': rudder_cmd}; return controls

# --------------------------------------------------------------------------
# Base Aircraft Model Class (Refactored, includes LQR setup/exec)
# --------------------------------------------------------------------------
class Aircraft6DOF:
    """ 기본 6DOF 항공기 모델 (JSON 로딩, LQR/PID 제어기 설정/실행 로직 포함) """
    def __init__(self, initial_state, json_config_path, control_config=None):
        if not os.path.exists(json_config_path): raise FileNotFoundError(f"Config file not found: {json_config_path}")
        with open(json_config_path, 'r') as f: config = json.load(f);
        params = config['parameters']; self.mass = float(params['mass_kg']); self.inertia_tensor = np.array(params['inertia_tensor_kgm2'], dtype=float); self.inv_inertia_tensor = np.linalg.inv(self.inertia_tensor)
        self.ref_area = float(params['ref_area_m2']); self.ref_chord = float(params['ref_chord_m']); self.ref_span = float(params['ref_span_m']); self.g = G
        self.max_g = params.get('max_g', 7.0); self.max_elevator_rad = params.get('max_elevator_rad', np.deg2rad(25.0)); self.max_aileron_rad = params.get('max_aileron_rad', np.deg2rad(21.5)); self.max_rudder_rad = params.get('max_rudder_rad', np.deg2rad(30.0))
        self.aero_data = config['aerodynamics']; self.engine_data = config['engine']; self.aircraft_type = config.get('aircraft_type', 'Generic Aircraft')
        self.initial_state = np.array(initial_state, dtype=float); self.state = self.initial_state.copy()

        # Initialize Aero Interpolator (subclass should define which one)
        self.aero_interpolator = self._initialize_aero_interpolator()

        # Control setup
        self.control_config = control_config if control_config else {'type': 'LQR' if 'lqr_params' in config else None} # Default LQR if params exist
        if self.control_config['type'] == 'LQR' and 'lqr_params' not in self.control_config:
             self.control_config['params'] = config.get('lqr_params', {}) # Load LQR params from main JSON if not in control_config
        elif self.control_config['type'] == 'PID' and 'pid_params' not in self.control_config:
             self.control_config['params'] = config.get('pid_params', {}) # Load PID params from main JSON

        self.controllers = {}; self.lqr_gain = None; self.lqr_trim_state = None; self.lqr_trim_controls = None
        self.lqr_state_indices = []; self.lqr_control_keys = []; self.lqr_target_state_selected = None
        # BFM AI Target State (for LQR tracking or PID setpoints) - BFM AI will set this
        self.bfm_target_state_selected = None # Dictionary like {'pitch': rad, 'roll': rad, 'airspeed': m/s} or state vector slice

        self._setup_controllers()
        print(f"[Info] {self.aircraft_type} (MaxG:{self.max_g}) 6DOF Base Model Initialized. Ctrl: {self.control_config.get('type')}")

    def _initialize_aero_interpolator(self):
        # Base class uses standard interpolator, subclasses can override
        print("[Warning] Using base AeroCoefficientInterpolator. Subclass should specify.")
        return AeroCoefficientInterpolator(self.aero_data)

    # Abstract methods for engine/aero derivatives (must be implemented by subclasses)
    def _get_aero_derivatives(self, alpha_rad, mach): return self.aero_data.get('derivatives', {}) # Return from JSON or empty
    def _get_engine_force_moment(self, state, controls): raise NotImplementedError("Subclass must implement engine model.")

    # Dynamics, Gravity, Aero Forces/Moments methods remain largely the same
    # ... (dynamics, _get_gravity_force, _get_aero_forces_moments from F16 model) ...
    def _get_aero_forces_moments(self, state, controls):
        x, y, z, u, v, w, phi, theta, psi, p, q, r = state; velocities_body = np.array([u, v, w]); altitude = -z; temp, pressure, rho, sound_speed = get_isa_conditions(altitude); V = np.linalg.norm(velocities_body)
        if V < 1e-3: alpha, beta, mach, q_bar = 0.0, 0.0, 0.0, 0.0
        else: alpha = math.atan2(w, u); beta = math.asin(np.clip(v / V, -1.0, 1.0)); mach = V / sound_speed; q_bar = 0.5 * rho * V**2
        delta_e=np.clip(controls.get('elevator',0.0), -self.max_elevator_rad, self.max_elevator_rad); delta_a=np.clip(controls.get('aileron',0.0), -self.max_aileron_rad, self.max_aileron_rad); delta_r=np.clip(controls.get('rudder',0.0), -self.max_rudder_rad, self.max_rudder_rad); alpha_dot = 0.0
        if V < 1.0: p_bar, q_bar_dim, r_bar, alpha_dot_bar = 0.0, 0.0, 0.0, 0.0
        else: p_bar=p*self.ref_span/(2*V); q_bar_dim=q*self.ref_chord/(2*V); r_bar=r*self.ref_span/(2*V); alpha_dot_bar=alpha_dot*self.ref_chord/(2*V)
        CL_base, CD_base, Cm_base = self.aero_interpolator.get_coeffs(mach, alpha); derivs = self._get_aero_derivatives(alpha, mach) # Use potentially overridden method
        CL=CL_base + derivs.get('CLq',0)*q_bar_dim + derivs.get('CLde',0)*delta_e + derivs.get('CLalpha_dot',0.0)*alpha_dot_bar; CD=CD_base + derivs.get('CDde',0.0)*delta_e
        Cm=Cm_base + derivs.get('Cmq',0)*q_bar_dim + derivs.get('Cmde',0)*delta_e + derivs.get('Cmalpha_dot',0.0)*alpha_dot_bar; CY=derivs.get('CYb',0)*beta + derivs.get('CYp',0)*p_bar + derivs.get('CYr',0)*r_bar + derivs.get('CYda',0.0)*delta_a + derivs.get('CYdr',0)*delta_r
        Cl=derivs.get('Clb',0)*beta + derivs.get('Clp',0)*p_bar + derivs.get('Clr',0)*r_bar + derivs.get('Clda',0)*delta_a + derivs.get('Cldr',0.0)*delta_r; Cn=derivs.get('Cnb',0)*beta + derivs.get('Cnp',0)*p_bar + derivs.get('Cnr',0)*r_bar + derivs.get('Cnda',0.0)*delta_a + derivs.get('Cndr',0)*delta_r
        ca, sa = math.cos(alpha), math.sin(alpha); Fax = q_bar*self.ref_area*(-CD*ca+CL*sa); Fay = q_bar*self.ref_area*CY; Faz = q_bar*self.ref_area*(-CD*sa-CL*ca); Force_Aero_body = np.array([Fax, Fay, Faz]); Moment_Aero_body = q_bar*self.ref_area*np.array([self.ref_span*Cl, self.ref_chord*Cm, self.ref_span*Cn]); return Force_Aero_body, Moment_Aero_body
    def _get_gravity_force(self, state):
        phi, theta, psi = state[6:9]; Fg_ned = np.array([0, 0, self.mass*self.g]); cph, sph=np.cos(phi), np.sin(phi); cth, sth=np.cos(theta), np.sin(theta); cps, sps=np.cos(psi), np.sin(psi)
        C_n_b = np.array([[cth*cps, cth*sps, -sth],[sph*sth*cps-cph*sps, sph*sth*sps+cph*cps, sph*cth],[cph*sth*cps+sph*sps, cph*sth*sps-sph*cps, cph*cth]]); Fg_body = C_n_b @ Fg_ned; return Fg_body
    def dynamics(self, t, state, controls_func):
        controls = controls_func(t); Fa_body, Ma_body = self._get_aero_forces_moments(state, controls); Ft_body, Mt_body = self._get_engine_force_moment(state, controls); Fg_body = self._get_gravity_force(state)
        F_total_body = Fa_body+Ft_body+Fg_body; M_total_body = Ma_body+Mt_body; x, y, z, u, v, w, phi, theta, psi, p, q, r = state; vel_body = np.array([u,v,w]); rates_body = np.array([p,q,r])
        d_uvw = F_total_body/self.mass - np.cross(rates_body, vel_body); omega_cross_I_omega = np.cross(rates_body, self.inertia_tensor @ rates_body); d_pqr = self.inv_inertia_tensor @ (M_total_body - omega_cross_I_omega)
        cph, sph=np.cos(phi), np.sin(phi); cth, sth=np.cos(theta), np.sin(theta); cps, sps=np.cos(psi), np.sin(psi); C_b_n = np.array([[cth*cps, sph*sth*cps-cph*sps, cph*sth*cps+sph*sps],[cth*sps, sph*sth*sps+cph*cps, cph*sth*sps-sph*cps],[-sth, sph*cth, cph*cth]])
        d_xyz = C_b_n @ vel_body
        if abs(abs(theta)-np.pi/2)<1e-4: d_phi_theta_psi=np.zeros(3)
        else: sec_theta=1/max(cth, 1e-8); tan_theta=sth*sec_theta; T_euler=np.array([[1, sph*tan_theta, cph*tan_theta],[0, cph, -sph],[0, sph*sec_theta, cph*sec_theta]]); d_phi_theta_psi = T_euler @ rates_body
        d_state_dt = np.concatenate((d_xyz, d_uvw, d_phi_theta_psi, d_pqr)); return d_state_dt

    # Controller Setup and Execution Logic (moved to base class)
    def _setup_controllers(self):
        ctype = self.control_config.get('type'); params = self.control_config.get('params', {})
        dt = self.control_config.get('dt', 0.02) # dt might be needed for PID discrete update
        if ctype == 'PID':
            print("[Info] Initializing PID controllers for base model."); lim_ele=(-self.max_elevator_rad, self.max_elevator_rad); lim_ail=(-self.max_aileron_rad, self.max_aileron_rad); lim_rud=(-self.max_rudder_rad, self.max_rudder_rad); lim_thr=(0.0, 1.0)
            # Setup PID controllers based on 'params' (from JSON 'pid_params' or control_config)
            if 'pitch' in params: self.controllers['pitch'] = PIDController(params['pitch']['Kp'], params['pitch']['Ki'], params['pitch']['Kd'], setpoint=params['pitch'].get('setpoint', 0.0), dt=dt, output_limits=lim_ele)
            if 'roll' in params: self.controllers['roll'] = PIDController(params['roll']['Kp'], params['roll']['Ki'], params['roll']['Kd'], setpoint=params['roll'].get('setpoint', 0.0), dt=dt, output_limits=lim_ail)
            if 'airspeed' in params: self.controllers['airspeed'] = PIDController(params['airspeed']['Kp'], params['airspeed']['Ki'], params['airspeed']['Kd'], setpoint=params['airspeed'].get('setpoint', 200), dt=dt, output_limits=lim_thr) # Default 200m/s setpoint
            if 'beta' in params: self.controllers['beta'] = PIDController(params['beta']['Kp'], params['beta']['Ki'], params['beta']['Kd'], setpoint=0.0, dt=dt, output_limits=lim_rud)
            # Altitude/Heading control might require target pitch/roll logic outside PID
            if 'altitude' in params: alt_to_pitch_gain=params.get('altitude_to_pitch_gain', -0.01); self.controllers['altitude_target'] = lambda cur_alt: np.clip(alt_to_pitch_gain * (params['altitude']['setpoint'] - cur_alt), -np.deg2rad(15), np.deg2rad(15))
            if 'heading' in params: heading_to_roll_gain=params.get('heading_to_roll_gain', 1.0); self.controllers['heading_target'] = lambda cur_hdg: np.clip(heading_to_roll_gain * _normalize_angle(params['heading']['setpoint'] - cur_hdg), -np.deg2rad(30), np.deg2rad(30))

        elif ctype == 'LQR':
            print("[Info] Setting up LQR controller for base model..."); trim_params = params.get('trim_condition', {'speed': 200, 'altitude': 5000}) # Default trim
            trim_state_found, trim_controls_found, trim_success = find_trim_condition_optimized(self, trim_params['speed'], trim_params['altitude'])
            if not trim_success: print("[Error] Trim find failed for LQR."); self.control_config['type'] = None; return
            self.lqr_trim_state = trim_state_found; self.lqr_trim_controls = trim_controls_found
            # Use LQR state/control definitions from params, default to longitudinal
            self.lqr_state_indices = params.get('state_indices', [3, 5, 10, 7]) # u, w, q, theta
            self.lqr_control_keys = params.get('control_keys', ['elevator', 'throttle'])
            A, B = linearize_dynamics(self, self.lqr_trim_state, self.lqr_trim_controls, self.lqr_state_indices, self.lqr_control_keys)
            if np.isnan(A).any() or np.isnan(B).any(): print("[Error] NaN in Jacobians. Abort LQR."); self.control_config['type'] = None; return
            Q_diag = params.get('Q_diag', [0.1, 0.1, 1.0, 1.0]); R_diag = params.get('R_diag', [1.0, 0.5]) # Default Q/R
            Q = np.diag(Q_diag); R = np.diag(R_diag)
            if Q.shape[0]!=len(self.lqr_state_indices) or R.shape[0]!=len(self.lqr_control_keys): print("[Error] LQR Q/R dim mismatch."); self.control_config['type'] = None; return
            try: P = scipy.linalg.solve_continuous_are(A, B, Q, R); self.lqr_gain = np.linalg.inv(R) @ B.T @ P; self.lqr_target_state_selected = self.lqr_trim_state[self.lqr_state_indices]; print(f"[Info] LQR gain K computed.")
            except Exception as e: print(f"[Error] LQR setup failed: {e}"); self.control_config['type'] = None
        else: print(f"[Info] No controller specified or setup failed.")

    def get_control_inputs(self, t, state, target_state_selected=None):
        """ Calculates control inputs based on configured controller (PID or LQR) and optional BFM target """
        ctype = self.control_config.get('type')
        # Start with trim controls if LQR, else zeros
        controls = self.lqr_trim_controls.copy() if ctype == 'LQR' and self.lqr_trim_controls else {'throttle': 0.0, 'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0}
        limits = {'elevator': (-self.max_elevator_rad, self.max_elevator_rad), 'throttle': (0.0, 1.0), 'aileron': (-self.max_aileron_rad, self.max_aileron_rad), 'rudder': (-self.max_rudder_rad, self.max_rudder_rad)}

        if ctype == 'PID':
            u,v,w=state[3:6]; V=np.linalg.norm([u,v,w]); beta=math.asin(np.clip(v/V, -1, 1)) if V>1e-3 else 0; alt=-state[2]; pitch=state[7]; speed=V; roll=state[6]; hdg=state[8]
            # Update setpoints based on BFM targets if provided
            if target_state_selected:
                if 'pitch' in self.controllers and 'pitch' in target_state_selected: self.controllers['pitch'].set_setpoint(target_state_selected['pitch'])
                if 'roll' in self.controllers and 'roll' in target_state_selected: self.controllers['roll'].set_setpoint(target_state_selected['roll'])
                if 'airspeed' in self.controllers and 'airspeed' in target_state_selected: self.controllers['airspeed'].set_setpoint(target_state_selected['airspeed'])
                # Add other target updates...
            else: # Use default PID setpoints (potentially from config)
                 if 'altitude_target' in self.controllers and 'pitch' in self.controllers: self.controllers['pitch'].set_setpoint(self.controllers['altitude_target'](alt))
                 if 'heading_target' in self.controllers and 'roll' in self.controllers: self.controllers['roll'].set_setpoint(self.controllers['heading_target'](hdg))

            # Calculate PID outputs
            if 'pitch' in self.controllers: controls['elevator'] = self.controllers['pitch'].update(pitch)
            if 'airspeed' in self.controllers: controls['throttle'] = self.controllers['airspeed'].update(speed)
            if 'roll' in self.controllers: controls['aileron'] = self.controllers['roll'].update(roll)
            if 'beta' in self.controllers: controls['rudder'] = self.controllers['beta'].update(beta)

        elif ctype == 'LQR' and self.lqr_gain is not None:
            current_lqr_state = state[self.lqr_state_indices]
            # Use BFM target if provided, otherwise use trim target
            target = target_state_selected if target_state_selected is not None else self.lqr_target_state_selected
            state_error = current_lqr_state - target
            control_deviation = -self.lqr_gain @ state_error
            for i, key in enumerate(self.lqr_control_keys): controls[key] = self.lqr_trim_controls[key] + control_deviation[i]
            # Set non-LQR controls to trim or zero
            all_keys = ['throttle', 'elevator', 'aileron', 'rudder']
            for key in all_keys:
                if key not in self.lqr_control_keys: controls[key] = self.lqr_trim_controls.get(key, 0.0)

        # Final clipping based on limits
        for key in controls:
             if key in limits: controls[key] = np.clip(controls[key], limits[key][0], limits[key][1])
        return controls

    # Inherited dynamics, simulate (solve_ivp version), step, _get_gravity_force are fine
    def simulate(self, t_span, t_eval, controls_func):
        # Fixed step simulation for controlled scenarios
        print(f"[Info] Starting {self.aircraft_type} Controlled sim for t={t_span}"); dt = t_eval[1]-t_eval[0] if len(t_eval)>1 else 0.1; assert dt>0
        num_steps=len(t_eval); states=np.zeros((len(self.initial_state), num_steps)); states[:,0]=self.initial_state; current_state=self.initial_state.copy()
        time_points=t_eval; control_inputs_history=[]
        for controller in self.controllers.values(): # Reset PIDs
            if hasattr(controller, 'reset'): controller.reset()
        for i in range(num_steps-1):
            t=time_points[i]
            # For fixed step sim, calculate controls here using current state
            current_controls=self.get_control_inputs(t, current_state) # Use internal controller
            control_inputs_history.append(current_controls.copy())
            self.step(dt, current_controls); current_state = self.state; states[:, i+1] = current_state
        print("[Info] Controlled Sim finished."); from scipy.integrate import OdeResult; sol=OdeResult(t=time_points, y=states); sol.control_inputs=control_inputs_history; return sol
    def _normalize_angle(self, angle): return _normalize_angle(angle)

    def step(self, dt, current_controls):
        """ RK4를 사용하여 한 스텝 상태를 업데이트합니다. """
        t=0; state_initial=self.state.copy()

        # 여기가 dynamics_wrapper 입니다! step 메소드 내부에 정의되어 있습니다.
        def dynamics_wrapper(t_ignore, state):
             ctrl_func = lambda t: current_controls if isinstance(current_controls, dict) else current_controls(t)
             return self.dynamics(t_ignore, state, ctrl_func)

        try:
            # 이 dynamics_wrapper를 사용하여 RK4 계산을 수행합니다.
            k1 = dynamics_wrapper(t, state_initial)
            k2 = dynamics_wrapper(t + dt/2, state_initial + dt/2 * k1)
            k3 = dynamics_wrapper(t + dt/2, state_initial + dt/2 * k2)
            k4 = dynamics_wrapper(t + dt, state_initial + dt * k3)
            next_state = state_initial + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        except Exception as e:
            print(f"[Error] RK4 step failed: {e}")
            next_state = state_initial

        next_state[2] = min(next_state[2], 0) # 고도 하한
        self.state = next_state # 내부 상태 업데이트
        return self.state

# --------------------------------------------------------------------------
# Specific Aircraft Model Classes (Inheriting from Aircraft6DOF)
# --------------------------------------------------------------------------
class F16_Model(Aircraft6DOF):
    """ F-16 Specific Model """
    def __init__(self, initial_state, json_config_path="f16_data_v2.json", control_config=None):
        super().__init__(initial_state, json_config_path, control_config)

    def _initialize_aero_interpolator(self):
        print("[Info] Initializing F-16 Aero Interpolator.")
        return AeroCoefficientInterpolator(self.aero_data)

    def _get_engine_force_moment(self, state, controls):
        """ F-16 엔진 모델 구현 """
        x, y, z, u, v, w, *_ = state
        throttle = np.clip(controls.get('throttle', 0.0), 0.0, 1.0)
        altitude = -z
        V = np.linalg.norm([u, v, w])
        
        # 대기 조건 계산
        temp, press, rho, sound_speed = get_isa_conditions(altitude)
        mach = V / sound_speed if V > 1e-3 else 0.0
        
        # 엔진 데이터에서 필요한 값들 추출
        max_thrust_sl = self.engine_data['max_thrust_sl_N']
        alt_density_exp = self.engine_data.get('alt_density_exp', 1.0)
        mach_poly_sub = self.engine_data.get('mach_factor_subsonic_poly', [1.0])
        mach_poly_super = self.engine_data.get('mach_factor_supersonic_poly', [1.0])
        thrust_offset = np.array(self.engine_data.get('thrust_offset_m', [0,0,0]), dtype=float)
        
        # 고도와 마하수에 따른 추력 보정
        density_factor = (rho / RHO0)**alt_density_exp
        mach_factor = np.polyval(mach_poly_sub[::-1], mach) if mach < 1.0 else np.polyval(mach_poly_super[::-1], mach)
        
        # 최종 추력 계산
        thrust = throttle * max_thrust_sl * density_factor * mach_factor
        thrust_force = np.array([thrust, 0, 0])
        thrust_moment = np.cross(thrust_offset, thrust_force)
        
        return thrust_force, thrust_moment

class UAV_Model(Aircraft6DOF):
    """ UAV Specific Model """
    def __init__(self, initial_state, json_config_path="uav_data_v2.json", control_config=None):
        super().__init__(initial_state, json_config_path, control_config)

    def _initialize_aero_interpolator(self):
        print("[Info] Initializing UAV Aero Interpolator.")
        return UAV_AeroCoefficientInterpolator(self.aero_data)

    def _get_engine_force_moment(self, state, controls):
        # UAV specific engine logic (propeller)
        throttle = np.clip(controls.get('throttle', 0.0), 0.0, 1.0); u, v, w = state[3:6]; V = np.linalg.norm([u, v, w])
        max_power=self.engine_data['max_power_W']; prop_efficiency=self.engine_data['prop_efficiency']; min_v = self.engine_data.get('min_thrust_speed_mps', 4.0); thrust_offset = np.array(self.engine_data.get('thrust_offset_m', [0,0,0]), dtype=float)
        power = throttle * max_power; thrust = prop_efficiency * power / max(V, min_v); thrust_force = np.array([thrust, 0, 0]); thrust_moment = np.cross(thrust_offset, thrust_force); return thrust_force, thrust_moment
    # Can override other methods like aero derivatives if needed

# --- RL Policy Example ---
class SimpleRLPolicy(nn.Module):
    def __init__(self, state_dim, action_dim): super().__init__(); self.fc = nn.Linear(state_dim, action_dim)
    def forward(self, x): return torch.tanh(self.fc(x))

# --------------------------------------------------------------------------
# Model 4: CombatAgent & MultiAgent_Simulation (Refactored)
# --------------------------------------------------------------------------
class SensorModel:
    """ 센서 모델 (LOS 체크 Placeholder 추가) """
    def __init__(self, max_range, fov_deg): self.max_range = max_range; self.fov_rad = np.deg2rad(fov_deg)

    def check_los(self, pos1, pos2):
        """ Line of Sight Check Placeholder """
        # TODO: Implement actual LOS check using terrain data or earth curvature
        # For now, assume always clear
        return True

    def detect(self, self_state, other_states):
        detected_agents = []; self_pos = self_state[0:3]; phi, theta, psi = self_state[6:9]; cth, sth=np.cos(theta), np.sin(theta); cps, sps=np.cos(psi), np.sin(psi)
        forward_vector_ned = np.array([cth*cps, cth*sps, -sth]); norm_fwd = np.linalg.norm(forward_vector_ned);
        if norm_fwd < 1e-6: return []
        forward_vector_ned /= norm_fwd
        for agent_id, other_state in other_states.items():
            other_pos = other_state[0:3]; vector_to_other = other_pos - self_pos; distance = np.linalg.norm(vector_to_other)
            if distance > 1e-6 and distance <= self.max_range:
                vector_to_other_normalized = vector_to_other / distance; dot_prod = np.dot(forward_vector_ned, vector_to_other_normalized); angle_rad = np.arccos(np.clip(dot_prod, -1.0, 1.0))
                if abs(angle_rad) <= self.fov_rad / 2:
                    # === LOS Check Added ===
                    is_los_clear = self.check_los(self_pos, other_pos)
                    # =======================
                    if is_los_clear: detected_agents.append({'id': agent_id, 'state_observed': other_state.copy(), 'distance': distance, 'relative_bearing_rad': angle_rad})
        return detected_agents

class CombatAgent:
    """ 다중 에이전트용 Combat Agent 클래스 (Aircraft 모델 포함) """
    def __init__(self, agent_id, team_id, initial_state,
                 aircraft_model_class, # F16_Model or UAV_Model class
                 json_config_path, # Path to JSON for the aircraft model
                 control_config=None, # Optional override for controller config
                 sensor_config=None,
                 policy_type='bfm_state_machine', # Default to our BFM AI
                 rl_policy_net=None):

        self.id = agent_id; self.team_id = team_id; self.alive = True
        # Instantiate the specific aircraft model
        self.model = aircraft_model_class(initial_state, json_config_path, control_config)
        self.state = self.model.state # Direct access to model's state

        self.sensor_config = sensor_config if sensor_config else {'range': 20000, 'fov': 120} # Default sensor
        self.sensor = SensorModel(max_range=self.sensor_config['range'], fov_deg=self.sensor_config['fov'])
        self.policy_type = policy_type; self.rl_policy_net = rl_policy_net; self.detected_info_cache = []
        self.current_controls = {'throttle':0.0, 'elevator':0.0, 'aileron':0.0, 'rudder':0.0}
        self.bfm_state = STATE_NEUTRAL # Add BFM state tracking to agent
        self.bfm_target_state = None # Store BFM desired target for LQR
        self.shared_enemy_detections = [] # 팀 내 공유된 적 탐지 정보

        print(f"[Info] Combat Agent {self.id} (Team:{self.team_id}, Type:{self.model.aircraft_type}) initialized with policy: {self.policy_type}")

    def perceive(self, all_agent_states):
        if not self.alive: return []
        other_states = {aid: agent['state'] for aid, agent in all_agent_states.items() if aid != self.id and agent['alive']}
        self.detected_info_cache = self.sensor.detect(self.state, other_states); return self.detected_info_cache

    def decide(self, mission_goal=None, all_agent_states=None):
        """ High-level decision making. Calls BFM AI or other policies. """
        if not self.alive: return {'throttle': 0.0, 'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0}

        # Find the closest opponent for BFM state machine
        opponent_state = None
        closest_enemy_dist = np.inf
        enemies = [info for aid, info in all_agent_states.items() if info.get('team_id') != self.team_id and info['alive']] if all_agent_states else []
        if enemies:
            # Use sensor cache if available, otherwise use true state (for simplicity here)
            visible_enemies = [e for e in enemies if any(d['id'] == e['state'][0] for d in self.detected_info_cache)] # How to link ID? Need agent ID in state dict
            # Let's pass true state for now for BFM AI, assuming target is selected
            if enemies: # Find closest overall enemy
                 closest_enemy = min(enemies, key=lambda e: np.linalg.norm(self.state[0:3] - e['state'][0:3]))
                 opponent_state = closest_enemy['state']

        if self.policy_type == 'bfm_state_machine':
            if opponent_state is not None:
                # Call BFM state machine logic, passing the *model object*
                bfm_controls, next_bfm_state = get_state_machine_bfm_controls(self, opponent_state, self.bfm_state)
                self.bfm_state = next_bfm_state
                # The returned 'bfm_controls' now contains the final actuator commands calculated by LQR/PID inside the model's get_control_inputs
                self.current_controls = bfm_controls
            else: # No opponent, maybe patrol or follow mission goal using controller
                self.bfm_state = STATE_NEUTRAL
                # Example: Hold current attitude/speed using LQR/PID
                target_state_for_controller = None # Use trim target
                self.current_controls = self.model.get_control_inputs(0, self.state, target_state_for_controller)

        elif self.policy_type == 'rl':
             # ... (RL policy logic) ...
             if self.rl_policy_net is None: raise ValueError("RL policy net missing.")
             # Prepare input based on self.state, self.detected_info_cache, mission_goal etc.
             rl_input = self._prepare_rl_input(self.state, self.detected_info_cache, mission_goal, all_agent_states)
             rl_tensor = torch.FloatTensor(rl_input).unsqueeze(0)
             self.rl_policy_net.eval()
             with torch.no_grad(): action_output = self.rl_policy_net(rl_tensor)
             self.current_controls = self._interpret_rl_action(action_output) # This should output target controls
        else: # Default: Hold attitude or simple behavior
            target_state_for_controller = None
            self.current_controls = self.model.get_control_inputs(0, self.state, target_state_for_controller)

        return self.current_controls

    # _prepare_rl_input, _interpret_rl_action (Need to be implemented if using RL)
    def _prepare_rl_input(self, self_state, detected_info, mission_goal, all_agent_states): raise NotImplementedError
    def _interpret_rl_action(self, action_output): raise NotImplementedError

    def step(self, dt):
        """ Step the internal aircraft model forward """
        if not self.alive: return self.state
        # The 'decide' method sets self.current_controls
        self.model.step(dt, self.current_controls)
        self.state = self.model.state # Sync agent state with model state
        return self.state

    def check_collision_or_damage(self, all_agent_states):
        """ 충돌 및 피해 체크 """
        if not self.alive: 
            return
        
        collision_radius = 10.0  # 충돌 반경 약간 증가
        self_pos = self.state[0:3]  # 자신의 위치 벡터
        
        for other_id, other_agent_info in all_agent_states.items():
            if other_id != self.id and other_agent_info['alive']:
                other_pos = other_agent_info['state'][0:3]
                if np.linalg.norm(self_pos - other_pos) < collision_radius:
                    self.alive = False
                    print(f"[Event] Agent {self.id} collided with Agent {other_id}.")
                    return

class MultiAgent_Simulation:
    """ 다중 에이전트 전투 시뮬레이션 환경 """
    def __init__(self, agents, mission_goals=None):
        # Store agent objects in a dictionary
        self.agents = {agent.id: agent for agent in agents}
        # Maintain a separate dictionary for state information needed by others
        self.agent_states = {aid: {'state': agent.state.copy(), 'alive': agent.alive, 'team_id': agent.team_id}
                             for aid, agent in self.agents.items()}
        self.mission_goals = mission_goals if mission_goals else {}; self.time = 0.0; self.history = []

    def run(self, duration, dt):
        num_steps=int(duration/dt); print(f"[Info] Starting Multi-Agent Sim {list(self.agents.keys())} ({duration}s, dt={dt}s)"); self._record_state()
        for i in range(num_steps):
            self.time = (i * dt) # Use time at beginning of step for logging
            # Log less frequently
            if i % int(max(1, 10/dt)) == 0: alive=[aid for aid,info in self.agent_states.items() if info['alive']]; print(f"\n--- Sim Time: {self.time:.2f}s | Alive: {len(alive)} ({alive}) ---")

            current_states_snapshot = {aid: info for aid, info in self.agent_states.items()}
            agent_controls = {}

            # Perceive & Decide for all living agents first
            for agent_id, agent in self.agents.items():
                if self.agent_states[agent_id]['alive']:
                    agent.perceive(current_states_snapshot) # Update internal cache
                    mission=self.mission_goals.get(agent_id)
                    # Decide sets agent.current_controls based on perception and mission
                    agent_controls[agent_id] = agent.decide(mission, current_states_snapshot)

            # Step all living agents using the decided controls
            for agent_id, agent in self.agents.items():
                if self.agent_states[agent_id]['alive']:
                    controls = agent_controls.get(agent_id, {}) # Get controls decided previously
                    agent.step(dt) # Step uses internal self.current_controls set by decide
                    self.agent_states[agent_id]['state'] = agent.state.copy() # Update shared state info

            # Check Collision/Damage after stepping everyone
            current_states_after_step = {aid: info for aid, info in self.agent_states.items()}
            destroyed=[]
            for agent_id, agent in self.agents.items():
                if self.agent_states[agent_id]['alive']:
                    was_alive = agent.alive; agent.check_collision_or_damage(current_states_after_step)
                    if was_alive and not agent.alive: self.agent_states[agent_id]['alive'] = False; destroyed.append(agent_id)
            # Handle mutual destruction if needed

            self._record_state() # Record state at end of interval t+dt
            if not any(info['alive'] for info in self.agent_states.values()): print("[Info] All agents destroyed."); break
        print("[Info] Multi-Agent Sim finished."); return self.history

    def _record_state(self):
        # Log time corresponding to the *end* of the interval (t+dt)
        state_snapshot={'time': self.time + (self.history[0]['time'] if self.history else 0) + (0.05 if self.history else 0) , # Hacky way to get dt, fix this
                        'agents': deepcopy(self.agent_states)}
        # Ensure state is list for JSON
        for agent_info in state_snapshot['agents'].values():
             agent_info['state'] = agent_info['state'].tolist()
        self.history.append(state_snapshot)

# --------------------------------------------------------------------------
# BFM State Machine AI Logic (v4 - LQR Target Setting)
# --------------------------------------------------------------------------
# BFM 상태 정의
STATE_NEUTRAL = 0; STATE_OFFENSIVE = 1; STATE_DEFENSIVE = 2; STATE_RECOVERY = 3
# 지면 충돌 방지 임계값
ALTITUDE_CRITICAL = 500.0; ALTITUDE_RECOVERED = 800.0

# --- 상태 전환 규칙 헬퍼 (이전과 동일) ---
def opponent_is_threat(bfm_params, aspect_threshold_deg, range_threshold_m):
    aspect = bfm_params.get('aspect_deg', 180.0); rng = bfm_params.get('range_m', 99999.0); return aspect < aspect_threshold_deg and rng < range_threshold_m
def i_am_offensive(bfm_params, aot_threshold_deg, range_threshold_m):
    aot = bfm_params.get('aot_deg', 180.0); rng = bfm_params.get('range_m', 99999.0); return aot < aot_threshold_deg and rng < range_threshold_m

# --- 상태별 행동 로직 함수들 (LQR 목표 설정 방식으로 수정) ---
def set_neutral_targets(aircraft_model, opponent_state, bfm_params):
    target_state_dict = {}; self_state = aircraft_model.state; vector_to_opponent_body = state_ned_to_body(opponent_state[0:3] - self_state[0:3], self_state[6:9]); norm_los_b = np.linalg.norm(vector_to_opponent_body)
    if norm_los_b > 1e-6: lat_err_norm = vector_to_opponent_body[1] / norm_los_b; target_state_dict['roll'] = np.clip(-lat_err_norm * np.pi / 4, -np.deg2rad(45), np.deg2rad(45))
    target_state_dict['airspeed'] = np.linalg.norm(self_state[3:6]); target_state_dict['altitude'] = -self_state[2]
    aircraft_model.bfm_target_state_selected = target_state_dict; controls = aircraft_model.get_control_inputs(0, self_state, target_state_dict); return controls

def set_offensive_targets(aircraft_model, opponent_state, bfm_params):
    target_state_dict = {}; self_state = aircraft_model.state; vector_to_opponent_body = state_ned_to_body(opponent_state[0:3] - self_state[0:3], self_state[6:9]); norm_los_b = np.linalg.norm(vector_to_opponent_body)
    if norm_los_b > 1e-6: lat_err_norm = vector_to_opponent_body[1] / norm_los_b; target_state_dict['roll'] = np.clip(-lat_err_norm * np.pi / 2, -np.deg2rad(70), np.deg2rad(70))
    target_state_dict['airspeed'] = np.linalg.norm(self_state[3:6]) * 1.05
    aircraft_model.bfm_target_state_selected = target_state_dict; controls = aircraft_model.get_control_inputs(0, self_state, target_state_dict); return controls

def set_defensive_targets(aircraft_model, opponent_state, bfm_params):
    target_state_dict = {}; self_state = aircraft_model.state; vector_to_opponent_body = state_ned_to_body(opponent_state[0:3] - self_state[0:3], self_state[6:9]); norm_los_b = np.linalg.norm(vector_to_opponent_body)
    if norm_los_b > 1e-6: lat_err_norm = vector_to_opponent_body[1] / norm_los_b; target_state_dict['roll'] = np.sign(-lat_err_norm) * np.deg2rad(100)
    else: target_state_dict['roll'] = self_state[6]
    target_state_dict['pitch_rate'] = np.sign(self_state[7]) * np.deg2rad(30) if abs(self_state[7]) > 0.1 else np.deg2rad(30)
    target_state_dict['throttle'] = 1.0
    aircraft_model.bfm_target_state_selected = target_state_dict; controls = aircraft_model.get_control_inputs(0, self_state, target_state_dict); controls['throttle'] = 1.0; return controls

def set_recovery_targets(aircraft_model):
    target_state_dict = {'roll': 0.0, 'pitch': np.deg2rad(10.0), 'throttle': 1.0}
    aircraft_model.bfm_target_state_selected = target_state_dict; controls = aircraft_model.get_control_inputs(0, aircraft_model.state, target_state_dict); controls['throttle'] = 1.0; return controls

def get_state_machine_bfm_controls(agent, opponent_state, current_bfm_state):
    """ 상태 머신 기반 BFM 제어 로직 (Agent 객체 및 Shared SA 사용) """
    current_state = agent.state; aircraft_id = agent.id; current_alt = -current_state[2]
    next_bfm_state = current_bfm_state; state_changed = False
    aircraft_model = agent.model

    # 1. 최우선 순위: 지면 충돌 회피
    if current_alt < ALTITUDE_CRITICAL:
        if current_bfm_state != STATE_RECOVERY: state_changed = True
        next_bfm_state = STATE_RECOVERY
    elif current_bfm_state == STATE_RECOVERY and current_alt > ALTITUDE_RECOVERED:
        next_bfm_state = STATE_NEUTRAL; state_changed = True
    elif current_bfm_state != STATE_RECOVERY:
        # === Shared SA 기반 판단 ===
        is_self_threatened_by_shared_SA = False
        for detection in agent.shared_enemy_detections:
             threat_state = detection['state_observed']
             threat_params = calculate_bfm_parameters(current_state, threat_state)
             if opponent_is_threat(threat_params, 60, 5000):
                 is_self_threatened_by_shared_SA = True
                 break

        bfm_params = calculate_bfm_parameters(current_state, opponent_state) if opponent_state is not None else {}
        is_threatened = opponent_is_threat(bfm_params, 70, 12000) or is_self_threatened_by_shared_SA
        is_offensive = i_am_offensive(bfm_params, 60, 18000) if opponent_state is not None else False

        threat_escape_range=12000+6000; threat_escape_aspect=70+50
        adv_lose_range=18000+3000; adv_lose_aot=60+30
        prev_state = current_bfm_state
        if current_bfm_state == STATE_NEUTRAL:
            if is_offensive: next_bfm_state = STATE_OFFENSIVE
            elif is_threatened: next_bfm_state = STATE_DEFENSIVE
        elif current_bfm_state == STATE_OFFENSIVE:
            if not i_am_offensive(bfm_params, adv_lose_aot, adv_lose_range): next_bfm_state = STATE_NEUTRAL
            elif is_threatened: next_bfm_state = STATE_DEFENSIVE
        elif current_bfm_state == STATE_DEFENSIVE:
            if not opponent_is_threat(bfm_params, threat_escape_aspect, threat_escape_range) and not is_self_threatened_by_shared_SA:
                if is_offensive: next_bfm_state = STATE_OFFENSIVE
                else: next_bfm_state = STATE_NEUTRAL
        if prev_state != next_bfm_state: state_changed = True

    if next_bfm_state == STATE_RECOVERY: controls = set_recovery_targets(aircraft_model)
    elif next_bfm_state == STATE_OFFENSIVE and opponent_state is not None: controls = set_offensive_targets(aircraft_model, opponent_state, bfm_params)
    elif next_bfm_state == STATE_DEFENSIVE and opponent_state is not None: controls = set_defensive_targets(aircraft_model, opponent_state, bfm_params)
    else:
         opponent_state_for_neutral = opponent_state if opponent_state is not None else current_state
         controls = set_neutral_targets(aircraft_model, opponent_state_for_neutral, bfm_params if opponent_state is not None else {})

    return controls, next_bfm_state

# --- 1v1 시뮬레이션 메인 루프 (수정 없음) ---
def run_1v1_simulation_sm(aircraft1, aircraft2, initial_state1, initial_state2, duration, dt):
    """ 1 대 1 교전 시뮬레이션 실행 (Agent 객체 및 상태 머신 AI 사용) """
    aircraft1.state = initial_state1.copy(); aircraft1.bfm_state = STATE_NEUTRAL # Reset state
    aircraft2.state = initial_state2.copy(); aircraft2.bfm_state = STATE_NEUTRAL
    time_points = np.arange(0, duration + dt, dt); num_steps = len(time_points); history = []
    print(f"[Info] Starting 1v1 SM Simulation: {aircraft1.id} vs {aircraft2.id} ({duration}s, dt={dt}s)")
    current_time = 0.0
    for i in range(num_steps):
        t = time_points[i]
        # --- Simulate Perfect Info Sharing for 1v1 (Simplified) ---
        aircraft1.shared_enemy_detections = aircraft1.perceive({aircraft2.id: {'state': aircraft2.state, 'alive': aircraft2.alive}}) # Only see opponent
        aircraft2.shared_enemy_detections = aircraft2.perceive({aircraft1.id: {'state': aircraft1.state, 'alive': aircraft1.alive}})
        aircraft1.shared_friendly_states = {}; aircraft2.shared_friendly_states = {} # No friendlies in 1v1
        # ----------------------------------------------------------
        controls1 = aircraft1.decide() # Decide uses internal shared SA
        controls2 = aircraft2.decide()
        current_state1 = aircraft1.step(dt); current_state2 = aircraft2.step(dt)
        current_time = t
        log_entry = {'time': current_time,
                     'aircraft1': {'state': current_state1.copy(), 'controls': controls1.copy(), 'bfm_state': aircraft1.bfm_state},
                     'aircraft2': {'state': current_state2.copy(), 'controls': controls2.copy(), 'bfm_state': aircraft2.bfm_state}}
        history.append(log_entry)
        pos1=current_state1[0:3]; pos2=current_state2[0:3]; distance=np.linalg.norm(pos1-pos2)
        alt1_msl = -current_state1[2]; alt2_msl = -current_state2[2]
        if distance < 50: print(f"[Event] Collision @ t={current_time:.2f}s! Dist: {distance:.2f}m"); break
        if distance > 40000: print(f"[Event] Separated @ t={current_time:.2f}s. Dist: {distance:.1f}m"); break
        if alt1_msl < 10 or alt2_msl < 10: print(f"[Event] Ground impact @ t={current_time:.2f}s (Alt1:{alt1_msl:.1f}m, Alt2:{alt2_msl:.1f}m)"); break
        if i % int(max(1, 5/dt)) == 0: print(f"  t={current_time:.1f}s, Dist={distance:.0f}m, Alt1={alt1_msl:.0f}m, Alt2={alt2_msl:.0f}m, St1={aircraft1.bfm_state}, St2={aircraft2.bfm_state}")
    if history: print(f"[Info] 1v1 SM Sim finished at t={history[-1]['time']:.2f}s")
    else: print("[Info] 1v1 SM Sim did not run.")
    return history

def get_simple_bfm_controls(self_state, opponent_state, target_speed_kts=450, max_g=7.0):
    """
    매우 기본적인 BFM 제어 로직 예시: 상대방 방향으로 선회 시도 + 속도/고도 유지 시도.
    출력은 제어 입력 딕셔너리 (단위는 모델 dynamics가 기대하는 단위, 예: 라디안).
    """
    controls = {'throttle': 0.5, 'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0}
    # 제어 표면 최대 각도 (예시, 모델과 일치 필요)
    MAX_ELEVATOR_RAD = np.deg2rad(25.0)
    MAX_AILERON_RAD = np.deg2rad(21.5)
    MAX_RUDDER_RAD = np.deg2rad(30.0)

    self_pos = self_state[0:3]; opponent_pos = opponent_state[0:3]
    self_vel_body = self_state[3:6]; phi, theta, psi = self_state[6:9]; p, q, r = self_state[9:12]
    vector_to_opponent_ned = opponent_pos - self_pos; distance = np.linalg.norm(vector_to_opponent_ned)

    if distance < 100: return controls # Close range safety

    # --- 상대방 방향 벡터 (Body Frame) ---
    vector_to_opponent_body = state_ned_to_body(vector_to_opponent_ned, self_state[6:9])
    norm_los_b = np.linalg.norm(vector_to_opponent_body)
    if norm_los_b < 1e-6: return controls
    vector_to_opponent_body_normalized = vector_to_opponent_body / norm_los_b

    # --- 제어 로직 ---
    # 1. 롤 제어: 상대방을 좌우 중앙 (Body Y=0) 으로
    lateral_error_norm = vector_to_opponent_body_normalized[1] # Y component
    target_roll = np.clip(-lateral_error_norm * np.pi / 2 * 1.5, -np.pi/2, np.pi/2) # Kp=1.5
    roll_error = _normalize_angle(target_roll - phi)
    target_p = np.clip(roll_error * 2.0, -np.deg2rad(120), np.deg2rad(120)) # Max roll rate 120 deg/s
    p_error = target_p - p
    controls['aileron'] = np.clip(p_error * 0.1, -MAX_AILERON_RAD, MAX_AILERON_RAD) # Kp_ail=0.1

    # 2. 피치 제어: 상대방을 상하 중앙 (Body Z=0) 으로, G 제한 고려
    vertical_error_norm = vector_to_opponent_body_normalized[2] # Z component (Body Z+ is down)
    target_q = np.clip(vertical_error_norm * 1.0, -np.deg2rad(45), np.deg2rad(45)) # Kp_q=1.0, Max pitch rate 45 deg/s
    q_error = target_q - q
    V = np.linalg.norm(self_vel_body)
    load_factor = np.sqrt(1 + (q*V/G)**2) if V > 1 else 1 # Simplified G
    g_allowance = max(0, (max_g - load_factor) / max_g) if max_g > 0 else 0
    controls['elevator'] = np.clip(q_error * 0.25 * g_allowance, -MAX_ELEVATOR_RAD, MAX_ELEVATOR_RAD) # Kp_ele=0.25

    # 3. 스로틀 제어: 목표 속도 유지
    target_speed_mps = target_speed_kts * 0.5144
    current_speed = V
    speed_error = target_speed_mps - current_speed
    # 현재 스로틀 값을 읽어와서 조절하는 방식이 더 안정적일 수 있음
    current_throttle = controls.get('throttle', 0.5) # 현재 값 또는 기본값
    controls['throttle'] = np.clip(current_throttle + speed_error * 0.005, 0.05, 1.0) # Simple P/I-like control

    # 4. 러더 제어: Beta=0 유지
    beta = math.asin(np.clip(self_vel_body[1]/V,-1,1)) if V>1e-3 else 0; beta_error = 0.0 - beta
    controls['rudder'] = np.clip(beta_error * 0.5, -MAX_RUDDER_RAD, MAX_RUDDER_RAD) # Kp_rud=0.5

    return controls

def execute_neutral_controls(self_state, opponent_state, bfm_params):
    current_alt = -self_state[2]; max_g = 7.0 if current_alt > 1500 else max(1.5, 7.0 * (current_alt / 1500)**0.5)
    controls = get_simple_bfm_controls(self_state, opponent_state, target_speed_kts=450, max_g=max_g); return controls

def execute_offensive_controls(self_state, opponent_state, bfm_params):
    """ 공격 상태: Lag Pursuit 로직으로 추적 """
    # 1. 기본 제어 한계 및 G-Load 설정
    MAX_ELEVATOR_RAD = np.deg2rad(25.0)
    MAX_AILERON_RAD = np.deg2rad(21.5)
    MAX_RUDDER_RAD = np.deg2rad(30.0)
    current_alt = -self_state[2]
    max_g = 8.0 if current_alt > 1500 else max(1.5, 8.0 * (current_alt / 1500)**0.5)

    # 2. 상태 벡터 추출
    self_pos = self_state[0:3]
    self_vel_body = self_state[3:6]
    self_euler = self_state[6:9]
    self_rates = self_state[9:12]
    V_self = np.linalg.norm(self_vel_body)
    
    opp_pos = opponent_state[0:3]
    opp_vel_body = opponent_state[3:6]
    opp_euler = opponent_state[6:9]
    opp_vel_ned = state_body_to_ned(opp_vel_body, opp_euler)
    V_opp = np.linalg.norm(opp_vel_body)

    # 3. Lag Pursuit 계산
    # 3.1 상대 위치 및 속도 계산
    vector_to_opp = opp_pos - self_pos
    R = np.linalg.norm(vector_to_opp)
    if R < 100: # 근접 시 안전 조치
        return {'throttle': 0.5, 'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0}

    # 3.2 Lag Angle 계산 (상대 속도에 따라 조정)
    # 속도비에 따라 Lag Angle 조정 (더 빠를수록 더 큰 Lag Angle)
    velocity_ratio = V_self / max(V_opp, 1.0)
    base_lag_angle = np.deg2rad(30)  # 기본 30도
    lag_angle = base_lag_angle * min(velocity_ratio, 1.5)  # 최대 45도까지

    # 3.3 목표 조준점 계산 (상대방 위치에서 Lag Angle만큼 뒤)
    opp_heading = math.atan2(opp_vel_ned[1], opp_vel_ned[0])
    lag_point = opp_pos - np.array([
        R * 0.3 * math.cos(opp_heading),
        R * 0.3 * math.sin(opp_heading),
        0  # 고도는 유지
    ])

    # 4. 제어 입력 계산
    # 4.1 Lag Point로의 방향 벡터 (Body Frame)
    vector_to_lag_ned = lag_point - self_pos
    vector_to_lag_body = state_ned_to_body(vector_to_lag_ned, self_euler)
    norm_lag_b = np.linalg.norm(vector_to_lag_body)
    if norm_lag_b < 1e-6:
        return {'throttle': 0.5, 'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0}
    
    # 4.2 Roll 제어 (Lag Point를 향해)
    lat_err_norm = vector_to_lag_body[1] / norm_lag_b
    target_roll = np.clip(-lat_err_norm * np.pi * 0.8, -np.deg2rad(60), np.deg2rad(60))
    roll_error = _normalize_angle(target_roll - self_euler[0])
    target_p = np.clip(roll_error * 2.5, -np.deg2rad(120), np.deg2rad(120))
    p_error = target_p - self_rates[0]
    aileron_cmd = np.clip(p_error * 0.15, -MAX_AILERON_RAD, MAX_AILERON_RAD)

    # 4.3 Pitch 제어 (G-Load 제한 적용)
    vert_err_norm = vector_to_lag_body[2] / norm_lag_b
    target_q = np.clip(vert_err_norm * 1.2, -np.deg2rad(30), np.deg2rad(30))
    q_error = target_q - self_rates[1]
    load_factor = np.sqrt(1 + (self_rates[1]*V_self/G)**2) if V_self > 1 else 1
    g_allowance = max(0, (max_g - load_factor) / max_g)
    elevator_cmd = np.clip(q_error * 0.3 * g_allowance, -MAX_ELEVATOR_RAD, MAX_ELEVATOR_RAD)

    # 4.4 속도 제어 (상대방보다 약간 빠르게)
    target_speed = min(V_opp * 1.1, 480 * 0.5144)  # 최대 480kts
    speed_error = (target_speed - V_self) * 0.01
    throttle_cmd = np.clip(0.5 + speed_error, 0.1, 1.0)

    # 4.5 Yaw 제어 (Beta 최소화)
    beta = math.asin(np.clip(self_vel_body[1]/V_self, -1, 1)) if V_self > 1e-3 else 0
    beta_error = 0.0 - beta
    rudder_cmd = np.clip(beta_error * 0.5, -MAX_RUDDER_RAD, MAX_RUDDER_RAD)

    controls = {
        'throttle': throttle_cmd,
        'elevator': elevator_cmd,
        'aileron': aileron_cmd,
        'rudder': rudder_cmd
    }
    return controls

def execute_defensive_controls(self_state, opponent_state, bfm_params):
    MAX_ELEVATOR_RAD=np.deg2rad(25.0); MAX_AILERON_RAD=np.deg2rad(21.5); MAX_RUDDER_RAD=np.deg2rad(30.0); current_alt = -self_state[2]
    max_g_allowed = 9.0 if current_alt > 1000 else max(1.5, 9.0 * (current_alt / 1000)**0.5)
    vector_to_opponent_body = state_ned_to_body(opponent_state[0:3] - self_state[0:3], self_state[6:9])
    norm_los_b = np.linalg.norm(vector_to_opponent_body)
    if norm_los_b < 1e-6: return {'throttle': 1.0, 'elevator': 0.0, 'aileron': 0.0, 'rudder': 0.0} # Avoid division by zero
    lat_err_norm = vector_to_opponent_body[1] / norm_los_b
    target_roll = np.sign(-lat_err_norm) * np.deg2rad(100); roll_error = _normalize_angle(target_roll - self_state[6])
    target_p = np.clip(roll_error*3.0, -np.deg2rad(150), np.deg2rad(150)); p_error = target_p - self_state[9]; aileron_cmd = np.clip(p_error*0.15, -MAX_AILERON_RAD, MAX_AILERON_RAD)
    elevator_cmd = -MAX_ELEVATOR_RAD * (max_g_allowed / 9.0) # Max pull attempt, limited by allowed G
    throttle_cmd = 1.0; rudder_cmd = np.clip(-lat_err_norm * 0.1, -MAX_RUDDER_RAD, MAX_RUDDER_RAD)
    controls = {'throttle': throttle_cmd, 'elevator': elevator_cmd, 'aileron': aileron_cmd, 'rudder': rudder_cmd}; return controls

def execute_recovery_controls(self_state):
    # print(f"  > Executing Recovery!") # 로그 축소
    MAX_ELEVATOR_RAD=np.deg2rad(25.0); MAX_AILERON_RAD=np.deg2rad(21.5); MAX_RUDDER_RAD=np.deg2rad(30.0)
    phi, theta, psi = self_state[6:9]; p, q, r = self_state[9:12]; V = np.linalg.norm(self_state[3:6])
    target_roll = 0.0; roll_error = _normalize_angle(target_roll - phi); target_p = np.clip(roll_error * 2.5, -np.deg2rad(120), np.deg2rad(120)); p_error = target_p - p; aileron_cmd = np.clip(p_error * 0.1, -MAX_AILERON_RAD, MAX_AILERON_RAD)
    target_pitch = np.deg2rad(10.0); pitch_error = target_pitch - theta; target_q = np.clip(pitch_error * 1.0, -np.deg2rad(20), np.deg2rad(20)); q_error = target_q - q
    load_factor = np.sqrt(1 + (q*V/G)**2) if V > 1 else 1; max_g_recovery = 4.0; g_allowance = max(0, (max_g_recovery - load_factor) / max_g_recovery)
    elevator_cmd = np.clip(q_error * 0.25 * g_allowance, -MAX_ELEVATOR_RAD, MAX_ELEVATOR_RAD)
    throttle_cmd = 1.0
    beta = math.asin(np.clip(self_state[4]/V,-1,1)) if V>1e-3 else 0; beta_error = 0.0 - beta; rudder_cmd = np.clip(beta_error * 0.5, -MAX_RUDDER_RAD, MAX_RUDDER_RAD)
    controls = {'throttle': throttle_cmd, 'elevator': elevator_cmd, 'aileron': aileron_cmd, 'rudder': rudder_cmd}; return controls
    


class AeroCoefficientInterpolator:
    """ JSON 데이터 기반 공력 계수 보간 """
    def __init__(self, aero_data):
        lookup_tables = aero_data['lookup_tables']; mach = np.array(lookup_tables['mach_values'], dtype=float); alpha_deg = np.array(lookup_tables['alpha_values_deg'], dtype=float)
        alpha_rad = np.deg2rad(alpha_deg); cl = np.array(lookup_tables['cl_table'], dtype=float); cd = np.array(lookup_tables['cd_table'], dtype=float); cm = np.array(lookup_tables['cm_table'], dtype=float)
        if not (cl.shape == cd.shape == cm.shape == (len(mach), len(alpha_rad))): raise ValueError(f"Aero table shapes mismatch")
        if not np.all(np.diff(mach) > 0): raise ValueError("Mach values must be strictly increasing.");
        if not np.all(np.diff(alpha_rad) > 0): raise ValueError("Alpha values must be strictly increasing.")
        self.mach_range = (mach[0], mach[-1]); self.alpha_range = (alpha_rad[0], alpha_rad[-1])
        self.cl_interpolator = scipy.interpolate.RectBivariateSpline(mach, alpha_rad, cl, kx=1, ky=1); self.cd_interpolator = scipy.interpolate.RectBivariateSpline(mach, alpha_rad, cd, kx=1, ky=1); self.cm_interpolator = scipy.interpolate.RectBivariateSpline(mach, alpha_rad, cm, kx=1, ky=1)
    def get_coeffs(self, mach, alpha_rad):
        mach = np.clip(mach, self.mach_range[0], self.mach_range[1]); alpha_rad = np.clip(alpha_rad, self.alpha_range[0], self.alpha_range[1])
        cl = self.cl_interpolator(mach, alpha_rad, grid=False); cd = self.cd_interpolator(mach, alpha_rad, grid=False); cm = self.cm_interpolator(mach, alpha_rad, grid=False)
        return float(cl), float(cd), float(cm)

class UAV_AeroCoefficientInterpolator(AeroCoefficientInterpolator):
    """ UAV 전용 공력 계수 보간기 """
    def __init__(self, aero_data):
        super().__init__(aero_data=aero_data)

def linearize_dynamics(model, x0, u0_dict, state_indices, control_keys):
    """ 수치적 선형화 (A, B 행렬 계산) """
    n_states_total=len(x0); n_states_selected=len(state_indices); n_controls_selected=len(control_keys); epsilon=1e-5; A=np.zeros((n_states_selected,n_states_selected)); B=np.zeros((n_states_selected,n_controls_selected))
    for j, idx in enumerate(state_indices):
        x_plus=x0.copy(); x_plus[idx]+=epsilon; x_minus=x0.copy(); x_minus[idx]-=epsilon
        try: f_plus=model.dynamics(0,x_plus,lambda t:u0_dict); f_minus=model.dynamics(0,x_minus,lambda t:u0_dict); deriv=(f_plus-f_minus)/(2*epsilon); A[:,j]=deriv[state_indices]
        except Exception as e: print(f"Error lin A for state {idx}: {e}"); A[:, j]=np.nan
    for j, key in enumerate(control_keys):
        u_plus=u0_dict.copy(); u_plus[key]+=epsilon; u_minus=u0_dict.copy(); u_minus[key]-=epsilon
        try: f_plus=model.dynamics(0,x0,lambda t:u_plus); f_minus=model.dynamics(0,x0,lambda t:u_minus); deriv=(f_plus-f_minus)/(2*epsilon); B[:,j]=deriv[state_indices]
        except Exception as e: print(f"Error lin B for ctrl {key}: {e}"); B[:, j]=np.nan
    if np.isnan(A).any() or np.isnan(B).any(): print("[Warning] NaN detected in Jacobians A or B.")
    # print(f"[Info] Linearized dynamics. A:{A.shape}, B:{B.shape}") # 로그 축소
    return A, B


# --- PID 컨트롤러 클래스 ---
class PIDController:
    """ 간단한 PID 제어기 (Anti-windup 포함) """
    def __init__(self, Kp, Ki, Kd, setpoint, dt, output_limits=(-np.inf, np.inf)):
        self.Kp=Kp; self.Ki=Ki; self.Kd=Kd; self.setpoint=setpoint; self.dt=dt; self._integral=0; self._previous_error=0; self.output_limits=output_limits
        int_limit_factor=0.5; self.integral_limits=(-np.inf, np.inf)
        if self.Ki != 0: lower_lim = self.output_limits[0]*int_limit_factor/self.Ki if self.output_limits[0]>-np.inf else -np.inf; upper_lim = self.output_limits[1]*int_limit_factor/self.Ki if self.output_limits[1]<np.inf else np.inf; self.integral_limits = (min(lower_lim, upper_lim), max(lower_lim, upper_lim))
    def update(self, current_value):
        error = self.setpoint - current_value; self._integral += error*self.dt; self._integral = np.clip(self._integral, self.integral_limits[0], self.integral_limits[1])
        derivative = (error - self._previous_error) / self.dt if self.dt > 0 else 0; output = self.Kp*error + self.Ki*self._integral + self.Kd*derivative; self._previous_error = error; output = np.clip(output, self.output_limits[0], self.output_limits[1]); return output
    def reset(self): self._integral=0; self._previous_error=0
    def set_setpoint(self, setpoint): self.setpoint = setpoint

def trim_residuals(vars_to_solve, V_target, gamma_target_rad, altitude_target, model):
    """ 트림 솔버용 잔차 함수 (u_dot, w_dot, q_dot) """
    alpha, delta_e, throttle = vars_to_solve; theta = alpha + gamma_target_rad
    state_guess = np.array([0,0,-altitude_target, V_target*np.cos(alpha),0,V_target*np.sin(alpha), 0,theta,0, 0,0,0])
    controls_guess = {'throttle': throttle, 'elevator': delta_e, 'aileron': 0.0, 'rudder': 0.0}
    try: d_state_dt = model.dynamics(0, state_guess, lambda t: controls_guess); residuals = d_state_dt[[3, 5, 10]]
    except Exception as e: residuals = np.array([1e6, 1e6, 1e6])
    return residuals

def find_trim_condition_optimized(model, target_speed, target_altitude, target_gamma_deg=0.0):
    """ scipy.optimize.root를 사용하여 트림 조건 최적화 """
    gamma_rad = np.deg2rad(target_gamma_deg); temp, press, rho, sound_speed = get_isa_conditions(target_altitude); q_bar_est = 0.5 * rho * target_speed**2
    CL_est = (model.mass * G * np.cos(gamma_rad)) / max(q_bar_est * model.ref_area, 1e-6)
    alpha_guess = np.clip(CL_est/(2*np.pi), model.aero_interpolator.alpha_range[0]+0.01, model.aero_interpolator.alpha_range[1]-0.01)
    delta_e_guess = 0.0; throttle_guess = 0.5; initial_guess = [alpha_guess, delta_e_guess, throttle_guess]
    sol = scipy.optimize.root(fun=trim_residuals, x0=initial_guess, args=(target_speed, gamma_rad, target_altitude, model), method='lm', tol=1e-7)
    if sol.success and np.all(np.abs(sol.fun) < 1e-4):
        alpha_trim, delta_e_trim, throttle_trim = sol.x; throttle_trim = np.clip(throttle_trim, 0.0, 1.0); theta_trim = alpha_trim + gamma_rad
        trim_state = np.array([0,0,-target_altitude, target_speed*np.cos(alpha_trim),0,target_speed*np.sin(alpha_trim), 0,theta_trim,0, 0,0,0])
        trim_controls = {'throttle': throttle_trim, 'elevator': delta_e_trim, 'aileron': 0.0, 'rudder': 0.0}
        print(f"  > Trim Found: Alpha={np.rad2deg(alpha_trim):.3f}deg, DeltaE={np.rad2deg(delta_e_trim):.3f}deg, Thr={throttle_trim:.4f}")
        return trim_state, trim_controls, True
    else: print(f"[Error] Trim solver failed. Success:{sol.success}, Msg:{sol.message}, Residuals:{sol.fun}"); return None, None, False
    
# ==========================================================================
# Main Execution Block (Multi-Agent 2v2 Example)
# ==========================================================================
if __name__ == "__main__":
    print(f"Current Time: {datetime.now()}")
    f16_config = "/mnt/hdd2/attoman/workspace/git/combat_sim/aero_db/f16_data_v2.json"; uav_config = "/mnt/hdd2/attoman/workspace/git/combat_sim/aero_db/uav_data_v2.json"
    if not os.path.exists(f16_config): print(f"[ERROR] F-16 config not found: {f16_config}"); exit()
    if not os.path.exists(uav_config): print(f"[ERROR] UAV config file not found: {uav_config}"); exit()

    # === 2 vs 2 Scenario: 2 F-16 (Blue) vs 2 UAV (Red) ===
    alt_m = 6000 # m
    spd_f16_mps = 450 * 0.5144
    spd_uav_mps = 180 * 0.5144
    sep_m = 25000 # m (Initial separation along North axis)
    lat_sep_m = 2000 # m (Lateral separation within team)

    # --- Blue Team (F-16s) ---
    # Blue 1: Starts South-West, heading North-East
    init_state_b1 = np.array([-sep_m/2, -lat_sep_m/2, -alt_m, spd_f16_mps, 0, 0, 0, 0, np.deg2rad(45), 0, 0, 0])
    # Blue 2: Starts South-East, heading North-West
    init_state_b2 = np.array([-sep_m/2, +lat_sep_m/2, -alt_m, spd_f16_mps, 0, 0, 0, 0, np.deg2rad(-45), 0, 0, 0])

    # --- Red Team (UAVs) ---
    # Red 1: Starts North-West, heading South-East
    init_state_r1 = np.array([+sep_m/2, -lat_sep_m/2, -(alt_m+500), spd_uav_mps, 0, 0, 0, 0, np.deg2rad(180-45), 0, 0, 0])
    # Red 2: Starts North-East, heading South-West
    init_state_r2 = np.array([+sep_m/2, +lat_sep_m/2, -(alt_m+500), spd_uav_mps, 0, 0, 0, 0, np.deg2rad(180+45), 0, 0, 0])

    # --- LQR Control Config (Example - needs tuning!) ---
    # Might want different tuning for F-16 and UAV in their respective JSONs or passed here
    lqr_control_config = {
        'type': 'LQR',
        'params': { # Load detailed params from JSON by default
            'trim_condition': {'speed': 200, 'altitude': 6000}, # Adjust default trim if needed
            'state_indices': [3, 5, 10, 7, 6, 9], # u, w, q, theta, phi, p ? (Example longitudinal + roll)
            'control_keys': ['elevator', 'throttle', 'aileron'], # Example controls
            # Q/R matrices should ideally be loaded from JSON 'lqr_params'
            'Q_diag': [0.1, 0.1, 1.0, 1.0, 0.5, 0.5], # Example Q weights
            'R_diag': [5.0, 0.5, 5.0]                # Example R weights
        }
    }
    # PID might be simpler for basic control during BFM AI development
    pid_control_config = {
        'type': 'PID',
        'dt': 0.05,
        'params': { # Load from JSON 'pid_params' or define here
            'pitch': {'Kp': 1.5, 'Ki': 0.3, 'Kd': 0.1},
            'roll': {'Kp': 1.8, 'Ki': 0.2, 'Kd': 0.1},
            'airspeed': {'Kp': 0.05, 'Ki': 0.01, 'Kd': 0.0},
            'beta': {'Kp': 0.5, 'Ki': 0.0, 'Kd': 0.0}
            # Altitude/Heading targets handled by BFM AI setting PID setpoints
        }
    }
    # --- Select Controller ---
    active_control_config = pid_control_config # Use PID for potentially simpler integration with BFM targets first
    # active_control_config = lqr_control_config # Switch to LQR later

    # --- Agent Creation ---
    agent_b1 = CombatAgent("Blue1", "Blue", init_state_b1, F16_Model, f16_config, control_config=active_control_config)
    agent_b2 = CombatAgent("Blue2", "Blue", init_state_b2, F16_Model, f16_config, control_config=active_control_config)
    agent_r1 = CombatAgent("Red1", "Red", init_state_r1, UAV_Model, uav_config, control_config=active_control_config)
    agent_r2 = CombatAgent("Red2", "Red", init_state_r2, UAV_Model, uav_config, control_config=active_control_config)

    # --- Simulation Setup & Run ---
    simulation = MultiAgent_Simulation(agents=[agent_b1, agent_b2, agent_r1, agent_r2])
    sim_duration = 180; time_step = 0.05
    simulation_history = simulation.run(duration=sim_duration, dt=time_step)

    # --- Visualization (Multi-Agent) ---
    if simulation_history:
        state_names_plot = {0:"Neutral", 1:"Offensive", 2:"Defensive", 3:"Recovery"}
        agent_ids = list(simulation.agents.keys())
        colors = {'Blue': 'b', 'Red': 'r'}
        styles = {'Blue1': '-', 'Blue2': '--', 'Red1': '-', 'Red2': '--'}
        markers = {'Blue1': 'o', 'Blue2': '^', 'Red1': 's', 'Red2': 'v'}

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 16))
        fig.suptitle("Multi-Agent BFM Simulation (2 F-16 vs 2 UAV)")

        # Trajectories
        for agent_id in agent_ids:
            team = simulation.agents[agent_id].team_id
            col = colors.get(team, 'k')
            sty = styles.get(agent_id, ':')
            mkr = markers.get(agent_id, '.')
            agent_hist = [entry['agents'].get(agent_id) for entry in simulation_history if entry['agents'].get(agent_id)]
            pos_x = [e['state'][0] for e in agent_hist if e['alive']] # North
            pos_y = [e['state'][1] for e in agent_hist if e['alive']] # East
            axs[0].plot(pos_y, pos_x, color=col, linestyle=sty, label=f'{agent_id}')
            if pos_x: # Plot start marker
                 axs[0].plot(pos_y[0], pos_x[0], color=col, marker=mkr, markersize=8)
        axs[0].set_xlabel("East (m)"); axs[0].set_ylabel("North (m)"); axs[0].legend(); axs[0].grid(True); axs[0].axis('equal'); axs[0].set_title("Trajectories (Top Down)")

        # Altitudes
        for agent_id in agent_ids:
            team = simulation.agents[agent_id].team_id
            col = colors.get(team, 'k'); sty = styles.get(agent_id, ':')
            times = [entry['time'] for entry in simulation_history if entry['agents'].get(agent_id)]
            alt = [-entry['agents'][agent_id]['state'][2] for entry in simulation_history if entry['agents'].get(agent_id)]
            axs[1].plot(times, alt, color=col, linestyle=sty, label=f'Alt {agent_id}')
        axs[1].axhline(ALTITUDE_CRITICAL, color='grey', linestyle=':', label=f'Crit Alt'); axs[1].axhline(ALTITUDE_RECOVERED, color='grey', linestyle='--', label=f'Rec Alt');
        axs[1].set_xlabel("Time (s)"); axs[1].set_ylabel("Altitude (m MSL)"); axs[1].legend(); axs[1].grid(True); axs[1].set_ylim(bottom=0)

        # BFM States
        for agent_id in agent_ids:
            team = simulation.agents[agent_id].team_id
            col = colors.get(team, 'k'); mkr = markers.get(agent_id, '.')
            times = [entry['time'] for entry in simulation_history if entry['agents'].get(agent_id)]
            bfm_state = [entry['agents'][agent_id].get('bfm_state', -1) for entry in simulation_history if entry['agents'].get(agent_id)] # Get bfm_state if exists
            axs[2].plot(times, bfm_state, color=col, marker=mkr, linestyle='None', markersize=3, label=f'State {agent_id}')
        axs[2].set_xlabel("Time (s)"); axs[2].set_ylabel("BFM State"); axs[2].legend(); axs[2].grid(True); axs[2].set_yticks(list(state_names_plot.keys())); axs[2].set_yticklabels(list(state_names_plot.values()))

        plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig("./2v2_bfm_sm_log.png"); plt.show()

        # Optional: Save log data
        try:
            log_filename = "2v2_bfm_sm_log.json"
            # History already contains serializable data due to deepcopy and list conversion in _record_state
            with open(log_filename, "w") as f: json.dump(simulation_history, f, indent=2); print(f"[Info] Sim history saved to {log_filename}")
        except Exception as e: print(f"[Error] Failed to save sim log: {e}")
    else: print("[Info] No simulation history.")
    
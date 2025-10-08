#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure Pursuit Dataset Generator (geometric method)

生成三类文件：
1) 纯跟踪完整数据 JSON：轨迹、状态、转角等（--out-json）
2) 扁平 CSV：便于表格查看（--out-csv）
3) LLM I/O NDJSON：每行包含一次与 CARLA 侧一致的“发送输入”和“期望输出”（--out-ndjson）
   send_data 结构为：
   {
     "psi_state": <航向角_度>,
     "x_state": <x_米>,
     "y_state": <y_米>,
     "waypoints": [{"x": <米>, "y": <米>}, ...],   // 共 n_waypoints 个点
     "n_waypoints": <整数>
   }
   expected_output_deg 为 LLM 期望返回的“转向角（度）”，可直接作为 qwen_output。

CSV 格式：
- waypoints.csv: 列为 x,y
- states.csv:    列为 t,x,y,yaw_deg,v

用法示例：
  # 默认（内置椭圆轨迹 + 合成状态）
  python pure_pursuit_make_dataset.py

  # 指定输入与参数
  python pure_pursuit_make_dataset.py \
    --waypoints path/to/waypoints.csv \
    --states path/to/states.csv \
    --lookahead 8.0 --wheelbase 2.7 \
    --n-waypoints 20 \
    --out-json out.json --out-csv out.csv --out-ndjson llm_io.ndjson

Author: your assistant
License: MIT
"""

import os
import math
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


# ------------------ Data structures ------------------
@dataclass
class State:
    t: float
    x: float
    y: float
    yaw: float  # radians
    v: float    # m/s


# ------------------ Math helpers ------------------
def normalize_angle(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def cumulative_lengths(pts: np.ndarray) -> np.ndarray:
    """Cumulative arc length of polyline points."""
    d = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s


def find_target_idx(x: float, y: float, s_path: np.ndarray, path: np.ndarray,
                    last_idx: int, lookahead: float) -> Tuple[int, int]:
    """
    返回 (target_idx, nearest_idx)：
    - nearest_idx：当前最近点索引（在闭合路径附近窗口内搜索）
    - target_idx：沿路径前进 lookahead 弧长后的索引
    """
    window = 200
    n = len(path)
    start = max(last_idx - window, 0)
    end = min(last_idx + window, n - 1)

    d = np.hypot(path[start:end, 0] - x, path[start:end, 1] - y)
    nearest_local = int(np.argmin(d))
    nearest_idx = start + nearest_local

    target_s = s_path[nearest_idx] + lookahead
    total_len = s_path[-1]
    if target_s > total_len:
        target_s -= total_len
    target_idx = int(np.searchsorted(s_path, target_s) % n)
    return target_idx, nearest_idx


def pure_pursuit_delta(state: State, target_xy, lookahead: float, L: float) -> float:
    """
    纯跟踪几何法计算前轮转角（弧度）。
    state.yaw 为车辆朝向（弧度），目标点为 (tx, ty)。
    """
    tx, ty = target_xy
    dx = tx - state.x
    dy = ty - state.y
    alpha = math.atan2(dy, dx) - state.yaw
    alpha = normalize_angle(alpha)
    delta = math.atan2(2.0 * L * math.sin(alpha), lookahead)
    return delta


# ------------------ I/O helpers ------------------
def load_waypoints(path_csv: Optional[str]) -> np.ndarray:
    """
    加载路点：
    - 若提供 CSV，则读取 x,y 两列
    - 否则生成一个默认“椭圆+两段直线”的闭合轨迹
    """
    if path_csv and os.path.exists(path_csv):
        df = pd.read_csv(path_csv)
        if not {"x", "y"}.issubset(df.columns):
            raise ValueError("Waypoints CSV must contain columns: x,y")
        return df[["x", "y"]].values.astype(float)

    # Default oval (类似赛道)
    r = 30.0
    straight = 50.0
    n = 200
    theta1 = np.linspace(np.pi / 2, -np.pi / 2, n // 2)  # 右半圆
    theta2 = np.linspace(-np.pi / 2, np.pi / 2, n // 2)  # 左半圆
    arc1_x = straight / 2 + r * np.cos(theta1)
    arc1_y = r * np.sin(theta1)
    arc2_x = -straight / 2 + r * np.cos(theta2)
    arc2_y = r * np.sin(theta2)

    s_pts = 120
    top_x = np.linspace(-straight / 2, straight / 2, s_pts)
    top_y = np.full_like(top_x, r)
    bottom_x = np.linspace(straight / 2, -straight / 2, s_pts)
    bottom_y = np.full_like(bottom_x, -r)

    x = np.concatenate([top_x, arc1_x, bottom_x, arc2_x])
    y = np.concatenate([top_y, arc1_y, bottom_y, arc2_y])
    return np.vstack([x, y]).T


def load_states(path_csv: Optional[str], default_speed: float, dt: float, sim_time: float,
                path_waypoints: np.ndarray, lookahead: float, wheelbase: float) -> List[State]:
    """
    加载或合成车辆状态时间序列：
    - 若有 CSV，则读取 t,x,y,yaw_deg,v
    - 否则在轨迹附近合成一段简单的运动（用纯跟踪推进朝向）
    """
    if path_csv and os.path.exists(path_csv):
        df = pd.read_csv(path_csv)
        required = {"t", "x", "y", "yaw_deg", "v"}
        if not required.issubset(df.columns):
            raise ValueError("States CSV must contain columns: t,x,y,yaw_deg,v")
        states: List[State] = []
        for _, row in df.iterrows():
            states.append(State(
                t=float(row["t"]),
                x=float(row["x"]),
                y=float(row["y"]),
                yaw=math.radians(float(row["yaw_deg"])),
                v=float(row["v"]),
            ))
        return states

    # 合成：从第一点附近开始，以恒定速度前进，并用纯跟踪的转角推进航向
    s_path = cumulative_lengths(path_waypoints)
    state = State(
        t=0.0,
        x=float(path_waypoints[0, 0] - 2.0),
        y=float(path_waypoints[0, 1] - 5.0),
        yaw=0.0,
        v=float(default_speed),
    )
    states = [state]
    last_idx = 0
    steps = int(sim_time / dt)
    for _ in range(steps):
        target_idx, last_idx = find_target_idx(state.x, state.y, s_path, path_waypoints, last_idx, lookahead)
        tx, ty = path_waypoints[target_idx]
        delta = pure_pursuit_delta(state, (tx, ty), lookahead, wheelbase)

        # 简单单轨运动学推进
        x_next = state.x + state.v * math.cos(state.yaw) * dt
        y_next = state.y + state.v * math.sin(state.yaw) * dt
        yaw_next = normalize_angle(state.yaw + (state.v / wheelbase) * math.tan(delta) * dt)

        state = State(t=state.t + dt, x=x_next, y=y_next, yaw=yaw_next, v=state.v)
        states.append(state)
    return states


def extract_ahead_waypoints(path: np.ndarray, start_idx: int, n: int) -> List[dict]:
    """
    从闭合路径中取连续 n 个点（从 start_idx 往前），打包为 [{"x":..., "y":...}, ...]。
    该结构能直接 json 序列化，并与 CARLA 侧 socket 发送格式一致。
    """
    m = len(path)
    pts: List[dict] = []
    for i in range(n):
        j = (start_idx + i) % m
        x, y = float(path[j, 0]), float(path[j, 1])
        pts.append({"x": x, "y": y})
    return pts


# ------------------ Main builder ------------------
def build_dataset(waypoints_csv: Optional[str],
                  states_csv: Optional[str],
                  wheelbase: float,
                  lookahead: float,
                  dt: float,
                  sim_time: float,
                  default_speed: float,
                  out_json: str,
                  out_csv: str,
                  n_waypoints: int,
                  out_ndjson: str):
    """
    计算纯跟踪记录，并输出 JSON / CSV / NDJSON 三种文件。
    NDJSON 每行包含：
      {
        "t": <时刻>,
        "send_data": {  # 与 CARLA 侧发送到 LLM 的结构一致
          "psi_state": <度>,
          "x_state": <米>,
          "y_state": <米>,
          "waypoints": [{"x":..,"y":..}, ...],
          "n_waypoints": <int>
        },
        "expected_output_deg": <float>  # LLM 期望返回的角度（度）
      }
    """
    path = load_waypoints(waypoints_csv)
    s_path = cumulative_lengths(path)
    states = load_states(states_csv, default_speed, dt, sim_time, path, lookahead, wheelbase)

    records = []
    llm_io_rows = []

    last_idx = 0
    for st in states:
        # 查找当前最近点与目标点
        target_idx, last_idx = find_target_idx(st.x, st.y, s_path, path, last_idx, lookahead)
        tx, ty = path[target_idx]
        delta = pure_pursuit_delta(st, (tx, ty), lookahead, wheelbase)  # 弧度
        delta_deg = math.degrees(delta)

        # 生成与 CARLA 侧一致的“LLM 输入”
        waypoints_list = extract_ahead_waypoints(path, last_idx, n_waypoints)
        llm_input = {
            "psi_state": round(math.degrees(st.yaw), 5),  # 度
            "x_state": round(st.x, 5),
            "y_state": round(st.y, 5),
            "waypoints": waypoints_list,                  # [{"x":..,"y":..}, ...]
            "n_waypoints": int(n_waypoints)
        }

        llm_io_rows.append({
            "t": round(st.t, 5),
            "send_data": llm_input,
            "expected_output_deg": round(delta_deg, 7)
        })

        # 详细记录（便于整体分析/可视化）
        records.append({
            "t": round(st.t, 5),
            "state": {
                "x": round(st.x, 5),
                "y": round(st.y, 5),
                "yaw_deg": round(math.degrees(st.yaw), 5),
                "v": round(st.v, 5),
            },
            "waypoint_target_idx": int(target_idx),
            "nearest_idx": int(last_idx),
            "target_point": {"x": round(float(tx), 5), "y": round(float(ty), 5)},
            "steering": {
                "delta_rad": round(float(delta), 7),
                "delta_deg": round(delta_deg, 7),
            }
        })

    # 汇总 JSON
    dataset = {
        "metadata": {
            "algorithm": "Pure Pursuit (geometric)",
            "wheelbase_m": float(wheelbase),
            "lookahead_m": float(lookahead),
            "dt_s": float(dt),
            "generated_with_states_file": bool(states_csv and os.path.exists(states_csv)),
            "generated_with_waypoints_file": bool(waypoints_csv and os.path.exists(waypoints_csv)),
            "n_waypoints_per_step": int(n_waypoints),
            "notes": "Waypoints CSV needs columns x,y; States CSV needs columns t,x,y,yaw_deg,v."
        },
        "waypoints": [{"x": float(x), "y": float(y)} for x, y in path],
        "records": records
    }

    # 写 JSON
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # 写 CSV
    rows = []
    for r in records:
        rows.append({
            "t": r["t"],
            "x": r["state"]["x"],
            "y": r["state"]["y"],
            "yaw_deg": r["state"]["yaw_deg"],
            "v": r["state"]["v"],
            "nearest_idx": r["nearest_idx"],
            "target_idx": r["waypoint_target_idx"],
            "target_x": r["target_point"]["x"],
            "target_y": r["target_point"]["y"],
            "delta_deg": r["steering"]["delta_deg"],
            "delta_rad": r["steering"]["delta_rad"],
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # 写 NDJSON（每行一条 LLM I/O）
    os.makedirs(os.path.dirname(out_ndjson) or ".", exist_ok=True)
    with open(out_ndjson, "w", encoding="utf-8") as f:
        for row in llm_io_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return out_json, out_csv, out_ndjson


# ------------------ CLI ------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate Pure Pursuit JSON/CSV dataset and LLM I/O NDJSON.")
    p.add_argument("--waypoints", type=str, default=None, help="Path to waypoints.csv (columns: x,y).")
    p.add_argument("--states", type=str, default=None, help="Path to states.csv (columns: t,x,y,yaw_deg,v).")
    p.add_argument("--wheelbase", type=float, default=2.9, help="Wheelbase in meters.")
    p.add_argument("--lookahead", type=float, default=6.0, help="Look-ahead distance in meters.")
    p.add_argument("--dt", type=float, default=0.05, help="Timestep (s) if synthesizing states.")
    p.add_argument("--sim-time", type=float, default=15.0, help="Sim duration (s) if synthesizing states.")
    p.add_argument("--default-speed", type=float, default=8.0, help="Speed (m/s) if synthesizing states.")
    p.add_argument("--n-waypoints", type=int, default=20, help="Number of waypoints to send each step.")
    p.add_argument("--out-json", type=str, default="pure_pursuit_dataset.json", help="Output JSON path.")
    p.add_argument("--out-csv", type=str, default="pure_pursuit_records.csv", help="Output CSV path.")
    p.add_argument("--out-ndjson", type=str, default="llm_io.ndjson", help="Per-step LLM I/O (NDJSON).")
    return p.parse_args()


def main():
    args = parse_args()
    out_json, out_csv, out_ndjson = build_dataset(
        waypoints_csv=args.waypoints,
        states_csv=args.states,
        wheelbase=args.wheelbase,
        lookahead=args.lookahead,
        dt=args.dt,
        sim_time=args.sim_time,
        default_speed=args.default_speed,
        out_json=args.out_json,
        out_csv=args.out_csv,
        n_waypoints=args.n_waypoints,
        out_ndjson=args.out_ndjson,
    )
    print("Wrote:", out_json)
    print("Wrote:", out_csv)
    print("Wrote:", out_ndjson)


if __name__ == "__main__":
    main()

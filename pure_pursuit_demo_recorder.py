# -*- coding: utf-8 -*-
"""
CARLA Pure Pursuit demonstration -> JSONL (inputs/outputs)
输入: psi_state, x_state, y_state, waypoints, n_waypoints
输出: steering_angle (rad)
"""

import os, sys, math, time, random, argparse
from typing import List, Tuple, Optional
import numpy as np
import simplejson as json

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default="Town03")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--duration-sec", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=42)

    # Pure Pursuit
    ap.add_argument("--wheelbase", type=float, default=2.8)      # L
    ap.add_argument("--lookahead-a", type=float, default=4.0)    # Ld = a + b*v
    ap.add_argument("--lookahead-b", type=float, default=0.6)
    ap.add_argument("--max-steer-deg", type=float, default=35.0) # 仅用于 steer 归一化
    ap.add_argument("--target-speed", type=float, default=10.0)  # m/s

    ap.add_argument("--jsonl", default="out/pure_pursuit_io.jsonl")
    ap.add_argument("--carla-egg", default=None)
    return ap.parse_args()

def setup_carla_egg(path: Optional[str]):
    if path and os.path.exists(path):
        sys.path.append(path); return
    for root in ["~/CARLA_0.9.14/PythonAPI/carla/dist",
                 "~/CARLA_0.9.13/PythonAPI/carla/dist"]:
        d = os.path.expanduser(root)
        if os.path.isdir(d):
            eggs = [os.path.join(d, f) for f in os.listdir(d) if f.startswith("carla-") and f.endswith(".egg")]
            if eggs:
                sys.path.append(sorted(eggs)[-1]); return

def world_to_body(x, y, yaw_deg, px, py):
    dyaw = math.radians(yaw_deg)
    dx, dy = x - px, y - py
    c, s = math.cos(-dyaw), math.sin(-dyaw)
    return dx * c - dy * s, dx * s + dy * c   # (ex, ey) 车体系

def pp_steer(L, Ld, ego_xy, ego_yaw_deg, target_xy):
    ex, ey = world_to_body(target_xy[0], target_xy[1], ego_yaw_deg, ego_xy[0], ego_xy[1])
    alpha = math.atan2(ey, ex)
    delta = math.atan2(2.0 * L * math.sin(alpha), Ld)  # 前轮转角（弧度，左正右负）
    return float(delta), float(alpha)

def main():
    args = parse_args()
    setup_carla_egg(args.carla_egg)
    import carla
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    os.makedirs(os.path.dirname(args.jsonl) or ".", exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed)

    client = carla.Client(args.host, args.port); client.set_timeout(10.0)
    world = client.load_world(args.town)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)
    tm = client.get_trafficmanager(); tm.set_synchronous_mode(True)

    bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    bp.set_attribute("role_name", "hero")
    spawns = world.get_map().get_spawn_points()
    start = random.choice(spawns)
    goal  = random.choice([s for s in spawns if s.location.distance(start.location) > 80.0])

    vehicle = world.spawn_actor(bp, start)
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(start.location + carla.Location(z=40), carla.Rotation(pitch=-90)))
    world.tick(); start_time = world.get_snapshot().timestamp.elapsed_seconds

    amap = world.get_map()
    grp = GlobalRoutePlanner(amap, sampling_resolution=2.0)
    route = grp.trace_route(start.location, goal.location)
    way_xy = [(wp.transform.location.x, wp.transform.location.y) for wp, _ in route]

    def cumdist(xy):
        d = [0.0]
        for i in range(1, len(xy)): d.append(d[-1] + float(np.hypot(xy[i][0]-xy[i-1][0], xy[i][1]-xy[i-1][1])))
        return np.asarray(d)
    cd = cumdist(way_xy)

    max_steer = math.radians(args.max-steer-deg) if hasattr(args, "max-steer-deg") else math.radians(args.max_steer_deg)

    def compute_Ld(v): return max(1.0, args.lookahead_a + args.lookahead_b * max(0.0, v))

    f = open(args.jsonl, "w", encoding="utf-8")
    idx = 0
    try:
        while True:
            world.tick()
            t = world.get_snapshot().timestamp.elapsed_seconds - start_time
            if t >= args.duration_sec: break

            tr = vehicle.get_transform(); vel = vehicle.get_velocity()
            px, py = tr.location.x, tr.location.y
            yaw_deg = tr.rotation.yaw
            speed = float(np.hypot(vel.x, vel.y))
            Ld = compute_Ld(speed)

            # 找到沿线路长 s 最近到 (px,py) 的索引，再向前找到 s+Ld 的点作为目标
            # 先邻近搜索（窗口 60 个点）
            search_from = idx
            best_i, best_d = search_from, 1e9
            for i in range(search_from, min(len(way_xy), search_from + 60)):
                d = float(np.hypot(way_xy[i][0]-px, way_xy[i][1]-py))
                if d < best_d: best_d, best_i = d, i
            s0 = cd[best_i]; sT = s0 + Ld
            j = best_i
            while j + 1 < len(way_xy) and cd[j] < sT: j += 1
            idx = j
            target_xy = way_xy[j]

            # 计算转向角（弧度）
            delta_rad, _ = pp_steer(args.wheelbase, Ld, (px, py), yaw_deg, target_xy)
            steer_norm = max(-1.0, min(1.0, delta_rad / max_steer))

            # 施加控制（跟速 + 转向）
            throttle = max(0.0, min(1.0, (args.target_speed - speed) * 0.2))
            brake = 0.0
            if speed - args.target_speed > 1.0:
                brake, throttle = min(0.5, (speed - args.target_speed) * 0.1), 0.0
            vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer_norm, brake=brake))

            # —— 写一条「输入→输出」样本 —— #
            # 输入：psi/x/y + 当前时刻的局部 waypoints 切片（示例取前方 20 个点）
            horizon = 20
            wps = way_xy[j : min(len(way_xy), j + horizon)]
            sample = {
                "psi_state": math.radians(yaw_deg) / 1.0,  # 存弧度
                "x_state": px,
                "y_state": py,
                "waypoints": [[float(x), float(y)] for (x, y) in wps],
                "n_waypoints": len(wps),
                # 输出：
                "steering_angle": float(delta_rad),        # 弧度
                # 可选字段：方便调试或训练时筛选
                "meta": {
                    "t": round(t, 4),
                    "speed": speed,
                    "Ld": Ld,
                    "route_idx": int(j),
                    "steer_norm": steer_norm
                }
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            # 到达终点
            if cd[-1] - cd[j] < 5.0: break

            # 跟随视角
            spectator.set_transform(carla.Transform(
                tr.location + carla.Location(x=-8*math.cos(math.radians(yaw_deg)),
                                             y=-8*math.sin(math.radians(yaw_deg)),
                                             z=4.0),
                carla.Rotation(pitch=-15, yaw=yaw_deg+180)))
    finally:
        f.close()
        try: vehicle.destroy()
        except: pass
        settings.synchronous_mode = False
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)

    print(f"[OK] samples saved to {args.jsonl}")

if __name__ == "__main__":
    main()

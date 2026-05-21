import re
import pandas as pd
import numpy as np
import os

# --- 物理常数与预处理配置 ---
GM = 3.986004418e14  # 地心引力常数
OMEGA_E = 7.2921151467e-5  # 地球自转角速度
CUT_OFF_ANGLE = 15.0  # 高度角遮蔽角（低于15度剔除）


def format_sat_id(sat_str):
    """统一卫星编号格式，将 '1' 或 'C1' 统一转为 'C01'"""
    sat_str = sat_str.strip()
    match = re.search(r'\d+', sat_str)
    if not match: return sat_str
    num = int(match.group())
    prefix = 'C'
    if 'G' in sat_str.upper():
        prefix = 'G'
    elif 'E' in sat_str.upper():
        prefix = 'E'
    elif 'R' in sat_str.upper():
        prefix = 'R'
    return f"{prefix}{num:02d}"


def calculate_sat_elevation(row, rec_pos):
    """计算卫星高度角（用于初步筛选低仰角卫星）"""
    if rec_pos == [0, 0, 0] or pd.isna(row['sqrtA']):
        return -90
    try:
        sqrtA, e, i0, M0, omega, OMEGA0, dn, odot, toe = \
            row['sqrtA'], row['e'], row['i0'], row['M0'], row['omega'], row['OMEGA0'], \
                row['Delta_n'], row['OMEGA_DOT'], row['toe']

        n = np.sqrt(GM) / (sqrtA ** 3) + dn
        Mk = M0 + n * 0
        Ek = Mk
        for _ in range(5): Ek = Mk + e * np.sin(Ek)

        vk = np.arctan2(np.sqrt(1 - e ** 2) * np.sin(Ek), np.cos(Ek) - e)
        uk = vk + omega
        rk = (sqrtA ** 2) * (1 - e * np.cos(Ek))

        xk_orb, yk_orb = rk * np.cos(uk), rk * np.sin(uk)
        Omega_k = OMEGA0 - OMEGA_E * toe

        sx = xk_orb * np.cos(Omega_k) - yk_orb * np.cos(i0) * np.sin(Omega_k)
        sy = xk_orb * np.sin(Omega_k) + yk_orb * np.cos(i0) * np.cos(Omega_k)
        sz = yk_orb * np.sin(i0)

        dx = np.array([sx, sy, sz]) - np.array(rec_pos)
        rho_rec = np.linalg.norm(rec_pos)
        lat, lon = np.arcsin(rec_pos[2] / rho_rec), np.arctan2(rec_pos[1], rec_pos[0])
        up = np.cos(lat) * np.cos(lon) * dx[0] + np.cos(lat) * np.sin(lon) * dx[1] + np.sin(lat) * dx[2]
        return np.degrees(np.arcsin(up / np.linalg.norm(dx)))
    except:
        return -90


def parse_nav_v5(nav_path):
    """升级版星历解析：完整提取包含 Cuc, Cus, Crc, Crs, Cic, Cis 在内的全部摄动参数"""
    if not os.path.exists(nav_path): return pd.DataFrame()
    eph_list = []
    with open(nav_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    header_end = next((i for i, l in enumerate(lines) if "END OF HEADER" in l), -1) + 1
    idx = header_end
    while idx < len(lines):
        line = lines[idx]
        if re.match(r'^[A-Z]\d{2}', line):
            try:
                prn = format_sat_id(line[0:3])
                dt = pd.to_datetime(
                    f"{line[4:8]}-{line[9:11]}-{line[12:14]} {line[15:17]}:{line[18:20]}:{int(float(line[21:23]))}")
                af0 = float(line[23:42].replace('D', 'E'))
                af1 = float(line[42:61].replace('D', 'E'))
                af2 = float(line[61:80].replace('D', 'E'))

                block = []
                for j in range(1, 8):
                    l = lines[idx + j]
                    for k in range(4):
                        fld = l[4 + k * 19: 4 + (k + 1) * 19].strip()
                        if fld: block.append(float(fld.replace('D', 'E')))

                # 完整映射 RINEX 广播星历的所有核心参数与摄动参数
                eph_list.append({
                    'Satellite_ID': prn, 'Nav_Time': dt, 'af0': af0, 'af1': af1, 'af2': af2,
                    'Crs': block[1], 'Delta_n': block[2], 'M0': block[3],
                    'Cuc': block[4], 'e': block[5], 'Cus': block[6], 'sqrtA': block[7],
                    'toe': block[8], 'Cic': block[9], 'OMEGA0': block[10], 'Cis': block[11],
                    'i0': block[12], 'Crc': block[13], 'omega': block[14], 'OMEGA_DOT': block[15],
                    'IDOT': block[16]
                })
            except:
                pass
            idx += 8
        else:
            idx += 1
    return pd.DataFrame(eph_list)


def parse_obs_v4(obs_path):
    if not os.path.exists(obs_path): return pd.DataFrame(), [0, 0, 0]
    rec_pos, obs_list, obs_types = [0, 0, 0], [], []
    with open(obs_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    header_end = 0
    for i, line in enumerate(lines):
        if "APPROX POSITION XYZ" in line: rec_pos = [float(x) for x in line[:60].split()]
        if "# / TYPES OF OBSERV" in line: obs_types.extend(line[:60].split()[1:])
        if "END OF HEADER" in line: header_end = i + 1; break
    c1_idx = obs_types.index('C1') if 'C1' in obs_types else -1
    s1_idx = obs_types.index('S1') if 'S1' in obs_types else -1
    curr = header_end
    while curr < len(lines):
        line = lines[curr]
        match = re.match(r'^\s*(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+([\d.]+)', line)
        if not match: curr += 1; continue
        try:
            yy = int(match.group(1))
            year = 2000 + yy
            dt = pd.to_datetime(
                f"{year}-{match.group(2).zfill(2)}-{match.group(3).zfill(2)} {match.group(4).zfill(2)}:{match.group(5).zfill(2)}:{int(float(match.group(6)))}")
            num_sats = int(line[29:32])
            sat_line_cnt = (num_sats + 11) // 12
            sats = []
            for j in range(sat_line_cnt):
                s_line = lines[curr + j][32:68]
                for k in range(0, len(s_line.strip()), 3):
                    sid = s_line[k:k + 3].strip()
                    if sid: sats.append(format_sat_id(sid))
            curr += sat_line_cnt
            for sat in sats:
                data_block = ""
                l_per_sat = (len(obs_types) + 4) // 5
                for _ in range(l_per_sat):
                    if curr < len(lines): data_block += lines[curr][:80].ljust(80); curr += 1
                vals = [data_block[k * 16: k * 16 + 14].strip() for k in range(len(obs_types))]
                c1_val = float(vals[c1_idx]) if c1_idx != -1 and vals[c1_idx] else 0
                if c1_val > 0:
                    obs_list.append({'Epoch_Time': dt, 'Satellite_ID': sat, 'C1_Pseudorange': c1_val,
                                     'S1_SNR': vals[s1_idx] if s1_idx != -1 else ""})
        except:
            curr += 1; continue
    return pd.DataFrame(obs_list), rec_pos


if __name__ == "__main__":
    nav_f, obs_f = 'nav.txt', 'obs.txt'
    print("1. 正在深度解析原始文件（提取完整轨道与摄动参数）...")
    df_nav = parse_nav_v5(nav_f)
    df_obs, rec_pos = parse_obs_v4(obs_f)

    if not df_nav.empty and not df_obs.empty:
        # 2. 时间对齐合并
        df_nav = df_nav.sort_values(['Nav_Time', 'Satellite_ID']).drop_duplicates(['Nav_Time', 'Satellite_ID'])
        df_obs = df_obs.sort_values(['Epoch_Time', 'Satellite_ID'])
        print("2. 正在进行时间流合并对齐...")
        df_final = pd.merge_asof(df_obs, df_nav, left_on='Epoch_Time', right_on='Nav_Time', by='Satellite_ID',
                                 direction='backward', tolerance=pd.Timedelta('4 hours'))

        # 3. 预处理过滤
        print("3. 正在执行预处理（粗差剔除、高度角初步筛选）...")
        df_final = df_final.dropna(subset=['sqrtA']).copy()
        df_final = df_final[(df_final['C1_Pseudorange'] > 1.8e7) & (df_final['C1_Pseudorange'] < 4.5e7)]

        # 计算高度角并按遮蔽角（15度）过滤
        df_final['Elevation'] = df_final.apply(lambda r: calculate_sat_elevation(r, rec_pos), axis=1)
        df_final = df_final[df_final['Elevation'] >= CUT_OFF_ANGLE]

        # 4. 排序对齐
        df_final = df_final.sort_values(['Epoch_Time', 'Satellite_ID'])

        # 保存为新的中间成果文件名
        output_name = 'gnss_preprocessed_upgrade_data.csv'
        df_final.to_csv(output_name, index=False, encoding='utf-8-sig')
        print(f"--- 预处理升级版完成！---")
        print(f"有效观测记录: {len(df_final)} 条")
        print(f"中间全摄动参数成果已保存至: {output_name}")
    else:
        print("解析失败，请检查数据文件是否正确。")
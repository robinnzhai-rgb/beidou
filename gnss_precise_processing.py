import os
import pandas as pd
import numpy as np

# --- 物理常数定义 (北斗系统 BDS 标准参数) ---
C_LIGHT = 299792458.0  # 真空光速 (m/s)
GM = 3.986004418e14  # 北斗地球引力常数 (m^3/s^2)
OMEGA_E = 7.2921151467e-5  # 地球自转角速度 (rad/s)
F_REL = -4.442807633e-10  # 相对论修正常数 [sec/sqrt(meter)]


def calculate_precise_sat_orbit_and_pseudorange(row):
    """
    针对单行数据，完整推算卫星 ECEF 坐标及各种物理改正量
    """
    try:
        # 1. 提取基础开普勒轨道根数与摄动参数
        sqrtA, e, i0, M0, omega, OMEGA0 = row['sqrtA'], row['e'], row['i0'], row['M0'], row['omega'], row['OMEGA0']
        dn, odot, idot, toe = row['Delta_n'], row['OMEGA_DOT'], row['IDOT'], row['toe']
        af0, af1, af2 = row['af0'], row['af1'], row['af2']

        # 提取第一步升级后保存在 CSV 中的 6 个核心轨道摄动参数
        Cuc = row['Cuc']
        Cus = row['Cus']
        Crc = row['Crc']
        Crs = row['Crs']
        Cic = row['Cic']
        Cis = row['Cis']

        # 2. 计算归化时间 tk (考虑 BDS 周跨越边界检查)
        # 将观测时间（字符串）转换为当天秒数
        t_obs = pd.to_datetime(row['Epoch_Time'])
        t_obs_sec = t_obs.hour * 3600 + t_obs.minute * 60 + t_obs.second
        tk = t_obs_sec - toe
        if tk > 302400:
            tk -= 604800
        elif tk < -302400:
            tk += 604800

        # 3. 计算平均角速度与平近点角 Mk
        n0 = np.sqrt(GM) / (sqrtA ** 3)
        n = n0 + dn
        Mk = M0 + n * tk

        # 4. 迭代求解开普勒方程（计算偏近点角 Ek）
        Ek = Mk
        for _ in range(10):
            Ek = Mk + e * np.sin(Ek)

        # 5. 求解卫星精密钟差（多项式基础钟差 + 相对论效应修正）—— 对应任务 2
        dt_sv = af0 + af1 * tk + af2 * (tk ** 2)
        dt_rel = F_REL * e * sqrtA * np.sin(Ek)
        total_dt_sv = dt_sv + dt_rel

        # 6. 计算真近点角 vk 与升交点角距 phi_k
        vk = np.arctan2(np.sqrt(1 - e ** 2) * np.sin(Ek), np.cos(Ek) - e)
        phi_k = vk + omega

        # 7. 轨道摄动修正量计算（周期修正）—— 对应任务 1
        du_k = Cuc * np.cos(2 * phi_k) + Cus * np.sin(2 * phi_k)
        dr_k = Crc * np.cos(2 * phi_k) + Crs * np.sin(2 * phi_k)
        di_k = Cic * np.cos(2 * phi_k) + Cis * np.sin(2 * phi_k)

        # 8. 计算修正后的倾角、半径和升交点角距
        uk = phi_k + du_k
        rk = (sqrtA ** 2) * (1 - e * np.cos(Ek)) + dr_k
        ik = i0 + idot * tk + di_k

        # 9. 计算轨道平面坐标并修正升交点经度 Omega_k
        xk_orb, yk_orb = rk * np.cos(uk), rk * np.sin(uk)
        Omega_k = OMEGA0 + (odot - OMEGA_E) * tk - OMEGA_E * toe

        # 10. 转换至 ECEF 坐标系下的空间位置 X, Y, Z —— 对应任务 1
        Sat_X = xk_orb * np.cos(Omega_k) - yk_orb * np.cos(ik) * np.sin(Omega_k)
        Sat_Y = xk_orb * np.sin(Omega_k) + yk_orb * np.cos(ik) * np.cos(Omega_k)
        Sat_Z = yk_orb * np.sin(ik)

        # 11. 大气延迟计算（对流层 Saastamoinen 模型 + 电离层经验模型）—— 对应任务 3
        E_rad = np.radians(row['Elevation'])
        tropo_delay = (0.002277 / np.sin(E_rad)) * 1013.25  # Saastamoinen 模型修正量（米）
        iono_delay = 5.0 / np.sin(E_rad + np.radians(2.5))  # 电离层简化经验修正量（米）

        # 12. 最终伪距修正方程 —— 对应任务 2 & 3
        P_raw = row['C1_Pseudorange']
        Corrected_P = P_raw + (total_dt_sv * C_LIGHT) - tropo_delay - iono_delay

        return pd.Series([Sat_X, Sat_Y, Sat_Z, total_dt_sv, tropo_delay, iono_delay, Corrected_P])
    except:
        return pd.Series([np.nan] * 7)


if __name__ == "__main__":
    # 已经修改输入文件名为第一步生成的升级版中间CSV文件
    input_file = 'gnss_preprocessed_upgrade_data.csv'
    output_file = 'gnss_final_corrected_data.csv'

    print("====== GNSS 精密物理修正解算程序 ======")
    if not os.path.exists(input_file):
        print(f"错误：未找到输入文件 '{input_file}'，请先确保预处理升级版脚本已成功运行并生成该文件。")
    else:
        print(f"1. 正在读取预处理升级版数据: {input_file} ...")
        df = pd.read_csv(input_file)

        print("2. 正在执行高精度物理修正解算...")
        print("   -> 任务1: 计算包含【轨道摄动修正】的卫星 ECEF 坐标 (Sat_X, Sat_Y, Sat_Z)")
        print("   -> 任务2: 完成【多项式修正+相对论效应】卫星钟差计算 (SV_Clock_Bias)")
        print("   -> 任务3: 采用【Saastamoinen与简化经验模型】扣除对流层及电离层延迟 (Corrected_P)")

        # 新增输出的高精度成果指标列名
        new_columns = ['Sat_X', 'Sat_Y', 'Sat_Z', 'SV_Clock_Bias', 'Tropo_Delay', 'Iono_Delay', 'Corrected_P']

        # 批量利用 DataFrame 的 apply 机制在内存中动态解算
        df[new_columns] = df.apply(calculate_precise_sat_orbit_and_pseudorange, axis=1)

        # 清除由于个别星历异常导致计算失败的死点
        df_clean = df.dropna(subset=['Sat_X']).copy()

        print(f"3. 正在将物理全修正成果保存至新文件: {output_file} ...")
        df_clean.to_csv(output_file, index=False, encoding='utf-8-sig')

        print("\n" + "=" * 45)
        print(f" 物理修正解算全部圆满完成！")
        print(f" 读取有效数据: {len(df)} 行 -> 最终高精度输出: {len(df_clean)} 行")
        print(f" 最终修正数据成果已完整保存在: {os.path.abspath(output_file)}")
        print("=" * 45)
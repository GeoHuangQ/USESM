import numpy as np
import pandas as pd
import math
import pylab as pl
import copy
import matplotlib.dates as mdate
import random
import pickle


def initiate_pop(pat_locator, initial_inf, initial_exp, pat_pop):
    n_pat = pat_locator.shape[0]
    temp = np.zeros(n_pat)
    initiate_data = {
        'nInf': copy.deepcopy(initial_inf),
        'nExp': copy.deepcopy(initial_exp),
        'nRec': copy.deepcopy(temp),
        'nTotal': pat_pop,
        'nRecoveredToday': copy.deepcopy(temp),
        'nInfectedToday': copy.deepcopy(temp),
        'nExposedToday': copy.deepcopy(temp),
        'nInfMovedToday': copy.deepcopy(temp),
    }
    return initiate_data


def roulette_selection(fitness):
    sum_fits = sum(fitness)
    rnd_point = random.uniform(0, sum_fits)
    accumulator = 0.0
    for ind, val in enumerate(fitness):
        accumulator += val
        if accumulator >= rnd_point:
            return ind


def recovery(initiate_data, rec_rate_pd, current_day):
    rec_rate = rec_rate_pd[rec_rate_pd["date"] == current_day].iloc[0, 1]
    if rec_rate > 0:
        rec_rate = 1 / rec_rate
    nInf = np.around(initiate_data["nInf"])
    for item in range(len(nInf)):
        initiate_data["nRecoveredToday"][item] = np.random.binomial(nInf[item], rec_rate, size=1)[0]
        initiate_data["nInf"][item] = initiate_data["nInf"][item] - initiate_data["nRecoveredToday"][item]
        initiate_data["nRec"][item] = initiate_data["nRec"][item] + initiate_data["nRecoveredToday"][item]
    return initiate_data


def exp_to_inf(initiate_data, exp_inf_rate):
    nExp = np.around(initiate_data["nExp"])
    for item in range(len(nExp)):
        initiate_data["nInfectedToday"][item] = np.random.binomial(nExp[item], exp_inf_rate, size=1)[0]
        initiate_data["nInf"][item] = initiate_data["nInf"][item] + initiate_data["nInfectedToday"][item]
        initiate_data["nExp"][item] = initiate_data["nExp"][item] - initiate_data["nInfectedToday"][item]
    return initiate_data


def sus_to_exp(initiate_data, exp_rate_pd, current_day, exposed_pop_inf_prop, control_df_in, movement_matrix,
               poi_impact, R0_pd, poi_flow_factors,flow_pop_area_distance_impact_np):
    pop_today = np.array(((initiate_data["nTotal"])[(initiate_data["nTotal"])["date"] == current_day]).iloc[:, 2])
    control_df_out_today = np.array((control_df_in[control_df_in["date"] == current_day]).iloc[0, 1:])
    movement_local_control = flow_pop_area_distance_impact_np * control_df_out_today
    R0_change_rate = poi_impact * poi_flow_factors[0] + movement_local_control * poi_flow_factors[1]
    R0_change = (max(R0_pd) - min(R0_pd)) * R0_change_rate/5
    exp_rate = exp_rate_pd[exp_rate_pd["date"] == current_day].iloc[0, 1]
    times = len(initiate_data["nExp"])
    for item in range(times):
        infectious_pop = initiate_data["nInfMovedToday"][item] + exposed_pop_inf_prop * initiate_data["nExp"][item]
        infectious_pop = int(np.around(infectious_pop))
        sus_rate = 1 - min(1, (
                (initiate_data["nInfMovedToday"][item] + initiate_data["nExp"][item] + initiate_data["nRec"][item]) /
                pop_today[item]))
        initiate_data["nExposedToday"][item] = np.sum(
            np.random.poisson(exp_rate + R0_change[item], infectious_pop)) * sus_rate
        initiate_data["nExp"][item] = initiate_data["nExp"][item] + initiate_data["nExposedToday"][item]
    return initiate_data


def movement_time_step(initiate_data, movement_matrix, current_day, control_df_out, control_df_in):
    n_pat = len(initiate_data["nExp"])
    movement_matrix_in = np.zeros((n_pat, n_pat))
    pop_today = np.array(((initiate_data["nTotal"])[(initiate_data["nTotal"])["date"] == current_day]).iloc[:, 2])
    sum_mob_1 = np.sum(movement_matrix, axis=1)
    sum_mob = sum_mob_1 - movement_matrix.diagonal()
    out_choose = np.zeros(n_pat)
    for row in range(n_pat):
        sum_mob_this = sum_mob[row]
        pop = pop_today[row]
        pro_out = sum_mob_this / pop
        inf_out = initiate_data["nInf"][row]
        if pro_out > 1:
            pro_out = 1
        out_choose[row] = np.random.binomial(inf_out, pro_out, size=1)[0]
    control_df_out_today = np.array((control_df_out[control_df_out["date"] == current_day]).iloc[0, 1:])
    control_df_in_today = np.array((control_df_in[control_df_in["date"] == current_day]).iloc[0, 1:])
    for row in range(n_pat):
        control_out = control_df_out_today[row]
        control_in = control_df_in_today[row]
        for col in range(n_pat):
            if col == row:
                movement_matrix_in[row, col] = movement_matrix[row, col] * control_in
            else:
                movement_matrix_in[row, col] = movement_matrix[row, col] * control_out
    sum_mob = np.sum(movement_matrix_in, axis=1)
    for row in range(n_pat):
        sum_mob_this = sum_mob[row]
        for col in range(n_pat):
            if sum_mob_this > 0:
                movement_matrix_in[row, col] = movement_matrix_in[row, col] / sum_mob_this
            else:
                movement_matrix_in[row, col] = 0
    movement_matrix_move = np.zeros((n_pat, n_pat))
    for row in range(n_pat):
        inf_out = initiate_data["nInf"][row]
        sum_out = int(out_choose[row])
        choose_pro = movement_matrix_in[row, :]
        choose_pro[row] = 0
        for inf_time in range(sum_out):
            choose_id = roulette_selection(choose_pro)
            movement_matrix_move[row, choose_id] = movement_matrix_move[row, choose_id] + 1
        flow_row_temp = np.sum(movement_matrix_move, axis=1)
        movement_matrix_move[row, row] = movement_matrix_move[row, row] + (inf_out - flow_row_temp[row])
    movement_matrix_move[np.diag_indices_from(movement_matrix_move)] = 0
    initiate_data["nInfMovedToday"] = initiate_data["nInf"] - np.sum(movement_matrix_move, axis=1) + np.sum(
        movement_matrix_move, axis=0)
    return initiate_data


def run_sim(initial_inf, pat_info, initial_exp, control_df_out, control_df_in, mob_mat, day_list, rec_rate_pd,
            exp_rate_pd, expose, exposed_pop_inf_prop, TSinday, poi_impact, pat_pop, R0_pd, poi_flow_factors,
            flow_pop_area_distance_impact):
    initiate_data = initiate_pop(pat_info, initial_inf, initial_exp, pat_pop)
    if TSinday > 1:
        expose_to_inf_rate = 1 / expose
        expose = 1 / (1 - math.exp(math.log(1 - expose_to_inf_rate) / TSinday))
    all_spread = np.zeros((len(initiate_data["nInf"]), len(day_list)))
    all_spread_today = np.zeros((len(initiate_data["nInf"]), len(day_list)))
    list_sim = []
    save_file = open("C:/Users/qiang/Desktop/movement_matrix_s.pkl", "rb")
    tup = pickle.load(save_file)
    save_file.close()
    for idx, current_day in enumerate(day_list):
        if idx == 0:
            list_sim.append(np.sum(initial_inf))
            continue
        flow_pop_area_distance_impact_np = (np.array(flow_pop_area_distance_impact[str(current_day)])).reshape(-1)
        movement_matrix = copy.deepcopy(tup[str(current_day)])
        initiate_data = movement_time_step(initiate_data, movement_matrix, current_day, control_df_out, control_df_in)
        initiate_data = sus_to_exp(initiate_data, exp_rate_pd, current_day, exposed_pop_inf_prop, control_df_in,
                                   movement_matrix, poi_impact, R0_pd,poi_flow_factors,flow_pop_area_distance_impact_np)
        initiate_data = exp_to_inf(initiate_data, 1 / expose)
        initiate_data = recovery(initiate_data, rec_rate_pd, current_day)
        all_spread_today[:, idx] = initiate_data["nInfectedToday"]
        all_spread[:, idx] = initiate_data["nInf"]
        list_sim.append(np.sum(initiate_data["nInfectedToday"]))
    return np.array(list_sim), all_spread_today


def poi_process(den_poi, den_poi_factors):
    num_poi = den_poi.shape[1]
    den_poi_norm = (den_poi.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))).values
    for i in range(num_poi):
        den_poi_norm[:, i] = den_poi_norm[:, i] * den_poi_factors[i]
    poi_impact = np.sum(den_poi_norm, axis=1)
    return poi_impact


def flow_quantify(pat_locator, mob_mat, pat_pop):
    area = (np.array(pat_locator[["area"]])).reshape(-1)
    day_pd = pd.read_excel(pop_path, sheet_name="data")
    day_list = day_pd['date'].tolist()
    flow_pressure = {}
    for idx, current_day in enumerate(day_list):
        flow_temp = np.zeros(len(area))
        mob_mat_inside = mob_mat[(mob_mat["date"] == current_day) & (mob_mat["fr_pat"] == mob_mat["to_pat"])]
        pat_pop_inside = pat_pop[pat_pop["date"] == current_day]
        for item in range(len(area)):
            try:
                flow_size = mob_mat_inside[mob_mat_inside["fr_pat"] == item].iloc[0, 3]
                distance_size = mob_mat_inside[mob_mat_inside["fr_pat"] == item].iloc[0, 5]
                pop_size = pat_pop_inside[pat_pop_inside["pat_id"] == item].iloc[0, 2]
                area_size = pat_locator[pat_locator["pat_id"] == item].iloc[0, 3]
                pop_d = pop_size / area_size
                flow_d = flow_size / pop_size
                distance_area = distance_size
                k_1 = pop_d * flow_d * distance_area
                flow_temp[item] = k_1
            except:
                flow_temp[item] = 0
        area_max = flow_temp.max()
        for col in range(len(area)):
            if flow_temp[col] <= 1:
                flow_temp[col] = 0
            else:
                flow_temp[col] = np.log10(flow_temp[col]) / np.log10(area_max)
        flow_pressure[str(current_day)] = flow_temp
    flow_re = pd.DataFrame(flow_pressure)
    return flow_re


if __name__ == '__main__':
    pop_path = 'C:/Users/qiang/Desktop/pop_path.xlsx'
    pop_data = 'C:/Users/qiang/Desktop/pop_data.csv'
    flow_data = 'C:/Users/qiang/Desktop/flow_data.csv'
    pat_locator_out = pd.read_excel(pop_path, sheet_name="pat_locator")
    pat_pop_out = pd.read_csv(pop_data, encoding="gbk")
    initial_inf_out = (np.array(pd.read_excel(pop_path, sheet_name="initial_inf"))).reshape(-1)
    initial_exp_out = (np.array(pd.read_excel(pop_path, sheet_name="initial_exp"))).reshape(-1)
    control_df_out_out = pd.read_excel(pop_path, sheet_name="control_df_out")
    control_df_in_out = pd.read_excel(pop_path, sheet_name="control_df_in")
    mob_mat_out = pd.read_csv(flow_data, encoding="gbk")
    day_pd = pd.read_excel(pop_path, sheet_name="day_pd")
    day_list_out = day_pd['date'].tolist()
    rec_rate_pd_out = pd.read_excel(pop_path, sheet_name="rec_rate")
    exp_rate_pd_out = pd.read_excel(pop_path, sheet_name="R0")
    R0_pd_out = [1.4, 3.9]
    den_poi_out = pd.read_excel(pop_path, sheet_name="den_poi")
    den_poi_factors_out = [0.4, 0.1, 0.1, 0.1, 0.1, 0.2]
    expose_out = 5.2
    poi_impact_out = poi_process(den_poi_out, den_poi_factors_out)
    poi_flow_factors_out = [0.5, 0.5]
    flow_pop_area_distance_impact_out = flow_quantify(pat_locator_out, mob_mat_out, pat_pop_out)
    exposed_pop_inf_prop_out  = 0
    TSinday_out = 1
    spread_today_Rs, spread_today_matrix = run_sim(initial_inf_out, pat_locator_out, initial_exp_out,control_df_out_out, control_df_in_out, mob_mat_out,day_list_out,rec_rate_pd_out, exp_rate_pd_out, expose_out,exposed_pop_inf_prop_out, TSinday_out, poi_impact_out,pat_pop_out, R0_pd_out, poi_flow_factors_out,flow_pop_area_distance_impact_out)
    save_file = open("C:/Users/qiang/Desktop/data_patch2/" + "control_" +  "_rec_" + ".pkl", "wb")
    pickle.dump(spread_today_matrix, save_file)
    save_file.close()

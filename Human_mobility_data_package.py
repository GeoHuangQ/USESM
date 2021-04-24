import numpy as np
import pandas as pd
import pickle


pop_path = 'C:/Users/qiang/Desktop/pop_path.xlsx'
flow_data = 'C:/Users/qiang/Desktop/flow_data.csv'
mob_mat = pd.read_csv(flow_data, encoding="gbk")
initial_inf_out = (np.array(pd.read_excel(pop_path, sheet_name="initial_inf"))).reshape(-1)
day_pd = pd.read_excel(pop_path, sheet_name="data")
day_list = day_pd['date'].tolist()
temp = {}
for idx, current_day in enumerate(day_list):
    n_pat = len(initial_inf_out)
    movement_matrix = np.zeros((n_pat, n_pat))
    mob_mat_today = mob_mat[mob_mat["date"] == current_day]
    for row in range(n_pat):
        for col in range(n_pat):
            try:
                mob_data = (mob_mat_today[(mob_mat_today["fr_pat"] == row) & (mob_mat_today["to_pat"] == col)]).iloc[
                    0, 3]
                movement_matrix[row, col] = mob_data
            except:
                movement_matrix[row, col] = 0
    temp[str(current_day)] = movement_matrix
    print(current_day)

save_file = open("C:/Users/qiang/Desktop/movement_matrix_s.pkl", "wb")
pickle.dump(temp,  save_file)
save_file.close()





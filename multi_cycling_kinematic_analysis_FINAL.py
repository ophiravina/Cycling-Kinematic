
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.stats import zscore, shapiro, f_oneway, friedmanchisquare, ttest_rel
from statsmodels.stats.multitest import multipletests

# GUI for selecting files and power levels
selected_files = []

def add_file():
    filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if filepath:
        power = simpledialog.askinteger("Power Level", "Enter power level in Watts:", minvalue=1)
        if power:
            selected_files.append({"path": filepath, "power": power})
            status_label.config(text=f"{len(selected_files)} file(s) selected.")

def finish_selection():
    if not selected_files:
        messagebox.showwarning("No Files", "No files selected.")
        return
    root.quit()

root = tk.Tk()
root.title("Kinematic Analysis by Power Level")
frame = tk.Frame(root, padx=20, pady=20)
frame.pack()
tk.Button(frame, text="Add New File", command=add_file, width=30).pack(pady=5)
tk.Button(frame, text="Finish and Start Analysis", command=finish_selection, width=30).pack(pady=5)
status_label = tk.Label(frame, text="No files selected yet.", fg="blue")
status_label.pack(pady=10)
root.mainloop()
root.destroy()

# Utility functions
def find_header_and_axis_rows(filepath):
    joint_keywords = ["l_ankle", "r_ankle", "l_hip", "r_hip", "l_knee", "r_knee"]
    axis_keywords = ["x", "y", "z"]
    header_row = axis_row = None
    for i in range(30):
        try:
            row = pd.read_csv(filepath, skiprows=i, nrows=1, header=None).iloc[0].astype(str).str.lower()
            if header_row is None and sum(any(k in cell for cell in row) for k in joint_keywords) >= 4:
                header_row = i
            elif header_row is not None and sum(any(a in cell for cell in row) for a in axis_keywords) >= 3:
                axis_row = i
                break
        except:
            continue
    if header_row is not None and axis_row is not None:
        return header_row, axis_row
    raise ValueError("Header and axis rows not found.")

def compute_cross_correlation(left_series, right_series, sampling_rate=50):
    l, r = zscore(left_series), zscore(right_series)
    corr = correlate(r, l, mode='full')
    lags = np.arange(-len(l)+1, len(r))
    r_max = np.max(corr) / len(l)
    lag_index = np.argmax(corr)
    lag_time = lags[lag_index] / sampling_rate
    cycle_duration = len(l) / sampling_rate
    lag_degrees = (lag_time / cycle_duration) * 360
    return r_max, lag_degrees

def compute_nsi_series(left_series, right_series):
    left_norm = (left_series - left_series.min()) / (left_series.max() - left_series.min()) + 1
    right_norm = (right_series - right_series.min()) / (right_series.max() - right_series.min()) + 1
    return (right_norm - left_norm) / (0.5 * (right_norm + left_norm)) * 100

def analyze_file(filepath, power):
    header_row, axis_row = find_header_and_axis_rows(filepath)
    raw_df = pd.read_csv(filepath, sep="\t", skiprows=axis_row + 1, engine='python')

    joint_labels = ["L_ANKLE", "L_ANKLE", "L_ANKLE", "L_HIP", "L_HIP", "L_HIP", "L_KNEE", "L_KNEE", "L_KNEE",
                    "R_ANKLE", "R_ANKLE", "R_ANKLE", "R_HIP", "R_HIP", "R_HIP", "R_KNEE", "R_KNEE", "R_KNEE"]
    dof_axes = ['Flex/Ext_X', 'Abd/Add_Y', 'Int/Ext Rot_Z'] * 6
    column_labels = [f"{joint}_{dof}" for joint, dof in zip(joint_labels, dof_axes)]

    joint_data = raw_df.iloc[:, 1:len(column_labels)+1].copy()
    joint_data.columns = column_labels
    df_angles = joint_data.copy()

    results = []
    for joint in ['ANKLE', 'HIP', 'KNEE']:
        for axis in ['Flex/Ext_X', 'Abd/Add_Y', 'Int/Ext Rot_Z']:
            l_col, r_col = f"L_{joint}_{axis}", f"R_{joint}_{axis}"
            if l_col in df_angles.columns and r_col in df_angles.columns:
                rom_l = df_angles[l_col].max() - df_angles[l_col].min()
                rom_r = df_angles[r_col].max() - df_angles[r_col].min()
                l_min, l_max = df_angles[l_col].min(), df_angles[l_col].max()
                r_min, r_max = df_angles[r_col].min(), df_angles[r_col].max()
                nsi = compute_nsi_series(df_angles[l_col], df_angles[r_col])
                nsi_mean, nsi_abs = nsi.mean(), nsi.abs().mean()
                rmax, tlag = compute_cross_correlation(df_angles[l_col], df_angles[r_col])
                results.append({
                    "Joint": joint, "DOF": axis, "Power": power,
                    "L-ROM": rom_l, "R-ROM": rom_r, "r_max": rmax, "τ_lag (deg)": tlag,
                    "NSI_mean (%)": nsi_mean, "NSI_abs_mean (%)": nsi_abs,
                    "L-min": l_min, "L-max": l_max, "R-min": r_min, "R-max": r_max
                })
    return pd.DataFrame(results)

# ========== Run analysis ==========
all_results = []
for entry in selected_files:
    df = analyze_file(entry["path"], entry["power"])
    all_results.append(df)

df_all = pd.concat(all_results, ignore_index=True)
base_path = os.path.dirname(selected_files[0]["path"])
csv_path = os.path.join(base_path, "Combined_Kinematic_Results.csv")
df_all.to_csv(csv_path, index=False)

# ========== Generate comparison plots ==========
plot_dir = os.path.join(base_path, "Comparison_Plots")
os.makedirs(plot_dir, exist_ok=True)

metrics = ["L-ROM", "R-ROM", "NSI_mean (%)", "NSI_abs_mean (%)", "r_max", "τ_lag (deg)"]
for metric in metrics:
    for (joint, dof), group in df_all.groupby(["Joint", "DOF"]):
        group = group.sort_values("Power")
        plt.figure(figsize=(6, 4))
        plt.plot(group["Power"], group[metric], marker="o")
        plt.title(f"{metric} vs Power - {joint} {dof}")
        plt.xlabel("Power (W)"); plt.ylabel(metric)
        plt.grid(True); plt.tight_layout()
        safe_dof = dof.replace("/", "-").replace(" ", "_")
        safe_metric = metric.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
        filename = f"{joint}_{safe_dof}_{safe_metric}.png"
        plt.savefig(os.path.join(plot_dir, filename))
        plt.close()

# ========== Statistical summary ==========
stat_summary = []
for (joint, dof), group in df_all.groupby(["Joint", "DOF"]):
    powers = sorted(group["Power"].unique())
    row_base = {"Joint": joint, "DOF": dof}
    for metric in metrics:
        summary = row_base.copy()
        summary["Metric"] = metric
        values_by_power = [group[group["Power"] == p][metric].values for p in powers]
        all_vals = np.concatenate(values_by_power)
        for i, p in enumerate(powers):
            v = values_by_power[i]
            summary[f"Mean_{p}W"] = round(np.mean(v), 3) if len(v) > 0 else "N.A."
            summary[f"STD_{p}W"] = round(np.std(v), 3) if len(v) > 0 else "N.A."
        if len(all_vals) >= 3:
            stat_norm, p_norm = shapiro(all_vals)
            summary["Normality p"] = round(p_norm, 4)
            summary["Normal?"] = p_norm > 0.05
        else:
            summary["Normality p"] = "N.A."
            summary["Normal?"] = "N.A."
        try:
            if summary["Normal?"] is True:
                stat, p_val = f_oneway(*values_by_power)
                summary["Test"] = "ANOVA"; summary["p-value"] = round(p_val, 4)
            elif summary["Normal?"] is False:
                stat, p_val = friedmanchisquare(*values_by_power)
                summary["Test"] = "Friedman"; summary["p-value"] = round(p_val, 4)
            else:
                summary["Test"] = "N.A."; summary["p-value"] = "N.A."
        except:
            summary["Test"] = "N.A."; summary["p-value"] = "N.A."

        # Pairwise
        pairs = [(i, j) for i in range(len(powers)) for j in range(i+1, len(powers))]
        pvals, labels = [], []
        for i, j in pairs:
            v1, v2 = values_by_power[i], values_by_power[j]
            if len(v1) == len(v2) and len(v1) >= 2:
                try:
                    stat, p = ttest_rel(v1, v2)
                    pvals.append(p); labels.append(f"{powers[i]}W vs {powers[j]}W")
                except: continue
        if pvals:
            _, pvals_corr, _, _ = multipletests(pvals, method='bonferroni')
            for lbl, pcorr in zip(labels, pvals_corr):
                summary[f"Pairwise {lbl}"] = round(pcorr, 4)
        else:
            summary["Pairwise comparisons"] = "N.A."
        stat_summary.append(summary)

# Save summary
stat_df = pd.DataFrame(stat_summary)
with open(csv_path, 'a', encoding='utf-8') as f:
    f.write("\n\nStatistical Summary Table\n")
stat_df.to_csv(csv_path, mode='a', index=False)

print(f"✅ Results saved to: {csv_path}")
print(f"📊 Plots saved to: {plot_dir}")

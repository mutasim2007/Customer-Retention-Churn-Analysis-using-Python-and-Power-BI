# ─────────────────────────────────────────────
#  Customer Retention & Churn Analysis
#  Task 2 — Future Interns Data Science
#  Mutasim Ahmed | 2025
# ─────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──
df = pd.read_excel('/home/claude/Customer_Data_MutasimAhmed.xlsx',
                   sheet_name='Customer_Data')
df['SignupDate'] = pd.to_datetime(df['SignupDate'])
df['ChurnDate']  = pd.to_datetime(df['ChurnDate'], errors='coerce')

print(f"Loaded {len(df)} customers")
print(df['Status'].value_counts())
print(f"\nChurn Rate: {(df['Status']=='Churned').mean():.1%}")

# ── Cohort Retention Matrix ──
df['CohortMonth'] = df['SignupDate'].dt.to_period('M')
cohorts = df.groupby('CohortMonth')['CustomerID'].count().reset_index()
cohorts.columns = ['CohortMonth','CohortSize']

retention_data = []
for cohort_month in df['CohortMonth'].unique():
    cohort_df = df[df['CohortMonth'] == cohort_month]
    size = len(cohort_df)
    for m in range(1, 19):
        active = (cohort_df['TenureMonths'] >= m).sum()
        retention_data.append({
            'CohortMonth': cohort_month,
            'Month': m,
            'Retained': active,
            'RetentionRate': active / size
        })

ret_df = pd.DataFrame(retention_data)
pivot = ret_df.pivot(index='CohortMonth', columns='Month', values='RetentionRate')
pivot = pivot.sort_index().head(18)

# ─────────────────────────────────────────────
#  FIGURE 1 — COHORT HEATMAP
# ─────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(16, 8))
fig1.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

import matplotlib.colors as mcolors
cmap = plt.cm.RdYlGn
norm = mcolors.Normalize(vmin=0, vmax=1)

for i, (idx, row) in enumerate(pivot.iterrows()):
    for j, val in enumerate(row):
        if pd.notna(val):
            color = cmap(norm(val))
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color, ec='white', lw=1.5))
            txt_color = 'white' if val < 0.35 or val > 0.75 else '#1a1a1a'
            ax.text(j + 0.5, i + 0.5, f'{val:.0%}',
                    ha='center', va='center', fontsize=8.5,
                    fontweight='600', color=txt_color, fontfamily='DejaVu Sans')

ax.set_xlim(0, 18)
ax.set_ylim(0, len(pivot))
ax.set_xticks([x + 0.5 for x in range(18)])
ax.set_xticklabels([f'M{m}' for m in range(1, 19)], fontsize=9, color='#444')
ax.set_yticks([y + 0.5 for y in range(len(pivot))])
ax.set_yticklabels([str(p) for p in pivot.index], fontsize=9, color='#444')
ax.set_xlabel('Month After Signup', fontsize=11, color='#333', labelpad=8)
ax.set_ylabel('Cohort (Signup Month)', fontsize=11, color='#333', labelpad=8)
ax.set_title('Customer Cohort Retention Heatmap\nRetention Rate by Signup Cohort and Month',
             fontsize=14, fontweight='bold', color='#1F4E79', pad=14)

# colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
cbar.set_label('Retention Rate', fontsize=9, color='#444')
cbar.ax.tick_params(labelsize=8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig('/home/claude/cohort_heatmap.png', dpi=150, bbox_inches='tight',
            facecolor='#FAFAFA')
plt.close()
print("cohort_heatmap.png saved")

# ─────────────────────────────────────────────
#  FIGURE 2 — FULL DASHBOARD
# ─────────────────────────────────────────────
fig2 = plt.figure(figsize=(20, 14))
fig2.patch.set_facecolor('#F0F4F8')

gs = GridSpec(3, 4, figure=fig2, hspace=0.45, wspace=0.35,
              top=0.88, bottom=0.06, left=0.06, right=0.97)

BLUE   = '#1F4E79'
TEAL   = '#2E86AB'
RED    = '#E63946'
GREEN  = '#2DC653'
AMBER  = '#F4A261'
LIGHT  = '#EBF3FB'
GRAY   = '#6B7280'

# ── Header ──
fig2.text(0.5, 0.94, 'Customer Retention & Churn Analysis Dashboard',
          ha='center', va='center', fontsize=18, fontweight='bold',
          color='white',
          bbox=dict(boxstyle='round,pad=0.5', facecolor=BLUE, edgecolor=BLUE))
fig2.text(0.5, 0.905, 'Task 2  ·  Future Interns Data Science  ·  Mutasim Ahmed  ·  2025',
          ha='center', va='center', fontsize=10, color=GRAY)

# ── KPI Cards ──
churned  = df[df['Status']=='Churned']
active   = df[df['Status']=='Active']
churn_rt = len(churned)/len(df)
avg_clv  = df['CLV'].mean()
avg_nps  = df['NPSScore'].mean()
mrr      = active['MonthlyRevenue'].sum()

kpis = [
    ('Total Customers', f'{len(df):,}',         BLUE),
    ('Churn Rate',      f'{churn_rt:.1%}',       RED),
    ('Monthly Revenue', f'${mrr:,.0f}',          GREEN),
    ('Avg CLV',         f'${avg_clv:,.0f}',      TEAL),
    ('Avg NPS Score',   f'{avg_nps:.1f}/10',     AMBER),
    ('Active Customers',f'{len(active):,}',      GREEN),
    ('Churned',         f'{len(churned):,}',     RED),
    ('Avg Tenure',      f'{active["TenureMonths"].mean():.1f} mo', TEAL),
]

kpi_ax_positions = [
    (0.04, 0.825, 0.10, 0.06), (0.155, 0.825, 0.10, 0.06),
    (0.27, 0.825, 0.10, 0.06), (0.385, 0.825, 0.10, 0.06),
    (0.50, 0.825, 0.10, 0.06), (0.615, 0.825, 0.10, 0.06),
    (0.73, 0.825, 0.10, 0.06), (0.845, 0.825, 0.10, 0.06),
]
for (label, value, color), (x, y, w, h) in zip(kpis, kpi_ax_positions):
    ax_kpi = fig2.add_axes([x, y, w, h])
    ax_kpi.set_facecolor('white')
    for spine in ax_kpi.spines.values():
        spine.set_edgecolor('#D1D5DB')
        spine.set_linewidth(0.8)
    ax_kpi.set_xticks([]); ax_kpi.set_yticks([])
    ax_kpi.add_patch(mpatches.FancyBboxPatch((0,0.6), 1, 0.08,
        boxstyle='square', facecolor=color, edgecolor='none', transform=ax_kpi.transAxes))
    ax_kpi.text(0.5, 0.78, value, ha='center', va='center',
                fontsize=14, fontweight='bold', color=color,
                transform=ax_kpi.transAxes)
    ax_kpi.text(0.5, 0.22, label, ha='center', va='center',
                fontsize=7.5, color=GRAY, transform=ax_kpi.transAxes)

# ── Chart 1: Churn by Plan ──
ax1 = fig2.add_subplot(gs[0, 0])
plan_churn = df.groupby('Plan').apply(
    lambda x: (x['Status']=='Churned').mean()).reindex(['Basic','Standard','Premium'])
bars = ax1.bar(plan_churn.index, plan_churn.values * 100,
               color=[RED, AMBER, GREEN], edgecolor='white', linewidth=1.5, width=0.55)
for bar, val in zip(bars, plan_churn.values):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
             f'{val:.1%}', ha='center', va='bottom', fontsize=9, fontweight='600')
ax1.set_title('Churn Rate by Plan', fontweight='bold', color=BLUE, fontsize=11)
ax1.set_ylabel('Churn Rate (%)', fontsize=8.5, color=GRAY)
ax1.set_facecolor('#FAFAFA')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(labelsize=9)
ax1.set_ylim(0, 60)

# ── Chart 2: Churn Reasons ──
ax2 = fig2.add_subplot(gs[0, 1])
reason_counts = churned['ChurnReason'].value_counts().head(6)
colors_bar = [BLUE, TEAL, '#4B8BBA', '#6BA3C8', '#8CBBD6', '#AED3E4']
h_bars = ax2.barh(reason_counts.index, reason_counts.values,
                  color=colors_bar, edgecolor='white', height=0.6)
for bar, val in zip(h_bars, reason_counts.values):
    ax2.text(val + 0.5, bar.get_y()+bar.get_height()/2,
             str(val), va='center', fontsize=8.5, fontweight='600', color=BLUE)
ax2.set_title('Top Churn Reasons', fontweight='bold', color=BLUE, fontsize=11)
ax2.set_xlabel('Count', fontsize=8.5, color=GRAY)
ax2.set_facecolor('#FAFAFA')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(labelsize=8)

# ── Chart 3: Monthly Churn Trend ──
ax3 = fig2.add_subplot(gs[0, 2:])
churned2 = churned.copy()
churned2['ChurnMonth'] = churned2['ChurnDate'].dt.to_period('M')
monthly_churn = churned2.groupby('ChurnMonth').size().reset_index(name='Count')
monthly_churn = monthly_churn[monthly_churn['ChurnMonth'].astype(str) >= '2022-01']
months_str = [str(m) for m in monthly_churn['ChurnMonth']]
ax3.fill_between(range(len(months_str)), monthly_churn['Count'],
                 alpha=0.18, color=RED)
ax3.plot(range(len(months_str)), monthly_churn['Count'],
         color=RED, linewidth=2.5, marker='o', markersize=4, markerfacecolor='white',
         markeredgewidth=2)
step = max(1, len(months_str)//8)
ax3.set_xticks(range(0, len(months_str), step))
ax3.set_xticklabels([months_str[i] for i in range(0, len(months_str), step)],
                    rotation=30, ha='right', fontsize=8)
ax3.set_title('Monthly Churn Trend (2022–2024)', fontweight='bold', color=BLUE, fontsize=11)
ax3.set_ylabel('Customers Churned', fontsize=8.5, color=GRAY)
ax3.set_facecolor('#FAFAFA')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.tick_params(labelsize=8.5)

# ── Chart 4: Retention by Region ──
ax4 = fig2.add_subplot(gs[1, 0])
region_ret = df.groupby('Region').apply(
    lambda x: (x['Status']=='Active').mean()).sort_values()
colors_r = [RED if v < 0.65 else AMBER if v < 0.75 else GREEN for v in region_ret.values]
h2 = ax4.barh(region_ret.index, region_ret.values * 100,
              color=colors_r, edgecolor='white', height=0.6)
for bar, val in zip(h2, region_ret.values):
    ax4.text(val*100 + 0.3, bar.get_y()+bar.get_height()/2,
             f'{val:.1%}', va='center', fontsize=8.5, fontweight='600')
ax4.set_title('Retention Rate by Region', fontweight='bold', color=BLUE, fontsize=11)
ax4.set_xlabel('Retention Rate (%)', fontsize=8.5, color=GRAY)
ax4.set_xlim(0, 105)
ax4.set_facecolor('#FAFAFA')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.tick_params(labelsize=8)

# ── Chart 5: CLV by Plan ──
ax5 = fig2.add_subplot(gs[1, 1])
clv_plan = df.groupby('Plan')['CLV'].mean().reindex(['Basic','Standard','Premium'])
ax5.bar(clv_plan.index, clv_plan.values,
        color=[TEAL, BLUE, '#0D2B4A'], edgecolor='white', linewidth=1.5, width=0.55)
for i, (plan, val) in enumerate(clv_plan.items()):
    ax5.text(i, val + 2, f'${val:.0f}', ha='center', fontsize=9, fontweight='600', color=BLUE)
ax5.set_title('Avg Customer Lifetime Value by Plan', fontweight='bold', color=BLUE, fontsize=11)
ax5.set_ylabel('Avg CLV ($)', fontsize=8.5, color=GRAY)
ax5.set_facecolor('#FAFAFA')
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.tick_params(labelsize=9)

# ── Chart 6: Engagement vs Churn scatter ──
ax6 = fig2.add_subplot(gs[1, 2])
colors_s = [RED if s == 'Churned' else TEAL for s in df['Status']]
ax6.scatter(df['EngagementScore'], df['TenureMonths'],
            c=colors_s, alpha=0.35, s=18, edgecolors='none')
ax6.set_title('Engagement Score vs Tenure', fontweight='bold', color=BLUE, fontsize=11)
ax6.set_xlabel('Engagement Score', fontsize=8.5, color=GRAY)
ax6.set_ylabel('Tenure (Months)', fontsize=8.5, color=GRAY)
ax6.set_facecolor('#FAFAFA')
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.tick_params(labelsize=8.5)
legend_elems = [mpatches.Patch(color=TEAL, label='Active'),
                mpatches.Patch(color=RED, label='Churned')]
ax6.legend(handles=legend_elems, fontsize=8, loc='upper left')

# ── Chart 7: NPS Distribution ──
ax7 = fig2.add_subplot(gs[1, 3])
nps_active  = active['NPSScore'].value_counts().sort_index()
nps_churned = churned['NPSScore'].value_counts().sort_index()
x = np.arange(1, 11)
w = 0.4
ax7.bar(x - w/2, [nps_active.get(i,0) for i in x],
        width=w, color=TEAL, label='Active', edgecolor='white')
ax7.bar(x + w/2, [nps_churned.get(i,0) for i in x],
        width=w, color=RED, label='Churned', edgecolor='white')
ax7.set_title('NPS Score Distribution', fontweight='bold', color=BLUE, fontsize=11)
ax7.set_xlabel('NPS Score (1–10)', fontsize=8.5, color=GRAY)
ax7.set_ylabel('Customers', fontsize=8.5, color=GRAY)
ax7.set_facecolor('#FAFAFA')
ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)
ax7.tick_params(labelsize=8.5)
ax7.legend(fontsize=8)

# ── Chart 8: Cohort Heatmap (mini) ──
ax8 = fig2.add_subplot(gs[2, :])
pivot_mini = pivot.head(12).iloc[:, :12]
import matplotlib.colors as mcolors
cmap2 = plt.cm.RdYlGn
norm2 = mcolors.Normalize(vmin=0.1, vmax=1.0)
for i, (idx, row) in enumerate(pivot_mini.iterrows()):
    for j, val in enumerate(row):
        if pd.notna(val):
            color = cmap2(norm2(val))
            ax8.add_patch(plt.Rectangle((j, i), 1, 1, color=color, ec='white', lw=1.2))
            tc = 'white' if val < 0.35 or val > 0.78 else '#1a1a1a'
            ax8.text(j+0.5, i+0.5, f'{val:.0%}',
                     ha='center', va='center', fontsize=7.5,
                     fontweight='600', color=tc)
ax8.set_xlim(0, 12); ax8.set_ylim(0, len(pivot_mini))
ax8.set_xticks([x+0.5 for x in range(12)])
ax8.set_xticklabels([f'M{m}' for m in range(1,13)], fontsize=8.5)
ax8.set_yticks([y+0.5 for y in range(len(pivot_mini))])
ax8.set_yticklabels([str(p) for p in pivot_mini.index], fontsize=8.5)
ax8.set_title('Cohort Retention Heatmap — First 12 Months by Signup Cohort',
              fontweight='bold', color=BLUE, fontsize=11)
for sp in ax8.spines.values(): sp.set_visible(False)

plt.savefig('/home/claude/churn_dashboard.png', dpi=150,
            bbox_inches='tight', facecolor='#F0F4F8')
plt.close()
print("churn_dashboard.png saved")

# ── Print key stats ──
print("\n=== KEY FINDINGS ===")
print(f"Overall churn rate: {churn_rt:.1%}")
for plan in ['Basic','Standard','Premium']:
    r = (df[df['Plan']==plan]['Status']=='Churned').mean()
    print(f"  {plan} churn: {r:.1%}")
print(f"\nTop churn reason: {churned['ChurnReason'].value_counts().index[0]}")
print(f"Avg CLV (all): ${avg_clv:,.0f}")
print(f"Avg tenure (active): {active['TenureMonths'].mean():.1f} months")
print(f"Avg NPS: {avg_nps:.1f}")

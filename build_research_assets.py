from pathlib import Path
import json,io,zipfile
import numpy as np,pandas as pd
from scipy.stats import friedmanchisquare,wilcoxon,pearsonr,spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score,confusion_matrix
from src.multimodal_core import feature_cols,MODALITIES,FS
ROOT=Path(__file__).resolve().parent;D=ROOT/'data'
f=pd.read_csv(D/'maus_multimodal_trial_features.csv',dtype={'subject':str})
quality=pd.read_csv(D/'signal_quality_by_trial.csv',dtype={'subject':str})
participants=pd.read_csv(D/'participant_context.csv',dtype={'subject':str})
nasa=pd.read_csv(D/'nasa_tlx_by_trial.csv',dtype={'subject':str})
perf=[]
with zipfile.ZipFile(D/'MAUS.zip') as z:
 for sid in sorted(f.subject.unique()):
  try:record=pd.read_excel(io.BytesIO(z.read(f'MAUS/Data/Raw_data/{sid}/record.xlsx')),sheet_name=0)
  except:record=pd.DataFrame()
  for t,tn in enumerate([0,2,3,2,3,0],1):
   if record.empty: continue
   rsp=pd.to_numeric(record.get(f'b{t}.Rsp',pd.Series(dtype=float)),errors='coerce').dropna()
   cr=pd.to_numeric(record.get(f'b{t}.CRsp',pd.Series(dtype=float)),errors='coerce').dropna()
   rt=pd.to_numeric(record.get(f'b{t}.RT',pd.Series(dtype=float)),errors='coerce');rt=rt[rt>0]
   perf.append({'subject':sid,'trial':t,'task_n':tn,'task_condition':f'{tn}-back','rsp_field_mean':float(rsp.mean()) if len(rsp) else None,'crsp_field_mean':float(cr.mean()) if len(cr) else None,'mean_response_time_ms':float(rt.mean()) if len(rt) else None,'median_response_time_ms':float(rt.median()) if len(rt) else None,'positive_rt_records':int(len(rt))})
perf=pd.DataFrame(perf);perf.to_csv(D/'task_response_log_summary.csv',index=False)
# Within-participant repeated-measures evidence. NASA-TLX is the subjective reference, never a model input.
measure_map={
 'NASA-TLX adjusted workload':'nasa_tlx_adjusted',
 'ECG heart rate':'ecg_hr_mean_bpm_mean',
 'ECG SDNN':'ecg_hrv_sd_ms_mean',
 'ECG RMSSD':'ecg_hrv_rmssd_ms_mean',
 'Fingertip PPG heart rate':'fppg_hr_mean_bpm_mean',
 'Fingertip PPG SDNN':'fppg_hrv_sd_ms_mean',
 'Fingertip PPG RMSSD':'fppg_hrv_rmssd_ms_mean',
 'Wrist PPG heart rate':'wppg_hr_mean_bpm_mean',
 'Wrist PPG SDNN':'wppg_hrv_sd_ms_mean',
 'Wrist PPG RMSSD':'wppg_hrv_rmssd_ms_mean',
 'GSR tonic level':'gsr_tonic_mean_mean',
 'GSR phasic variability':'gsr_phasic_sd_mean',
 'GSR SCR rate':'gsr_scr_rate_min_mean',
}
analysis=[];subject_task=[]
for label,col in measure_map.items():
 src=nasa if col=='nasa_tlx_adjusted' else f
 d=src.groupby(['subject','task_n'])[col].mean().unstack('task_n').reindex(columns=[0,2,3])
 for sid,row in d.iterrows():subject_task.append({'subject':sid,'measure':label,'0-back':row.get(0,np.nan),'2-back':row.get(2,np.nan),'3-back':row.get(3,np.nan)})
 d=d.dropna()
 if len(d)<3:continue
 stat,p=friedmanchisquare(d[0],d[2],d[3]);pair=[]
 for a,b in [(0,2),(0,3),(2,3)]:
  try:_,pp=wilcoxon(d[a],d[b],zero_method='wilcox',alternative='two-sided',method='auto')
  except:pp=np.nan
  diff=d[b]-d[a];pos=int((diff>0).sum());neg=int((diff<0).sum());rbc=(pos-neg)/(pos+neg) if pos+neg else 0
  pair.append({'contrast':f'{a}-back vs {b}-back','median_difference':float(diff.median()),'p_value':float(pp),'rank_biserial':float(rbc)})
 analysis.append({'measure':label,'feature':col,'n_subjects':int(len(d)),'mean_0back':float(d[0].mean()),'mean_2back':float(d[2].mean()),'mean_3back':float(d[3].mean()),'friedman_chi2':float(stat),'friedman_p':float(p),'kendall_w':float(stat/(len(d)*2)),'pairwise':pair})
if analysis:
 _,adj,_,_=multipletests([x['friedman_p'] for x in analysis],method='fdr_bh')
 for x,p in zip(analysis,adj):x['friedman_p_fdr_bh']=float(p)
pd.DataFrame(subject_task).to_csv(D/'subject_task_level_evidence.csv',index=False)
# Cross-sensor agreement of beat-to-beat intervals: ECG-derived RRI vs finger/wrist PPI.
agreement=[]
for ref,other,label in [('ibi_ecg_mean_ms','ibi_fppg_mean_ms','ECG vs fingertip PPG'),('ibi_ecg_mean_ms','ibi_wppg_mean_ms','ECG vs wrist PPG'),('ibi_fppg_mean_ms','ibi_wppg_mean_ms','Fingertip vs wrist PPG')]:
 a=f[ref].to_numpy(float);b=f[other].to_numpy(float);mask=np.isfinite(a)&np.isfinite(b);a=a[mask];b=b[mask];diff=b-a
 agreement.append({'comparison':label,'n_trials':int(len(diff)),'pearson_r':float(pearsonr(a,b).statistic),'spearman_r':float(spearmanr(a,b).statistic),'mean_bias_ms':float(diff.mean()),'loa_lower_ms':float(diff.mean()-1.96*diff.std(ddof=1)),'loa_upper_ms':float(diff.mean()+1.96*diff.std(ddof=1)),'mae_ms':float(np.mean(np.abs(diff))),'rmse_ms':float(np.sqrt(np.mean(diff**2)))})
# PPG reference baseline metrics are read from the existing independently grouped benchmark report and cross-checked against its OOF predictions.
old_report=json.loads((D/'model_report.json').read_text())
old_oof=pd.read_csv(D/'loso_oof_predictions.csv',dtype={'subject':str})
y=(old_oof.actual_group=='Elevated').astype(int);score=old_oof.oof_score.to_numpy();threshold=old_oof.fold_threshold.to_numpy();pred=(score>=threshold).astype(int)
ppg_ref={'model':'RandomForestClassifier','validation':'22-fold Leave-One-Subject-Out with grouped inner-threshold selection','accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),'roc_auc':float(roc_auc_score(y,score)),'confusion_matrix':confusion_matrix(y,pred).tolist(),'threshold_median':float(np.median(threshold)),'feature_count':84}
# Multimodal compact benchmark outputs, generated with the same ExtraTrees/feature-selection pipeline per modality.
comparison=[]
for mod in ['ecg','fppg','wppg','gsr','ibi','fusion']:
 path=D / f'modality_metrics_{mod}.json'
 if path.exists():comparison.append(json.loads(path.read_text()))
# If no per-modality metric cache yet, derive from the already stored summary in the caller's build.
# PCA uses the compact 100-feature panel; it is a visual projection, not a predictive validation result.
cols=feature_cols(f,'fusion');X=SimpleImputer(strategy='median',keep_empty_features=True).fit_transform(f[cols]);X=StandardScaler().fit_transform(X);coords=PCA(n_components=3,random_state=42).fit_transform(X)
pca=pd.DataFrame(coords,columns=['PC1','PC2','PC3']);pca.insert(0,'subject',f.subject.astype(str));pca['trial']=f.trial;pca['task_n']=f.task_n;pca['task_condition']=f.task_condition;pca['nasa_tlx']=f.nasa_tlx;pca['group']=f.group;pca.to_csv(D/'multimodal_pca.csv',index=False)
quality_summary={}
for mod in ['ecg','fppg','wppg','gsr']:
 quality_summary[mod]={'mean_coverage_pct':float(quality[f'{mod}_coverage_pct'].mean()),'min_coverage_pct':float(quality[f'{mod}_coverage_pct'].min()),'median_samples':int(quality[f'{mod}_samples'].median()),'sampling_rate_hz':FS[mod]}
report={'dataset':{'name':'MAUS — A Dataset for Mental Workload Assessment on N-back Task Using Wearable Sensor','doi':'10.21227/q4td-yd35','participants':int(f.subject.nunique()),'trials':int(len(f)),'trial_conditions':{'0-back':44,'2-back':44,'3-back':44},'trial_duration_seconds':300,'modalities':MODALITIES,'sampling_rates_hz':FS,'derived_streams':['RRI_inf: ECG-derived inter-beat intervals','PPI_inf: fingertip-PPG-derived pulse intervals','PPI_pix: wrist-PPG-derived pulse intervals'],'subjective_context':['NASA-TLX adjusted workload per trial','PSQI participant-level sleep-quality score'],'behavioural_context':'N-back response-log fields and positive reaction-time records from record.xlsx'},'model_reference':{'fingertip_ppg_baseline':ppg_ref,'multimodal_fusion_note':'The four-sensor feature-fusion pilot is included as an exploratory comparison. It did not outperform the prior fingertip-PPG reference in subject-independent discrimination; the software therefore presents sensor-level evidence and fusion transparently rather than claiming that fusion is superior.'},'statistical_evidence':analysis,'ibi_sensor_agreement':agreement,'signal_quality':quality_summary,'limitations':['The N-back condition is an experimental task-demand proxy, not a direct measurement of a person’s internal mental state.','NASA-TLX and PSQI are reference/context variables and are not model predictors.','The 22-participant MAUS cohort is small and uses one acquisition protocol.','Subject-independent results require external validation on an independent cohort and hardware.','The application is a research demonstration, not a medical device or clinical diagnostic system.']}
(D/'research_evidence.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('saved evidence: measures',len(analysis),'agreement',len(agreement),'pca',pca.shape)
print('NASA TLX',analysis[0])
print('agreement',agreement)

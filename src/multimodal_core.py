from __future__ import annotations
import io, json, zipfile, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; ARCHIVE=DATA/'MAUS.zip'
FEATURES_PATH=DATA/'maus_multimodal_trial_features.csv'; OOF_PATH=DATA/'multimodal_loso_predictions.csv'; REPORT_PATH=DATA/'multimodal_model_report.json'
PCA_PATH=DATA/'multimodal_pca.csv'; IMPORTANCE_PATH=DATA/'multimodal_feature_importance.csv'; COMPARISON_PATH=DATA/'modality_comparison.json'
NASA_PATH=DATA/'nasa_tlx_by_trial.csv'; BEHAVIOUR_PATH=DATA/'task_performance_by_trial.csv'; QUALITY_PATH=DATA/'signal_quality_by_trial.csv'; PARTICIPANT_PATH=DATA/'participant_context.csv'
FS={'ecg':256,'fppg':256,'gsr':256,'wppg':100}
FILES={'ecg':'inf_ecg.csv','fppg':'inf_ppg.csv','gsr':'inf_gsr.csv','wppg':'pixart.csv'}
TASKS=['0-back','2-back','3-back','2-back','3-back','0-back']; TASK_N=[0,2,3,2,3,0]
MODALITIES={
 'ecg':{'label':'ECG','full':'Single-lead electrocardiogram','fs':256,'source':'ProComp Infiniti'},
 'fppg':{'label':'Fingertip PPG','full':'Palmar fingertip photoplethysmography','fs':256,'source':'ProComp Infiniti'},
 'wppg':{'label':'Wrist PPG','full':'Wrist-worn PixArt photoplethysmography','fs':100,'source':'PixArt PPG watch'},
 'gsr':{'label':'GSR / EDA','full':'Galvanic skin response / electrodermal activity','fs':256,'source':'ProComp Infiniti'},
}

def read_csv(zf,name):
 try:return pd.read_csv(io.BytesIO(zf.read(name)))
 except Exception:return pd.DataFrame()
def finite(x):
 a=np.asarray(x,dtype=float).reshape(-1);return a[np.isfinite(a)]
def safe_num(x):
 try:
  v=float(x);return v if np.isfinite(v) else np.nan
 except:return np.nan
def filt(x,fs,lo,hi=None,order=3):
 if len(x)<fs*3:return x-np.median(x)
 ny=fs/2
 if hi is None:sos=butter(2,min(lo,ny*.9),btype='lowpass',fs=fs,output='sos')
 else:sos=butter(order,[max(.05,lo),min(hi,ny*.9)],btype='bandpass',fs=fs,output='sos')
 try:return sosfiltfilt(sos,x)
 except:return x-np.median(x)
def slope(x):
 if len(x)<3:return 0.
 t=np.linspace(-1.0,1.0,len(x),dtype=float); t-=t.mean(); y=np.asarray(x,dtype=float); y=y-y.mean(); den=float(np.dot(t,t)); return float(np.dot(t,y)/den) if den>0 else 0.
def interval_stats(ibi,prefix):
 x=finite(ibi);x=x[(x>=300)&(x<=2000)]
 if len(x)<3:return {f'{prefix}_{k}':np.nan for k in ['mean_ms','sd_ms','median_ms','iqr_ms','rmssd_ms','pnn50','cv','slope','count']}
 d=np.diff(x)
 return {f'{prefix}_mean_ms':float(x.mean()),f'{prefix}_sd_ms':float(x.std(ddof=1)),f'{prefix}_median_ms':float(np.median(x)),f'{prefix}_iqr_ms':float(np.percentile(x,75)-np.percentile(x,25)),f'{prefix}_rmssd_ms':float(np.sqrt(np.mean(d*d))),f'{prefix}_pnn50':float(np.mean(np.abs(d)>50)),f'{prefix}_cv':float(x.std()/(x.mean()+1e-9)),f'{prefix}_slope':slope(x),f'{prefix}_count':int(len(x))}
def common_stats(x,fs,prefix):
 x=finite(x)
 if len(x)<fs*5:return {}
 lo,hi=np.percentile(x,[.5,99.5]);x=np.clip(x,lo,hi);d=np.diff(x);med=np.median(x);mad=np.median(np.abs(x-med))*1.4826+1e-9;z=np.clip((x-med)/mad,-12,12)
 # Compact spectral signature computed on a decimated copy to keep the full pipeline fast.
 stride=max(1,len(z)//1800); zz=z[::stride]; f=np.fft.rfftfreq(len(zz),d=stride/fs); p=np.abs(np.fft.rfft(zz))**2; total=p.sum()+1e-12
 ent=float(-np.sum((p/total)*np.log(p/total+1e-12))/np.log(max(2,len(p))))
 def ratio(a,b):
  m=(f>=a)&(f<b);return float(p[m].sum()/total) if m.any() else 0.
 return {f'{prefix}_mean':float(x.mean()),f'{prefix}_std':float(x.std()),f'{prefix}_iqr':float(np.percentile(x,75)-np.percentile(x,25)),f'{prefix}_range_95_05':float(np.percentile(x,95)-np.percentile(x,5)),f'{prefix}_skew':float(skew(z,bias=False)) if np.std(z)>1e-9 else 0.,f'{prefix}_kurtosis':float(kurtosis(z,bias=False)) if np.std(z)>1e-9 else 0.,f'{prefix}_line_length':float(np.mean(np.abs(d))),f'{prefix}_diff_std':float(np.std(d)),f'{prefix}_slope':slope(x),f'{prefix}_spectral_entropy':ent,f'{prefix}_low_power_ratio':ratio(.1,.8),f'{prefix}_high_power_ratio':ratio(3,15)}
def peak_intervals(filtered,fs,kind):
 if kind=='ecg':
  qrs=np.abs(filt(filtered,fs,5,25,order=2)); n=max(1,int(fs*.06)); env=np.convolve(qrs,np.ones(n)/n,mode='same'); prom=max(np.std(env)*.30,1e-5); peaks,_=find_peaks(env,distance=int(fs*.34),prominence=prom)
 else:
  med=np.median(filtered);mad=np.median(np.abs(filtered-med))*1.4826+1e-9;z=np.clip((filtered-med)/mad,-10,10);prom=max(.18,np.std(z)*.22);peaks,_=find_peaks(z,distance=int(fs*.34),prominence=prom)
 ibi=np.diff(peaks)*(1000/fs);ibi=ibi[(ibi>=300)&(ibi<=2000)]
 return peaks,ibi
def window_features(raw,fs,mod):
 x=finite(raw)
 if len(x)<fs*5:return {}
 if mod=='ecg':
  y=filt(x,fs,.5,40);out=common_stats(y,fs,'ecg');peaks,ibi=peak_intervals(y,fs,'ecg');out.update(interval_stats(ibi,'ecg_hrv'));out['ecg_hr_mean_bpm']=float(60000/np.mean(ibi)) if len(ibi) else np.nan;out['ecg_hr_sd_bpm']=float(np.std(60000/ibi)) if len(ibi)>2 else np.nan;out['ecg_qrs_rate_min']=len(peaks)/(len(x)/fs)*60;out['ecg_qrs_amp_median']=float(np.median(np.abs(y[peaks]))) if len(peaks) else np.nan;return out
 if mod in ('fppg','wppg'):
  y=filt(x,fs,.45,8 if mod=='fppg' else 5);out=common_stats(y,fs,mod);peaks,ibi=peak_intervals(y,fs,'ppg');out.update(interval_stats(ibi,f'{mod}_hrv'));out[f'{mod}_hr_mean_bpm']=float(60000/np.mean(ibi)) if len(ibi) else np.nan;out[f'{mod}_hr_sd_bpm']=float(np.std(60000/ibi)) if len(ibi)>2 else np.nan;out[f'{mod}_pulse_rate_min']=len(peaks)/(len(x)/fs)*60
  if len(peaks):
   amps=[]
   for pk in peaks:
    st=max(0,pk-int(fs*.35));tr=st+int(np.argmin(y[st:pk+1]));amps.append(y[pk]-y[tr])
   out[f'{mod}_pulse_amp_median']=float(np.median(amps));out[f'{mod}_pulse_amp_sd']=float(np.std(amps))
  else:out[f'{mod}_pulse_amp_median']=np.nan;out[f'{mod}_pulse_amp_sd']=np.nan
  return out
 if mod=='gsr':
  tonic=filt(x,fs,.05,None);phasic=x-tonic;out=common_stats(x,fs,'gsr');out.update({'gsr_tonic_mean':float(tonic.mean()),'gsr_tonic_sd':float(tonic.std()),'gsr_tonic_slope':slope(tonic),'gsr_phasic_sd':float(phasic.std()),'gsr_phasic_range':float(np.percentile(phasic,95)-np.percentile(phasic,5))});prom=max(np.std(phasic)*.7,1e-4);peaks,props=find_peaks(phasic,distance=fs,prominence=prom);out['gsr_scr_rate_min']=len(peaks)/(len(x)/fs)*60;out['gsr_scr_amp_median']=float(np.median(props.get('prominences',[np.nan]))) if len(peaks) else np.nan;out['gsr_scr_amp_sd']=float(np.std(props.get('prominences',[np.nan]))) if len(peaks) else np.nan;return out
 return {}
def signal_features(signal,fs,mod):
 x=finite(signal);need=fs*300
 if len(x)<need:x=np.pad(x,(0,need-len(x)),mode='edge')
 x=x[:need];step=fs*50;wins=[window_features(x[i*step:(i+1)*step],fs,mod) for i in range(6)];df=pd.DataFrame(wins);out={}
 for c in df.columns:
  v=pd.to_numeric(df[c],errors='coerce').to_numpy(dtype=float);g=v[np.isfinite(v)]
  if len(g):out[f'{c}_mean']=float(g.mean());out[f'{c}_sd']=float(g.std());out[f'{c}_slope']=slope(g);out[f'{c}_delta']=float(g[-1]-g[0])
 return out
def get_signal(zf,sid,file,idx):
 df=read_csv(zf,f'MAUS/Data/Raw_data/{sid}/{file}')
 if df.empty or idx>=df.shape[1]:return np.array([])
 return pd.to_numeric(df.iloc[:,idx],errors='coerce').interpolate(limit_direction='both').to_numpy(dtype=float)
def get_rest(zf,sid,mod):
 if mod=='wppg':df=read_csv(zf,f'MAUS/Data/Raw_data/{sid}/pixart_resting.csv');col=df.columns[0] if len(df.columns) else None
 else:df=read_csv(zf,f'MAUS/Data/Raw_data/{sid}/inf_resting.csv');col={'ecg':'Resting_ECG','fppg':'Resting_PPG','gsr':'Resting_GSR'}[mod]
 if df.empty or col not in df:return np.array([])
 return pd.to_numeric(df[col],errors='coerce').interpolate(limit_direction='both').to_numpy(dtype=float)
def get_ibi(zf,sid,trial):
 sources={'RRI_inf':'ibi_ecg','PPI_inf':'ibi_fppg','PPI_pix':'ibi_wppg'};out={}
 for src,prefix in sources.items():
  vals=[]
  for seg in range(1,4):
   df=read_csv(zf,f'MAUS/Data/IBI_sequence/{sid}/trial_{trial}_{seg}.csv')
   if src in df:vals.extend(pd.to_numeric(df[src],errors='coerce').dropna().tolist())
  out.update(interval_stats(vals,prefix))
 return out
def get_nasa(zf,sid):
 df=read_csv(zf,f'MAUS/Subjective_rating/{sid}/NASA_TLX.csv')
 return [safe_num(df.iloc[-1,i]) if not df.empty and df.shape[1]>i else np.nan for i in range(1,7)]
def get_psqi(zf,sid):
 df=read_csv(zf,f'MAUS/Subjective_rating/{sid}/PSQI.csv');return safe_num(df.iloc[0,0]) if not df.empty else np.nan
def get_behaviour(df,trial):
 if df.empty:return {'rsp_field_mean':np.nan,'crsp_field_mean':np.nan,'mean_response_time_ms':np.nan,'median_response_time_ms':np.nan,'response_count':np.nan}
 rsp=pd.to_numeric(df.get(f'b{trial}.Rsp',pd.Series(dtype=float)),errors='coerce').dropna();cr=pd.to_numeric(df.get(f'b{trial}.CRsp',pd.Series(dtype=float)),errors='coerce').dropna();rt=pd.to_numeric(df.get(f'b{trial}.RT',pd.Series(dtype=float)),errors='coerce');rt=rt[rt>0]
 # MAUS documentation identifies these as response-log fields but does not define their coding; do not relabel them as accuracy.
 return {'rsp_field_mean':float(rsp.mean()) if len(rsp) else np.nan,'crsp_field_mean':float(cr.mean()) if len(cr) else np.nan,'mean_response_time_ms':float(rt.mean()) if len(rt) else np.nan,'median_response_time_ms':float(rt.median()) if len(rt) else np.nan,'response_count':int(len(rt))}
def subjects(zf):return sorted({n.split('/')[3] for n in zf.namelist() if n.startswith('MAUS/Data/Raw_data/') and n.endswith('/inf_ecg.csv')})
def build_dataset(subject_ids=None):
 import gc, time
 if subject_ids is None:
  with zipfile.ZipFile(ARCHIVE) as zf0: subject_ids=subjects(zf0)
 else: subject_ids=[str(x).zfill(3) for x in subject_ids]
 rows=[];nasa_rows=[];perf_rows=[];quality_rows=[];participant_rows=[]
 for sid in subject_ids:
  tic=time.time(); print(f"[CogniSense] Extracting participant {sid}",flush=True)
  with zipfile.ZipFile(ARCHIVE) as zf:
   psqi=get_psqi(zf,sid);participant_rows.append({'subject':sid,'psqi':psqi});nasa=get_nasa(zf,sid)
   try:record=pd.read_excel(io.BytesIO(zf.read(f'MAUS/Data/Raw_data/{sid}/record.xlsx')),sheet_name=0)
   except:record=pd.DataFrame()
   rest={}
   for mod,fs in FS.items():
    a=get_rest(zf,sid,mod)
    # Only a near-complete resting recording is used for rest-relative calibration.
    rest[mod]=signal_features(a,fs,mod) if len(a)>=fs*270 else {}
   signals={mod:[get_signal(zf,sid,file,i) for i in range(6)] for mod,file in FILES.items()}
   for idx in range(6):
    trial=idx+1;tn=TASK_N[idx];row={'subject':sid,'trial':trial,'task_n':tn,'task_condition':TASKS[idx],'group':'Low' if tn==0 else 'Elevated','psqi':psqi};q={'subject':sid,'trial':trial,'task_condition':TASKS[idx]}
    for mod,fs in FS.items():
     a=signals[mod][idx];expected=fs*300;valid=finite(a);q[f'{mod}_samples']=len(valid);q[f'{mod}_coverage_pct']=min(100,len(valid)/expected*100);q[f'{mod}_missing_pct']=max(0,100-len(valid)/expected*100)
     if len(valid)>=fs*270:
      f=signal_features(valid,fs,mod);row.update(f);base=rest.get(mod,{})
      for key,value in f.items():
       if key in base and np.isfinite(value) and np.isfinite(base[key]):
        delta=float(value-base[key]);row[f'{key}_rest_delta']=delta
        # Standardized rest-relative change supports participant-specific calibration.
        base_sd_key=key.rsplit('_',1)[0]+'_sd' if key.endswith('_mean') else None
        denom=base.get(base_sd_key,np.nan) if base_sd_key else np.nan
        if np.isfinite(denom) and abs(denom)>1e-8:row[f'{key}_rest_z']=float(delta/denom)
    row.update(get_ibi(zf,sid,trial));perf=get_behaviour(record,trial);row.update({f'behaviour_{k}':v for k,v in perf.items()});nasa_score=nasa[idx] if idx<len(nasa) else np.nan;row['nasa_tlx']=nasa_score
    nasa_rows.append({'subject':sid,'trial':trial,'task_n':tn,'task_condition':TASKS[idx],'nasa_tlx_adjusted':nasa_score})
    perf_rows.append({'subject':sid,'trial':trial,'task_n':tn,'task_condition':TASKS[idx],**perf});rows.append(row);quality_rows.append(q)
  print(f"[CogniSense] Participant {sid} complete in {time.time()-tic:.1f}s",flush=True)
  gc.collect()
 return (pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan),pd.DataFrame(nasa_rows),pd.DataFrame(perf_rows),pd.DataFrame(quality_rows),pd.DataFrame(participant_rows))
def feature_cols(df,mod):
 # A compact, physiology-led feature panel avoids fitting hundreds of weakly informative
 # descriptors on only 132 trials. The full extracted feature matrix remains available for audit/export.
 keys={
  'ecg':['ecg_std','ecg_range_95_05','ecg_spectral_entropy','ecg_hr_mean_bpm','ecg_hr_sd_bpm','ecg_hrv_mean_ms','ecg_hrv_sd_ms','ecg_hrv_rmssd_ms','ecg_hrv_pnn50','ecg_qrs_rate_min'],
  'fppg':['fppg_std','fppg_range_95_05','fppg_spectral_entropy','fppg_hr_mean_bpm','fppg_hr_sd_bpm','fppg_hrv_mean_ms','fppg_hrv_sd_ms','fppg_hrv_rmssd_ms','fppg_hrv_pnn50','fppg_pulse_rate_min','fppg_pulse_amp_median'],
  'wppg':['wppg_std','wppg_range_95_05','wppg_spectral_entropy','wppg_hr_mean_bpm','wppg_hr_sd_bpm','wppg_hrv_mean_ms','wppg_hrv_sd_ms','wppg_hrv_rmssd_ms','wppg_hrv_pnn50','wppg_pulse_rate_min','wppg_pulse_amp_median'],
  'gsr':['gsr_std','gsr_range_95_05','gsr_spectral_entropy','gsr_tonic_mean','gsr_tonic_sd','gsr_phasic_sd','gsr_phasic_range','gsr_scr_rate_min','gsr_scr_amp_median'],
 }
 cols=[]
 if mod in keys:
  for key in keys[mod]:
   for suffix in ['_mean','_slope','_rest_delta','_rest_z']:
    name=key+suffix
    if name in df.columns and name not in cols:cols.append(name)
 elif mod=='ibi':
  for source in ['ibi_ecg','ibi_fppg','ibi_wppg']:
   for stat in ['mean_ms','sd_ms','rmssd_ms','pnn50','cv','slope']:
    name=f'{source}_{stat}'
    if name in df.columns:cols.append(name)
 elif mod=='fusion':
  for m in ['ecg','fppg','wppg','gsr','ibi']:
   cols.extend(feature_cols(df,m))
 return cols
def pipe(n):
 return Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('variance',VarianceThreshold(0.0)),('select',SelectKBest(lambda X,y:mutual_info_classif(X,y,random_state=42),k=max(1,min(18,n)))),('model',ExtraTreesClassifier(n_estimators=160,min_samples_leaf=2,max_features=.65,class_weight='balanced',random_state=42,n_jobs=1))])
def evaluate(df,cols,target='binary',keep=False):
 y=(df.task_n.astype(int)!=0).astype(int).to_numpy() if target=='binary' else df.task_n.astype(int).to_numpy();labels=[0,1] if target=='binary' else [0,2,3];names=['Low','Elevated'] if target=='binary' else ['0-back','2-back','3-back'];groups=df.subject.astype(str).to_numpy();X=df[cols].replace([np.inf,-np.inf],np.nan).to_numpy(float);pred=np.full(len(df),-1);probs=np.full((len(df),len(labels)),np.nan);imps={}
 for tr,te in LeaveOneGroupOut().split(X,y,groups):
  m=pipe(X.shape[1]);m.fit(X[tr],y[tr]);pred[te]=m.predict(X[te]);pp=m.predict_proba(X[te]);classes=list(m.named_steps['model'].classes_)
  for j,l in enumerate(labels):
   if l in classes:probs[te,j]=pp[:,classes.index(l)]
  support=m.named_steps['select'].get_support();vcols=np.asarray(cols)[m.named_steps['variance'].get_support()];selected=vcols[support]
  for name,val in zip(selected,m.named_steps['model'].feature_importances_):imps.setdefault(name,[]).append(float(val))
 metrics={'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro',labels=labels,zero_division=0)),'macro_precision':float(precision_score(y,pred,average='macro',labels=labels,zero_division=0)),'macro_recall':float(recall_score(y,pred,average='macro',labels=labels,zero_division=0)),'confusion_matrix':confusion_matrix(y,pred,labels=labels).tolist(),'class_names':names,'roc_auc':float(roc_auc_score(y,probs[:,1])) if target=='binary' else None,'n_samples':len(y),'n_subjects':len(np.unique(groups)),'n_features_input':len(cols)}
 oof=None
 if keep:
  oof=pd.DataFrame({'subject':df.subject.astype(str),'trial':df.trial.astype(int),'task_n':df.task_n.astype(int),'task_condition':df.task_condition,'actual_group':np.where(y==0,'Low','Elevated') if target=='binary' else df.task_condition,'predicted_group':np.where(pred==0,'Low','Elevated') if target=='binary' else np.array(names)[[labels.index(p) for p in pred]],'oof_score':probs[:,1] if target=='binary' else np.nan,'correct':pred==y,'nasa_tlx':df.nasa_tlx})
 imp=pd.DataFrame([{'feature':k,'importance_mean':np.mean(v),'folds_selected':len(v)} for k,v in imps.items()]).sort_values('importance_mean',ascending=False)
 return metrics,oof,imp
def run_analysis(force=False):
 DATA.mkdir(parents=True,exist_ok=True)
 if FEATURES_PATH.exists() and REPORT_PATH.exists() and not force:return load_report()
 features,nasa,perf,quality,participants=build_dataset()
 if len(features)!=132:raise RuntimeError(f'Expected 132 participant-trials; found {len(features)}')
 sets={m:feature_cols(features,m) for m in ['ecg','fppg','wppg','gsr','ibi']};sets['fusion']=feature_cols(features,'fusion');comparison=[];oof=None;importance=None;fusion_bin=fusion_multi=None
 for mod,cols in sets.items():
  bm,bo,bi=evaluate(features,cols,'binary',keep=(mod=='fusion'))
  mm,_,_=evaluate(features,cols,'multiclass',keep=False) if mod=='fusion' else ({},None,None)
  comparison.append({'modality':mod,'label':'Multimodal fusion' if mod=='fusion' else ('IBI / HRV' if mod=='ibi' else MODALITIES[mod]['label']),'n_features':len(cols),'binary':bm,'multiclass':mm})
  if mod=='fusion':oof=bo;importance=bi;fusion_bin=bm;fusion_multi=mm
 imputed=SimpleImputer(strategy='median',keep_empty_features=True).fit_transform(features[sets['fusion']]);scaled=StandardScaler().fit_transform(imputed);coords=PCA(n_components=3,random_state=42).fit_transform(scaled)
 pca=pd.DataFrame(coords,columns=['PC1','PC2','PC3']);pca.insert(0,'subject',features.subject.astype(str));pca['trial']=features.trial;pca['task_n']=features.task_n;pca['task_condition']=features.task_condition;pca['nasa_tlx']=features.nasa_tlx;pca['group']=features.group
 for df,path in [(features,FEATURES_PATH),(oof,OOF_PATH),(pca,PCA_PATH),(importance,IMPORTANCE_PATH),(nasa,NASA_PATH),(perf,BEHAVIOUR_PATH),(quality,QUALITY_PATH),(participants,PARTICIPANT_PATH)]:df.to_csv(path,index=False)
 COMPARISON_PATH.write_text(json.dumps(comparison,indent=2),encoding='utf-8')
 report={'project':'CogniSense AI — Multimodal Cognitive Workload Observatory','dataset':'MAUS: A Dataset for Mental Workload Assessment on N-back Task Using Wearable Sensor','doi':'10.21227/q4td-yd35','subjects':int(features.subject.nunique()),'trials':int(len(features)),'task_counts':{str(k):int(v) for k,v in features.task_n.value_counts().sort_index().items()},'modalities':MODALITIES,'sampling_rates_hz':FS,'primary_validation':'Leave-One-Subject-Out (22 outer folds); held-out participant excluded from model fitting and feature selection','binary_target':'0-back = Low; 2-back + 3-back = Elevated task-demand condition','multiclass_target':'0-back, 2-back, and 3-back task conditions','binary_metrics':fusion_bin,'multiclass_metrics':fusion_multi,'modality_comparison':comparison,'feature_count':len(sets['fusion']),'notes':['NASA-TLX is an external subjective workload reference and is excluded from predictive features.','PSQI is participant-level sleep context and is excluded from predictive features.','Task response and reaction-time measures are descriptive behavioural outcomes and are excluded from predictive features.','IBI streams (RRI_inf, PPI_inf, PPI_pix) are derived inter-beat intervals, not additional physical sensors.','Rest-relative deltas are calculated only when a valid resting recording is available; missing values are imputed inside each training fold.','All performance estimates are internal results from a small, single-dataset cohort; external validation is required.','This is a research prototype, not a medical device or clinical diagnostic system.']}
 REPORT_PATH.write_text(json.dumps(report,indent=2),encoding='utf-8');return report
def load_report():return json.loads(REPORT_PATH.read_text(encoding='utf-8'))
def load_data():
 return {'features':pd.read_csv(FEATURES_PATH,dtype={'subject':str}),'oof':pd.read_csv(OOF_PATH,dtype={'subject':str}),'pca':pd.read_csv(PCA_PATH,dtype={'subject':str}),'importance':pd.read_csv(IMPORTANCE_PATH),'comparison':pd.read_json(COMPARISON_PATH),'nasa':pd.read_csv(NASA_PATH,dtype={'subject':str}),'performance':pd.read_csv(BEHAVIOUR_PATH,dtype={'subject':str}),'quality':pd.read_csv(QUALITY_PATH,dtype={'subject':str}),'participants':pd.read_csv(PARTICIPANT_PATH,dtype={'subject':str})}
def get_raw_trial(subject,trial,window=1,seconds=20):
 sid=str(subject).zfill(3);trial=int(trial);window=int(window)
 if trial not in range(1,7) or window not in range(1,7):raise ValueError('Trial and window must be between 1 and 6')
 result={'subject':sid,'trial':trial,'window':window,'task_condition':TASKS[trial-1],'start_seconds':(window-1)*50,'signals':{}}
 with zipfile.ZipFile(ARCHIVE) as zf:
  for mod,file in FILES.items():
   fs=FS[mod];a=get_signal(zf,sid,file,trial-1);start=(window-1)*50*fs;seg=a[start:min(len(a),start+seconds*fs)] if len(a) else np.array([]);stride=max(1,int(np.ceil(len(seg)/1600))) if len(seg) else 1;v=seg[::stride]
   result['signals'][mod]={'time':(np.arange(len(v))*stride/fs).round(4).tolist(),'value':np.round(v,6).tolist(),'fs':fs,'samples':int(len(seg)),'unit':'dataset native units'}
  ibi={'RRI_inf':[],'PPI_inf':[],'PPI_pix':[]}
  for sg in range(1,4):
   d=read_csv(zf,f'MAUS/Data/IBI_sequence/{sid}/trial_{trial}_{sg}.csv')
   for c in ibi:
    if c in d:ibi[c].extend(pd.to_numeric(d[c],errors='coerce').dropna().tolist())
  result['ibi']={k:np.round(finite(v),3).tolist() for k,v in ibi.items()}
 return result

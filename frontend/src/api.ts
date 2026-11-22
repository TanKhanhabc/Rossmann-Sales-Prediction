const PREFIX = '/api';
async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(PREFIX + url, init);
  if (!res.ok) { const body = await res.json().catch(() => ({})); throw new Error(body.detail ?? `HTTP ${res.status}`); }
  return res.json() as Promise<T>;
}
export type Summary = {train_rows: number; test_rows: number; stores_total: number; stores_in_test: number; date_start: string; date_end: string; test_start: string; test_end: string; total_sales: number; mean_sales_open: number; daily_average_revenue: number; missing_test_open: number; top_stores: {store: number; sales: number}[]};
export type Store = {store: number; store_type: string; assortment: string; competition_distance: number | null};
export type TestStore = Store & {start_date: string; end_date: string};
export type Point = {date: string; sales: number; customers: number | null};
export type ForecastSchedule = {date: string; promo: number; open: number; school_holiday: number; state_holiday: '0' | 'a' | 'b' | 'c'};
export type ForecastPoint = ForecastSchedule & {store: number; predicted_sales: number; currency: string; schedule_source: 'user' | 'test.csv' | 'assumption'};
export type Forecast = {store: number; start_date: string; days: number; total_predicted_sales: number; known_schedule_days: number; assumed_schedule_days: number; user_schedule_days: number; points: ForecastPoint[]; warning: string};
export type TestPredictionPoint = {date: string; predicted_sales: number; open: number; promo: number; state_holiday: '0' | 'a' | 'b' | 'c'; school_holiday: number};
export type TestPredictions = {store: number; start_date: string; end_date: string; total_predicted_sales: number; points: TestPredictionPoint[]};
export type Model = {trained: boolean; validation?: {mae: number; rmse: number; rmspe: number; wape: number}; baseline?: {mae: number; rmse: number; rmspe: number}; validation_dates?: string[]; train_rows?: number; important_features?: {feature:string;importance:number}[]};
export type CvModel = {name:string;train_mae:number;val_mae:number;val_mae_sd:number;rmse:number;r2:number;within_1sd:boolean;selected:boolean;best_params:string};
export type HoldoutModel = {name:string;mae:number;rmse:number;rmspe:number;wape:number;mae_ci95:number[];rmse_ci95:number[]};
export type SubgroupRow = {[key:string]: string|number};
export type FeatureImportance = {feature:string;importance_mae_increase:number;importance_sd:number};
export type TopError = {[key:string]: string|number};
export type SelectionRow = {[key:string]: string|number|boolean};
export type ModelComparison = {
 models:CvModel[];
 best_model:string|null;
 stages:{
  cross_validation:CvModel[];
  holdout:HoldoutModel[];
  subgroups:{store_type?:SubgroupRow[];day_of_week?:SubgroupRow[];promo?:SubgroupRow[]};
  feature_importance:FeatureImportance[];
  top_errors?:TopError[];
  selection?:SelectionRow[];
 };
 interval_summary?:{[key:string]:string|number}[];
 metadata?:{fit_rows?:number;validation_rows?:number;validation_dates?:string[];[key:string]:unknown};
 selected_reason?:string;
 message?:string;
 missing_files?:string[];
 sources?:{cv:string;holdout:string;selection:string};
};
export type DataProfile = {
  train: {rows:number; columns:string[]; missing:{[key:string]:number}};
  test: {rows:number; columns:string[]; missing_open_original:number};
  store: {rows:number; columns:string[]; missing:{[key:string]:number}};
  weather: {rows:number; columns:string[]; imputed_measurements:number};
  notes:string[];
};
export type PredictInput = {store: number; date: string; promo: number; open: number; state_holiday: string; school_holiday: number};
export type PredictResult = {store:number;date:string;predicted_sales:number;open:number;promo:number;currency:string};
export const api = {
  summary: () => request<Summary>('/overview'),
  dataProfile: () => request<DataProfile>('/data-profile'),
  stores: () => request<Store[]>('/stores'),
  testStores: () => request<TestStore[]>('/test-stores'),
  testPredictions: (store: number, startDate: string, endDate: string) => request<TestPredictions>(`/test-predictions?store=${store}&start_date=${startDate}&end_date=${endDate}`),
  model: () => request<Model>('/model-info'),
  modelComparison: () => request<ModelComparison>('/model-comparison'),
  forecast: (store: number, startDate: string, days: number, schedule: ForecastSchedule[]) => request<Forecast>('/forecast', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({store, start_date: startDate, days, schedule})}),
  predict: (data: PredictInput) => request<PredictResult>('/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)}),
};

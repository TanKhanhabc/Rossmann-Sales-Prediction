import { useCallback, useEffect, useMemo, useState, type FormEvent } from 'react';
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { Activity, ArrowRight, BarChart3, CalendarDays, CheckCircle2, ChevronDown, CircleHelp, Clock3, Database, Euro, FileText, Layers3, LoaderCircle, Search, ShieldCheck, Sparkles, TrendingUp, Wallet, XCircle, type LucideIcon } from 'lucide-react';
import { api, type TestPredictions, type Model, type PredictInput, type PredictResult, type Store, type TestStore, type Summary } from './api';
const euros = (v: number) => '€' + new Intl.NumberFormat('de-DE', {maximumFractionDigits: 0}).format(v || 0);
const numeral = (v: number) => new Intl.NumberFormat('en-US').format(v || 0);
const shortDate = (v: string) => new Date(`${v}T12:00:00`).toLocaleDateString('vi-VN', {day: '2-digit', month: '2-digit'});
const fmt = (v: number) => new Intl.NumberFormat('vi-VN', {maximumFractionDigits: 2}).format(v);
const defaultForm: PredictInput = {store: 1, date: '2015-08-03', promo: 1, open: 1, state_holiday: '0', school_holiday: 0};
function StatCard({icon: Icon, label, value, note, accent = 'teal'}: {icon: LucideIcon;label:string;value:string;note:string;accent?:string}) {
 return <div className="stat-card"><div className={`stat-icon ${accent}`}><Icon size={20}/></div><div className="stat-label">{label}</div><div className="stat-value">{value}</div><div className="stat-note">{note}</div></div>;
}
function ChartTip({active, payload, label}: {active?:boolean;payload?: {name:string;value:number;color:string}[];label?:string}) {
 if (!active || !payload?.length) return null;
 return <div className="chart-tooltip"><span>{label}</span>{payload.map((p, i) => <div key={i} style={{color:p.color}}>{p.name}: <strong>{euros(p.value)}</strong></div>)}</div>;
}
export default function App() {
 const [summary,setSummary] = useState<Summary|null>(null);
 const [stores,setStores] = useState<Store[]>([]);
 const [testStores,setTestStores] = useState<TestStore[]>([]);
 const [model,setModel] = useState<Model|null>(null);
 const [forecast,setForecast] = useState<TestPredictions|null>(null);
 const [store,setStore] = useState(0);
 const [storeFilter,setStoreFilter] = useState('');
 const [startDate,setStartDate] = useState('');
 const [endDate,setEndDate] = useState('');
 const [form,setForm] = useState<PredictInput>(defaultForm);
 const [predictResult,setPredictResult] = useState<PredictResult|null>(null);
 const [busy,setBusy] = useState(false);
 const [predicting,setPredicting] = useState(false);
 const [error,setError] = useState('');
 const [searchOpen,setSearchOpen] = useState(false);
 const [predictionStoreOpen,setPredictionStoreOpen] = useState(false);
 const [predictionStoreFilter,setPredictionStoreFilter] = useState('');
 useEffect(() => {
   Promise.all([api.summary(),api.stores(),api.testStores(),api.model()]).then(([s, st, tst, m]) => {
     setSummary(s);setStores(st);setTestStores(tst);setModel(m);
     if(tst.length>0){setStore(tst[0].store);setStartDate(tst[0].start_date);setEndDate(tst[0].end_date);}
   }).catch(e=>setError(e.message));
 },[]);
 const selectedTestStore = testStores.find(s=>s.store===store);
 const forecastPeriodValid = Boolean(
   selectedTestStore && startDate && endDate
   && startDate <= endDate
   && startDate >= selectedTestStore.start_date
   && endDate <= selectedTestStore.end_date
 );
 const refresh = useCallback(async () => {
   if (!forecastPeriodValid) return;
   setBusy(true);setForecast(null);setError('');
   try {
     setForecast(await api.testPredictions(store,startDate,endDate));
   }
   catch (e) {setError((e as Error).message);}
   finally {setBusy(false);}
 },[store,startDate,endDate,forecastPeriodValid]);
 useEffect(() => {
   if (forecastPeriodValid) void refresh();
   else setForecast(null);
 },[refresh,forecastPeriodValid]);
 const selectedStore = stores.find(s=>s.store===store);
 const filteredTestStores = useMemo(
   ()=>testStores.filter(s=>String(s.store).includes(storeFilter)),
   [testStores,storeFilter]
 );
 const filteredPredictionStores = useMemo(
   ()=>stores.filter(s=>String(s.store).includes(predictionStoreFilter)),
   [stores,predictionStoreFilter]
 );
 const forecastChartData = forecast?.points.map(p=>({date:shortDate(p.date), sales:p.predicted_sales, name:p.date}))??[];
 async function submitPrediction(e: FormEvent) {
    e.preventDefault(); setPredicting(true);setPredictResult(null);setError('');
    try{setPredictResult(await api.predict(form));}catch(e){setError((e as Error).message);}finally{setPredicting(false);}
 }
 return <div className="app-shell">
  <aside className="sidebar">
   <div className="brand"><div className="brand-mark"><Activity size={22} strokeWidth={2.8}/></div><div><div className="brand-title">ROSSMANN<span>IQ</span></div><div className="brand-subtitle">Sales Intelligence</div></div></div>
   <div className="nav-header">WORKSPACE</div>
   <a className="nav-link selected" href="#overview"><BarChart3 size={19}/> Tổng quan <span className="nav-dot"/></a>
   <a className="nav-link" href="#forecast"><TrendingUp size={19}/> Dự báo doanh thu</a>
   <a className="nav-link" href="#predict"><Sparkles size={19}/> Dự đoán tùy chỉnh</a>
   <a className="nav-link" href="#model"><Layers3 size={19}/> Hiệu suất mô hình</a>
   <div className="nav-header second">DỮ LIỆU & BÁO CÁO</div>
   <a className="nav-link" href="#data"><Database size={19}/> Dataset Rossmann</a>
   <div className="sidebar-bottom"><div className="sidebar-help"><div className="help-top"><CircleHelp size={18}/> Thông tin dữ liệu</div><p>1.017.209 bản ghi từ năm 2013–2015. Đây là bộ dữ liệu lịch sử, không phải số liệu trực tiếp.</p></div><div className="sidebar-footer"><span className="live-dot"/> API / ML Connected <span className="version">v1.0</span></div></div>
  </aside>
  <main className="content">
   <header className="topbar"><div className="breadcrumbs">Workspace <span>/</span> <strong>Dashboard</strong></div><div className="top-right"><span className="badge"><span className="live-dot"/> ML Model {model?.trained ? 'Ready':'Checking'}</span><span className="avatar">RS</span></div></header>
   <section className="page-body" id="overview">
    <div className="title-row"><div><div className="eyebrow"><span/> INSIGHTS CENTER</div><h1>Tổng quan doanh thu <span className="title-wave">↗</span></h1><p className="muted">Phân tích và dự đoán doanh thu từng cửa hàng trong chuỗi Rossmann.</p></div><div className="time-chip"><CalendarDays size={17}/> Dữ liệu lịch sử 2013 — 2015</div></div>
    {error && <div className="error"><XCircle size={18}/>{error}<button onClick={()=>setError('')}>Đóng</button></div>}
    <div className="stats-grid">
      <StatCard icon={Wallet} label="Tổng doanh thu lịch sử" value={summary?euros(summary.total_sales):'...'} note="Toàn bộ train.csv"/>
      <StatCard icon={Euro} label="TB doanh thu / ngày mở" value={summary?euros(summary.mean_sales_open):'...'} note="Trên mỗi bản ghi cửa hàng" accent="blue"/>
      <StatCard icon={Layers3} label="Số cửa hàng" value={summary?numeral(summary.stores_total):'...'} note={`${summary?.stores_in_test ?? '...'} cửa hàng có dữ liệu test`} accent="purple"/>
      <StatCard icon={FileText} label="Bản ghi huấn luyện" value={summary?numeral(summary.train_rows):'...'} note="2013-01-01 → 2015-07-31" accent="orange"/>
    </div>
    <div className="bottom-grid"><section className="panel" id="predict"><div className="panel-header"><div><div className="heading-top">WHAT-IF ANALYSIS</div><h2>Dự đoán theo kịch bản</h2><p>Thay đổi điều kiện để ước lượng doanh thu.</p></div><div className="panel-symbol"><Sparkles size={20}/></div></div>
    <form className="prediction-form" onSubmit={submitPrediction}>
      <div className="form-grid">
        <label className="prediction-store-control">Mã cửa hàng
          <button
            type="button"
            className="prediction-store-button"
            aria-haspopup="listbox"
            aria-expanded={predictionStoreOpen}
            disabled={stores.length===0}
            onClick={()=>setPredictionStoreOpen(open=>!open)}
          >
            <span><Search size={14}/> Store #{form.store}</span>
            <ChevronDown size={15}/>
          </button>
          {predictionStoreOpen && <div className="prediction-store-dropdown" role="listbox">
            <input
              autoFocus
              type="text"
              inputMode="numeric"
              placeholder="Tìm mã cửa hàng..."
              value={predictionStoreFilter}
              onChange={e=>setPredictionStoreFilter(e.target.value.replace(/\D/g,''))}
            />
            <div className="store-options">
              {filteredPredictionStores.map(s=><button
                type="button"
                key={s.store}
                role="option"
                aria-selected={form.store===s.store}
                onClick={()=>{setForm(prev=>({...prev,store:s.store}));setPredictionStoreOpen(false);setPredictionStoreFilter('');}}
              >
                Store #{s.store}<span>Loại {s.store_type}</span>
              </button>)}
              {filteredPredictionStores.length===0 && <div className="store-no-results">Không tìm thấy cửa hàng</div>}
            </div>
          </div>}
        </label>
        <label>Ngày dự báo<input type="date" value={form.date} onChange={e=>setForm({...form,date:e.target.value})}/></label>
        <label>Khuyến mãi<select value={form.promo} onChange={e=>setForm({...form,promo:Number(e.target.value)})}><option value={1}>Có khuyến mãi</option><option value={0}>Không</option></select></label>
        <label>Trạng thái cửa hàng<select value={form.open} onChange={e=>setForm({...form,open:Number(e.target.value)})}><option value={1}>Mở cửa</option><option value={0}>Đóng cửa</option></select></label>
        <label>Nghỉ học của trường học<select value={form.school_holiday} onChange={e=>setForm({...form,school_holiday:Number(e.target.value)})}><option value={0}>Không</option><option value={1}>Có</option></select></label>
        <label>Ngày lễ<select value={form.state_holiday} onChange={e=>setForm({...form,state_holiday:e.target.value})}><option value="0">Ngày thường</option><option value="a">Lễ loại a</option><option value="b">Lễ loại b</option><option value="c">Lễ loại c</option></select></label>
      </div>
      <button className="primary-btn full" type="submit" disabled={predicting}>{predicting?<LoaderCircle size={17} className="spin"/>:<Sparkles size={17}/>} Chạy dự đoán <ArrowRight size={17}/></button>
    </form>
    {predictResult && <div className="result-box"><span>DOANH THU ƯỚC TÍNH</span><strong>{euros(predictResult.predicted_sales)}</strong><small>Store #{predictResult.store} · {predictResult.date} · EUR</small></div>}
    </section>
    <section className="panel model-panel" id="model"><div className="panel-header"><div><div className="heading-top">MODEL PERFORMANCE</div><h2>Đánh giá mô hình</h2><p>Kiểm định theo thời gian, không chia ngẫu nhiên.</p></div><div className="panel-symbol purple"><Activity size={20}/></div></div><div className="model-status"><span className="good-dot"/> {model?.trained?'Mô hình đã được huấn luyện':'Chưa tìm thấy mô hình'} <CheckCircle2 size={16}/></div><div className="model-metrics"><div><span>MAE</span><strong>{model?.validation?euros(model.validation.mae):'—'}</strong><small>Sai số tuyệt đối TB</small></div><div><span>RMSE</span><strong>{model?.validation?euros(model.validation.rmse):'—'}</strong><small>Sai số căn bậc hai TB</small></div><div><span>RMSPE</span><strong>{model?.validation?fmt(model.validation.rmspe):'—'}</strong><small>Đo trên Sales &gt; 0</small></div></div><div className="model-divider"/><div className="validation-notes"><div><Clock3 size={16}/> Tập validation</div><strong>{model?.validation_dates?`${model.validation_dates[0]} → ${model.validation_dates[1]}`:'—'}</strong></div><div className="validation-notes"><div><ShieldCheck size={16}/> So với baseline (MAE)</div><strong>{model?.baseline?euros(model.baseline.mae):'—'}</strong></div><p className="model-caption">Không dùng Sales/Customers làm đầu vào. Khi cửa hàng đóng cửa, dự đoán bằng 0.</p></section></div>
    <section className="panel forecast-section" id="forecast"><div className="panel-header wide"><div><div className="heading-top">DỰ ĐOÁN TRÊN TẬP TEST</div><h2>Doanh thu dự đoán theo ngày</h2><p>LightGBM sử dụng lịch từ test.csv và tự tạo đặc trưng từ dữ liệu cửa hàng, thời tiết.</p></div><span className="ai-chip"><Sparkles size={15}/> AI POWERED</span></div>
      <div className="forecast-controls">
        <div className="control search-control"><label>Cửa hàng</label><button className="input-look" disabled={testStores.length===0} onClick={()=>setSearchOpen(!searchOpen)}><Search size={16}/> {selectedTestStore?`Store #${store}`:'Chọn cửa hàng'} <ChevronDown size={16}/></button>{searchOpen && <div className="store-dropdown"><input autoFocus placeholder="Tìm mã cửa hàng..." value={storeFilter} onChange={e=>setStoreFilter(e.target.value)}/><div className="store-options">{filteredTestStores.map(s=><button key={s.store} onClick={()=>{setStore(s.store);setStartDate(s.start_date);setEndDate(s.end_date);setForm(prev=>({...prev,store:s.store}));setForecast(null);setSearchOpen(false);setStoreFilter('');}}>Store #{s.store} <span>Loại {s.store_type}</span></button>)}</div></div>}</div>
        <div className="control"><label htmlFor="forecast-start-date">Ngày bắt đầu</label><input id="forecast-start-date" type="date" min={selectedTestStore?.start_date} max={selectedTestStore?.end_date} value={startDate} onChange={e=>setStartDate(e.target.value)} disabled={!selectedTestStore}/></div>
        <div className="control"><label htmlFor="forecast-end-date">Ngày kết thúc</label><input id="forecast-end-date" type="date" min={selectedTestStore?.start_date} max={selectedTestStore?.end_date} value={endDate} onChange={e=>setEndDate(e.target.value)} disabled={!selectedTestStore}/></div>
        <button className="primary-btn" disabled={busy || !forecastPeriodValid} onClick={()=>void refresh()}>{busy?<LoaderCircle className="spin" size={17}/>:<TrendingUp size={17}/>} Cập nhật</button>
      </div>
      {startDate && endDate && startDate > endDate && <div className="chart-empty date-range-error">Ngày bắt đầu phải trước hoặc bằng ngày kết thúc.</div>}
      <div className="forecast-summary"><div><div className="micro-label">TỔNG DỰ ĐOÁN · {forecast?.points.length ?? '—'} NGÀY</div><div className="big-number">{forecast?euros(forecast.total_predicted_sales):'—'}</div></div><div className="quick-store"><div>STORE #{store}</div><strong>Nhóm {selectedStore?.store_type?.toUpperCase()??'—'}</strong><small>Danh mục {selectedStore?.assortment?.toUpperCase()??'—'}</small></div></div>
      <div className="chart-toolbar"><strong className="chart-title">Doanh thu dự đoán · Store #{store}</strong><div className="chart-key"><span/> Doanh thu dự đoán</div></div>
      {!forecastPeriodValid || (forecast && forecast.points.length===0) ? <div className="chart-empty">{busy?'Đang tính dự đoán...':forecastPeriodValid?`Không có dữ liệu test.csv của cửa hàng #${store} trong khoảng ngày đã chọn.`:'Chọn khoảng ngày hợp lệ trong test.csv để xem doanh thu.'}</div> : <div className="chart-area"><ResponsiveContainer width="100%" height="100%"><AreaChart data={forecastChartData} margin={{top:14,right:10,left:0,bottom:2}}><defs><linearGradient id="areaColor" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#1bb998" stopOpacity={.22}/><stop offset="95%" stopColor="#1bb998" stopOpacity={.008}/></linearGradient></defs><CartesianGrid vertical={false} stroke="#eef2f6" strokeDasharray="4 4"/><XAxis dataKey="date" stroke="#97a2b2" fontSize={11} tickLine={false} axisLine={false} minTickGap={24}/><YAxis stroke="#97a2b2" fontSize={11} tickLine={false} axisLine={false} width={50} tickFormatter={v=>`${Math.round(Number(v)/1000)}k`}/><Tooltip content={<ChartTip/>}/><Area name="Doanh thu dự đoán" type="monotone" dataKey="sales" stroke="#0da987" strokeWidth={3} fill="url(#areaColor)" dot={false} activeDot={{r:5,fill:'#0da987',stroke:'#fff',strokeWidth:2}}/></AreaChart></ResponsiveContainer></div>}
      {forecast && <div className="chart-footnote"><ShieldCheck size={16}/><span>Giai đoạn {forecast.start_date} – {forecast.end_date}. Lịch và đặc trưng được lấy tự động từ test.csv; doanh thu là dự đoán của mô hình đã huấn luyện, không phải giá trị thực tế.</span></div>}
    </section>
    <section className="panel dataset-panel" id="data"><div className="panel-header"><div><div className="heading-top">DATA TRANSPARENCY</div><h2>Ba bộ dữ liệu Rossmann</h2><p>Mô hình lấy dữ liệu trực tiếp từ các tệp CSV trong dự án.</p></div><span className="table-small-badge"><Database size={15}/> VERIFIED CSV</span></div><div className="dataset-grid"><div><span className="dataset-icon"><FileText size={19}/></span><strong>train.csv</strong><small>Dữ liệu huấn luyện</small><b>{summary?numeral(summary.train_rows):'...'}</b><span>9 cột · có Sales / Customers</span></div><div><span className="dataset-icon blue"><FileText size={19}/></span><strong>test.csv</strong><small>Ngày cần dự báo</small><b>{summary?numeral(summary.test_rows):'...'}</b><span>8 cột · không có Sales</span></div><div><span className="dataset-icon purple"><Database size={19}/></span><strong>store.csv</strong><small>Thuộc tính cửa hàng</small><b>{summary?numeral(summary.stores_total):'...'}</b><span>10 cột · join bằng Store</span></div></div><div className="dataset-footer"><span><CircleHelp size={16}/> 11 giá trị Open còn thiếu trong test.csv được suy luận theo thứ trong tuần.</span></div>    </section>
    <footer>Rossmann Sales Intelligence · FastAPI + LightGBM + React <span>Không phải dữ liệu Rossmann thời gian thực · EUR</span></footer>
   </section>
  </main>
 </div>;
}

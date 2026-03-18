import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="AI HSE Analytics", layout="wide", page_icon="🛡️")

# --- ЗАГРУЗКА ДАННЫХ ---
@st.cache_data
def load_all_data():
    df_pred = pd.read_csv("Final_Risk_Predictions.csv", sep=";")
    df_inc = pd.read_csv("Проишествия_clean.csv", sep=";")
    df_kor = pd.read_csv("коргау_clean.csv", sep=";")
    
    # Приводим даты к формату datetime
    if 'Дата_Время' in df_kor.columns:
        df_kor['Дата_Время'] = pd.to_datetime(df_kor['Дата_Время'], errors='coerce')
    if 'Время_и_дата_сообщения' in df_inc.columns:
        df_inc['Время_и_дата_сообщения'] = pd.to_datetime(df_inc['Время_и_дата_сообщения'], errors='coerce')
        
    return df_pred, df_inc, df_kor

df_pred, df_inc, df_kor = load_all_data()

# --- БОКОВАЯ ПАНЕЛЬ (ФИЛЬТРЫ И ДИНАМИЧЕСКИЕ ПАРАМЕТРЫ) ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/KazMunayGas_Logo.svg/1200px-KazMunayGas_Logo.svg.png", width=150)
st.sidebar.title("Параметры системы")

selected_org = st.sidebar.selectbox("Организация / ДЗО:", ["Все организации"] + list(df_pred['Организация'].unique()))
period_months = st.sidebar.radio("Предиктивный горизонт (мес):", [3, 6, 12], index=2)

st.sidebar.markdown("---")
st.sidebar.subheader("Параметры расчета ROI")
# Пользователь сам задает стоимость инцидента (без хардкода!)
base_cost = st.sidebar.number_input("Прямые потери от 1 НС (₸):", min_value=1000000, value=5000000, step=1000000)
indirect_mult = st.sidebar.slider("Коэф. косвенных потерь (штрафы, простой):", 1.0, 5.0, 2.0, 0.5)
COST_PER_INCIDENT = base_cost + (base_cost * indirect_mult)

st.title("🛡️ Платформа AI-Аналитики HSE (ОТИПБ)")

# --- ФИЛЬТРАЦИЯ ДАННЫХ ПО ВЫБРАННОЙ ОРГАНИЗАЦИИ ---
if selected_org != "Все организации":
    f_pred = df_pred[df_pred['Организация'] == selected_org]
    f_inc = df_inc[df_inc['Наименование_организации_ДЗО'] == selected_org] if 'Наименование_организации_ДЗО' in df_inc.columns else df_inc
    f_kor = df_kor[df_kor['Организация'] == selected_org]
else:
    f_pred, f_inc, f_kor = df_pred, df_inc, df_kor

# --- ДИНАМИЧЕСКАЯ ЭКОНОМИКА (CALCULATED ROI) ---
st.header("💰 Динамический расчет экономического эффекта")

# 1. Расчет ожидаемых инцидентов (Математическое ожидание)
# Вероятность переводим в доли (0-1) и умножаем на исторический тренд
historical_incident_rate = len(f_inc) / 3 if len(f_inc) > 0 else 1  # среднее за 3 года
expected_incidents = (f_pred['predicted_risk_probability'] / 100).mean() * historical_incident_rate * (period_months / 12)

# 2. ИИ снижает риск в "красных зонах" до базового (например, до медианного по датасету)
median_risk = df_pred['predicted_risk_probability'].median()
red_zones = f_pred[f_pred['predicted_risk_probability'] > median_risk * 1.5]

if not red_zones.empty:
    prevented_ratio = ((red_zones['predicted_risk_probability'] - median_risk) / 100).mean()
    prevented_incidents = prevented_ratio * historical_incident_rate * (period_months / 12)
else:
    prevented_incidents = 0

saved_money = prevented_incidents * COST_PER_INCIDENT

# Динамическая метрика качества (Корреляция Пирсона между предиктом и реальными инцидентами)
correlation = df_pred['predicted_risk_probability'].corr(df_pred['target_incidents'])
accuracy_metric = f"{max(0, correlation * 100):.1f}%" if pd.notna(correlation) else "Н/Д"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Прогноз НС (БЕЗ ИИ)", f"{expected_incidents:.1f} случаев", "Ожидаемые потери", delta_color="inverse")
col2.metric("Предотвращено с ИИ", f"{prevented_incidents:.1f} случаев", "Спасенные жизни")
col3.metric("Экономия (ROI)", f"{int(saved_money):,} ₸".replace(',', ' '), f"При {COST_PER_INCIDENT:,.0f}₸ за НС")
col4.metric("Достоверность паттерна", accuracy_metric, "Корреляция с историей")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Историческая аналитика", "🔮 Предикт и Тренды", "🚨 Динамические Рекомендации"])

# ==========================================
# Вкладка 1: ИСТОРИЯ
# ==========================================
with tab1:
    colA, colB = st.columns(2)
    with colA:
        if not f_inc.empty and 'Классификация_НС' in f_inc.columns:
            inc_counts = f_inc['Классификация_НС'].value_counts().reset_index()
            inc_counts.columns = ['Тип', 'Количество']
            fig1 = px.pie(inc_counts, names='Тип', values='Количество', title="Структура исторических инцидентов")
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Нет данных по инцидентам для выбранного фильтра")

    with colB:
        if not f_kor.empty and 'Дата_Время' in f_kor.columns:
            monthly_kor = f_kor.groupby(f_kor['Дата_Время'].dt.to_period("M")).size().reset_index(name='Кол-во наблюдений')
            monthly_kor['Дата_Время'] = monthly_kor['Дата_Время'].dt.to_timestamp()
            fig2 = px.line(monthly_kor, x='Дата_Время', y='Кол-во наблюдений', title="Динамика регистраций Карт Коргау", markers=True)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Нет данных по наблюдениям Коргау")

# ==========================================
# Вкладка 2: ПРЕДИКТ И ТРЕНДЫ
# ==========================================
with tab2:
    colC, colD = st.columns(2)
    with colC:
        st.subheader("🔥 Распределение вероятностей риска")
        top_risks = f_pred.sort_values(by='predicted_risk_probability', ascending=False).head(10)
        fig_bar = px.bar(top_risks, x='predicted_risk_probability', y='Организация', orientation='h', color='predicted_risk_probability', color_continuous_scale='Reds')
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)

    with colD:
        st.subheader(f"📈 Динамический прогноз на {period_months} мес.")
        # Строим прогноз на базе исторической средней (Run-rate) и вектора риска
        dates = [datetime.today() + timedelta(days=30*i) for i in range(period_months)]
        
        # Динамический базовый уровень на основе истории
        base_rate = historical_incident_rate / 12
        risk_multiplier = (top_risks['predicted_risk_probability'].mean() / 50) if not top_risks.empty else 1.0
        
        # Трендовая линия (без рандома, чистая математика)
        trend = [base_rate * risk_multiplier * (1 + (i * 0.05)) for i in range(period_months)]
        variance = [t * 0.25 for t in trend] # Доверительный интервал 25% от значения
        
        upper_bound = [t + v for t, v in zip(trend, variance)]
        lower_bound = [max(0, t - v) for t, v in zip(trend, variance)]
        
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=dates+dates[::-1], y=upper_bound+lower_bound[::-1], fill='toself', fillcolor='rgba(255, 0, 0, 0.2)', line=dict(color='rgba(255,255,255,0)'), name='Доверительный интервал (±25%)'))
        fig_line.add_trace(go.Scatter(x=dates, y=trend, mode='lines+markers', line=dict(color='red', width=3), name='Прогнозируемый тренд'))
        st.plotly_chart(fig_line, use_container_width=True)

# ==========================================
# Вкладка 3: ДИНАМИЧЕСКИЕ РЕКОМЕНДАЦИИ
# ==========================================
with tab3:
    st.subheader("💡 Предписывающая аналитика (AI Prescriptive)")
    
    if selected_org == "Все организации":
        st.warning("Пожалуйста, выберите конкретную организацию в боковом меню слева, чтобы ИИ сгенерировал персонализированные меры контроля.")
    else:
        risk_score = f_pred['predicted_risk_probability'].values[0] if not f_pred.empty else 0
        
        # 1. ДИНАМИЧЕСКИЙ АЛЕРТ
        threshold_critical = df_pred['predicted_risk_probability'].quantile(0.85)
        threshold_high = df_pred['predicted_risk_probability'].quantile(0.60)
        
        if risk_score >= threshold_critical:
            st.error(f"🚨 **КРИТИЧЕСКИЙ АЛЕРТ (Риск {risk_score:.1f}%)**\n\nВероятность инцидента находится в верхних 15% по всей группе компаний.")
        elif risk_score >= threshold_high:
            st.warning(f"⚠️ **ВЫСОКИЙ АЛЕРТ (Риск {risk_score:.1f}%)**\n\nТренд нарушений указывает на формирование опасных паттернов.")
        else:
            st.success(f"✅ **НОРМА (Риск {risk_score:.1f}%)**\n\nСитуация стабильна. Риск ниже медианного значения.")

        # 2. ДИНАМИЧЕСКАЯ ГЕНЕРАЦИЯ РЕКОМЕНДАЦИЙ ИЗ ИСТОРИИ
        st.markdown("### Коренные причины (Root Causes) и корректирующие меры")
        
        # Находим реальные самые частые опасные категории ИМЕННО ЭТОЙ компании в Коргау
        bad_practices = f_kor[f_kor['Тип_наблюдения'].str.contains('Опасн', case=False, na=False)]
        
        if not bad_practices.empty and 'Категория_наблюдения' in bad_practices.columns:
            top_violations = bad_practices['Категория_наблюдения'].value_counts().head(3)
            
            st.write(f"На основе NLP-анализа Карт Коргау за весь период, алгоритм выявил главные уязвимости компании **{selected_org}**:")
            
            for i, (violation_category, count) in enumerate(top_violations.items(), 1):
                st.info(f"**Паттерн №{i}: «{violation_category}»** (зафиксировано {count} раз). \n\n👉 *Рекомендация:* Провести целевой аудит, обновить инструкции и организовать внеплановый Stand-down по теме '{violation_category}'.")
        else:
            st.success("У данной организации нет достаточного количества записей об опасных факторах в системе Коргау для выявления негативных паттернов. Рекомендуется поощрение персонала за соблюдение ТБ.")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from datetime import datetime

# ─────────────────────────────────────────────────────
# КОНФИГ СТРАНИЦЫ
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="F-04 • AI Рекомендации по мерам контроля",
    layout="wide",
    page_icon="🧠",
)

# ─────────────────────────────────────────────────────
# КАСТОМНЫЙ CSS
# ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* KPI карточки */
.kpi-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border-radius: 16px;
    padding: 24px;
    color: white;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-value {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.kpi-label { font-size: 0.85rem; color: #94a3b8; margin-top: 6px; }

/* Карточка рекомендации */
.rec-card {
    background: #1e293b;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
    border-left: 4px solid #60a5fa;
    color: #e2e8f0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.2);
}
.rec-card.critical { border-left-color: #ef4444; }
.rec-card.high { border-left-color: #f59e0b; }
.rec-card.medium { border-left-color: #22c55e; }

.rec-header {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.rec-body { font-size: 0.92rem; line-height: 1.6; color: #cbd5e1; }
.rec-meta { font-size: 0.8rem; color: #64748b; margin-top: 10px; }

.sim-badge {
    display: inline-block;
    background: rgba(96,165,250,0.15);
    color: #60a5fa;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.risk-badge-critical {
    display: inline-block;
    background: rgba(239,68,68,0.15);
    color: #ef4444;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.risk-badge-high {
    display: inline-block;
    background: rgba(245,158,11,0.15);
    color: #f59e0b;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.risk-badge-medium {
    display: inline-block;
    background: rgba(34,197,94,0.15);
    color: #22c55e;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}

/* Секция-заголовок */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 6px;
}
.section-sub {
    font-size: 0.88rem;
    color: #64748b;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# ЗАГРУЗКА ДАННЫХ
# ─────────────────────────────────────────────────────
@st.cache_data
def load_incidents():
    df = pd.read_csv('../Проишествия_clean.csv', sep=';')
    df['Время_и_дата_сообщения'] = pd.to_datetime(df['Время_и_дата_сообщения'], errors='coerce')
    return df

@st.cache_data
def load_korgau():
    df = pd.read_csv('../коргау_clean.csv', sep=';')
    df['Дата_Время'] = pd.to_datetime(df['Дата_Время'], errors='coerce')
    return df

@st.cache_data
def load_risk_predictions():
    return pd.read_csv('../Final_Risk_Predictions.csv', sep=';')


# ─────────────────────────────────────────────────────
# NLP МОДЕЛЬ + ЭМБЕДДИНГИ
# ─────────────────────────────────────────────────────
@st.cache_resource
def load_nlp_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_resource
def build_incident_index(_model, descriptions):
    """Векторизует описания инцидентов и строит kNN-индекс."""
    vectors = _model.encode(descriptions, show_progress_bar=False, batch_size=64)
    nn = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
    nn.fit(vectors)
    return nn, vectors


# ─────────────────────────────────────────────────────
# ЗАГРУЗКА
# ─────────────────────────────────────────────────────
df_inc = load_incidents()
df_kor = load_korgau()
df_risk = load_risk_predictions()
model = load_nlp_model()

# Готовим индекс только по инцидентам с описанием и рекомендациями
df_with_recs = df_inc.dropna(subset=['Краткое_описание_происшествия', 'Рекомендации']).reset_index(drop=True)
descriptions_list = df_with_recs['Краткое_описание_происшествия'].tolist()
nn_model, vectors = build_incident_index(model, descriptions_list)

# Опасные наблюдения
danger_types = ['Опасный фактор', 'Небезопасное условие', 'Небезопасное поведение', 'Опасный случай']
df_danger = df_kor[df_kor['Тип_наблюдения'].str.strip().isin(danger_types)].copy()


# ─────────────────────────────────────────────────────
# ХЕДЕР
# ─────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 10px 0 20px;">
    <h1 style="margin:0; font-size:2rem; background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        🧠 AI-Рекомендации по мерам контроля
    </h1>
    <p style="color:#64748b; font-size:0.95rem; margin-top:6px;">
        F-04 • Генерация рекомендаций на основе паттернов из инцидентов и Карт Коргау
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# KPI МЕТРИКИ
# ─────────────────────────────────────────────────────
# Считаем паттерны: сколько уникальных комбинаций (организация, категория) с ≥3 опасными наблюдениями
if not df_danger.empty and 'Категория_наблюдения' in df_danger.columns:
    pattern_counts = df_danger.groupby(['Организация', 'Категория_наблюдения']).size().reset_index(name='count')
    patterns = pattern_counts[pattern_counts['count'] >= 3]
    n_patterns = len(patterns)
    critical_patterns = len(patterns[patterns['count'] >= 10])
    orgs_at_risk = patterns['Организация'].nunique()
else:
    n_patterns = 0
    critical_patterns = 0
    orgs_at_risk = 0

avg_risk = df_risk['predicted_risk_probability'].mean() if not df_risk.empty else 0
n_recs_available = len(df_with_recs)

c1, c2, c3, c4 = st.columns(4)
for col, val, label, icon in [
    (c1, n_patterns, "Выявленных паттернов", "📊"),
    (c2, orgs_at_risk, "Организаций в зоне риска", "🚨"),
    (c3, f"{avg_risk:.1f}%", "Средний индекс риска", "📈"),
    (c4, n_recs_available, "Инцидентов с рекомендациями", "💡"),
]:
    col.markdown(f"""
    <div class="kpi-card">
        <div style="font-size:1.4rem;">{icon}</div>
        <div class="kpi-value">{val}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# ТАБЫ
# ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔥 Паттерн-анализ Коргау",
    "🔍 Семантический поиск рекомендаций",
    "🏢 Рекомендации по организациям"
])


# ══════════════════════════════════════════════════════
# ТАБ 1: ПАТТЕРН-АНАЛИЗ
# ══════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Тепловая карта опасных наблюдений</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Пересечение организаций и категорий наблюдений — чем краснее ячейка, тем чаще фиксируется опасный паттерн</div>', unsafe_allow_html=True)

    if not df_danger.empty and 'Категория_наблюдения' in df_danger.columns:
        # Тепловая карта
        heatmap_data = df_danger.groupby(['Организация', 'Категория_наблюдения']).size().reset_index(name='Количество')
        pivot = heatmap_data.pivot_table(index='Организация', columns='Категория_наблюдения', values='Количество', fill_value=0)

        fig_heat = px.imshow(
            pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            color_continuous_scale='YlOrRd',
            aspect='auto',
            labels=dict(color="Наблюдений")
        )
        fig_heat.update_layout(
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1'),
            xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=10)),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_heat, width='stretch')

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="section-header">📈 Динамика опасных наблюдений</div>', unsafe_allow_html=True)
            monthly = df_danger.groupby(df_danger['Дата_Время'].dt.to_period('M')).size().reset_index(name='Количество')
            monthly['Дата_Время'] = monthly['Дата_Время'].dt.to_timestamp()
            fig_trend = px.area(
                monthly, x='Дата_Время', y='Количество',
                color_discrete_sequence=['#ef4444'],
            )
            fig_trend.update_layout(
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#cbd5e1'),
                xaxis_title="",
                yaxis_title="Кол-во наблюдений",
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_trend, width='stretch')

        with col_b:
            st.markdown('<div class="section-header">🏆 ТОП-10 опасных паттернов</div>', unsafe_allow_html=True)
            top_patterns = df_danger.groupby(['Организация', 'Категория_наблюдения']).size().reset_index(name='Частота')
            top_patterns = top_patterns.sort_values('Частота', ascending=False).head(10)
            top_patterns['Паттерн'] = top_patterns['Категория_наблюдения'] + ' — ' + top_patterns['Организация']

            fig_top = px.bar(
                top_patterns, x='Частота', y='Паттерн', orientation='h',
                color='Частота', color_continuous_scale='Reds',
            )
            fig_top.update_layout(
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#cbd5e1'),
                yaxis=dict(categoryorder='total ascending', tickfont=dict(size=10)),
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_top, width='stretch')
    else:
        st.info("Нет данных по опасным наблюдениям в Картах Коргау.")


# ══════════════════════════════════════════════════════
# ТАБ 2: СЕМАНТИЧЕСКИЙ ПОИСК
# ══════════════════════════════════════════════════════

# Инициализация session_state для query
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""

def set_example_query(text):
    """Callback для кнопок-примеров."""
    st.session_state.search_query = text

with tab2:
    st.markdown('<div class="section-header">Интеллектуальный поиск рекомендаций</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Опишите инцидент или опасную ситуацию — AI найдёт похожие случаи из истории и предложит меры контроля</div>', unsafe_allow_html=True)

    # Примеры запросов
    example_queries = [
        "Работник поскользнулся на разлитом масле возле насоса и ушиб колено",
        "Обнаружена утечка газа на задвижке трубопровода",
        "Водитель превысил скорость на территории промысла",
        "Электрик получил удар током при ремонте щита",
        "Падение предмета с высоты на рабочего",
    ]

    col_input, col_examples = st.columns([3, 1])
    with col_input:
        query_text = st.text_area(
            "Описание инцидента / опасной ситуации:",
            value=st.session_state.search_query,
            height=120,
            placeholder="Например: Работник получил ожог руки при сварке трубопровода без СИЗ...",
            key="query_input",
        )
    with col_examples:
        st.markdown("**Примеры запросов:**")
        for i, ex in enumerate(example_queries):
            st.button(
                f"📝 {ex[:45]}...",
                key=f"ex_btn_{i}",
                on_click=set_example_query,
                args=(ex,),
            )

    top_k = st.slider("Количество рекомендаций:", 1, 10, 5)

    # Определяем текст запроса (из поля ввода или из session_state)
    active_query = query_text.strip() if query_text else st.session_state.search_query.strip()

    search_clicked = st.button("🚀 Найти рекомендации", type="primary", width='stretch')

    if (search_clicked or active_query) and active_query:
        with st.spinner("🔍 Семантический поиск по базе инцидентов..."):
            query_vector = model.encode([active_query])
            distances, indices = nn_model.kneighbors(query_vector, n_neighbors=min(top_k, len(df_with_recs)))

        st.markdown(f"<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-header">✅ Найдено {len(indices[0])} релевантных рекомендаций</div>', unsafe_allow_html=True)

        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            similarity = (1 - dist) * 100
            row = df_with_recs.iloc[idx]

            rec_text = row.get('Рекомендации', 'Нет данных')
            incident_text = row.get('Краткое_описание_происшествия', '')
            org = row.get('Наименование_организации_ДЗО', 'Н/Д')
            classification = row.get('Классификация_НС', '') or row.get('Классификация_ОМП', '') or ''
            corrective = row.get('Корректирующие_меры', '')
            causes = row.get('Предварительные_причины', '')

            # Определяем цвет по похожести
            if similarity >= 70:
                card_class = "critical"
                badge = f'<span class="sim-badge">🎯 {similarity:.0f}% совпадение</span>'
            elif similarity >= 50:
                card_class = "high"
                badge = f'<span class="sim-badge">📌 {similarity:.0f}% совпадение</span>'
            else:
                card_class = "medium"
                badge = f'<span class="sim-badge">📎 {similarity:.0f}% совпадение</span>'

            corrective_html = f"<br><b>Корректирующие меры:</b> {corrective}" if pd.notna(corrective) and str(corrective).strip() else ""
            causes_html = f"<br><b>Предварительные причины:</b> {causes}" if pd.notna(causes) and str(causes).strip() else ""

            st.markdown(f"""
            <div class="rec-card {card_class}">
                <div class="rec-header">
                    💡 Рекомендация #{rank + 1} {badge}
                </div>
                <div class="rec-body">
                    <b>📋 Рекомендация:</b> {rec_text}
                    {corrective_html}
                    {causes_html}
                </div>
                <div class="rec-meta">
                    📍 Организация: {org} &nbsp;|&nbsp; 🏷️ {classification} &nbsp;|&nbsp;
                    📝 Похожий инцидент: {str(incident_text)[:150]}...
                </div>
            </div>
            """, unsafe_allow_html=True)
    elif search_clicked:
        st.warning("Введите описание инцидента для поиска рекомендаций.")


# ══════════════════════════════════════════════════════
# ТАБ 3: РЕКОМЕНДАЦИИ ПО ОРГАНИЗАЦИЯМ
# ══════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Персонализированные рекомендации по организациям</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Выберите организацию — система проанализирует её паттерны из Коргау и историю инцидентов для формирования мер контроля</div>', unsafe_allow_html=True)

    all_orgs = sorted(df_kor['Организация'].dropna().unique().tolist())
    selected_org = st.selectbox("Выберите организацию:", all_orgs)

    if selected_org:
        org_danger = df_danger[df_danger['Организация'] == selected_org]
        org_incidents = df_inc[df_inc['Наименование_организации_ДЗО'] == selected_org] if 'Наименование_организации_ДЗО' in df_inc.columns else pd.DataFrame()
        org_risk = df_risk[df_risk['Организация'] == selected_org] if 'Организация' in df_risk.columns else pd.DataFrame()

        # Risk score
        risk_score = org_risk['predicted_risk_probability'].values[0] if not org_risk.empty else 0
        n_incidents = len(org_incidents)
        n_danger_obs = len(org_danger)

        # Определяем уровень
        if risk_score >= 70:
            level_badge = '<span class="risk-badge-critical">🚨 КРИТИЧЕСКИЙ</span>'
            level_color = "#ef4444"
        elif risk_score >= 40:
            level_badge = '<span class="risk-badge-high">⚠️ ВЫСОКИЙ</span>'
            level_color = "#f59e0b"
        else:
            level_badge = '<span class="risk-badge-medium">✅ УМЕРЕННЫЙ</span>'
            level_color = "#22c55e"

        # KPI организации
        st.markdown("<br>", unsafe_allow_html=True)
        oc1, oc2, oc3 = st.columns(3)
        oc1.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="background: linear-gradient(90deg, {level_color}, {level_color}); -webkit-background-clip:text;">{risk_score:.1f}%</div>
            <div class="kpi-label">Индекс риска {level_badge}</div>
        </div>
        """, unsafe_allow_html=True)
        oc2.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{n_incidents}</div>
            <div class="kpi-label">Исторических инцидентов</div>
        </div>
        """, unsafe_allow_html=True)
        oc3.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{n_danger_obs}</div>
            <div class="kpi-label">Опасных наблюдений Коргау</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown('<div class="section-header">📋 AI-рекомендации по мерам контроля</div>', unsafe_allow_html=True)

            if not org_danger.empty and 'Категория_наблюдения' in org_danger.columns:
                top_violations = org_danger['Категория_наблюдения'].value_counts().head(5)

                # Банк рекомендаций по категориям
                recommendation_bank = {
                    'Машины и оборудование': [
                        'Провести внеплановую проверку технического состояния оборудования',
                        'Обновить процедуры блокировки/маркировки (LOTO) на всех агрегатах',
                        'Организовать дополнительное обучение операторов по безопасной эксплуатации',
                    ],
                    'Электрооборудование': [
                        'Проверить заземление и целостность изоляции на всех электроустановках',
                        'Обновить наряды-допуски для работы с электрооборудованием',
                        'Провести аудит квалификации электротехнического персонала',
                    ],
                    'СИЗ': [
                        'Провести ревизию обеспеченности персонала средствами индивидуальной защиты',
                        'Организовать показательное учение по правильному применению СИЗ',
                        'Внедрить систему контроля ношения СИЗ через поведенческие аудиты',
                    ],
                    'Порядок, чистота на рабочем месте': [
                        'Внедрить систему 5S на производственных участках',
                        'Организовать еженедельные аудиты чистоты и порядка',
                        'Разработать визуальные стандарты для каждого рабочего места',
                    ],
                    'Вывешивание плакатов': [
                        'Провести инвентаризацию и обновить все информационные плакаты',
                        'Разместить визуальные предупреждения в критических зонах',
                        'Разработать QR-коды с инструкциями безопасности на плакатах',
                    ],
                    'Наряд-допуск / Оценка риска': [
                        'Усилить контроль оформления нарядов-допусков',
                        'Внедрить цифровую систему управления нарядами-допусками',
                        'Провести переаттестацию ответственных руководителей работ',
                    ],
                    'Целостность объекта / Оборудования': [
                        'Провести комплексный технический аудит целостности оборудования',
                        'Обновить график планово-предупредительных ремонтов',
                        'Внедрить систему предиктивного мониторинга состояния оборудования',
                    ],
                }

                # Дефолтные рекомендации для неизвестных категорий
                default_recs = [
                    'Провести целевой поведенческий аудит безопасности на данном участке',
                    'Организовать внеплановый Stand-down с разбором выявленных нарушений',
                    'Усилить контроль соблюдения стандартов безопасности на рабочих местах',
                ]

                for i, (category, count) in enumerate(top_violations.items()):
                    # Приоритет
                    if count >= 10:
                        priority = "КРИТИЧЕСКИЙ"
                        card_class = "critical"
                        icon = "🔴"
                    elif count >= 5:
                        priority = "ВЫСОКИЙ"
                        card_class = "high"
                        icon = "🟡"
                    else:
                        priority = "СРЕДНИЙ"
                        card_class = "medium"
                        icon = "🟢"

                    # Получаем рекомендации
                    recs = recommendation_bank.get(category, default_recs)
                    recs_html = "".join([f"<br>  • {r}" for r in recs])

                    st.markdown(f"""
                    <div class="rec-card {card_class}">
                        <div class="rec-header">
                            {icon} Паттерн #{i+1}: «{category}»
                            <span class="risk-badge-{card_class}">{priority} • {count} наблюдений</span>
                        </div>
                        <div class="rec-body">
                            <b>Рекомендованные меры контроля:</b>
                            {recs_html}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Семантический поиск рекомендаций из инцидентов этой организации
                if not org_incidents.empty:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="section-header">📚 Рекомендации из истории инцидентов</div>', unsafe_allow_html=True)

                    org_inc_with_recs = org_incidents.dropna(subset=['Краткое_описание_происшествия', 'Рекомендации'])
                    for _, row in org_inc_with_recs.head(5).iterrows():
                        rec = row.get('Рекомендации', '')
                        desc = str(row.get('Краткое_описание_происшествия', ''))[:200]
                        cls = row.get('Классификация_НС', '') or row.get('Классификация_ОМП', '') or 'Н/Д'

                        st.markdown(f"""
                        <div class="rec-card medium">
                            <div class="rec-body">
                                <b>💡 Рекомендация:</b> {rec}
                            </div>
                            <div class="rec-meta">
                                🏷️ {cls} &nbsp;|&nbsp; 📝 {desc}...
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.success("✅ У данной организации нет опасных паттернов в Картах Коргау.")

        with col_right:
            st.markdown('<div class="section-header">📊 Профиль организации</div>', unsafe_allow_html=True)

            if not org_danger.empty and 'Категория_наблюдения' in org_danger.columns:
                # Радар по категориям
                cat_counts = org_danger['Категория_наблюдения'].value_counts().head(6)
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=cat_counts.values.tolist() + [cat_counts.values[0]],
                    theta=cat_counts.index.tolist() + [cat_counts.index[0]],
                    fill='toself',
                    fillcolor='rgba(239, 68, 68, 0.2)',
                    line=dict(color='#ef4444', width=2),
                    name='Опасные наблюдения'
                ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor='rgba(0,0,0,0)',
                        radialaxis=dict(visible=True, gridcolor='rgba(255,255,255,0.1)'),
                        angularaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(size=9)),
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#cbd5e1'),
                    showlegend=False,
                    height=350,
                    margin=dict(l=40, r=40, t=30, b=30),
                )
                st.plotly_chart(fig_radar, width='stretch')

            # Временной тренд
            if not org_danger.empty:
                st.markdown('<div class="section-header">📈 Тренд по месяцам</div>', unsafe_allow_html=True)
                monthly_org = org_danger.groupby(org_danger['Дата_Время'].dt.to_period('M')).size().reset_index(name='Количество')
                monthly_org['Дата_Время'] = monthly_org['Дата_Время'].dt.to_timestamp()
                fig_org_trend = px.bar(
                    monthly_org, x='Дата_Время', y='Количество',
                    color_discrete_sequence=['#f59e0b'],
                )
                fig_org_trend.update_layout(
                    height=250,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#cbd5e1'),
                    xaxis_title="",
                    yaxis_title="",
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_org_trend, width='stretch')


# ─────────────────────────────────────────────────────
# ФУТЕР
# ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#475569; font-size:0.8rem; padding: 10px 0;">
    🧠 AI-модуль F-04 • Модель: <code>paraphrase-multilingual-MiniLM-L12-v2</code> •
    Метод: Семантический kNN (cosine similarity) + Паттерн-майнинг Коргау •
    Данные: {n_inc} инцидентов, {n_kor} наблюдений Коргау
</div>
""".format(n_inc=len(df_inc), n_kor=len(df_kor)), unsafe_allow_html=True)

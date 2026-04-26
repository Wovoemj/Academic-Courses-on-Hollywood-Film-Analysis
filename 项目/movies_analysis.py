import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# 设置页面
st.set_page_config(page_title="电影数据分析仪表板", layout="wide", page_icon="🎬")
st.title("🎬 好莱坞电影数据分析仪表板")
st.markdown("探索电影产业的五大核心维度：行业趋势、投资回报、类型分析、导演影响力和演员号召力")

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

# 加载和缓存数据
@st.cache_data
def load_data():
    try:
        # 尝试从多个可能路径加载数据
        possible_paths = [
            'tmdb_5000_movies.csv',
            './data/tmdb_5000_movies.csv',
            '../data/tmdb_5000_movies.csv'
        ]
        
        movies_df = None
        for path in possible_paths:
            try:
                movies_df = pd.read_csv(path)
                credits_path = path.replace('movies', 'credits')
                credits_df = pd.read_csv(credits_path)
                st.sidebar.success(f"成功加载数据: {path}")
                break
            except FileNotFoundError:
                continue
        
        if movies_df is None:
            st.error("无法找到数据文件。请确保存在'tmdb_5000_movies.csv'和'tmdb_5000_credits.csv'文件")
            return pd.DataFrame()
        
        # 重命名列以便合并
        credits_df.rename(columns={'movie_id': 'id'}, inplace=True)
        
        # 合并数据集
        df = pd.merge(movies_df, credits_df, on='id')
        
        # 处理缺失值和无效值
        df = df[(df['budget'] > 10000) & (df['revenue'] > 10000)]  # 过滤掉极小的预算和收入值
        
        # 解析JSON字段的函数
        def parse_json_column(text, key='name'):
            if pd.isna(text) or text == '' or text == '[]':
                return []
            try:
                data = ast.literal_eval(text)
                return [item[key] for item in data] if isinstance(data, list) else []
            except (ValueError, SyntaxError):
                return []
        
        # 创建新的列表列
        df['genres_list'] = df['genres'].apply(lambda x: parse_json_column(x))
        df['keywords_list'] = df['keywords'].apply(lambda x: parse_json_column(x))
        
        # 提取导演信息
        def get_director(crew_text):
            if pd.isna(crew_text) or crew_text == '' or crew_text == '[]':
                return None
            try:
                crew_list = ast.literal_eval(crew_text)
                for person in crew_list:
                    if person.get('job') == 'Director':
                        return person.get('name')
            except (ValueError, SyntaxError):
                pass
            return None
        
        df['director'] = df['crew'].apply(get_director)
        
        # 提取演员信息 (前5名主要演员)
        def get_top_cast(cast_text, n=5):
            if pd.isna(cast_text) or cast_text == '' or cast_text == '[]':
                return []
            try:
                cast_list = ast.literal_eval(cast_text)
                # 按出场顺序排序并取前n个
                return [actor['name'] for actor in cast_list[:n]]
            except (ValueError, SyntaxError):
                return []
        
        df['cast_list'] = df['cast'].apply(lambda x: get_top_cast(x, 5))
        
        # 添加其他计算列
        df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
        df['profit'] = df['revenue'] - df['budget']
        df['profit_ratio'] = df['profit'] / df['budget']
        df['roi'] = df['profit_ratio'] * 100  # 投资回报率百分比
        
        # 过滤掉年份异常的数据
        df = df[(df['year'] >= 1900) & (df['year'] <= 2023)]
        
        # 修复列名问题 - 创建统一的title列
        # 优先使用title_x，如果没有则使用title_y
        if 'title_x' in df.columns:
            df['title'] = df['title_x']
        elif 'title_y' in df.columns:
            df['title'] = df['title_y']
        else:
            # 如果两个都没有，尝试使用original_title
            df['title'] = df.get('original_title', 'Unknown Title')
        
        return df
    except Exception as e:
        st.error(f"加载数据时出错: {e}")
        return pd.DataFrame()

# 显示数据加载进度
progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

for i in range(5):
    # 更新进度
    progress_bar.progress((i + 1) * 20)
    status_text.text(f"加载数据中... {20*(i+1)}%")
    time.sleep(0.1)  # 模拟加载时间

# 加载数据
df = load_data()

# 完成后清除进度条
progress_bar.empty()
status_text.empty()

# 检查数据是否加载成功
if df.empty:
    st.error("无法加载数据，请检查数据文件是否存在且格式正确")
    st.stop()

# 数据质量检查
if not df.empty:
    st.sidebar.info(f"数据范围: {int(df['year'].min())}-{int(df['year'].max())}")
    st.sidebar.info(f"电影总数: {len(df):,}")
    st.sidebar.info(f"有导演信息的电影: {df['director'].notna().sum():,}")

# 添加侧边栏控件
st.sidebar.header("控制面板")
st.sidebar.info("使用以下筛选器调整数据范围")

# 年份范围选择器
year_range = st.sidebar.slider(
    "选择年份范围:",
    min_value=int(df['year'].min()),
    max_value=int(df['year'].max()),
    value=(1990, 2015)
)

# 预算范围选择器（使用对数尺度）
min_budget, max_budget = st.sidebar.slider(
    "选择预算范围(美元):",
    min_value=int(np.log10(df['budget'].min())),
    max_value=int(np.log10(df['budget'].max())) + 1,
    value=(6, 9)  # 10^6 到 10^9
)
min_budget, max_budget = 10**min_budget, 10**max_budget

# 可视化选项
st.sidebar.subheader("可视化选项")
chart_theme = st.sidebar.selectbox("选择图表主题:", ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"])
color_scale = st.sidebar.selectbox("选择颜色方案:", ["Viridis", "Plasma", "Inferno", "Magma", "Cividis"])

# 根据用户选择过滤数据
filtered_df = df[
    (df['year'] >= year_range[0]) & 
    (df['year'] <= year_range[1]) &
    (df['budget'] >= min_budget) & 
    (df['budget'] <= max_budget)
]

# 获取颜色序列
try:
    color_sequence = getattr(px.colors.sequential, color_scale)
except AttributeError:
    color_sequence = px.colors.sequential.Viridis
    st.sidebar.warning(f"颜色方案'{color_scale}'未找到，使用默认方案'Viridis'")

# 展示关键指标
st.header("📊 关键指标概览")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("电影数量", f"{len(filtered_df):,}")
with col2:
    st.metric("平均预算", f"${filtered_df['budget'].mean():,.0f}")
with col3:
    st.metric("平均票房", f"${filtered_df['revenue'].mean():,.0f}")
with col4:
    st.metric("平均投资回报率", f"{filtered_df['roi'].mean():.1f}%")

# 添加统计摘要
st.subheader("统计摘要")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("最高预算电影", f"${filtered_df['budget'].max():,}")
    st.metric("最低预算电影", f"${filtered_df['budget'].min():,}")

with col2:
    st.metric("最高票房电影", f"${filtered_df['revenue'].max():,}")
    st.metric("最低票房电影", f"${filtered_df['revenue'].min():,}")

with col3:
    st.metric("最高评分电影", f"{filtered_df['vote_average'].max():.1f}")
    st.metric("最低评分电影", f"{filtered_df['vote_average'].min():.1f}")

# 创建标签页
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 宏观趋势", 
    "💰 投资回报", 
    "🎭 类型分析", 
    "🎬 导演分析", 
    "🌟 演员分析"
])

# 1. 宏观趋势分析
with tab1:
    st.header("电影行业宏观趋势分析")
    
    # 按年份分组计算指标
    yearly_data = filtered_df.groupby('year').agg({
        'id': 'count', 
        'budget': 'mean', 
        'revenue': 'mean',
        'vote_average': 'mean',
        'roi': 'mean'
    }).rename(columns={'id': 'movie_count'}).reset_index()
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('电影产量变化', '平均预算变化', '平均票房变化', '平均评分变化'),
        vertical_spacing=0.15
    )
    
    # 电影数量
    fig.add_trace(
        go.Scatter(x=yearly_data['year'], y=yearly_data['movie_count'], 
                  mode='lines+markers', name='电影数量', line=dict(color='blue')),
        row=1, col=1
    )
    
    # 平均预算
    fig.add_trace(
        go.Scatter(x=yearly_data['year'], y=yearly_data['budget'], 
                  mode='lines+markers', name='平均预算', line=dict(color='green')),
        row=1, col=2
    )
    
    # 平均票房
    fig.add_trace(
        go.Scatter(x=yearly_data['year'], y=yearly_data['revenue'], 
                  mode='lines+markers', name='平均票房', line=dict(color='red')),
        row=2, col=1
    )
    
    # 平均评分
    fig.add_trace(
        go.Scatter(x=yearly_data['year'], y=yearly_data['vote_average'], 
                  mode='lines+markers', name='平均评分', line=dict(color='purple')),
        row=2, col=2
    )
    
    # 更新布局
    fig.update_layout(
        height=600, 
        showlegend=False, 
        title_text="电影行业年度趋势分析",
        template=chart_theme
    )
    fig.update_xaxes(title_text="年份", row=2, col=1)
    fig.update_xaxes(title_text="年份", row=2, col=2)
    fig.update_yaxes(title_text="电影数量", row=1, col=1)
    fig.update_yaxes(title_text="平均预算(美元)", row=1, col=2)
    fig.update_yaxes(title_text="平均票房(美元)", row=2, col=1)
    fig.update_yaxes(title_text="平均评分", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 添加洞察分析
    st.subheader("趋势洞察")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **电影产量趋势**: 
        - 分析电影年产量的变化趋势
        - 识别产量高峰期和低谷期
        - 探讨可能的影响因素（经济、技术等）
        """)
    
    with col2:
        st.info("""
        **预算与票房关系**: 
        - 观察预算和票房的长期变化
        - 分析两者之间的相关性
        - 探讨高预算是否总能带来高票房
        """)

# 2. 投资回报分析
with tab2:
    st.header("电影投资与回报分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 预算与票房散点图
        fig = px.scatter(
            filtered_df, x='budget', y='revenue', 
            hover_data=['title', 'year', 'director'],  # 这里使用统一的title列
            title='预算与票房关系',
            labels={'budget': '预算(美元)', 'revenue': '票房(美元)'},
            trendline='lowess',
            color_discrete_sequence=color_sequence
        )
        fig.update_layout(template=chart_theme)
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)
        
        # 计算相关系数
        correlation = filtered_df['budget'].corr(filtered_df['revenue'])
        st.metric("预算与票房的相关系数", f"{correlation:.3f}")
    
    with col2:
        # 投资回报率分布
        fig = px.histogram(
            filtered_df, x='roi', 
            title='投资回报率分布',
            labels={'roi': '投资回报率(%)'},
            nbins=50,
            color_discrete_sequence=color_sequence
        )
        fig.update_layout(template=chart_theme)
        st.plotly_chart(fig, use_container_width=True)
        
        # 高回报电影分析
        high_roi_movies = filtered_df[filtered_df['roi'] > 1000].sort_values('roi', ascending=False)
        if not high_roi_movies.empty:
            st.write("超高回报电影案例:")
            for _, movie in high_roi_movies.head(3).iterrows():
                st.write(f"- **{movie['title']}** (ROI: {movie['roi']:.0f}%)")
    
    # 预算与评分的关系
    st.subheader("预算与电影评分的关系")
    fig = px.scatter(
        filtered_df, x='budget', y='vote_average', 
        trendline='lowess',
        title='预算与电影评分',
        labels={'budget': '预算(美元)', 'vote_average': '评分'},
        color_discrete_sequence=color_sequence
    )
    fig.update_layout(template=chart_theme)
    st.plotly_chart(fig, use_container_width=True)

# 3. 电影类型分析
with tab3:
    st.header("电影类型分析")
    
    # 将嵌套的genres_list展开
    all_genres = [genre for sublist in filtered_df['genres_list'] for genre in sublist]
    genre_count = Counter(all_genres)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 类型流行度
        genres_df = pd.DataFrame(genre_count.most_common(15), columns=['Genre', 'Count'])
        fig = px.bar(
            genres_df, x='Count', y='Genre', 
            orientation='h', title='最流行的电影类型',
            labels={'Count': '出现次数', 'Genre': '类型'},
            color_discrete_sequence=color_sequence
        )
        fig.update_layout(template=chart_theme)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 类型与平均票房
        genre_revenue = {}
        for genre in genre_count:
            # 计算包含该类型的电影的平均票房
            genre_movies = filtered_df[filtered_df['genres_list'].apply(lambda x: genre in x)]
            if len(genre_movies) > 5:  # 只考虑有足够样本的类型
                genre_revenue[genre] = genre_movies['revenue'].mean()
        
        # 转换为DataFrame并排序
        genre_revenue_df = pd.DataFrame(
            list(genre_revenue.items()), 
            columns=['Genre', 'Avg_Revenue']
        ).sort_values('Avg_Revenue', ascending=False).head(15)
        
        fig = px.bar(
            genre_revenue_df, x='Avg_Revenue', y='Genre', 
            orientation='h', title='各类型电影的平均票房',
            labels={'Avg_Revenue': '平均票房(美元)', 'Genre': '类型'},
            color_discrete_sequence=color_sequence
        )
        fig.update_layout(template=chart_theme)
        st.plotly_chart(fig, use_container_width=True)
    
    # 类型与投资回报率
    st.subheader("各类型电影的投资回报率")
    genre_roi = {}
    for genre in genre_count:
        genre_movies = filtered_df[filtered_df['genres_list'].apply(lambda x: genre in x)]
        if len(genre_movies) > 5:
            genre_roi[genre] = genre_movies['roi'].mean()
    
    genre_roi_df = pd.DataFrame(
        list(genre_roi.items()), 
        columns=['Genre', 'Avg_ROI']
    ).sort_values('Avg_ROI', ascending=False).head(15)
    
    fig = px.bar(
        genre_roi_df, x='Avg_ROI', y='Genre', 
        orientation='h', title='各类型电影的平均投资回报率',
        labels={'Avg_ROI': '平均投资回报率(%)', 'Genre': '类型'},
        color_discrete_sequence=color_sequence
    )
    fig.update_layout(template=chart_theme)
    st.plotly_chart(fig, use_container_width=True)

# 4. 导演分析
with tab4:
    st.header("导演影响力分析")
    
    # 过滤掉没有导演信息的行
    director_df = filtered_df[filtered_df['director'].notna()]
    
    # 计算每位导演的平均票房和电影数量
    director_stats = director_df.groupby('director').agg({
        'revenue': 'mean', 
        'id': 'count',
        'roi': 'mean',
        'vote_average': 'mean'
    }).rename(columns={'id': 'movie_count', 'revenue': 'avg_revenue'})
    
    # 过滤掉电影数量太少的导演
    director_stats = director_stats[director_stats['movie_count'] >= 3]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 平均票房最高的导演
        top_directors_revenue = director_stats.sort_values('avg_revenue', ascending=False).head(10)
        fig = px.bar(
            top_directors_revenue, x='avg_revenue', y=top_directors_revenue.index, 
            orientation='h', title='平均票房最高的导演 (作品≥3部)',
            labels={'avg_revenue': '平均票房(美元)', 'director': '导演'},
            color_discrete_sequence=color_sequence
        )
        fig.update_layout(template=chart_theme)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 投资回报率最高的导演
        top_directors_roi = director_stats.sort_values('roi', ascending=False).head(10)
        fig = px.bar(
            top_directors_roi, x='roi', y=top_directors_roi.index, 
            orientation='h', title='投资回报率最高的导演 (作品≥3部)',
            labels={'roi': '平均投资回报率(%)', 'director': '导演'},
            color_discrete_sequence=color_sequence
        )
        fig.update_layout(template=chart_theme)
        st.plotly_chart(fig, use_container_width=True)
    
    # 导演作品数量与平均票房的关系
    st.subheader("导演作品数量与平均票房的关系")
    fig = px.scatter(
        director_stats, x='movie_count', y='avg_revenue', 
        hover_data=[director_stats.index],
        title='导演作品数量 vs 平均票房',
        labels={'movie_count': '作品数量', 'avg_revenue': '平均票房(美元)'},
        trendline='lowess',
        color_discrete_sequence=color_sequence
    )
    fig.update_layout(template=chart_theme)
    st.plotly_chart(fig, use_container_width=True)

# 5. 演员分析
with tab5:
    st.header("演员号召力分析")
    
    # 展开演员列表
    actor_list = []
    for _, row in filtered_df.iterrows():
        for actor in row['cast_list']:
            actor_list.append({
                'actor': actor,
                'revenue': row['revenue'],
                'budget': row['budget'],
                'roi': row['roi'],
                'title': row['title']
            })
    
    actors_df = pd.DataFrame(actor_list)
    
    if not actors_df.empty:
        # 计算每位演员的平均票房和参演电影数量
        actor_stats = actors_df.groupby('actor').agg({
            'revenue': 'mean', 
            'roi': 'mean',
            'title': 'count'
        }).rename(columns={'title': 'movie_count', 'revenue': 'avg_revenue'})
        
        # 过滤掉参演电影太少的演员
        actor_stats = actor_stats[actor_stats['movie_count'] >= 3]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 平均票房最高的演员
            top_actors_revenue = actor_stats.sort_values('avg_revenue', ascending=False).head(15)
            fig = px.bar(
                top_actors_revenue, x='avg_revenue', y=top_actors_revenue.index, 
                orientation='h', title='平均票房最高的演员 (参演≥3部)',
                labels={'avg_revenue': '平均票房(美元)', 'actor': '演员'},
                color_discrete_sequence=color_sequence
            )
            fig.update_layout(template=chart_theme)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 投资回报率最高的演员
            top_actors_roi = actor_stats.sort_values('roi', ascending=False).head(15)
            fig = px.bar(
                top_actors_roi, x='roi', y=top_actors_roi.index, 
                orientation='h', title='投资回报率最高的演员 (参演≥3部)',
                labels={'roi': '平均投资回报率(%)', 'actor': '演员'},
                color_discrete_sequence=color_sequence
            )
            fig.update_layout(template=chart_theme)
            st.plotly_chart(fig, use_container_width=True)
        
        # 演员参演电影数量与平均票房的关系
        st.subheader("演员参演电影数量与平均票房的关系")
        fig = px.scatter(
            actor_stats, x='movie_count', y='avg_revenue', 
            hover_data=[actor_stats.index],
            title='演员参演数量 vs 平均票房',
            labels={'movie_count': '参演电影数量', 'avg_revenue': '平均票房(美元)'},
            trendline='lowess',
            color_discrete_sequence=color_sequence
        )
        fig.update_layout(template=chart_theme)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("没有足够的演员数据进行分析")

# 添加数据浏览部分
st.header("🔍 数据浏览")
search_term = st.text_input("搜索电影标题:", "")
if search_term:
    browse_df = filtered_df[filtered_df['title'].str.contains(search_term, case=False, na=False)]
else:
    browse_df = filtered_df

st.dataframe(browse_df[['title', 'year', 'director', 'budget', 'revenue', 'vote_average']].head(20))

# 添加下载数据选项
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(filtered_df[['title', 'year', 'director', 'budget', 'revenue', 'vote_average']])
st.download_button(
    label="下载筛选后的数据 (CSV)",
    data=csv,
    file_name="filtered_movie_data.csv",
    mime="text/csv",
)

# 页脚
st.markdown("---")
expander = st.expander("关于数据和分析方法")
expander.write("""
本仪表板使用TMDB 5000电影数据集，包含约5000部电影的信息。

**数据包括:**
- 电影基本信息(标题、年份、预算、收入等)
- 类型和关键词信息
- 导演和演员信息

**分析方法:**
- 投资回报率(ROI)计算: (收入-预算)/预算 * 100%
- 类型分析基于电影的类型标签
- 导演和演员分析基于其参与的电影表现

**局限性:**
- 数据可能不完整或包含错误
- 预算和收入数据可能不是通货膨胀调整后的
- 某些电影可能有多个导演或演员，分析中只考虑了主要导演和前5名演员
""")

st.caption("数据来源: TMDB 5000 Movie Dataset | 分析工具: Streamlit, Pandas, Plotly")
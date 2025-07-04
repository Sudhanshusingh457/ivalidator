import streamlit as st
import pandas as pd
import plotly.express as px
import os
import tempfile
from streamlit_extras.metric_cards import style_metric_cards

# --------------- Helper Functions ----------------

def show_data_types(df):
    type_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[col].dtype) for col in df.columns],
        "Null Values": df.isna().sum().values,
        "Unique Values": df.nunique().values
    })
    return type_info

def find_duplicates(df, columns):
    duplicates = df[df.duplicated(subset=columns, keep=False)]
    if not duplicates.empty:
        duplicates = duplicates.sort_values(by=columns)
        duplicate_counts = duplicates.groupby(columns).size().reset_index(name='Duplicate Count')
        return duplicates, duplicate_counts
    return pd.DataFrame(), pd.DataFrame()

def convert_column_types(df, column, target_type, date_format=None, verbose=False):
    try:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe")
        clean_series = df[column].copy()
        original_dtype = str(clean_series.dtype)
        original_sample = clean_series.head(3).tolist()
        if pd.api.types.is_object_dtype(clean_series):
            clean_series = clean_series.fillna('')
        if pd.api.types.is_object_dtype(clean_series):
            clean_series = (
                clean_series.astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
            )
        target_type = target_type.lower()
        if target_type in ['numeric', 'int', 'float']:
            clean_series = (
                clean_series.astype(str)
                .str.replace(r'[^\d\.\-eE,]', '', regex=True)
                .str.replace(',', '.')
                .str.replace(r'(?<=\d)\.(?=\d)', '.')
                .str.replace(r'\.(?=.*\.)', '')
                .str.replace(r'^\.', '0.')
                .str.replace(r'^-\.', '-0.')
            )
            result = pd.to_numeric(clean_series, errors='coerce')
            if target_type == 'int':
                result = result.astype('Int64')
        elif target_type in ['date', 'datetime']:
            if date_format:
                result = pd.to_datetime(clean_series, format=date_format, errors='coerce')
            else:
                result = pd.to_datetime(clean_series, errors='coerce')
            if target_type == 'date':
                result = result.dt.date
        elif target_type == 'string':
            result = (
                clean_series.astype(str).str.strip().str.normalize('NFKC')
                .str.replace(r'[\x00-\x1F\x7F-\x9F]', '', regex=True)
                .str.replace(r'\s+', ' ', regex=True)
            )
        elif target_type == 'category':
            result = clean_series.astype('category')
        else:
            raise ValueError(f"Unsupported target type: {target_type}")
        return result
    except Exception as e:
        if verbose:
            print(f"⚠️ Conversion error for {column}: {str(e)}")
        return df[column]

def validate_data_quality(df, columns):
    results = []
    for col in columns:
        col_stats = {
            'Column': col,
            'Data Type': str(df[col].dtype),
            'Null Values': df[col].isna().sum(),
            'Null Percentage': f"{df[col].isna().mean() * 100:.1f}%",
            'Unique Values': df[col].nunique(),
            'Sample Values': [str(val) for val in df[col].dropna().head(3)]
        }
        if pd.api.types.is_string_dtype(df[col]):
            col_stats.update({
                'Min Length': df[col].str.len().min(),
                'Max Length': df[col].str.len().max(),
                'Empty Strings': (df[col] == '').sum()
            })
        results.append(col_stats)
    return pd.DataFrame(results)

# --------------- Streamlit UI ----------------

st.set_page_config(
    page_title="Data Validator",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme toggle (light/dark mode)
if 'theme_mode' not in st.session_state:
    st.session_state['theme_mode'] = 'light'

with st.sidebar:
    theme = st.radio(
        'Theme',
        ['Light', 'Dark'],
        index=0 if st.session_state['theme_mode'] == 'light' else 1,
        help='Switch between light and dark mode.'
    )
    st.session_state['theme_mode'] = theme.lower()

# Apply theme CSS
if st.session_state['theme_mode'] == 'dark':
    st.markdown('''
        <style>
        .stApp { background-color: #181c20; }
        h1, h2, h3, h4, h5, h6, .stMarkdown, .stCaption, .stMetric, .stTabs, .stExpander, .stButton>button, label, .stTextInput label, .stSelectbox label, .stNumberInput label, .stRadio label, .stCheckbox label, .stFileUploader label, .stTextArea label {
            color: #fff !important;
        }
        .stDataFrame, .stExpander, .stTabs [data-baseweb="tab-list"], .stTabs [data-baseweb="tab"], .stTabs, .stCaption, .stSidebar, .stMetric, .stDataFrame, .stDataFrame th, .stDataFrame td {
            background-color: #fff !important;
            color: #181c20 !important;
            border-color: #444 !important;
        }
        .stSidebar, .css-1d391kg, .stSidebarContent {
            background-color: #23272e !important;
            color: #fff !important;
        }
        .stTextInput, .stSelectbox, .stNumberInput, .stRadio, .stCheckbox, .stFileUploader, .stTextArea {
            background-color: #fff !important;
            color: #181c20 !important;
            border-radius: 6px !important;
        }
        .stButton>button {
            background-color: #4682b4 !important;
            color: #fff !important;
        }
        .stButton>button:hover {
            background-color: #36648b !important;
        }
        .metric-card {
            background-color: #fff !important;
            border-left: 4px solid #4682b4 !important;
            border-radius: 8px !important;
            color: #181c20 !important;
        }
        .stDataFrame {
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .streamlit-expanderHeader {
            background-color: #fff !important;
            color: #181c20 !important;
        }
        </style>
    ''', unsafe_allow_html=True)
else:
    st.markdown('''
        <style>
        .stApp { background-color: #f0f8ff; }
        h1 { color: #1a3e72; border-bottom: 2px solid #4682b4; padding-bottom: 10px; }
        .css-1d391kg { background-color: #e6f2ff !important; }
        .stButton>button { background-color: #4682b4 !important; color: white !important; border-radius: 8px; padding: 8px 16px; border: none; transition: all 0.3s; }
        .stButton>button:hover { background-color: #36648b !important; transform: translateY(-2px); box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
        .stDataFrame { border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .metric-card { background-color: white !important; border-left: 4px solid #4682b4 !important; border-radius: 8px !important; }
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] { background-color: #e6f2ff; border-radius: 8px 8px 0 0; padding: 8px 16px; }
        .stTabs [aria-selected="true"] { background-color: #4682b4 !important; color: white !important; }
        .streamlit-expanderHeader { background-color: #e6f2ff; border-radius: 8px; padding: 8px 16px; }
        </style>
    ''', unsafe_allow_html=True)

col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/128/2519/2519393.png", width=80)
with col2:
    st.title("Data Validation Portal")
    st.caption("Validate crew data between Power BI reports and client systems ")

with st.sidebar:
    st.header("⚙️ Settings")

    st.header("📊 Data Options")
    show_quality_checks = st.checkbox(
        "Enable Advanced Data Quality Checks",
        value=True,
        help="Perform comprehensive data quality analysis"
    )
    st.divider()
    st.info("This tool helps maritime operators validate crew data between reporting systems and client records.")

# User selects how many rows to load for processing
max_rows = st.number_input(
    "How many rows to load/process at a time? (Context window)",
    min_value=100, max_value=1000000, value=10000, step=1000,
    help="Only this many rows will be loaded into memory at once. Use lower values for very large files."
)

# Helper for chunked reading
@st.cache_data(show_spinner=False)
def read_excel_chunked(uploaded_file, nrows):
    """
    Reads an uploaded file (Excel or CSV) and returns a DataFrame.
    Automatically detects file type by extension or MIME type.
    """
    import os
    import io
    # Try to get file name or fallback to default
    file_name = getattr(uploaded_file, 'name', None)
    if file_name:
        ext = os.path.splitext(file_name)[1].lower()
    else:
        ext = ''
    if ext in ['.xlsx', '.xls']:
        return pd.read_excel(uploaded_file, nrows=nrows, engine="openpyxl")
    elif ext == '.csv':
        return pd.read_csv(uploaded_file, nrows=nrows)
    else:
        # Try to sniff file type by reading a few bytes
        try:
            pos = uploaded_file.tell()
            sample = uploaded_file.read(2048)
            uploaded_file.seek(pos)
            if b',' in sample or b'\n' in sample:
                return pd.read_csv(uploaded_file, nrows=nrows)
            else:
                return pd.read_excel(uploaded_file, nrows=nrows, engine="openpyxl")
        except Exception:
            # Fallback to Excel
            return pd.read_excel(uploaded_file, nrows=nrows, engine="openpyxl")

@st.cache_data(show_spinner=False)
def read_sql_chunked(query, conn_str, nrows):
    import pyodbc
    conn = pyodbc.connect(conn_str)
    df = pd.read_sql(query, conn, chunksize=nrows)
    chunk = next(df)
    conn.close()
    return chunk

# Store large datasets in temp files
TEMP_DIR = tempfile.gettempdir()
def save_temp_csv(df, prefix):
    temp_path = os.path.join(TEMP_DIR, f"{prefix}_large_data.csv")
    df.to_csv(temp_path, index=False)
    return temp_path

def load_temp_csv(prefix, nrows):
    temp_path = os.path.join(TEMP_DIR, f"{prefix}_large_data.csv")
    if os.path.exists(temp_path):
        return pd.read_csv(temp_path, nrows=nrows)
    return None

# Safe defaults
df_report, df_client = None, None
report_file, client_file = None, None

st.subheader("📤 Select Data Sources")

report_source = st.radio("Power BI Data Source", ["Upload Excel", "SQL Server"], key="report_source")
client_source = st.radio("Client Data Source", ["Upload Excel", "SQL Server"], key="client_source")

# ------------------ Power BI Dataset ------------------
if report_source == "Upload Excel":
    report_file = st.file_uploader("Upload Power BI Excel file", type=["xlsx"], key="report_file")
    if report_file:
        df_report = read_excel_chunked(report_file, max_rows)
        st.success(f"✅ Loaded {len(df_report)} rows from Power BI Excel file")
        temp_path = save_temp_csv(df_report, "report")
        st.caption(f"Data stored at: {temp_path}")

elif report_source == "SQL Server":
    st.markdown("**Power BI SQL Settings**")
    sql_host_r = st.text_input("SQL Server Host (Power BI)", value="localhost", key="sql_host_r")
    sql_db_r = st.text_input("Database (Power BI)", key="sql_db_r")
    sql_user_r = st.text_input("Username (Power BI)", key="sql_user_r")
    sql_pass_r = st.text_input("Password (Power BI)", type="password", key="sql_pass_r")
    sql_query_r = st.text_area("SQL Query (Power BI)", height=100, key="sql_query_r")
    if st.button("Load Power BI Data from SQL"):
        try:
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={sql_host_r};DATABASE={sql_db_r};UID={sql_user_r};PWD={sql_pass_r}"
            )
            df_report = read_sql_chunked(sql_query_r, conn_str, max_rows)
            st.success(f"✅ Loaded {len(df_report)} rows from Power BI SQL Server")
            temp_path = save_temp_csv(df_report, "report")
            st.caption(f"Data stored at: {temp_path}")
        except Exception as e:
            st.error(f"Power BI SQL Load Failed: {e}")

# ------------------ Client Dataset ------------------
if client_source == "Upload Excel":
    client_file = st.file_uploader("Upload Client Excel file", type=["xlsx"], key="client_file")
    if client_file:
        df_client = read_excel_chunked(client_file, max_rows)
        st.success(f"✅ Loaded {len(df_client)} rows from Client Excel file")
        temp_path = save_temp_csv(df_client, "client")
        st.caption(f"Data stored at: {temp_path}")

elif client_source == "SQL Server":
    st.markdown("**Client SQL Settings**")
    sql_host_c = st.text_input("SQL Server Host (Client)", value="localhost", key="sql_host_c")
    sql_db_c = st.text_input("Database (Client)", key="sql_db_c")
    sql_user_c = st.text_input("Username (Client)", key="sql_user_c")
    sql_pass_c = st.text_input("Password (Client)", type="password", key="sql_pass_c")
    sql_query_c = st.text_area("SQL Query (Client)", height=100, key="sql_query_c")
    if st.button("Load Client Data from SQL"):
        try:
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={sql_host_c};DATABASE={sql_db_c};UID={sql_user_c};PWD={sql_pass_c}"
            )
            df_client = read_sql_chunked(sql_query_c, conn_str, max_rows)
            st.success(f"✅ Loaded {len(df_client)} rows from Client SQL Server")
            temp_path = save_temp_csv(df_client, "client")
            st.caption(f"Data stored at: {temp_path}")
        except Exception as e:
            st.error(f"Client SQL Load Failed: {e}")

# If both datasets are loaded, proceed
if df_report is not None and df_client is not None:
    try:
        with st.expander("🔍 Preview Data", expanded=True):
            preview_tab1, preview_tab2 = st.tabs(["Power BI Data", "Client Data"])
            with preview_tab1:
                if df_report is not None:
                    st.dataframe(df_report.head(), use_container_width=True)
                    st.caption(f"Shape: {df_report.shape[0]} rows × {df_report.shape[1]} columns")
                else:
                    st.info("No Power BI data loaded yet. Please upload a file or load from SQL.")
            with preview_tab2:
                if df_client is not None:
                    st.dataframe(df_client.head(), use_container_width=True)
                    st.caption(f"Shape: {df_client.shape[0]} rows × {df_client.shape[1]} columns")
                else:
                    st.info("No Client data loaded yet. Please upload a file or load from SQL.")
        st.subheader("🔎 Comparison Setup")
        st.markdown("**Select columns to compare**")
        col_sel_col1, col_sel_col2 = st.columns(2)
        with col_sel_col1:
            report_columns = st.multiselect(
                "Power BI Columns", 
                df_report.columns,
                help="Select one or more columns to compare",
                key="report_cols"
            )
        with col_sel_col2:
            client_columns = st.multiselect(
                "Client Columns", 
                df_client.columns,
                help="Select corresponding columns from client data",
                key="client_cols"
            )
        if report_columns and client_columns and len(report_columns) == len(client_columns):
            with st.expander("🔧 Data Type Matching", expanded=True):
                st.markdown("<h5 style='margin-bottom:0'>Column Type Conversion</h5>", unsafe_allow_html=True)
                st.caption("Choose the target type for each column. Preview and data type shown for context.")
                type_mapping = []
                col_pairs = list(zip(report_columns, client_columns))
                for i, (r_col, c_col) in enumerate(col_pairs):
                    r_type = str(df_report[r_col].dtype)
                    c_type = str(df_client[c_col].dtype)
                    r_preview = ', '.join([str(x) for x in df_report[r_col].dropna().head(3)])
                    c_preview = ', '.join([str(x) for x in df_client[c_col].dropna().head(3)])
                    # Determine type options for each column
                    if 'date' in r_type.lower() or 'date' in c_type.lower():
                        options = ['datetime', 'date', 'string']
                    elif 'int' in r_type or 'float' in r_type or 'int' in c_type or 'float' in c_type:
                        options = ['numeric', 'string']
                    else:
                        options = ['string', 'category']
                    colA, colB = st.columns(2)
                    with colA:
                        st.markdown(f"<div style='background:#f8fafd;padding:10px;border-radius:8px;border:1px solid #e0e6ef;margin-bottom:8px'>"
                                    f"<b>Power BI:</b> <span style='color:#4682b4'>{r_col}</span> "
                                    f"<span style='background:#e6f2ff;border-radius:4px;padding:2px 6px;font-size:12px;color:#1a3e72'>Type: {r_type}</span><br>"
                                    f"<span style='font-size:12px;color:#888'>Preview: {r_preview}</span>"
                                    f"</div>", unsafe_allow_html=True)
                        target_type_r = st.selectbox(
                            "Convert to:",
                            options,
                            key=f"type_conv_report_{i}"
                        )
                    with colB:
                        st.markdown(f"<div style='background:#f8fafd;padding:10px;border-radius:8px;border:1px solid #e0e6ef;margin-bottom:8px'>"
                                    f"<b>Client:</b> <span style='color:#4682b4'>{c_col}</span> "
                                    f"<span style='background:#e6f2ff;border-radius:4px;padding:2px 6px;font-size:12px;color:#1a3e72'>Type: {c_type}</span><br>"
                                    f"<span style='font-size:12px;color:#888'>Preview: {c_preview}</span>"
                                    f"</div>", unsafe_allow_html=True)
                        target_type_c = st.selectbox(
                            "Convert to:",
                            options,
                            key=f"type_conv_client_{i}"
                        )
                    type_mapping.append((r_col, target_type_r, c_col, target_type_c))
                if st.button("Apply Type Conversions", help="Convert columns to selected types"):
                    with st.spinner("Applying type conversions..."):
                        for r_col, target_type_r, c_col, target_type_c in type_mapping:
                            df_report[r_col] = convert_column_types(df_report, r_col, target_type_r)
                            df_client[c_col] = convert_column_types(df_client, c_col, target_type_c)
                    st.success("Data type conversions applied!")
            with st.expander("📊 Data Types (Post-Conversion)", expanded=True):
                type_tab1, type_tab2 = st.tabs(["Power BI Data Types", "Client Data Types"])
                with type_tab1:
                    st.dataframe(show_data_types(df_report[report_columns]), use_container_width=True)
                with type_tab2:
                    st.dataframe(show_data_types(df_client[client_columns]), use_container_width=True)
            if show_quality_checks:
                with st.expander("🧹 Data Quality Analysis", expanded=True):
                    qual_tab1, qual_tab2 = st.tabs(["Power BI Quality", "Client Quality"])
                    with qual_tab1:
                        st.markdown("### Power BI Data Quality Report")
                        report_quality = validate_data_quality(df_report, report_columns)
                        st.dataframe(report_quality, use_container_width=True)
                    with qual_tab2:
                        st.markdown("### Client Data Quality Report")
                        client_quality = validate_data_quality(df_client, client_columns)
                        st.dataframe(client_quality, use_container_width=True)
            with st.expander("🔎 Duplicate Detection", expanded=False):
                dup_tab1, dup_tab2 = st.tabs(["Power BI Duplicates", "Client Duplicates"])
                with dup_tab1:
                    report_duplicates, report_dup_counts = find_duplicates(df_report, report_columns)
                    if not report_duplicates.empty:
                        st.warning(f"Found {len(report_duplicates)} duplicate rows")
                        st.dataframe(report_dup_counts, use_container_width=True)
                        st.download_button(
                            "📥 Download Power BI Duplicates",
                            report_duplicates.to_csv(index=False).encode('utf-8'),
                            "powerbi_duplicates.csv",
                            "text/csv",
                            key="dl_pbi_dupes"
                        )
                    else:
                        st.success("No duplicates found in Power BI data")
                with dup_tab2:
                    client_duplicates, client_dup_counts = find_duplicates(df_client, client_columns)
                    if not client_duplicates.empty:
                        st.warning(f"Found {len(client_duplicates)} duplicate rows")
                        st.dataframe(client_dup_counts, use_container_width=True)
                        st.download_button(
                            "📥 Download Client Duplicates",
                            client_duplicates.to_csv(index=False).encode('utf-8'),
                            "client_duplicates.csv",
                            "text/csv",
                            key="dl_client_dupes"
                        )
                    else:
                        st.success("No duplicates found in Client data")
            join_type = st.selectbox(
                "Comparison Method",
                ["Left Join (Find mismatches)", "Inner Join (Find matches)", "Outer Join (Find all differences)"],
                help="Select how to compare the datasets"
            )
            if join_type == "Left Join (Find mismatches)":
                merged = df_report.merge(df_client, left_on=report_columns, right_on=client_columns, how='left', indicator=True)
                mismatches = merged[merged['_merge'] == 'left_only'][report_columns]
                matches = None
            elif join_type == "Inner Join (Find matches)":
                merged = df_report.merge(df_client, left_on=report_columns, right_on=client_columns, how='inner')
                matches = merged[report_columns + client_columns]
                mismatches = None
            else:
                merged = df_report.merge(df_client, left_on=report_columns, right_on=client_columns, how='outer', indicator=True)
                mismatches = merged[merged['_merge'] != 'both']
                report_only = merged[merged['_merge'] == 'left_only'][report_columns]
                client_only = merged[merged['_merge'] == 'right_only'][client_columns]
            st.subheader("📊 Validation Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Report Rows", len(df_report), help="Number of rows in Power BI data")
            with col2:
                st.metric("Total Client Rows", len(df_client), help="Number of rows in Client data")
            if join_type == "Inner Join (Find matches)":
                with col3:
                    st.metric("Matching Rows Found", len(matches), help="Rows present in both datasets")
                st.success(f"✅ Found {len(matches)} matching records")
                with st.expander("🔍 View Matching Records", expanded=True):
                    st.dataframe(matches, use_container_width=True)
                st.download_button(
                    "📥 Download Matching Records",
                    matches.to_csv(index=False).encode('utf-8'),
                    "matching_records.csv",
                    "text/csv",
                    key='download-matches',
                    help="Download all matching records"
                )
            else:
                with col3:
                    st.metric("Mismatches Found", len(mismatches) if mismatches is not None else 0, help="Rows with discrepancies between datasets")
                if mismatches is not None and len(mismatches) > 0:
                    st.warning(f"⚠️ Found {len(mismatches)} mismatches")
                    if join_type == "Outer Join (Find all differences)":
                        tab_mismatch, tab_only_client, tab_stats = st.tabs(["Unmatched Records", "Only in Client", "Statistics"])
                        with tab_mismatch:
                            st.markdown("**Records only in Power BI Report**")
                            st.dataframe(report_only, use_container_width=True)
                            st.download_button(
                                "📥 Download Only in Power BI",
                                report_only.to_csv(index=False).encode('utf-8'),
                                "only_in_powerbi.csv",
                                "text/csv",
                                key='download-only-powerbi',
                                help="Download records only in Power BI report"
                            )
                        with tab_only_client:
                            st.markdown("**Records only in Client Data**")
                            st.dataframe(client_only, use_container_width=True)
                            st.download_button(
                                "📥 Download Only in Client",
                                client_only.to_csv(index=False).encode('utf-8'),
                                "only_in_client.csv",
                                "text/csv",
                                key='download-only-client',
                                help="Download records only in Client data"
                            )
                        with tab_stats:
                            st.markdown(f"- **Total Differences Found**: {len(mismatches)}")
                            st.markdown(f"- **Only in Power BI**: {len(report_only)}")
                            st.markdown(f"- **Only in Client Data**: {len(client_only)}")
                            pie_data = pd.DataFrame({
                                "Status": ["Matching", "Only in Power BI", "Only in Client"],
                                "Count": [
                                    len(df_report) - len(report_only),
                                    len(report_only),
                                    len(client_only)
                                ]
                            })
                            fig = px.pie(
                                pie_data,
                                values='Count',
                                names='Status',
                                title='Data Comparison Distribution',
                                color='Status',
                                color_discrete_map={
                                    'Matching': '#4682b4',
                                    'Only in Power BI': '#ffa07a',
                                    'Only in Client': '#ff7f50'
                                }
                            )
                            fig.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        tab_mismatch, tab_stats = st.tabs(["Unmatched Records", "Statistics"])
                        with tab_mismatch:
                            st.dataframe(mismatches, use_container_width=True)
                            st.download_button(
                                "📥 Download All Differences",
                                mismatches.to_csv(index=False).encode('utf-8'),
                                "all_differences.csv",
                                "text/csv",
                                key='download-csv',
                                help="Download all mismatched records"
                            )
                        with tab_stats:
                            match_pct = 100 * (len(df_report) - len(mismatches)) / len(df_report)
                            st.markdown(f"- **Match Percentage**: {match_pct:.1f}%")
                            st.markdown(f"- **Mismatch Percentage**: {100 - match_pct:.1f}%")
                            pie_data = pd.DataFrame({
                                "Status": ["Matching", "Mismatched"],
                                "Count": [len(df_report) - len(mismatches), len(mismatches)]
                            })
                            fig = px.pie(
                                pie_data,
                                values='Count',
                                names='Status',
                                title='Match/Mismatch Distribution',
                                color='Status',
                                color_discrete_map={
                                    'Matching': '#4682b4',
                                    'Mismatched': '#ff7f50'
                                }
                            )
                            fig.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("✅ Perfect match! All records align.")
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")

# --- Modern Loading Animation ---
loading_css = '''
<style>
#modern-loader {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(24,28,32,0.85);
  z-index: 9999;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: opacity 0.5s;
}
.loader-anim {
  border: 6px solid #e0e6ef;
  border-top: 6px solid #4682b4;
  border-radius: 50%;
  width: 60px;
  height: 60px;
  animation: spin 1s linear infinite;
}
@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style>
'''

# --- Modern Button Animation ---
button_css = '''
<style>
.stButton>button {
  transition: background 0.2s, box-shadow 0.2s, transform 0.1s;
  box-shadow: 0 2px 8px rgba(70,130,180,0.08);
}
.stButton>button:active {
  transform: scale(0.97);
  box-shadow: 0 1px 4px rgba(70,130,180,0.12);
}
</style>
'''

# --- Card Hover Animation ---
card_css = '''
<style>
.modern-card {
  transition: box-shadow 0.2s, transform 0.2s;
  box-shadow: 0 2px 8px rgba(70,130,180,0.08);
  border-radius: 10px;
  background: #fff;
  padding: 18px 20px;
  margin-bottom: 18px;
}
.modern-card:hover {
  box-shadow: 0 6px 24px rgba(70,130,180,0.18);
  transform: translateY(-2px) scale(1.01);
}
</style>
'''

# st.markdown(loading_css, unsafe_allow_html=True)
# st.markdown(button_css, unsafe_allow_html=True)
# st.markdown(card_css, unsafe_allow_html=True)

# # --- Show loader on heavy actions ---
# def show_loader():
#     st.markdown('''<div id="modern-loader"><div class="loader-anim"></div></div>''', unsafe_allow_html=True)
# def hide_loader():
#     st.markdown('''<style>#modern-loader{display:none !important;}</style>''', unsafe_allow_html=True)

# # Example usage: show_loader() before a heavy operation, hide_loader() after
# # You can wrap heavy operations like file upload, SQL load, or type conversion with these

# # --- Modern Card Wrapper for UI Sections ---
# def modern_card(content_func, *args, **kwargs):
#     st.markdown('<div class="modern-card">', unsafe_allow_html=True)
#     content_func(*args, **kwargs)
#     st.markdown('</div>', unsafe_allow_html=True)



style_metric_cards(background_color="#f0f8ff", border_left_color="#4682b4")

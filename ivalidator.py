import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards


def get_ai_response(prompt, provider, model, api_key):
    """Get AI response from specified provider."""
    if provider == "Hugging Face":
        return call_huggingface_api(prompt, model, api_key)
    return call_openai_api(prompt, model, api_key)


def call_huggingface_api(prompt, model, api_key):
    """Call Hugging Face API with given parameters."""
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": 512}}
        )
        if response.status_code == 200:
            return response.json()[0]["generated_text"].replace(prompt, "").strip()
        return f"❌ Hugging Face Error ({response.status_code}): {response.text}"
    except Exception as e:
        return f"❌ Hugging Face Exception: {str(e)}"


def call_openai_api(prompt, model, api_key):
    """Call OpenAI API with given parameters."""
    try:
        openai_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5
        }
        response = requests.post(openai_url, headers=headers, json=body)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        return f"❌ OpenAI Error ({response.status_code}): {response.text}"
    except Exception as e:
        return f"❌ OpenAI Exception: {str(e)}"


def show_data_types(df):
    """Display the data types of all columns in a DataFrame."""
    return pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[col].dtype) for col in df.columns],
        "Null Values": df.isna().sum().values
    })


def find_duplicates(df, columns):
    """Find and return duplicate rows based on specified columns."""
    duplicates = df[df.duplicated(subset=columns, keep=False)]
    if not duplicates.empty:
        duplicates = duplicates.sort_values(by=columns)
        duplicate_counts = duplicates.groupby(columns).size().reset_index(name='Duplicate Count')
        return duplicates, duplicate_counts
    return pd.DataFrame(), pd.DataFrame()


def convert_column_types(df, column, target_type, date_format=None, verbose=False):
    """Convert column to specified type with error handling."""
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
                clean_series
                .astype(str)
                .str.strip()
                .str.replace(r'\s+', ' ', regex=True)
            )
            
        target_type = target_type.lower()
        
        if target_type in ['numeric', 'int', 'float']:
            clean_series = (
                clean_series
                .astype(str)
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
                date_formats = [
                    '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y',
                    '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
                    '%Y.%m.%d', '%d.%m.%Y', '%m.%d.%Y',
                    '%Y%m%d', '%d%m%Y', '%m%d%Y',
                    '%b %d, %Y', '%d %b %Y', '%B %d, %Y'
                ]
                
                for fmt in date_formats:
                    try:
                        result = pd.to_datetime(clean_series, format=fmt, errors='raise')
                        break
                    except:
                        continue
                else:
                    result = pd.to_datetime(clean_series, errors='coerce')
            
            if target_type == 'date':
                result = result.dt.date
            
        elif target_type == 'string':
            result = (
                clean_series
                .astype(str)
                .str.strip()
                .str.normalize('NFKC')
                .str.replace(r'[\x00-\x1F\x7F-\x9F]', '', regex=True)
                .str.replace(r'\s+', ' ', regex=True)
            )
            
        elif target_type == 'category':
            result = clean_series.astype('category')
            
        else:
            raise ValueError(f"Unsupported target type: {target_type}")
        
        if verbose:
            converted_count = len(result) - result.isna().sum()
            original_non_null = len(clean_series) - clean_series.isna().sum()
            
            print(f"Conversion report for column '{column}':")
            print(f"  Original dtype: {original_dtype}")
            print(f"  Target type:    {target_type}")
            print(f"  Sample before:  {original_sample}")
            print(f"  Sample after:   {result.head(3).tolist()}")
            print(f"  Success rate:   {converted_count}/{original_non_null} "
                 f"({converted_count/original_non_null:.1%}) values converted")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"⚠️ Conversion error for {column}: {str(e)}")
        return df[column]


def main():
    """Main application function."""
    st.set_page_config(
        page_title="Maritime Data Validator",
        page_icon="🚢",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        .stApp { background-color: #f0f8ff; }
        h1 { 
            color: #1a3e72;
            border-bottom: 2px solid #4682b4;
            padding-bottom: 10px; 
        }
        .css-1d391kg { background-color: #e6f2ff !important; }
        .stButton>button {
            background-color: #4682b4 !important;
            color: white !important;
            border-radius: 8px;
            padding: 8px 16px;
            border: none;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #36648b !important;
            transform: translateY(-2px);
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        .stDataFrame {
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .metric-card {
            background-color: white !important;
            border-left: 4px solid #4682b4 !important;
            border-radius: 8px !important;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #e6f2ff;
            border-radius: 8px 8px 0 0;
            padding: 8px 16px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4682b4 !important;
            color: white !important;
        }
        .streamlit-expanderHeader {
            background-color: #e6f2ff;
            border-radius: 8px;
            padding: 8px 16px;
        }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/128/2519/2519393.png", width=80)
    with col2:
        st.title("Maritime Data Validation Portal")
        st.caption("Validate crew data between Power BI reports and client systems")

    if "type_conversions_applied" not in st.session_state:
        st.session_state.type_conversions_applied = False
    if "df_report_converted" not in st.session_state:
        st.session_state.df_report_converted = None
    if "df_client_converted" not in st.session_state:
        st.session_state.df_client_converted = None
    if "type_mapping" not in st.session_state:
        st.session_state.type_mapping = []
    if "date_formats" not in st.session_state:
        st.session_state.date_formats = {}

    with st.sidebar:
        st.header("⚙️ Settings")
        st.divider()
        st.header("🔑 API Key Management")
        ai_provider = st.radio("AI Provider", ["Hugging Face", "OpenAI"], index=0)

        HF_API_KEY = OPENAI_API_KEY = hf_model = openai_model = ""

        if ai_provider == "Hugging Face":
            HF_API_KEY = st.text_input("Hugging Face API Key", type="password")
            hf_model = st.selectbox("Model", ["google/flan-t5-large", "HuggingFaceH4/zephyr-7b-beta"])
        else:
            OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")
            openai_model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"])

        if (ai_provider == "Hugging Face" and HF_API_KEY) or (ai_provider == "OpenAI" and OPENAI_API_KEY):
            st.success("✅ API configured successfully!")

        st.divider()
        st.info("This tool helps maritime operators validate crew data between reporting systems and client records.")

    st.subheader("📤 Data Upload")
    upload_col1, upload_col2 = st.columns(2)
    with upload_col1:
        with st.expander("Power BI Report", expanded=True):
            report_file = st.file_uploader("Upload Power BI Excel file", type=["xlsx"], label_visibility="collapsed")
            if report_file:
                st.success("Power BI data loaded successfully")

    with upload_col2:
        with st.expander("Client Records", expanded=True):
            client_file = st.file_uploader("Upload Client Excel file", type=["xlsx"], label_visibility="collapsed")
            if client_file:
                st.success("Client data loaded successfully")

    if report_file and client_file:
        if not st.session_state.type_conversions_applied:
            df_report = pd.read_excel(report_file)
            df_client = pd.read_excel(client_file)
            st.session_state.df_report_original = df_report.copy()
            st.session_state.df_client_original = df_client.copy()
        else:
            df_report = st.session_state.df_report_converted
            df_client = st.session_state.df_client_converted
        
        with st.expander("🔍 Preview Data", expanded=True):
            preview_tab1, preview_tab2 = st.tabs(["Power BI Data", "Client Data"])
            
            with preview_tab1:
                st.dataframe(df_report.head(), use_container_width=True)
                st.caption(f"Shape: {df_report.shape[0]} rows × {df_report.shape[1]} columns")
                
            with preview_tab2:
                st.dataframe(df_client.head(), use_container_width=True)
                st.caption(f"Shape: {df_client.shape[0]} rows × {df_client.shape[1]} columns")

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
            with st.expander("🔍 Selected Columns Preview", expanded=True):
                preview_tab1, preview_tab2 = st.tabs(["Power BI Selected Columns", "Client Selected Columns"])
                
                with preview_tab1:
                    st.dataframe(df_report[report_columns].head(10), use_container_width=True)
                    
                with preview_tab2:
                    st.dataframe(df_client[client_columns].head(10), use_container_width=True)
            
            with st.expander("📊 Current Data Types", expanded=True):
                type_tab1, type_tab2 = st.tabs(["Power BI Data Types", "Client Data Types"])
                
                with type_tab1:
                    st.dataframe(show_data_types(df_report[report_columns]), use_container_width=True)
                    
                with type_tab2:
                    st.dataframe(show_data_types(df_client[client_columns]), use_container_width=True)

            with st.expander("🔧 Data Type Matching", expanded=True):
                st.markdown("**Match column data types between datasets**")
                
                current_cols = set(zip(report_columns, client_columns))
                session_cols = {(item[0], item[1]) for item in st.session_state.type_mapping}
                
                if not st.session_state.type_mapping or current_cols != session_cols:
                    st.session_state.type_mapping = []
                    for r_col, c_col in zip(report_columns, client_columns):
                        r_type = str(df_report[r_col].dtype)
                        c_type = str(df_client[c_col].dtype)
                        
                        if 'date' in r_type.lower() or 'date' in c_type.lower() or 'datetime' in r_type.lower() or 'datetime' in c_type.lower():
                            target_type = 'datetime'
                            type_options = ['datetime', 'date', 'string']
                        elif 'int' in r_type or 'float' in r_type or 'int' in c_type or 'float' in c_type:
                            target_type = 'numeric'
                            type_options = ['numeric', 'int', 'float', 'string']
                        else:
                            target_type = 'string'
                            type_options = ['string', 'category']
                        
                        st.session_state.type_mapping.append((r_col, c_col, target_type, type_options))
                
                conversion_needed = False
                for i, (r_col, c_col, default_type, type_options) in enumerate(st.session_state.type_mapping):
                    r_type = str(df_report[r_col].dtype)
                    c_type = str(df_client[c_col].dtype)
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{r_col}** ({r_type}) ↔ **{c_col}** ({c_type})")
                    
                    with col2:
                        selected_type = st.selectbox(
                            f"Convert to:",
                            type_options,
                            index=type_options.index(default_type) if default_type in type_options else 0,
                            key=f"type_conv_{i}",
                            label_visibility="collapsed"
                        )
                        
                        st.session_state.type_mapping[i] = (r_col, c_col, selected_type, type_options)
                        
                        if r_type != selected_type or c_type != selected_type:
                            conversion_needed = True
                    
                    if selected_type in ['date', 'datetime']:
                        date_format_map = {
                            "DD-MM-YYYY": "%d-%m-%Y",
                            "MM-DD-YYYY": "%m-%d-%Y", 
                            "YYYY-MM-DD": "%Y-%m-%d",
                            "Auto-detect": None
                        }
                        
                        selected_format = st.selectbox(
                            f"Date format for {r_col} ↔ {c_col}:",
                            options=list(date_format_map.keys()),
                            key=f"date_fmt_{r_col}_{c_col}",
                            help="Select the date format or use auto-detect",
                            index=3
                        )
                        
                        st.session_state.date_formats[(r_col, c_col)] = date_format_map[selected_format]
                    
                    st.divider()
                
                if conversion_needed or not st.session_state.type_conversions_applied:
                    if st.button("Apply Type Conversions", help="Convert columns to selected types"):
                        df_report_converted = st.session_state.df_report_original.copy()
                        df_client_converted = st.session_state.df_client_original.copy()
                        
                        for r_col, c_col, target_type, _ in st.session_state.type_mapping:
                            date_format = st.session_state.date_formats.get((r_col, c_col), None)
                            df_report_converted[r_col] = convert_column_types(df_report_converted, r_col, target_type, date_format)
                            df_client_converted[c_col] = convert_column_types(df_client_converted, c_col, target_type, date_format)
                        
                        st.session_state.df_report_converted = df_report_converted
                        st.session_state.df_client_converted = df_client_converted
                        st.session_state.type_conversions_applied = True
                        
                        df_report = df_report_converted
                        df_client = df_client_converted
                        
                        st.success("Data type conversions applied successfully!")
                        st.rerun()
                else:
                    st.success("Type conversions have been applied! ✅")
                    
                # Move this section OUTSIDE the "Data Type Matching" expander
                if st.session_state.type_conversions_applied:
                    st.subheader("📊 Validation Method")
                    
                    join_type = st.selectbox(
                        "Comparison Method",
                        ["Left Join (Find mismatches)", 
                        "Inner Join (Find matches)",
                        "Outer Join (Find all differences)"],  
                        help="Select how to compare the datasets"
                    )
                    
                    st.info("""
                    **Comparison Methods:**
                    - **Left Join**: Find records in Power BI that aren't in Client data
                    - **Inner Join**: Show only matching records between both systems
                    - **Outer Join**: Show all differences (missing in either system)
                    """)

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
             
            if st.session_state.type_conversions_applied and 'join_type' in locals():
                st.subheader("📊 Validation Results")
                
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
                        st.metric("Mismatches Found", len(mismatches) if mismatches is not None else 0, 
                                help="Rows with discrepancies between datasets")
                    
                    if mismatches is not None and len(mismatches) > 0:
                        st.warning(f"⚠️ Found {len(mismatches)} mismatches")
                        tab_mismatch, tab_stats = st.tabs(["Unmatched Records", "Statistics"])

                        with tab_mismatch:
                            if join_type == "Outer Join (Find all differences)":
                                st.markdown("**Records only in Power BI Report**")
                                st.dataframe(report_only, use_container_width=True)
                                
                                st.markdown("**Records only in Client Data**")
                                st.dataframe(client_only, use_container_width=True)
                            else:
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
                            if join_type == "Outer Join (Find all differences)":
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

                    st.subheader("🤖 AI-Powered Insights")
                    if st.toggle("Enable Advanced Analysis", help="Get AI-powered insights about your data"):
                        with st.form("ai_analysis_form"):
                            user_input = st.text_area(
                                "Ask about the data:",
                                "What could be causing these mismatches in maritime crew data?",
                                help="Ask any question about your data analysis"
                            )
                            
                            submitted = st.form_submit_button("Analyze", 
                                help="Get AI analysis of your data discrepancies")
                            
                            if submitted and ((ai_provider == "Hugging Face" and HF_API_KEY) or 
                                        (ai_provider == "OpenAI" and OPENAI_API_KEY)):
                                prompt = f"""
You are a maritime operations analyst reviewing crew data discrepancies between a Power BI report and client records.

Key Findings:
- Comparing columns {report_columns} (Power BI) with {client_columns} (Client)
- Using {join_type} method
- Found {len(matches) if join_type == "Inner Join (Find matches)" else len(mismatches) if mismatches is not None else 0} {'matches' if join_type == "Inner Join (Find matches)" else 'mismatches'}
- Data types: {show_data_types(df_report[report_columns]).to_dict()}
- {'Duplicate rows detected in Power BI data' if not report_duplicates.empty else 'No duplicates in Power BI data'}
- {'Duplicate rows detected in Client data' if not client_duplicates.empty else 'No duplicates in Client data'}

User Question: {user_input}

Provide a technical analysis considering:
1. Data type compatibility between systems
2. Common maritime data issues
3. System integration challenges
4. Impact of duplicate records
5. Practical resolution steps

Structure your response with clear headings and bullet points.
"""
                                with st.spinner("🔍 Analyzing with maritime domain expertise..."):
                                    api_key = HF_API_KEY if ai_provider == "Hugging Face" else OPENAI_API_KEY
                                    model = hf_model if ai_provider == "Hugging Face" else openai_model
                                    ai_response = get_ai_response(prompt, ai_provider, model, api_key)
                                    
                                    st.markdown("### AI Analysis Results")
                                    st.markdown("---")
                                    st.markdown(ai_response)
                                    st.markdown("---")
                                    
                            elif submitted:
                                st.error("Please configure your API key in the sidebar first")


if __name__ == "__main__":
    main()

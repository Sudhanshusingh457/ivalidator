import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards

# --------------- Helper Functions ----------------

def get_ai_response(prompt, provider, model, api_key):
    if provider == "Hugging Face":
        return call_huggingface_api(prompt, model, api_key)
    else:
        return call_openai_api(prompt, model, api_key)

def call_huggingface_api(prompt, model, api_key):
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": 512}}
        )
        if response.status_code == 200:
            return response.json()[0]["generated_text"].replace(prompt, "").strip()
        else:
            return f"❌ Hugging Face Error ({response.status_code}): {response.text}"
    except Exception as e:
        return f"❌ Hugging Face Exception: {str(e)}"

def call_openai_api(prompt, model, api_key):
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
        else:
            return f"❌ OpenAI Error ({response.status_code}): {response.text}"
    except Exception as e:
        return f"❌ OpenAI Exception: {str(e)}"

def show_data_types(df):
    """Display the data types of all columns in a DataFrame"""
    type_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[col].dtype) for col in df.columns],
        "Null Values": df.isna().sum().values
    })
    return type_info

def find_duplicates(df, columns):
    """Find and return duplicate rows based on specified columns"""
    duplicates = df[df.duplicated(subset=columns, keep=False)]
    if not duplicates.empty:
        duplicates = duplicates.sort_values(by=columns)
        duplicate_counts = duplicates.groupby(columns).size().reset_index(name='Duplicate Count')
        return duplicates, duplicate_counts
    return pd.DataFrame(), pd.DataFrame()

# --------------- Streamlit UI ----------------

st.set_page_config(
    page_title="Maritime Data Validator",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern styling
st.markdown("""
<style>
    /* General App Background and Heading Styling */
    .stApp { background-color: #f8f9fa; }
    h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }

    /* Custom Styling for Elements */
    .css-1v0mbdj { border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .stButton>button { background-color: #3498db; color: white; border-radius: 8px; padding: 8px 16px; border: none; }
    .stDataFrame { border-radius: 8px; }

    /* Alert Styling */
    .stAlert .stSuccess { background-color: #d4edda; color: #155724; border-radius: 8px; }
    .stAlert .stWarning { background-color: #fff3cd; color: #856404; border-radius: 8px; }
    .stAlert .stError { background-color: #f8d7da; color: #721c24; border-radius: 8px; }

    /* Styling for Plotly Chart */
    .stPlotlyChart {
        border-radius: 10px;
        background: white;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)


col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/128/2519/2519393.png", width=80)
with col2:
    st.title("Maritime Data Validation Portal")
    st.caption("Validate crew data between Power BI reports and client systems")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Join type selection
    join_type = st.selectbox(
        "Comparison Method",
        ["Left Join (Find mismatches)", 
         "Inner Join (Find matches)",
         "Anti Join (Find non-matches)"],
        help="Select how to compare the datasets"
    )
    
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

# Upload area
st.subheader("Data Upload")
with st.expander("📤 Upload Files", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        report_file = st.file_uploader("Power BI Report", type=["xlsx"])
    with col2:
        client_file = st.file_uploader("Client Records", type=["xlsx"])

if report_file and client_file:
    df_report = pd.read_excel(report_file)
    df_client = pd.read_excel(client_file)
    st.success("✅ Files uploaded successfully.")

    with st.expander("🔍 Preview Data"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Power BI Data Preview**")
            st.dataframe(df_report.head())
        with col2:
            st.markdown("**Client Data Preview**")
            st.dataframe(df_client.head())

    st.subheader("🔎 Comparison Setup")
    
    # Multi-column selection
    col1, col2 = st.columns(2)
    with col1:
        report_columns = st.multiselect(
            "Select Power BI Columns", 
            df_report.columns,
            help="Select one or more columns to compare"
        )
    with col2:
        client_columns = st.multiselect(
            "Select Client Columns", 
            df_client.columns,
            help="Select corresponding columns from client data"
        )

    if report_columns and client_columns and len(report_columns) == len(client_columns):
        # Data type display
        with st.expander("📊 Data Types", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Power BI Data Types**")
                st.dataframe(show_data_types(df_report[report_columns]))
            with col2:
                st.markdown("**Client Data Types**")
                st.dataframe(show_data_types(df_client[client_columns]))

        # Duplicate detection
        with st.expander("🔎 Duplicate Detection", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Power BI Duplicates**")
                report_duplicates, report_dup_counts = find_duplicates(df_report, report_columns)
                if not report_duplicates.empty:
                    st.warning(f"Found {len(report_duplicates)} duplicate rows")
                    st.dataframe(report_dup_counts)
                    st.download_button(
                        "📥 Download Power BI Duplicates",
                        report_duplicates.to_csv(index=False).encode('utf-8'),
                        "powerbi_duplicates.csv",
                        "text/csv"
                    )
                else:
                    st.success("No duplicates found")
            
            with col2:
                st.markdown("**Client Data Duplicates**")
                client_duplicates, client_dup_counts = find_duplicates(df_client, client_columns)
                if not client_duplicates.empty:
                    st.warning(f"Found {len(client_duplicates)} duplicate rows")
                    st.dataframe(client_dup_counts)
                    st.download_button(
                        "📥 Download Client Duplicates",
                        client_duplicates.to_csv(index=False).encode('utf-8'),
                        "client_duplicates.csv",
                        "text/csv"
                    )
                else:
                    st.success("No duplicates found")

        # Perform the join based on selected method
        if join_type == "Left Join (Find mismatches)":
            merged = df_report.merge(df_client, left_on=report_columns, right_on=client_columns, how='left', indicator=True)
            mismatches = merged[merged['_merge'] == 'left_only'][report_columns]
            matches = None
        elif join_type == "Inner Join (Find matches)":
            merged = df_report.merge(df_client, left_on=report_columns, right_on=client_columns, how='inner')
            matches = merged[report_columns + client_columns]
            mismatches = None
        else:  # Anti Join
            merged = df_report.merge(df_client, left_on=report_columns, right_on=client_columns, how='left', indicator=True)
            mismatches = merged[merged['_merge'] == 'left_only'][report_columns]
            matches = None

        st.subheader("📊 Validation Results")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Report Rows", len(df_report))
        with col2: st.metric("Total Client Rows", len(df_client))

        if join_type == "Inner Join (Find matches)":
            with col3: st.metric("Matching Rows Found", len(matches))
            st.success(f"✅ Found {len(matches)} matching records")
            with st.expander("🔍 View Matching Records", expanded=True):
                st.dataframe(matches)
            
            csv = matches.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Matching Records",
                csv,
                "matching_records.csv",
                "text/csv",
                key='download-matches'
            )
        else:
            with col3: st.metric("Mismatches Found", len(mismatches) if mismatches is not None else 0)
            if mismatches is not None and len(mismatches) > 0:
                st.warning(f"⚠️ Found {len(mismatches)} mismatches")
                tab_mismatch, tab_stats = st.tabs(["Mismatch Details", "Statistics"])

                with tab_mismatch:
                    st.dataframe(mismatches)
                    
                    csv = mismatches.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Mismatch Report",
                        csv,
                        "mismatch_report.csv",
                        "text/csv",
                        key='download-csv'
                    )

                with tab_stats:
                    if join_type == "Left Join (Find mismatches)":
                        match_pct = 100 * (len(df_report) - len(mismatches)) / len(df_report)
                        st.markdown(f"- **Match %**: {match_pct:.1f}%\n- **Mismatch %**: {100-match_pct:.1f}%")
                    else:
                        st.markdown(f"- **Total Non-Matches**: {len(mismatches)}")
                        
                    pie_data = pd.DataFrame({
                        "Status": ["Matching", "Mismatched"],
                        "Count": [len(df_report) - len(mismatches), len(mismatches)]
                    })

                    st.plotly_chart(
                        px.pie(
                            pie_data,
                            values='Count',
                            names='Status',
                            title='Match/Mismatch Distribution',
                            color='Status',
                            color_discrete_map={'Matching':'green','Mismatched':'red'}
                        ),
                        use_container_width=True
                    )
            else:
                st.success("✅ Perfect match! All records align.")

        st.subheader("🤖 AI-Powered Insights")
        if st.toggle("Enable Advanced Analysis"):
            with st.form("ai_analysis_form"):
                user_input = st.text_area(
                    "Ask about the data:",
                    "What could be causing these mismatches in maritime crew data?"
                )
                submitted = st.form_submit_button("Analyze")

                if submitted:
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
                        st.markdown(f"### AI Response:\n{ai_response}")

st.divider()

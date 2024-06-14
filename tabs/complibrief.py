import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def display():
    st.header("ComplianceBrief")

    # Example news section
    # st.write("**Today's Top Compliance Update:**")
    # st.write("1. New data privacy regulations introduced for 2024.")
    # st.write("2. Financial institutions face stricter AML requirements.")

    # Simulated data
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    files_uploaded = np.random.randint(20, 100, size=12)
    files_checked = np.random.randint(10, 50, size=12)
    approval_rate = np.random.uniform(70, 100, size=12)

    # DataFrame for bar charts
    data = pd.DataFrame({
        "Month": months,
        "Files Uploaded": files_uploaded,
        "Files Checked": files_checked,
        "Approval Rate (%)": approval_rate
    })


    #######################
    # Dashboard Main Panel
    col = st.columns((1.5, 4.5, 2), gap='medium')

    with col[0]:
        st.markdown('#### File Checked')
        st.metric(label='May-2024', value=100, delta=20)

        
        st.markdown('#### Approval Status')


        migrations_col = st.columns((0.2, 1, 0.2))
        with migrations_col[1]:
            # st.write("2024-May")
            # st.subheader("This Month's Approval Status")
            approval_status = ["Approved", "Partially Approved", "Not Approved"]
            approval_counts = np.random.randint(10, 50, size=3)
            approval_df = pd.DataFrame({
                "Status": approval_status,
                "Count": approval_counts
            })
            fig = px.pie(approval_df, values='Count', names='Status'
                        #  , title="Approval Status Distribution"
                         )
            fig.update_layout(
                autosize=False,
                width=250,
                height=280,
                margin=dict(l=0, r=0),
            )
            st.plotly_chart(fig)

    with col[1]:
        st.markdown('#### Files Checked by Month with Approval Rate')
        
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=data["Month"], 
                y=data["Files Checked"], 
                name="Files Checked", 
                marker_color='DodgerBlue'
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data["Month"], 
                y=data["Approval Rate (%)"], 
                name="Approval Rate (%)", 
                yaxis="y2", 
                mode="lines+markers",
                line=dict(color='orange')
            )
        )

        # Create axis objects
        fig.update_layout(
            xaxis=dict(domain=[0.1, 0.9]),
            yaxis=dict(title="Files Checked", titlefont=dict(color="DodgerBlue"), tickfont=dict(color="DodgerBlue")),
            yaxis2=dict(title="Approval Rate (%)", titlefont=dict(color="orange"), tickfont=dict(color="orange"), anchor="x", overlaying="y", side="right"),
            legend=dict(x=0.1, y=1.1, orientation="h"),
            margin=dict(l=30, r=30, t=0, b=0),
            width=680
        )

        st.plotly_chart(fig)
        

    with col[2]:
        st.markdown('#### Details')
        st.dataframe(data,
                     column_order=("Month", "Files Checked"),
                     hide_index=True,
                     width=None,
                    column_config={
                    "Month": st.column_config.TextColumn(
                        "Month",
                    ),
                    "Files Checked": st.column_config.ProgressColumn(
                        "Files Checked",
                        format="%f",
                        min_value=0,
                        max_value=max(data['Files Checked']),
                    )}
        )
        
        with st.expander('About', expanded=True):
            st.write('''
                - ComplianceBrief provides you with a daily brief of the most important compliance news.
                - Data Souce: [Compliance](https://www.some.link).
                - :orange[**New**]: Data privacy regulations updated for 2024.
                - :orange[**New**]: Financial institutions face stricter AML requirements.
                     
                ''')

 
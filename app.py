"""Streamlit web application for the Tax Document Processing System."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from src.config import Config
from src.data_loader import DataLoader
from src.document_processor import DocumentProcessor
from src.pattern_analyzer import PatternAnalyzer
from src.similarity_search import SimilaritySearchEngine
from src.tip_generator import TipGenerator

# Page configuration
st.set_page_config(
    page_title="Tax Optimization Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .tip-card {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ff7f0e;
    }
    .high-priority {
        border-left-color: #d62728 !important;
    }
    .medium-priority {
        border-left-color: #ff7f0e !important;
    }
    .low-priority {
        border-left-color: #2ca02c !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache all data."""
    try:
        config = Config()
        data_loader = DataLoader(config)
        
        # Load CSV data
        transactions_df, users_df, tax_filings_df = data_loader.load_csv_data()
        
        # Load processed results if available
        results_dir = Path("results")
        processed_data = {}
        
        if results_dir.exists():
            for file_name in ["patterns.json", "all_tips.json", "receipt_data.json", "payslip_data.json"]:
                file_path = results_dir / file_name
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        processed_data[file_name.replace('.json', '')] = json.load(f)
        
        return {
            'transactions': transactions_df,
            'users': users_df,
            'tax_filings': tax_filings_df,
            'processed': processed_data
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    """Main Streamlit application."""
    st.markdown('<h1 class="main-header">💰 Tax Optimization Assistant</h1>', unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    if data is None:
        st.error("Failed to load data. Please check your configuration and run the main processing script first.")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "User Analysis", "Pattern Discovery", "Document Processing", "Tax Tips", "Similarity Search"]
    )
    
    if page == "Overview":
        show_overview(data)
    elif page == "User Analysis":
        show_user_analysis(data)
    elif page == "Pattern Discovery":
        show_pattern_discovery(data)
    elif page == "Document Processing":
        show_document_processing(data)
    elif page == "Tax Tips":
        show_tax_tips(data)
    elif page == "Similarity Search":
        show_similarity_search(data)

def show_overview(data):
    """Show system overview and key metrics."""
    st.header("📊 System Overview")
    
    transactions_df = data['transactions']
    users_df = data['users']
    tax_filings_df = data['tax_filings']
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Users", len(users_df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Transactions", len(transactions_df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_amount = transactions_df['amount'].sum()
        st.metric("Total Amount", f"€{total_amount:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_deduction = tax_filings_df['total_deductions'].mean()
        st.metric("Avg Deductions", f"€{avg_deduction:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Data overview charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Spending by Category")
        category_spending = transactions_df.groupby('category')['amount'].sum().sort_values(ascending=True)
        fig = px.bar(
            x=category_spending.values,
            y=category_spending.index,
            orientation='h',
            title="Total Spending by Category"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("User Demographics")
        occupation_counts = users_df['occupation_category'].value_counts()
        fig = px.pie(
            values=occupation_counts.values,
            names=occupation_counts.index,
            title="Users by Occupation"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Monthly spending trends
    st.subheader("Monthly Spending Trends")
    transactions_df['month'] = pd.to_datetime(transactions_df['transaction_date']).dt.month
    monthly_spending = transactions_df.groupby('month')['amount'].sum()
    
    fig = px.line(
        x=monthly_spending.index,
        y=monthly_spending.values,
        title="Monthly Spending Pattern",
        markers=True
    )
    fig.update_xaxes(title="Month")
    fig.update_yaxes(title="Amount (€)")
    st.plotly_chart(fig, use_container_width=True)

def show_user_analysis(data):
    """Show detailed user analysis."""
    st.header("👤 User Analysis")
    
    users_df = data['users']
    transactions_df = data['transactions']
    tax_filings_df = data['tax_filings']
    
    # User selection
    selected_user = st.selectbox("Select a user", users_df['user_id'].tolist())
    
    if selected_user:
        user_info = users_df[users_df['user_id'] == selected_user].iloc[0]
        user_transactions = transactions_df[transactions_df['user_id'] == selected_user]
        user_tax_data = tax_filings_df[tax_filings_df['user_id'] == selected_user]
        
        # User profile
        st.subheader("User Profile")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.info(f"**Occupation:** {user_info['occupation_category']}")
        with col2:
            st.info(f"**Age:** {user_info['age_range']}")
        with col3:
            st.info(f"**Family Status:** {user_info['family_status']}")
        with col4:
            st.info(f"**Region:** {user_info['region']}")
        
        # Transaction analysis
        if not user_transactions.empty:
            st.subheader("Transaction Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Category breakdown
                category_spending = user_transactions.groupby('category')['amount'].sum()
                fig = px.pie(
                    values=category_spending.values,
                    names=category_spending.index,
                    title=f"Spending Categories for {selected_user}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Monthly spending
                user_transactions['month'] = pd.to_datetime(user_transactions['transaction_date']).dt.month
                monthly = user_transactions.groupby('month')['amount'].sum()
                fig = px.bar(
                    x=monthly.index,
                    y=monthly.values,
                    title="Monthly Spending Pattern"
                )
                fig.update_xaxes(title="Month")
                fig.update_yaxes(title="Amount (€)")
                st.plotly_chart(fig, use_container_width=True)
            
            # Transaction details
            st.subheader("Recent Transactions")
            st.dataframe(
                user_transactions.sort_values('transaction_date', ascending=False).head(10),
                use_container_width=True
            )
        
        # Tax analysis
        if not user_tax_data.empty:
            st.subheader("Tax Analysis")
            tax_info = user_tax_data.iloc[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Income", f"€{tax_info['total_income']:,.2f}")
            with col2:
                st.metric("Total Deductions", f"€{tax_info['total_deductions']:,.2f}")
            with col3:
                deduction_rate = (tax_info['total_deductions'] / tax_info['total_income']) * 100
                st.metric("Deduction Rate", f"{deduction_rate:.1f}%")

def show_pattern_discovery(data):
    """Show pattern discovery results."""
    st.header("🔍 Pattern Discovery")
    
    patterns = data['processed'].get('patterns', {})
    
    if not patterns:
        st.warning("No pattern analysis data found. Please run the main processing script first.")
        return
    
    # Pattern overview
    st.subheader("Discovered Patterns")
    
    # Transaction patterns
    if 'transaction_patterns' in patterns:
        st.subheader("Transaction Patterns")
        
        transaction_patterns = patterns['transaction_patterns']
        
        if 'category_statistics' in transaction_patterns:
            st.write("**Category Statistics:**")
            category_stats = transaction_patterns['category_statistics']
            
            # Convert to DataFrame for display
            stats_data = []
            for category, stats in category_stats.items():
                if isinstance(stats, dict) and 'amount' in stats:
                    amount_stats = stats['amount']
                    user_stats = stats.get('user_id', {})
                    stats_data.append({
                        'Category': category,
                        'Total Amount': f"€{amount_stats.get('sum', 0):,.2f}",
                        'Avg Amount': f"€{amount_stats.get('mean', 0):,.2f}",
                        'Transaction Count': amount_stats.get('count', 0),
                        'Unique Users': user_stats.get('nunique', 0)
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
    
    # Demographic patterns
    if 'demographic_patterns' in patterns:
        st.subheader("Demographic Patterns")
        
        demographic_patterns = patterns['demographic_patterns']
        
        if 'spending_by_occupation' in demographic_patterns:
            st.write("**Spending by Occupation:**")
            occupation_data = demographic_patterns['spending_by_occupation']
            
            # Create visualization
            occupations = list(occupation_data.keys())
            amounts = [occupation_data[occ].get('sum', 0) for occ in occupations]
            
            fig = px.bar(
                x=occupations,
                y=amounts,
                title="Total Spending by Occupation Category"
            )
            fig.update_xaxes(title="Occupation", tickangle=45)
            fig.update_yaxes(title="Amount (€)")
            st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal patterns
    if 'seasonal_patterns' in patterns:
        st.subheader("Seasonal Patterns")
        
        seasonal_patterns = patterns['seasonal_patterns']
        
        if 'monthly_category_spending' in seasonal_patterns:
            monthly_data = seasonal_patterns['monthly_category_spending']
            monthly_df = pd.DataFrame(monthly_data)
            
            if not monthly_df.empty:
                # Pivot for heatmap
                pivot_data = monthly_df.pivot(index='category', columns='month', values='amount')
                
                fig = px.imshow(
                    pivot_data.values,
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    title="Spending Heatmap by Category and Month",
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Clustering results
    if 'clustering_patterns' in patterns:
        st.subheader("User Clustering Analysis")
        
        clustering_patterns = patterns['clustering_patterns']
        
        if 'cluster_analysis' in clustering_patterns:
            cluster_analysis = clustering_patterns['cluster_analysis']
            
            cluster_data = []
            for cluster_id, info in cluster_analysis.items():
                cluster_data.append({
                    'Cluster': cluster_id,
                    'Size': info.get('size', 0),
                    'Avg Spending': f"€{info.get('avg_total_spending', 0):,.2f}",
                    'Avg Deduction Rate': f"{info.get('avg_deduction_rate', 0):.1f}%",
                    'Dominant Occupation': info.get('dominant_occupation', 'Unknown')
                })
            
            if cluster_data:
                cluster_df = pd.DataFrame(cluster_data)
                st.dataframe(cluster_df, use_container_width=True)

def show_document_processing(data):
    """Show document processing results."""
    st.header("📄 Document Processing")
    
    receipt_data = data['processed'].get('receipt_data', [])
    payslip_data = data['processed'].get('payslip_data', [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Receipt Processing")
        if receipt_data:
            st.success(f"Processed {len(receipt_data)} receipts")
            
            # Show receipt summary
            total_amount = sum(r.get('total_amount', 0) for r in receipt_data if r.get('total_amount'))
            st.metric("Total Receipt Value", f"€{total_amount:.2f}")
            
            # Show receipt details
            for i, receipt in enumerate(receipt_data):
                with st.expander(f"Receipt {i+1}: {receipt.get('file_name', 'Unknown')}"):
                    if 'error' in receipt:
                        st.error(f"Processing error: {receipt['error']}")
                    else:
                        st.write(f"**Date:** {receipt.get('receipt_date', 'Not found')}")
                        st.write(f"**Amount:** €{receipt.get('total_amount', 0):.2f}")
                        st.write(f"**Vendor:** {receipt.get('vendor_name', 'Not found')}")
                        if receipt.get('line_items'):
                            st.write("**Line Items:**")
                            for item in receipt['line_items'][:3]:  # Show first 3 items
                                st.write(f"- {item}")
        else:
            st.warning("No receipt data found. Run document processing first.")
    
    with col2:
        st.subheader("Payslip Processing")
        if payslip_data:
            st.success(f"Processed {len(payslip_data)} payslips")
            
            # Show payslip summary
            avg_gross = sum(p.get('gross_pay', 0) for p in payslip_data) / len(payslip_data) if payslip_data else 0
            st.metric("Average Gross Pay", f"€{avg_gross:.2f}")
            
            # Show payslip details
            for i, payslip in enumerate(payslip_data):
                with st.expander(f"Payslip {i+1}: {payslip.get('file_name', 'Unknown')}"):
                    if 'error' in payslip:
                        st.error(f"Processing error: {payslip['error']}")
                    else:
                        st.write(f"**Employee:** {payslip.get('employee_name', 'Not found')}")
                        st.write(f"**Employer:** {payslip.get('employer_name', 'Not found')}")
                        st.write(f"**Gross Pay:** €{payslip.get('gross_pay', 0):.2f}")
                        st.write(f"**Net Pay:** €{payslip.get('net_pay', 0):.2f}")
                        st.write(f"**Position:** {payslip.get('position', 'Not found')}")
                        st.write(f"**Department:** {payslip.get('department', 'Not found')}")
        else:
            st.warning("No payslip data found. Run document processing first.")

def show_tax_tips(data):
    """Show personalized tax tips."""
    st.header("💡 Tax Optimization Tips")
    
    users_df = data['users']
    all_tips = data['processed'].get('all_tips', {})
    
    if not all_tips:
        st.warning("No tax tips found. Run the tip generation process first.")
        return
    
    # User selection
    selected_user = st.selectbox("Select a user for tips", list(all_tips.keys()))
    
    if selected_user and selected_user in all_tips:
        user_tips = all_tips[selected_user]
        
        if not user_tips:
            st.info("No tips available for this user.")
            return
        
        # Tips overview
        st.subheader(f"Tips for User {selected_user}")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        total_savings = sum(tip.get('potential_savings', 0) for tip in user_tips)
        high_priority = len([tip for tip in user_tips if tip.get('priority') == 'HIGH'])
        medium_priority = len([tip for tip in user_tips if tip.get('priority') == 'MEDIUM'])
        
        with col1:
            st.metric("Total Potential Savings", f"€{total_savings:.2f}")
        with col2:
            st.metric("High Priority Tips", high_priority)
        with col3:
            st.metric("Medium Priority Tips", medium_priority)
        
        # Tips by priority
        priority_order = ['HIGH', 'MEDIUM', 'LOW']
        priority_colors = {'HIGH': 'high-priority', 'MEDIUM': 'medium-priority', 'LOW': 'low-priority'}
        
        for priority in priority_order:
            priority_tips = [tip for tip in user_tips if tip.get('priority') == priority]
            
            if priority_tips:
                st.subheader(f"{priority.title()} Priority Tips")
                
                for tip in priority_tips:
                    priority_class = priority_colors.get(priority, '')
                    
                    st.markdown(f'<div class="tip-card {priority_class}">', unsafe_allow_html=True)
                    st.write(f"**{tip.get('title', 'Untitled Tip')}**")
                    st.write(tip.get('description', 'No description available'))
                    
                    if tip.get('potential_savings', 0) > 0:
                        st.write(f"💰 **Potential Savings:** €{tip['potential_savings']:.2f}")
                    
                    st.write(f"🎯 **Confidence:** {tip.get('confidence', 0)*100:.0f}%")
                    
                    if tip.get('action_items'):
                        st.write("**Action Items:**")
                        for action in tip['action_items']:
                            st.write(f"• {action}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Tips by category
        st.subheader("Tips by Category")
        
        tips_by_category = {}
        for tip in user_tips:
            category = tip.get('category', 'Other')
            if category not in tips_by_category:
                tips_by_category[category] = []
            tips_by_category[category].append(tip)
        
        for category, category_tips in tips_by_category.items():
            with st.expander(f"{category} ({len(category_tips)} tips)"):
                for tip in category_tips:
                    st.write(f"**{tip.get('title')}** - €{tip.get('potential_savings', 0):.2f} potential savings")

def show_similarity_search(data):
    """Show similarity search functionality."""
    st.header("🔗 Similarity Search")
    
    transactions_df = data['transactions']
    
    st.subheader("Find Similar Transactions")
    
    # Search options
    search_type = st.radio("Search Type", ["By Transaction", "By Text Query", "By Category"])
    
    if search_type == "By Transaction":
        # Select a transaction to find similar ones
        transaction_options = []
        for _, row in transactions_df.head(20).iterrows():  # Limit for performance
            option = f"{row['transaction_id']}: {row['description']} - €{row['amount']:.2f}"
            transaction_options.append((option, row))
        
        if transaction_options:
            selected_option = st.selectbox("Select a transaction", [opt[0] for opt in transaction_options])
            
            if selected_option:
                selected_transaction = next(opt[1] for opt in transaction_options if opt[0] == selected_option)
                
                st.write("**Selected Transaction:**")
                st.write(f"- **Description:** {selected_transaction['description']}")
                st.write(f"- **Category:** {selected_transaction['category']}")
                st.write(f"- **Amount:** €{selected_transaction['amount']:.2f}")
                st.write(f"- **Vendor:** {selected_transaction['vendor']}")
                
                # Find similar transactions (simple similarity based on category and amount)
                similar_transactions = transactions_df[
                    (transactions_df['category'] == selected_transaction['category']) &
                    (transactions_df['amount'].between(
                        selected_transaction['amount'] * 0.8,
                        selected_transaction['amount'] * 1.2
                    )) &
                    (transactions_df['transaction_id'] != selected_transaction['transaction_id'])
                ].head(10)
                
                if not similar_transactions.empty:
                    st.subheader("Similar Transactions")
                    st.dataframe(similar_transactions[['transaction_id', 'description', 'category', 'amount', 'vendor']])
                else:
                    st.info("No similar transactions found.")
    
    elif search_type == "By Text Query":
        query = st.text_input("Enter search query", placeholder="e.g., medical expenses, office supplies")
        
        if query:
            # Simple text search in descriptions
            matching_transactions = transactions_df[
                transactions_df['description'].str.contains(query, case=False, na=False) |
                transactions_df['category'].str.contains(query, case=False, na=False) |
                transactions_df['vendor'].str.contains(query, case=False, na=False)
            ].head(20)
            
            if not matching_transactions.empty:
                st.subheader(f"Transactions matching '{query}'")
                st.dataframe(matching_transactions[['transaction_id', 'description', 'category', 'amount', 'vendor']])
            else:
                st.info(f"No transactions found matching '{query}'.")
    
    elif search_type == "By Category":
        categories = transactions_df['category'].unique().tolist()
        selected_category = st.selectbox("Select category", categories)
        
        if selected_category:
            category_transactions = transactions_df[transactions_df['category'] == selected_category]
            
            st.subheader(f"Transactions in '{selected_category}' category")
            st.write(f"Found {len(category_transactions)} transactions")
            
            # Show statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Amount", f"€{category_transactions['amount'].sum():,.2f}")
            with col2:
                st.metric("Average Amount", f"€{category_transactions['amount'].mean():,.2f}")
            with col3:
                st.metric("Unique Vendors", category_transactions['vendor'].nunique())
            
            # Show transactions
            st.dataframe(category_transactions.head(20))

if __name__ == "__main__":
    main()

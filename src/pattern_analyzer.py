"""Pattern analysis module for identifying patterns in tax data."""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class PatternAnalyzer:
    """Pattern analyzer for tax data and transactions."""
    
    def __init__(self, config):
        """Initialize pattern analyzer with configuration."""
        self.config = config
        self.patterns = {}
        
    def analyze_patterns(self, transactions_df: pd.DataFrame, users_df: pd.DataFrame, 
                        tax_filings_df: pd.DataFrame, receipt_data: List[Dict], 
                        payslip_data: List[Dict]) -> Dict[str, Any]:
        """
        Comprehensive pattern analysis across all data sources.
        
        Args:
            transactions_df: Transaction data
            users_df: User demographic data
            tax_filings_df: Tax filing data
            receipt_data: Extracted receipt data
            payslip_data: Extracted payslip data
            
        Returns:
            Dictionary containing all discovered patterns
        """
        logger.info("Starting comprehensive pattern analysis...")
        
        patterns = {}
        
        # 1. Transaction patterns
        patterns['transaction_patterns'] = self._analyze_transaction_patterns(transactions_df)
        
        # 2. User demographic patterns
        patterns['demographic_patterns'] = self._analyze_demographic_patterns(
            users_df, transactions_df, tax_filings_df
        )
        
        # 3. Tax optimization patterns
        patterns['tax_optimization_patterns'] = self._analyze_tax_optimization_patterns(
            transactions_df, tax_filings_df, users_df
        )
        
        # 4. Seasonal patterns
        patterns['seasonal_patterns'] = self._analyze_seasonal_patterns(transactions_df)
        
        # 5. Document patterns
        patterns['document_patterns'] = self._analyze_document_patterns(receipt_data, payslip_data)
        
        # 6. Clustering patterns
        patterns['clustering_patterns'] = self._perform_user_clustering(
            transactions_df, users_df, tax_filings_df
        )
        
        # 7. Deduction opportunity patterns
        patterns['deduction_opportunities'] = self._identify_deduction_opportunities(
            transactions_df, users_df
        )
        
        # Generate visualizations
        self._create_visualizations(patterns, transactions_df, users_df, tax_filings_df)
        
        self.patterns = patterns
        logger.info("Pattern analysis completed successfully")
        
        return patterns
    
    def _analyze_transaction_patterns(self, transactions_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze transaction patterns."""
        patterns = {}
          # Category distribution
        category_stats = transactions_df.groupby('category').agg({
            'amount': ['count', 'sum', 'mean', 'std'],
            'user_id': 'nunique'
        }).round(2)
        
        # Flatten MultiIndex columns to avoid tuple keys in JSON
        category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns.values]
        patterns['category_statistics'] = category_stats.to_dict()
        
        # Monthly spending patterns
        transactions_df['month'] = transactions_df['transaction_date'].dt.month
        monthly_spending = transactions_df.groupby(['user_id', 'month'])['amount'].sum().reset_index()
        patterns['monthly_spending_variance'] = monthly_spending.groupby('user_id')['amount'].std().to_dict()
        
        # Top vendors by category
        top_vendors = transactions_df.groupby(['category', 'vendor'])['amount'].sum().reset_index()
        patterns['top_vendors_by_category'] = {}
        for category in transactions_df['category'].unique():
            category_vendors = top_vendors[top_vendors['category'] == category].nlargest(5, 'amount')
            patterns['top_vendors_by_category'][category] = category_vendors.to_dict('records')
        
        # Repeat transaction patterns
        repeat_transactions = transactions_df.groupby(['user_id', 'vendor', 'category']).size().reset_index(name='frequency')
        patterns['repeat_transaction_patterns'] = repeat_transactions[repeat_transactions['frequency'] > 3].to_dict('records')
        
        return patterns
    
    def _analyze_demographic_patterns(self, users_df: pd.DataFrame, 
                                    transactions_df: pd.DataFrame, 
                                    tax_filings_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze demographic patterns."""
        patterns = {}
        
        # Merge data for analysis
        user_transactions = transactions_df.merge(users_df, on='user_id')
        user_tax_data = tax_filings_df.merge(users_df, on='user_id')
          # Spending by occupation
        occupation_spending = user_transactions.groupby('occupation_category')['amount'].agg(['sum', 'mean', 'count']).round(2)
        occupation_spending.columns = ['amount_sum', 'amount_mean', 'amount_count']
        patterns['spending_by_occupation'] = occupation_spending.to_dict()
        
        # Deduction patterns by demographics
        deduction_patterns = user_tax_data.groupby(['occupation_category', 'family_status']).agg({
            'total_deductions': ['mean', 'std'],
            'refund_amount': ['mean', 'std']
        }).round(2)
        deduction_patterns.columns = ['_'.join(col).strip() for col in deduction_patterns.columns.values]
        patterns['deduction_by_demographics'] = deduction_patterns.to_dict()
        
        # Regional patterns
        regional_patterns = user_tax_data.groupby('region').agg({
            'total_income': 'mean',
            'total_deductions': 'mean',
            'refund_amount': 'mean'
        }).round(2)
        patterns['regional_patterns'] = regional_patterns.to_dict()
        
        # Age group patterns
        age_patterns = user_transactions.groupby(['age_range', 'category'])['amount'].sum().reset_index()
        patterns['spending_by_age_category'] = age_patterns.to_dict('records')
        
        return patterns
    
    def _analyze_tax_optimization_patterns(self, transactions_df: pd.DataFrame,
                                         tax_filings_df: pd.DataFrame,
                                         users_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze tax optimization patterns."""
        patterns = {}
        
        # Calculate deduction potential by category
        deductible_categories = [
            'Work Equipment', 'Professional Development', 'Medical', 
            'Charitable Donations', 'Transportation'
        ]
        
        # User deductible spending
        user_deductible = transactions_df[
            transactions_df['category'].isin(deductible_categories)
        ].groupby('user_id')['amount'].sum().reset_index()
        user_deductible.columns = ['user_id', 'potential_deductions']
        
        # Merge with actual deductions
        tax_comparison = tax_filings_df.merge(user_deductible, on='user_id', how='left')
        tax_comparison['potential_deductions'] = tax_comparison['potential_deductions'].fillna(0)
        tax_comparison['deduction_gap'] = tax_comparison['potential_deductions'] - tax_comparison['total_deductions']
        
        patterns['deduction_gap_analysis'] = {
            'users_with_gap': len(tax_comparison[tax_comparison['deduction_gap'] > 100]),
            'average_gap': tax_comparison['deduction_gap'].mean(),
            'max_gap': tax_comparison['deduction_gap'].max(),
            'total_missed_deductions': tax_comparison[tax_comparison['deduction_gap'] > 0]['deduction_gap'].sum()
        }
        
        # Efficiency ratios
        tax_comparison['deduction_efficiency'] = (
            tax_comparison['total_deductions'] / tax_comparison['total_income'] * 100
        ).round(2)
        patterns['deduction_efficiency'] = tax_comparison['deduction_efficiency'].describe().to_dict()
        
        return patterns
    
    def _analyze_seasonal_patterns(self, transactions_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal spending patterns."""
        patterns = {}
        
        # Add time features
        transactions_df['month'] = transactions_df['transaction_date'].dt.month
        transactions_df['quarter'] = transactions_df['transaction_date'].dt.quarter
        transactions_df['year'] = transactions_df['transaction_date'].dt.year
        
        # Monthly patterns by category
        monthly_category = transactions_df.groupby(['month', 'category'])['amount'].sum().reset_index()
        patterns['monthly_category_spending'] = monthly_category.to_dict('records')
          # Quarterly patterns
        quarterly_spending = transactions_df.groupby('quarter')['amount'].agg(['sum', 'mean', 'count']).round(2)
        quarterly_spending.columns = ['amount_sum', 'amount_mean', 'amount_count']
        patterns['quarterly_patterns'] = quarterly_spending.to_dict()
        
        # Year-end spending spikes
        december_spending = transactions_df[transactions_df['month'] == 12]
        november_spending = transactions_df[transactions_df['month'] == 11]
        
        patterns['year_end_patterns'] = {
            'december_charitable_donations': december_spending[
                december_spending['category'] == 'Charitable Donations'
            ]['amount'].sum(),
            'november_charitable_donations': november_spending[
                november_spending['category'] == 'Charitable Donations'
            ]['amount'].sum(),
            'medical_q4_spending': transactions_df[
                (transactions_df['quarter'] == 4) & 
                (transactions_df['category'] == 'Medical')
            ]['amount'].sum()
        }
        
        return patterns
    
    def _analyze_document_patterns(self, receipt_data: List[Dict], 
                                 payslip_data: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in processed documents."""
        patterns = {}
        
        # Receipt patterns
        if receipt_data:
            receipt_amounts = [r.get('total_amount', 0) for r in receipt_data if r.get('total_amount')]
            patterns['receipt_analysis'] = {
                'total_receipts': len(receipt_data),
                'receipts_with_amounts': len(receipt_amounts),
                'average_receipt_amount': np.mean(receipt_amounts) if receipt_amounts else 0,
                'total_receipt_value': sum(receipt_amounts)
            }
            
            # Vendor patterns in receipts
            vendors = [r.get('vendor_name') for r in receipt_data if r.get('vendor_name')]
            patterns['receipt_vendors'] = list(set(vendors))
        
        # Payslip patterns
        if payslip_data:
            gross_pays = [p.get('gross_pay', 0) for p in payslip_data if p.get('gross_pay')]
            net_pays = [p.get('net_pay', 0) for p in payslip_data if p.get('net_pay')]
            
            patterns['payslip_analysis'] = {
                'total_payslips': len(payslip_data),
                'average_gross_pay': np.mean(gross_pays) if gross_pays else 0,
                'average_net_pay': np.mean(net_pays) if net_pays else 0,
                'average_deduction_rate': (
                    (np.mean(gross_pays) - np.mean(net_pays)) / np.mean(gross_pays) * 100
                    if gross_pays and net_pays else 0
                )
            }
            
            # Position and department patterns
            positions = [p.get('position') for p in payslip_data if p.get('position')]
            departments = [p.get('department') for p in payslip_data if p.get('department')]
            patterns['employment_patterns'] = {
                'positions': list(set(positions)),
                'departments': list(set(departments))
            }
        
        return patterns
    
    def _perform_user_clustering(self, transactions_df: pd.DataFrame,
                               users_df: pd.DataFrame,
                               tax_filings_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform user clustering based on spending and tax patterns."""
        patterns = {}
        
        # Create user feature matrix
        user_features = self._create_user_feature_matrix(transactions_df, users_df, tax_filings_df)
        
        if len(user_features) < 2:
            patterns['clustering_note'] = "Insufficient data for clustering"
            return patterns
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(user_features.select_dtypes(include=[np.number]))
          # Perform K-means clustering
        n_clusters = min(4, len(user_features))  # Ensure we don't have more clusters than users
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        user_features['cluster'] = clusters
          # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(n_clusters):
            cluster_users = user_features[user_features['cluster'] == cluster_id]
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'size': len(cluster_users),
                'avg_total_spending': cluster_users['total_spending'].mean(),
                'avg_deduction_rate': cluster_users['deduction_rate'].mean(),
                'dominant_occupation': cluster_users['occupation_category'].mode().iloc[0] if not cluster_users['occupation_category'].mode().empty else 'Unknown'
            }
        
        patterns['cluster_analysis'] = cluster_analysis
        patterns['user_clusters'] = user_features[['user_id', 'cluster']].to_dict('records')
        
        return patterns
    
    def _create_user_feature_matrix(self, transactions_df: pd.DataFrame,
                                  users_df: pd.DataFrame,
                                  tax_filings_df: pd.DataFrame) -> pd.DataFrame:
        """Create feature matrix for user clustering."""
        # User spending patterns
        user_spending = transactions_df.groupby('user_id').agg({
            'amount': ['sum', 'mean', 'count'],
            'category': lambda x: len(x.unique())
        }).round(2)
        
        # Flatten MultiIndex columns to avoid tuple keys
        user_spending.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in user_spending.columns.values]
        user_spending.columns = ['total_spending', 'avg_transaction', 'transaction_count', 'category_diversity']
        user_spending = user_spending.reset_index()
        
        # Category spending percentages
        category_spending = transactions_df.groupby(['user_id', 'category'])['amount'].sum().unstack(fill_value=0)
        category_percentages = category_spending.div(category_spending.sum(axis=1), axis=0) * 100
        category_percentages = category_percentages.reset_index()
        
        # Merge with user data
        user_features = user_spending.merge(users_df, on='user_id')
        user_features = user_features.merge(tax_filings_df[['user_id', 'total_income', 'total_deductions']], on='user_id')
        
        # Add derived features
        user_features['deduction_rate'] = (user_features['total_deductions'] / user_features['total_income'] * 100).round(2)
        user_features['spending_rate'] = (user_features['total_spending'] / user_features['total_income'] * 100).round(2)
        
        return user_features
    
    def _identify_deduction_opportunities(self, transactions_df: pd.DataFrame,
                                        users_df: pd.DataFrame) -> Dict[str, Any]:
        """Identify deduction opportunities for users."""
        opportunities = {}
        
        # Define deductible categories and their typical deduction rates
        deductible_categories = {
            'Work Equipment': 1.0,
            'Professional Development': 1.0,
            'Medical': 0.8,  # Assuming some medical expenses are deductible
            'Charitable Donations': 1.0,
            'Transportation': 0.6  # Assuming business-related transportation
        }
        
        for user_id in users_df['user_id'].unique():
            user_transactions = transactions_df[transactions_df['user_id'] == user_id]
            user_opportunities = {}
            
            for category, deduction_rate in deductible_categories.items():
                category_spending = user_transactions[
                    user_transactions['category'] == category
                ]['amount'].sum()
                
                if category_spending > 0:
                    potential_deduction = category_spending * deduction_rate
                    user_opportunities[category] = {
                        'total_spending': category_spending,
                        'potential_deduction': potential_deduction,
                        'deduction_rate': deduction_rate
                    }
            
            if user_opportunities:
                opportunities[user_id] = user_opportunities
        
        return opportunities
    
    def _create_visualizations(self, patterns: Dict[str, Any],
                             transactions_df: pd.DataFrame,
                             users_df: pd.DataFrame,
                             tax_filings_df: pd.DataFrame):
        """Create visualization files for patterns."""
        viz_dir = self.config.results_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Category spending distribution
        fig, ax = plt.subplots(figsize=(12, 8))
        category_totals = transactions_df.groupby('category')['amount'].sum().sort_values(ascending=True)
        category_totals.plot(kind='barh', ax=ax)
        ax.set_title('Total Spending by Category')
        ax.set_xlabel('Amount (€)')
        plt.tight_layout()
        plt.savefig(viz_dir / 'category_spending.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. User demographic analysis
        user_data = transactions_df.merge(users_df, on='user_id')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Spending by occupation
        occupation_spending = user_data.groupby('occupation_category')['amount'].sum()
        occupation_spending.plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Spending by Occupation')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Spending by region
        region_spending = user_data.groupby('region')['amount'].sum()
        region_spending.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Spending by Region')
        
        # Age group spending
        age_spending = user_data.groupby('age_range')['amount'].sum()
        age_spending.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Spending by Age Group')
        
        # Family status spending
        family_spending = user_data.groupby('family_status')['amount'].sum()
        family_spending.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Spending by Family Status')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'demographic_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Seasonal patterns
        transactions_df['month'] = transactions_df['transaction_date'].dt.month
        monthly_spending = transactions_df.groupby('month')['amount'].sum()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        monthly_spending.plot(kind='line', marker='o', ax=ax)
        ax.set_title('Monthly Spending Patterns')
        ax.set_xlabel('Month')
        ax.set_ylabel('Amount (€)')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.tight_layout()
        plt.savefig(viz_dir / 'seasonal_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {viz_dir}")
        
        return patterns

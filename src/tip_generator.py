"""Tax tip generation module for creating personalized recommendations."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class TipGenerator:
    """Generator for personalized tax optimization tips."""
    
    def __init__(self, config):
        """Initialize tip generator with configuration."""
        self.config = config
        
        # Define tax-deductible categories and their rules
        self.deduction_rules = {
            'Work Equipment': {
                'deduction_rate': 1.0,
                'min_amount': 50,
                'max_annual': 800,
                'description': 'Work-related equipment and tools'
            },
            'Professional Development': {
                'deduction_rate': 1.0,
                'min_amount': 100,
                'max_annual': 4000,
                'description': 'Courses, certifications, and training'
            },
            'Medical': {
                'deduction_rate': 0.8,
                'min_amount': 100,
                'max_annual': None,
                'description': 'Medical expenses above insurance coverage'
            },
            'Charitable Donations': {
                'deduction_rate': 1.0,
                'min_amount': 25,
                'max_annual': None,
                'description': 'Donations to registered charities'
            },
            'Transportation': {
                'deduction_rate': 0.6,
                'min_amount': 200,
                'max_annual': 2000,
                'description': 'Business-related transportation costs'
            }
        }
        
        # Tax brackets (simplified German tax system)
        self.tax_brackets = [
            (0, 10908, 0.0),
            (10909, 15999, 0.14),
            (16000, 62809, 0.24),
            (62810, 277825, 0.42),
            (277826, float('inf'), 0.45)
        ]
    
    def generate_tips_for_user(self, user_id: str, transactions_df: pd.DataFrame,
                              users_df: pd.DataFrame, tax_filings_df: pd.DataFrame,
                              patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate personalized tax optimization tips for a specific user.
        
        Args:
            user_id: Target user ID
            transactions_df: Transaction data
            users_df: User demographic data
            tax_filings_df: Tax filing data
            patterns: Discovered patterns from analysis
            
        Returns:
            List of personalized tips with confidence scores
        """
        tips = []
        
        # Get user data
        user_info = users_df[users_df['user_id'] == user_id].iloc[0] if not users_df[users_df['user_id'] == user_id].empty else None
        user_transactions = transactions_df[transactions_df['user_id'] == user_id]
        user_tax_data = tax_filings_df[tax_filings_df['user_id'] == user_id].iloc[0] if not tax_filings_df[tax_filings_df['user_id'] == user_id].empty else None
        
        if user_info is None or user_transactions.empty:
            return tips
        
        # Generate different types of tips
        tips.extend(self._generate_deduction_tips(user_id, user_transactions, user_tax_data, user_info))
        tips.extend(self._generate_timing_tips(user_id, user_transactions, patterns))
        tips.extend(self._generate_category_optimization_tips(user_id, user_transactions, user_info, patterns))
        tips.extend(self._generate_similar_user_tips(user_id, user_info, patterns))
        tips.extend(self._generate_compliance_tips(user_id, user_transactions, user_tax_data))
        
        # Sort tips by potential impact and confidence
        tips.sort(key=lambda x: x['potential_savings'] * x['confidence'], reverse=True)
        
        # Add tip IDs and priorities
        for i, tip in enumerate(tips):
            tip['tip_id'] = f"TIP_{user_id}_{i+1:03d}"
            tip['priority'] = self._calculate_priority(tip)
        
        return tips[:10]  # Return top 10 tips
    
    def _generate_deduction_tips(self, user_id: str, transactions_df: pd.DataFrame,
                               tax_data: pd.Series, user_info: pd.Series) -> List[Dict[str, Any]]:
        """Generate tips related to missed deductions."""
        tips = []
        
        for category, rules in self.deduction_rules.items():
            category_transactions = transactions_df[transactions_df['category'] == category]
            
            if category_transactions.empty:
                continue
            
            total_spending = category_transactions['amount'].sum()
            potential_deduction = min(total_spending * rules['deduction_rate'], 
                                    rules['max_annual'] or float('inf'))
            
            if potential_deduction >= rules['min_amount']:
                current_deductions = tax_data['total_deductions'] if tax_data is not None else 0
                
                # Estimate if this deduction was likely missed
                estimated_missed = max(0, potential_deduction - (current_deductions * 0.2))  # Assume 20% of deductions are from this category
                
                if estimated_missed > 50:  # Only suggest if significant impact
                    tax_bracket = self._get_tax_bracket(tax_data['total_income'] if tax_data is not None else 50000)
                    potential_savings = estimated_missed * tax_bracket
                    
                    tip = {
                        'type': 'deduction_opportunity',
                        'category': category,
                        'title': f"Maximize {category} Deductions",
                        'description': f"You spent €{total_spending:.2f} on {rules['description'].lower()}. You could potentially deduct €{potential_deduction:.2f}, saving approximately €{potential_savings:.2f} in taxes.",
                        'action_items': [
                            f"Gather receipts for all {category.lower()} expenses",
                            f"Ensure expenses total at least €{rules['min_amount']} to qualify",
                            "Consult with a tax professional to confirm eligibility"
                        ],
                        'potential_savings': potential_savings,
                        'confidence': 0.8,
                        'evidence': {
                            'total_spending': total_spending,
                            'transaction_count': len(category_transactions),
                            'average_transaction': category_transactions['amount'].mean()
                        }                    }
                    tips.append(tip)
        
        return tips
    
    def _generate_timing_tips(self, user_id: str, transactions_df: pd.DataFrame,
                            patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tips related to transaction timing."""
        tips = []
        
        # Check for December spending patterns - use .copy() to avoid warning
        transactions_copy = transactions_df.copy()
        transactions_copy['month'] = transactions_copy['transaction_date'].dt.month
        december_transactions = transactions_copy[transactions_copy['month'] == 12]
        charitable_december = december_transactions[december_transactions['category'] == 'Charitable Donations']
        
        if not charitable_december.empty:
            avg_charitable = charitable_december['amount'].mean()
            
            tip = {
                'type': 'timing_optimization',
                'category': 'Charitable Donations',
                'title': 'Optimize Year-End Charitable Giving',
                'description': f"You donated €{charitable_december['amount'].sum():.2f} in December. Consider spreading donations throughout the year for better cash flow management while maintaining the same tax benefits.",
                'action_items': [
                    "Set up monthly charitable giving instead of lump sum",
                    "Consider automatic deductions to spread giving evenly",
                    "Track donations throughout the year for tax planning"
                ],
                'potential_savings': avg_charitable * 0.1,  # Cash flow benefit
                'confidence': 0.7,
                'evidence': {
                    'december_donations': charitable_december['amount'].sum(),
                    'donation_count': len(charitable_december)
                }
            }
            tips.append(tip)
        
        # Check for medical expense timing
        medical_transactions = transactions_df[transactions_df['category'] == 'Medical']
        if not medical_transactions.empty:
            medical_by_month = medical_transactions.groupby('month')['amount'].sum()
            medical_variance = medical_by_month.std()
            
            if medical_variance > 100:  # High variance suggests irregular timing
                tip = {
                    'type': 'timing_optimization',
                    'category': 'Medical',
                    'title': 'Time Medical Expenses Strategically',
                    'description': f"Your medical expenses vary significantly by month (std: €{medical_variance:.2f}). Consider timing elective procedures to maximize tax benefits.",
                    'action_items': [
                        "Schedule elective procedures in high-income years",
                        "Consider FSA or HSA if available",
                        "Track all medical expenses throughout the year"
                    ],
                    'potential_savings': medical_variance * 0.2,
                    'confidence': 0.6,
                    'evidence': {
                        'monthly_variance': medical_variance,
                        'total_medical': medical_transactions['amount'].sum()
                    }
                }
                tips.append(tip)
        
        return tips
    
    def _generate_category_optimization_tips(self, user_id: str, transactions_df: pd.DataFrame,
                                           user_info: pd.Series, patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tips for optimizing spending in specific categories."""
        tips = []
        
        # Compare user spending to similar users
        user_occupation = user_info['occupation_category']
        occupation_patterns = patterns.get('demographic_patterns', {}).get('spending_by_occupation', {})
        
        if user_occupation in occupation_patterns:
            user_total_spending = transactions_df['amount'].sum()
            avg_occupation_spending = occupation_patterns.get(user_occupation, {}).get('sum', user_total_spending)
            
            if isinstance(avg_occupation_spending, dict):
                avg_occupation_spending = list(avg_occupation_spending.values())[0] if avg_occupation_spending else user_total_spending
            
            # Check for under-spending in deductible categories
            for category in self.deduction_rules.keys():
                user_category_spending = transactions_df[transactions_df['category'] == category]['amount'].sum()
                
                # Estimate typical spending for this occupation in this category
                category_ratio = 0.1 if category == 'Professional Development' else 0.05
                expected_spending = avg_occupation_spending * category_ratio
                
                if user_category_spending < expected_spending * 0.5:  # Significantly under-spending
                    potential_deduction = expected_spending * self.deduction_rules[category]['deduction_rate']
                    tax_bracket = self._get_tax_bracket(50000)  # Default bracket
                    potential_savings = potential_deduction * tax_bracket
                    
                    tip = {
                        'type': 'category_optimization',
                        'category': category,
                        'title': f"Consider Increasing {category} Investments",
                        'description': f"Similar {user_occupation.lower()} professionals typically spend €{expected_spending:.2f} on {category.lower()}. You spent €{user_category_spending:.2f}. Increasing investments in this area could provide tax benefits.",
                        'action_items': [
                            f"Research {category.lower()} opportunities relevant to your profession",
                            "Set aside budget for tax-deductible expenses",
                            "Track all related expenses carefully"
                        ],
                        'potential_savings': potential_savings,
                        'confidence': 0.5,
                        'evidence': {
                            'user_spending': user_category_spending,
                            'peer_average': expected_spending,
                            'occupation': user_occupation
                        }
                    }
                    tips.append(tip)
        
        return tips
    
    def _generate_similar_user_tips(self, user_id: str, user_info: pd.Series,
                                  patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tips based on similar users' successful strategies."""
        tips = []
        
        # Get clustering patterns
        clustering_patterns = patterns.get('clustering_patterns', {})
        user_clusters = clustering_patterns.get('user_clusters', [])
        
        # Find user's cluster
        user_cluster = None
        for cluster_info in user_clusters:
            if cluster_info['user_id'] == user_id:
                user_cluster = cluster_info['cluster']
                break
        
        if user_cluster is not None:
            cluster_analysis = clustering_patterns.get('cluster_analysis', {})
            cluster_info = cluster_analysis.get(f'cluster_{user_cluster}', {})
            
            avg_deduction_rate = cluster_info.get('avg_deduction_rate', 0)
            
            if avg_deduction_rate > 5:  # If cluster has good deduction rates
                tip = {
                    'type': 'peer_learning',
                    'category': 'General',
                    'title': 'Learn from Similar Taxpayers',
                    'description': f"Users with similar profiles achieve an average deduction rate of {avg_deduction_rate:.1f}%. Consider reviewing your deduction strategy.",
                    'action_items': [
                        "Review all possible deduction categories",
                        "Consider consulting with a tax professional",
                        "Implement better expense tracking systems"
                    ],
                    'potential_savings': avg_deduction_rate * 10,  # Rough estimate
                    'confidence': 0.4,
                    'evidence': {
                        'cluster_id': user_cluster,
                        'cluster_size': cluster_info.get('size', 0),
                        'avg_deduction_rate': avg_deduction_rate
                    }
                }
                tips.append(tip)
        
        return tips
    
    def _generate_compliance_tips(self, user_id: str, transactions_df: pd.DataFrame,
                                tax_data: pd.Series) -> List[Dict[str, Any]]:
        """Generate compliance and documentation tips."""
        tips = []
        
        # Check for large transactions without proper documentation
        large_transactions = transactions_df[transactions_df['amount'] > 500]
        deductible_large = large_transactions[large_transactions['category'].isin(self.deduction_rules.keys())]
        
        if not deductible_large.empty:
            tip = {
                'type': 'compliance',
                'category': 'Documentation',
                'title': 'Ensure Proper Documentation for Large Expenses',
                'description': f"You have {len(deductible_large)} transactions over €500 in deductible categories. Ensure you have proper receipts and documentation.",
                'action_items': [
                    "Collect and organize receipts for all large deductible expenses",
                    "Consider digital receipt management tools",
                    "Maintain detailed records of business purpose for each expense"
                ],
                'potential_savings': 0,  # Risk mitigation rather than savings
                'confidence': 0.9,
                'evidence': {
                    'large_transaction_count': len(deductible_large),
                    'total_large_amount': deductible_large['amount'].sum()
                }
            }
            tips.append(tip)
        
        # Check for missing quarterly payments (for high earners)
        if tax_data is not None and tax_data['total_income'] > 60000:
            tip = {
                'type': 'compliance',
                'category': 'Tax Planning',
                'title': 'Consider Quarterly Tax Planning',
                'description': f"With an income of €{tax_data['total_income']:,.2f}, consider quarterly tax planning to avoid penalties and improve cash flow.",
                'action_items': [
                    "Calculate estimated quarterly tax payments",
                    "Set up automatic quarterly payments if beneficial",
                    "Review tax strategy quarterly with a professional"
                ],
                'potential_savings': tax_data['total_income'] * 0.01,  # Penalty avoidance
                'confidence': 0.7,
                'evidence': {
                    'annual_income': tax_data['total_income']
                }
            }
            tips.append(tip)
        
        return tips
    
    def _get_tax_bracket(self, income: float) -> float:
        """Get marginal tax rate for given income."""
        for min_income, max_income, rate in self.tax_brackets:
            if min_income <= income <= max_income:
                return rate
        return 0.45  # Highest bracket
    
    def _calculate_priority(self, tip: Dict[str, Any]) -> str:
        """Calculate priority level for a tip."""
        impact_score = tip['potential_savings'] * tip['confidence']
        
        if impact_score > 200:
            return 'HIGH'
        elif impact_score > 50:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def generate_tip_report(self, user_id: str, tips: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive tip report for a user."""
        if not tips:
            return {
                'user_id': user_id,
                'total_tips': 0,
                'total_potential_savings': 0,
                'summary': "No optimization opportunities identified at this time."
            }
        
        total_savings = sum(tip['potential_savings'] for tip in tips)
        high_priority = len([tip for tip in tips if tip.get('priority') == 'HIGH'])
        medium_priority = len([tip for tip in tips if tip.get('priority') == 'MEDIUM'])
        low_priority = len([tip for tip in tips if tip.get('priority') == 'LOW'])
        
        # Group tips by type
        tips_by_type = {}
        for tip in tips:
            tip_type = tip['type']
            if tip_type not in tips_by_type:
                tips_by_type[tip_type] = []
            tips_by_type[tip_type].append(tip)
        
        return {
            'user_id': user_id,
            'total_tips': len(tips),
            'total_potential_savings': total_savings,
            'priority_breakdown': {
                'high': high_priority,
                'medium': medium_priority,
                'low': low_priority
            },
            'tips_by_type': tips_by_type,
            'top_recommendations': tips[:3],
            'summary': f"Identified {len(tips)} optimization opportunities with potential savings of €{total_savings:.2f}. Focus on {high_priority} high-priority items first."
        }

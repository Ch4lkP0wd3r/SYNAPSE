"""
SYNAPSE Composite Judgment Framework
Categorized Risk Indicators for High-Precision Detection
"""

import numpy as np
import pandas as pd
import hashlib

class CompositeCriteriaEngine:
    def __init__(self, df):
        self.df = df
        self.criteria_results = {}
        
    def evaluate_all(self):
        """Evaluate all 10 categories with forensic narratives and tactics"""
        results = {}
        all_narrs = [[] for _ in range(len(self.df))]
        all_tactics = [[] for _ in range(len(self.df))]
        
        def merge_signals(score_tuple):
            score, narr_list, tactic_list = score_tuple
            for i in range(len(self.df)):
                all_narrs[i].extend(narr_list[i])
                all_tactics[i].extend(tactic_list[i])
            return score

        # 1. Transactional Behavior
        results['behavioral'] = merge_signals(self._judge_transactional_behavior())
        
        # 2. Geographic Risk
        results['geographic'] = merge_signals(self._judge_geographic_risk())
        
        # 3. Temporal Patterns
        results['temporal'] = merge_signals(self._judge_temporal_patterns())
        
        # 4. Entity Relationship Risk (Stubbed)
        results['relationship'] = merge_signals(self._judge_entity_relationships())

        # 5. Metadata Risk
        results['metadata'] = merge_signals(self._judge_metadata_risk())
        
        # 6. Financial Consistency
        results['consistency'] = merge_signals(self._judge_financial_consistency())

        # 7. Graph-Structural Anomalies (Stubbed)
        results['graph'] = merge_signals(self._judge_graph_anomalies())

        # 8. Cross-Data Intelligence (Stubbed)
        results['intelligence'] = merge_signals(self._judge_cross_data_intel())

        # 9. Statistical & Machine-Learning Indicators (Stubbed)
        results['ml_indicators'] = merge_signals(self._judge_ml_indicators())

        # 10. Human / Contextual Cues (Stubbed)
        results['human_context'] = merge_signals(self._judge_human_context())
        
        # Aggregation Logic
        results['narrative_strings'] = pd.Series([" | ".join(n) for n in all_narrs], index=self.df.index)
        results['tactic_strings'] = pd.Series([" | ".join(list(set(t))) for t in all_tactics], index=self.df.index)
        
        return results

    def _judge_transactional_behavior(self):
        """I. Transactional Behavior with Narratives"""
        df = self.df
        scores = pd.Series(0, index=df.index)
        narrs = [[] for _ in range(len(df))]
        tactics = [[] for _ in range(len(df))]
        
        # 1. Amount exceeds average by 10x
        if 'sender_avg_amount' in df.columns:
            mask = df['amount'] > df['sender_avg_amount'] * 10
            scores[mask] += 40
            for i in np.where(mask)[0]:
                narrs[i].append(f"Amount (${df.iloc[i]['amount']:,.2f}) is 10x higher than entity's historical average.")
                tactics[i].append("Large-Scale Fund Movement")
            
        # 2. Round figure transactions repeated often
        mask_round = (df['amount'] % 1000 == 0)
        scores[mask_round] += 10
        for i in np.where(mask_round)[0]:
            narrs[i].append("Transaction uses perfectly round figures ($1,000 increments).")
            tactics[i].append("Structuring / Smurfing")
            
        # 3. Same sender -> many receivers within one day (removed from this method's explicit scoring)
        # if 'sender_frequency' in df.columns:
        #     scores += (df['sender_frequency'] > 10) * 15
            
        return scores.clip(0, 100), narrs, tactics

    def _judge_geographic_risk(self):
        """II. Geographic Risk with Narratives"""
        df = self.df
        scores = pd.Series(0, index=df.index)
        narrs = [[] for _ in range(len(df))]
        tactics = [[] for _ in range(len(df))]
        
        high_risk = ['North Korea', 'Iran', 'Cayman Islands']
        if 'country' in df.columns:
            mask = df['country'].isin(high_risk)
            scores[mask] += 50
            for i in np.where(mask)[0]:
                narrs[i].append(f"Transaction involves high-risk jurisdiction: {df.iloc[i]['country']}.")
                tactics[i].append("Offshore Tax Haven Routing")
                
        if 'vpn_detected' in df.columns:
            mask = (df['vpn_detected'] == 1)
            scores[mask] += 20
            for i in np.where(mask)[0]:
                narrs[i].append("Connection routed through known VPN/Proxy service.")
                tactics[i].append("Identity Obfuscation")
                
        return scores.clip(0, 100), narrs, tactics

    def _judge_temporal_patterns(self):
        """III. Temporal Patterns with Narratives"""
        df = self.df
        scores = pd.Series(0, index=df.index)
        narrs = [[] for _ in range(len(df))]
        tactics = [[] for _ in range(len(df))]
        
        if 'hour' in df.columns:
            # Odd hours (2AM-5AM)
            mask = (df['hour'] >= 2) & (df['hour'] <= 5)
            scores[mask] += 30
            for i in np.where(mask)[0]:
                narrs[i].append(f"Transaction executed at {df.iloc[i]['hour']}:00 AM (anomalous timeframe).")
                tactics[i].append("Night-time Rapid Exfiltration")
            
        # 1. Bot-like automation (Periodic pattern matching) - Removed
        # if 'date' in df.columns:
        #     df['minute'] = df['date'].dt.minute
        #     bot_score = df.groupby(['sender_id', 'minute']).size().max()
        #     if bot_score > 3:
        #         scores += 25
                
        # 2. Burst of transactions followed by silence
        if 'is_burst' in df.columns:
            mask = (df['is_burst'] == 1)
            scores[mask] += 20
            for i in np.where(mask)[0]:
                narrs[i].append("Transaction is part of a high-speed burst sequence.")
                tactics[i].append("Cashing Out / Rapid Layering")
            
        return scores.clip(0, 100), narrs, tactics

    def _judge_entity_relationships(self):
        """IV. Entity Relationship Risk with Narratives"""
        df = self.df
        scores = pd.Series(0, index=df.index)
        narrs = [[] for _ in range(len(df))]
        tactics = [[] for _ in range(len(df))]
        
        # 1. Receiver is a high-frequency hub (centrality proxy)
        if 'receiver_frequency' in df.columns:
            mask = df['receiver_frequency'] > 20
            scores[mask] += 25
            for i in np.where(mask)[0]:
                narrs[i].append(f"Counterparty is a high-frequency transaction hub.")
                tactics[i].append("Concentration Risk")
                
        # 2. Transaction between entities with high individual risk (Stubbed logic based on freq interaction)
        if 'freq_amount_interaction' in df.columns:
            mask = df['freq_amount_interaction'] > df['freq_amount_interaction'].mean() * 3
            scores[mask] += 20
            for i in np.where(mask)[0]:
                narrs[i].append("High-intensity entity relationship identified.")
                tactics[i].append("Coordinated Asset Movement")

        return scores.clip(0, 100), narrs, tactics

    def _judge_metadata_risk(self):
        """V. Metadata Risk with Narratives"""
        df = self.df
        scores = pd.Series(0, index=df.index)
        narrs = [[] for _ in range(len(df))]
        tactics = [[] for _ in range(len(df))]
        
        if 'auth_failures' in df.columns:
            mask = (df['auth_failures'] >= 3)
            scores[mask] += 30
            for i in np.where(mask)[0]:
                narrs[i].append(f"Significant authentication failures ({df.iloc[i]['auth_failures']}) before success.")
                tactics[i].append("Account Takeover / Credential Stuffing")
            
        # if 'proxy_detected' in df.columns: # Removed
        #     scores += (df['proxy_detected'] == 1) * 35
            
        return scores.clip(0, 100), narrs, tactics

    def _judge_financial_consistency(self):
        """VI. Financial Consistency with Narratives"""
        df = self.df
        scores = pd.Series(0, index=df.index)
        narrs = [[] for _ in range(len(df))]
        tactics = [[] for _ in range(len(df))]
        
        if 'sender_income' in df.columns:
            # Monthly volume > 3x income
            # For simplicity, using single transaction > 0.5x annual income
            mask = df['amount'] > (df['sender_income'] * 0.5)
            scores[mask] += 40
            for i in np.where(mask)[0]:
                narrs[i].append(f"Amount exceeds 50% of annual declared income (${df.iloc[i]['sender_income']:,.2f}).")
                tactics[i].append("Unexplained Wealth Movement")
            
        # if 'sender_purpose' in df.columns and 'sender_purpose' == 'PERSONAL': # Removed
        #     scores += (df['amount'] > 50000) * 15 # Personal account high value
            
        return scores.clip(0, 100), narrs, tactics

    def _judge_graph_anomalies(self):
        """VII. Graph-Structural Anomalies"""
        df = self.df
        scores = pd.Series(0, index=df.index)
        narrs = [[] for _ in range(len(df))]
        tactics = [[] for _ in range(len(df))]
        
        # 1. Potential Shell Entity indicators (from preprocessing/features)
        if 'amount_to_max_ratio' in df.columns:
            mask = df['amount_to_max_ratio'] > 0.95 # Consistent max-out transactions
            scores[mask] += 30
            for i in np.where(mask)[0]:
                narrs[i].append("Account shows signs of 'Pass-Through' shell behavior.")
                tactics[i].append("Shell Company Layering")
                
        return scores.clip(0, 100), narrs, tactics

    def _judge_cross_data_intel(self):
        """VIII. Cross-Data Intelligence"""
        # Simulated watchlist match
        return pd.Series(0, index=self.df.index), [[] for _ in range(len(self.df))], [[] for _ in range(len(self.df))]

    def _judge_ml_indicators(self):
        """IX. Statistical & Machine-Learning Indicators"""
        df = self.df
        scores = pd.Series(0, index=df.index)
        narrs = [[] for _ in range(len(df))]
        tactics = [[] for _ in range(len(df))]
        
        # 1. Benford Deviation
        if 'benford_deviation' in df.columns:
            mask = df['benford_deviation'] > 0.8
            scores[mask] += 25
            for i in np.where(mask)[0]:
                narrs[i].append("Transaction amount violates Benford's Law distribution.")
                tactics[i].append("Data Manipulation Detection")
                
        # 2. Log-Amount Outliers
        if 'log_amount' in df.columns:
            mean_log = df['log_amount'].mean()
            std_log = df['log_amount'].std()
            mask = (df['log_amount'] > mean_log + 3 * std_log)
            scores[mask] += 30
            for i in np.where(mask)[0]:
                narrs[i].append("Statistical volume outlier identified.")
                tactics[i].append("Anomalous Value Transfer")
            
        return scores.clip(0, 100), narrs, tactics

    def _judge_human_context(self):
        """X. Human / Contextual Cues"""
        # Simulated refusal to provide docs or rapid closure
        return pd.Series(0, index=self.df.index), [[] for _ in range(len(self.df))], [[] for _ in range(len(self.df))]

def get_judgment_scoreboard(df):
    """
    Main entry point for generating the 10-category judgment scoreboard
    with full forensic context.
    """
    engine = CompositeCriteriaEngine(df)
    results = engine.evaluate_all()
    
    # Calculate overall Composite Score
    overall_score = pd.Series(0.0, index=df.index)
    categories = ['behavioral', 'geographic', 'temporal', 'relationship', 'metadata', 'consistency', 'graph', 'ml_indicators']
    for cat in categories:
        overall_score += results[cat] * (1.0 / len(categories))
        
    results['overall'] = overall_score
    return results

"""
Graph Builder Module for SYNAPSE
Creates transaction network graphs and detects suspicious clusters
"""

import networkx as nx
import pandas as pd
from collections import defaultdict

def build_transaction_graph(df):
    """
    Build a directed graph from transaction data
    
    Args:
        df: Transaction DataFrame
        
    Returns:
        nx.DiGraph: Transaction network graph
    """
    G = nx.DiGraph()
    
    # Add nodes and edges
    for idx, row in df.iterrows():
        sender = row['sender_id']
        receiver = row['receiver_id']
        amount = row['amount']
        risk_score = row.get('risk_score', 0)
        is_suspicious = row.get('is_suspicious', False)
        
        # Add nodes if they don't exist
        if not G.has_node(sender):
            G.add_node(sender, node_type='entity', total_sent=0, total_received=0, risk_score=0)
        if not G.has_node(receiver):
            G.add_node(receiver, node_type='entity', total_sent=0, total_received=0, risk_score=0)
        
        # Update node attributes
        G.nodes[sender]['total_sent'] += amount
        G.nodes[receiver]['total_received'] += amount
        
        # Add or update edge
        if G.has_edge(sender, receiver):
            G[sender][receiver]['weight'] += amount
            G[sender][receiver]['count'] += 1
            if is_suspicious:
                G[sender][receiver]['suspicious_count'] += 1
        else:
            G.add_edge(sender, receiver, 
                      weight=amount, 
                      count=1, 
                      suspicious_count=1 if is_suspicious else 0)
    
    # Calculate node risk scores based on connected transactions
    for node in G.nodes():
        # Get all transactions involving this node
        node_txns = df[(df['sender_id'] == node) | (df['receiver_id'] == node)]
        if len(node_txns) > 0:
            G.nodes[node]['risk_score'] = node_txns['risk_score'].mean()
        else:
            G.nodes[node]['risk_score'] = 0
    
    return G

def detect_circular_flows(G, max_cycle_length=5):
    """
    Detect circular money flows (potential layering schemes)
    
    Args:
        G: Transaction graph
        max_cycle_length: Maximum cycle length to detect
        
    Returns:
        list: List of detected cycles
    """
    cycles = []
    
    try:
        # Find simple cycles (limited to avoid performance issues)
        simple_cycles = list(nx.simple_cycles(G))
        
        # Filter cycles by length and analyze
        for cycle in simple_cycles:
            if len(cycle) <= max_cycle_length:
                # Calculate total flow in cycle
                total_flow = 0
                for i in range(len(cycle)):
                    sender = cycle[i]
                    receiver = cycle[(i + 1) % len(cycle)]
                    if G.has_edge(sender, receiver):
                        total_flow += G[sender][receiver]['weight']
                
                cycles.append({
                    'nodes': cycle,
                    'length': len(cycle),
                    'total_flow': total_flow,
                    'avg_flow': total_flow / len(cycle)
                })
        
        # Sort by total flow (descending)
        cycles.sort(key=lambda x: x['total_flow'], reverse=True)
        
    except Exception as e:
        print(f"Error detecting cycles: {e}")
    
    return cycles[:10]  # Return top 10 cycles

def identify_shell_entities(G, threshold_ratio=10):
    """
    Identify potential shell entities (high pass-through, low retention)
    
    Args:
        G: Transaction graph
        threshold_ratio: Ratio threshold for flagging
        
    Returns:
        list: List of potential shell entities
    """
    shell_entities = []
    
    for node in G.nodes():
        total_sent = G.nodes[node]['total_sent']
        total_received = G.nodes[node]['total_received']
        
        # Shell entities typically receive and send similar amounts quickly
        if total_received > 0 and total_sent > 0:
            # Calculate pass-through ratio
            pass_through_ratio = min(total_sent, total_received) / max(total_sent, total_received)
            
            # High pass-through (close to 1.0) suggests shell behavior
            if pass_through_ratio > 0.8:
                out_degree = G.out_degree(node)
                in_degree = G.in_degree(node)
                
                shell_entities.append({
                    'entity_id': node,
                    'total_received': total_received,
                    'total_sent': total_sent,
                    'pass_through_ratio': pass_through_ratio,
                    'in_degree': in_degree,
                    'out_degree': out_degree,
                    'risk_score': G.nodes[node]['risk_score']
                })
    
    # Sort by risk score
    shell_entities.sort(key=lambda x: x['risk_score'], reverse=True)
    
    return shell_entities

def get_high_risk_subgraph(G, risk_threshold=70):
    """
    Extract subgraph containing only high-risk nodes and edges
    
    Args:
        G: Full transaction graph
        risk_threshold: Minimum risk score for inclusion
        
    Returns:
        nx.DiGraph: Subgraph of high-risk entities
    """
    high_risk_nodes = [node for node in G.nodes() 
                       if G.nodes[node]['risk_score'] >= risk_threshold]
    
    subgraph = G.subgraph(high_risk_nodes).copy()
    
    return subgraph

def calculate_graph_metrics(G):
    """
    Calculate network metrics for the transaction graph
    
    Args:
        G: Transaction graph
        
    Returns:
        dict: Graph metrics
    """
    metrics = {
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_in_degree': sum(dict(G.in_degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        'avg_out_degree': sum(dict(G.out_degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
    }
    
    # Find most connected nodes
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    if in_degrees:
        metrics['most_received_entity'] = max(in_degrees, key=in_degrees.get)
        metrics['max_in_degree'] = in_degrees[metrics['most_received_entity']]
    
    if out_degrees:
        metrics['most_sending_entity'] = max(out_degrees, key=out_degrees.get)
        metrics['max_out_degree'] = out_degrees[metrics['most_sending_entity']]
    
    return metrics

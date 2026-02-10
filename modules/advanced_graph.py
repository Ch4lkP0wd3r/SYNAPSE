"""
Advanced Graph Analytics Module for SYNAPSE - Multi-Layer Network Intelligence
PageRank, community detection, centrality measures, and advanced pattern recognition
"""

import networkx as nx
import numpy as np
from collections import defaultdict

def calculate_advanced_centrality(G):
    """
    Calculate advanced centrality measures for network analysis
    
    Args:
        G: NetworkX graph
        
    Returns:
        dict: Centrality metrics for each node
    """
    centrality_metrics = {}
    
    try:
        # PageRank - identifies influential nodes
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
        
        # Betweenness centrality - identifies bridge nodes
        betweenness = nx.betweenness_centrality(G)
        
        # Closeness centrality - identifies well-connected nodes
        closeness = nx.closeness_centrality(G)
        
        # Degree centrality
        degree_centrality = nx.degree_centrality(G)
        
        # Combine metrics
        for node in G.nodes():
            centrality_metrics[node] = {
                'pagerank': pagerank.get(node, 0),
                'betweenness': betweenness.get(node, 0),
                'closeness': closeness.get(node, 0),
                'degree_centrality': degree_centrality.get(node, 0),
                'influence_score': (
                    pagerank.get(node, 0) * 40 +
                    betweenness.get(node, 0) * 30 +
                    closeness.get(node, 0) * 20 +
                    degree_centrality.get(node, 0) * 10
                )
            }
    except Exception as e:
        print(f"Error calculating centrality: {e}")
    
    return centrality_metrics

def detect_communities(G):
    """
    Detect communities/clusters in the transaction network
    
    Args:
        G: NetworkX graph
        
    Returns:
        dict: Community assignments and metrics
    """
    try:
        # Convert to undirected for community detection
        G_undirected = G.to_undirected()
        
        # Use Louvain method for community detection
        from networkx.algorithms import community
        
        communities = community.greedy_modularity_communities(G_undirected)
        
        # Assign community IDs
        community_map = {}
        community_stats = []
        
        for idx, comm in enumerate(communities):
            comm_nodes = list(comm)
            
            # Calculate community statistics
            subgraph = G.subgraph(comm_nodes)
            total_flow = sum(G[u][v]['weight'] for u, v in subgraph.edges() if 'weight' in G[u][v])
            
            community_stats.append({
                'community_id': idx,
                'size': len(comm_nodes),
                'nodes': comm_nodes[:10],  # Top 10 nodes
                'total_flow': total_flow,
                'density': nx.density(subgraph),
                'avg_degree': sum(dict(subgraph.degree()).values()) / len(comm_nodes) if len(comm_nodes) > 0 else 0
            })
            
            for node in comm_nodes:
                community_map[node] = idx
        
        # Sort by total flow
        community_stats.sort(key=lambda x: x['total_flow'], reverse=True)
        
        return {
            'num_communities': len(communities),
            'community_map': community_map,
            'community_stats': community_stats
        }
    
    except Exception as e:
        print(f"Error detecting communities: {e}")
        return {
            'num_communities': 0,
            'community_map': {},
            'community_stats': []
        }

def identify_key_players(G, centrality_metrics):
    """
    Identify key players in the network (multi-layer targeting)
    
    Args:
        G: NetworkX graph
        centrality_metrics: Centrality scores
        
    Returns:
        list: Key players ranked by importance
    """
    key_players = []
    
    for node in G.nodes():
        if node in centrality_metrics:
            metrics = centrality_metrics[node]
            
            key_players.append({
                'entity_id': node,
                'influence_score': metrics['influence_score'],
                'pagerank': metrics['pagerank'],
                'betweenness': metrics['betweenness'],
                'total_sent': G.nodes[node].get('total_sent', 0),
                'total_received': G.nodes[node].get('total_received', 0),
                'risk_score': G.nodes[node].get('risk_score', 0),
                'connections': G.degree(node),
                'role': classify_node_role(G, node, metrics)
            })
    
    # Sort by influence score
    key_players.sort(key=lambda x: x['influence_score'], reverse=True)
    
    return key_players[:25]  # Top 25 key players

def classify_node_role(G, node, metrics):
    """
    Classify the role of a node in the network (Gen 4 Enhanced)
    
    Args:
        G: NetworkX graph
        node: Node ID
        metrics: Centrality metrics
        
    Returns:
        str: Role classification
    """
    in_degree = G.in_degree(node)
    out_degree = G.out_degree(node)
    betweenness = metrics['betweenness']
    
    # Fan-In: High in-degree from unique sources
    # Fan-Out: High out-degree to unique destinations
    
    if in_degree > 15 and out_degree < 5:
        return 'COLLECTOR_NODE' # Fan-In
    elif out_degree > 15 and in_degree < 5:
        return 'DISPERSION_NODE' # Fan-Out
    elif betweenness > 0.15:
        return 'STRATEGIC_BROKER'
    elif in_degree > 10 and out_degree > 10:
        return 'NETWORK_HUB'
    elif out_degree > 5 and in_degree < 2:
        return 'SMURFER_SOURCE' # Likely starting point of smurfing
    else:
        return 'PARTICIPANT'

def detect_fan_patterns(G, threshold=10):
    """
    Detect Fan-In and Fan-Out patterns (Gen 4)
    
    Args:
        G: NetworkX graph
        threshold: Degree threshold for "Fan" classification
        
    Returns:
        list: Detected fan patterns
    """
    fan_patterns = []
    
    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        
        if in_deg >= threshold:
            fan_patterns.append({
                'type': 'FAN_IN (Collector)',
                'entity_id': node,
                'degree': in_deg,
                'total_volume': G.nodes[node].get('total_received', 0),
                'suspicion_score': min(100, in_deg * 5)
            })
            
        if out_deg >= threshold:
            fan_patterns.append({
                'type': 'FAN_OUT (Distributor)',
                'entity_id': node,
                'degree': out_deg,
                'total_volume': G.nodes[node].get('total_sent', 0),
                'suspicion_score': min(100, out_deg * 5)
            })
            
    return sorted(fan_patterns, key=lambda x: x['suspicion_score'], reverse=True)

def detect_layering_schemes(G, max_depth=5):
    """
    Detect layering schemes (complex multi-hop transactions)
    
    Args:
        G: NetworkX graph
        max_depth: Maximum path length to analyze
        
    Returns:
        list: Detected layering schemes
    """
    layering_schemes = []
    
    try:
        # Find all simple paths up to max_depth
        for source in G.nodes():
            for target in G.nodes():
                if source != target:
                    try:
                        paths = list(nx.all_simple_paths(G, source, target, cutoff=max_depth))
                        
                        for path in paths:
                            if len(path) >= 3:  # At least 3 hops
                                # Calculate total flow through path
                                total_flow = 0
                                for i in range(len(path) - 1):
                                    if G.has_edge(path[i], path[i+1]):
                                        total_flow += G[path[i]][path[i+1]].get('weight', 0)
                                
                                if total_flow > 10000:  # Significant amount
                                    layering_schemes.append({
                                        'path': path,
                                        'length': len(path),
                                        'total_flow': total_flow,
                                        'avg_hop': total_flow / (len(path) - 1),
                                        'suspicion_score': min(100, len(path) * 15 + np.log1p(total_flow) / 10)
                                    })
                    except:
                        continue  # Skip if too many paths
        
        # Remove duplicates and sort
        layering_schemes.sort(key=lambda x: x['suspicion_score'], reverse=True)
        
        return layering_schemes[:15]  # Top 15 schemes
    
    except Exception as e:
        print(f"Error detecting layering: {e}")
        return []

def calculate_network_resilience(G):
    """
    Calculate network resilience metrics
    
    Args:
        G: NetworkX graph
        
    Returns:
        dict: Resilience metrics
    """
    try:
        # Calculate various resilience metrics
        resilience = {
            'num_components': nx.number_weakly_connected_components(G),
            'largest_component_size': len(max(nx.weakly_connected_components(G), key=len)),
            'average_clustering': nx.average_clustering(G.to_undirected()),
            'diameter': 0,  # Will calculate if possible
        }
        
        # Try to calculate diameter (can be expensive)
        try:
            largest_cc = max(nx.weakly_connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            resilience['diameter'] = nx.diameter(subgraph.to_undirected())
        except:
            resilience['diameter'] = 0
        
        return resilience
    
    except Exception as e:
        print(f"Error calculating resilience: {e}")
        return {}

def find_suspicious_subgraphs(G, min_size=3, max_size=10):
    """
    Find suspicious subgraphs (tightly connected groups)
    
    Args:
        G: NetworkX graph
        min_size: Minimum subgraph size
        max_size: Maximum subgraph size
        
    Returns:
        list: Suspicious subgraphs
    """
    suspicious_subgraphs = []
    
    try:
        # Find k-cores (densely connected subgraphs)
        for k in range(2, 6):
            try:
                k_core = nx.k_core(G.to_undirected(), k)
                
                if len(k_core) >= min_size and len(k_core) <= max_size:
                    # Calculate metrics
                    total_flow = sum(G[u][v].get('weight', 0) for u, v in k_core.edges() if G.has_edge(u, v))
                    avg_risk = np.mean([G.nodes[node].get('risk_score', 0) for node in k_core.nodes()])
                    
                    suspicious_subgraphs.append({
                        'type': f'{k}-CORE',
                        'nodes': list(k_core.nodes()),
                        'size': len(k_core),
                        'edges': k_core.number_of_edges(),
                        'total_flow': total_flow,
                        'avg_risk_score': avg_risk,
                        'density': nx.density(k_core),
                        'suspicion_score': min(100, k * 20 + avg_risk / 2)
                    })
            except:
                continue
        
        # Sort by suspicion score
        suspicious_subgraphs.sort(key=lambda x: x['suspicion_score'], reverse=True)
        
        return suspicious_subgraphs[:10]  # Top 10
    
    except Exception as e:
        print(f"Error finding subgraphs: {e}")
        return []
